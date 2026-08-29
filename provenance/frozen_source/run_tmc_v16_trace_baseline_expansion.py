"""Retrospective DE/DTS baseline addendum on frozen UZH/M3ED replays.

The runner reuses the exact public-trace construction and oracle used by
``run_uzh_trace_replay.py`` and ``run_m3ed_trace_replay.py``.  The two new
policies receive only the current observations of selected streams:

* DE-CD-Whittle-CV transfers the controlled-study diminishing-exploration
  schedule and resettable UCB-CV learner.
* DTS-Whittle-CV transfers the controlled-study discounted CV posterior and
  per-slot Thompson draw.

The alpha and gamma values are read from the disjoint-pilot choices recorded
in ``results/tmc_v16_baseline_expansion.json``.  Neither public-flight replay
is used for tuning.  Because the original replay outcomes were already
inspected, all results are explicitly retrospective matched addenda.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.change_detection_cv import (
    ResettableOnlineCVMomentEstimator,
    TwoWindowCVMomentDetector,
)
from core.m3ed_pose import load_all_falcon_pose
from core.sim import W_from_pack, coeff_pack, poly_cost, topn_mask
from core.uzh_fpv_replay_v2 import load_all_replay_traces
from run_tmc_v16_baseline_expansion import (
    DE_ALPHA_CANDIDATES,
    DTS_GAMMA_CANDIDATES,
    DiscountedCVMomentPosterior,
    DiminishingExplorationSchedule,
)
from run_uzh_trace_replay import (
    bootstrap_mean_ci,
    make_episode,
    physical_to_theta,
    run_method,
)


SCHEMA_VERSION = "TMC_V16_TRACE_BASELINE_EXPANSION_v1"
CONTROLLED_SCHEMA_VERSION = "TMC_V16_BASELINE_EXPANSION_v1"
METHODS = ("de_cd_whittle_cv", "dts_whittle_cv")

K = 12
N = 3
LENGTH = 640
N0 = 8
H = LENGTH - N0
DETECTOR_WINDOW = 8
DETECTOR_THRESHOLD = 5.0

UZH_SEEDS = tuple(range(410001, 410031))
M3ED_SEEDS = tuple(range(420001, 420031))
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEEDS = {"uzh_fpv": 916247, "m3ed_falcon": 926247}
POSTERIOR_STREAM_TAGS = {"uzh_fpv": 0x55A1, "m3ed_falcon": 0x33ED}


def _initial_observations(physical: np.ndarray, n0: int) -> np.ndarray:
    """Return the existing trace protocol's (R,B,K,2) initialization."""
    return np.transpose(
        physical[:, :n0, :], (1, 0, 2)
    )[:, None, :, :]


def _selected_observation(
    current: np.ndarray, selected: np.ndarray
) -> np.ndarray:
    """Mask current physical moments before any learner sees them."""
    observation = np.zeros_like(current)
    observation[selected] = current[selected]
    return observation


def _slot_truth(
    current: np.ndarray, ages: np.ndarray, budget: int
):
    """Evaluation-only true cost, index, and top-N action."""
    theta_true = physical_to_theta(current)
    coefficients = coeff_pack(1.0, theta_true)
    true_index = W_from_pack(ages, coefficients)
    oracle_mask = topn_mask(true_index, budget)
    slot_cost = float(poly_cost(ages, coefficients).sum())
    return slot_cost, true_index, oracle_mask


def _base_state(physical: np.ndarray, n0: int):
    arms, length, dimensions = physical.shape
    if dimensions != 2:
        raise ValueError("physical trace must have two CV moment coordinates")
    if not 1 <= n0 < length:
        raise ValueError("n0 must leave at least one evaluation slot")
    if not np.all(np.isfinite(physical)):
        raise ValueError("physical trace must be finite")
    if np.any(physical < 0.01) or np.any(physical > 1.0):
        raise ValueError("physical trace leaves the frozen [0.01,1] box")
    return arms, length - n0, np.ones((1, arms), dtype=float)


def run_de_trace_policy(
    physical: np.ndarray,
    alpha: float,
    budget: int = N,
    n0: int = N0,
):
    """Run DE-CD-Whittle-CV under the frozen selected-only trace interface."""
    arms, horizon, ages = _base_state(physical, n0)
    initial = _initial_observations(physical, n0)
    estimator = ResettableOnlineCVMomentEstimator(
        initial,
        confidence_scale=3.0,
        variance_floor=0.01,
        variance_ceiling=1.0,
    )
    detector = TwoWindowCVMomentDetector(
        1,
        arms,
        window=DETECTOR_WINDOW,
        threshold=DETECTOR_THRESHOLD,
    )
    all_selected = np.ones((1, arms), dtype=bool)
    for initial_block in initial:
        detector.update(all_selected, initial_block)

    schedule = DiminishingExplorationSchedule(
        batches=1,
        arms=arms,
        budget=budget,
        alpha=alpha,
    )
    total_cost = 0.0
    rank_loss = 0.0
    alarms = 0
    forced_slots = 0
    selected_observation_blocks = 0
    current_gap = np.zeros(arms, dtype=int)
    max_gap = np.zeros(arms, dtype=int)
    started = time.perf_counter()

    for h in range(horizon):
        current = physical[:, n0 + h, :][None, :, :]
        slot_cost, true_index, oracle_mask = _slot_truth(
            current, ages, budget
        )
        total_cost += slot_cost

        _, optimistic_theta = estimator.effective_box
        base_selected = topn_mask(
            W_from_pack(
                ages, coeff_pack(1.0, optimistic_theta)
            ),
            budget,
        )
        forced_mask = schedule.mask(h + 1)
        if forced_mask.any():
            selected = forced_mask
            forced_slots += 1
        else:
            selected = base_selected

        rank_loss += float(
            np.where(oracle_mask, true_index, 0.0).sum()
            - np.where(selected, true_index, 0.0).sum()
        )
        observation = _selected_observation(current, selected)
        detection = detector.update(selected, observation)
        alarm_mask = detection["alarms"]
        alarms += int(alarm_mask.sum())
        estimator.update_and_reset(
            selected, observation, detection
        )
        schedule.reset(alarm_mask.any(axis=1), h + 1)

        selected_observation_blocks += int(selected.sum())
        current_gap = np.where(
            selected[0], 0, current_gap + 1
        )
        max_gap = np.maximum(max_gap, current_gap)
        ages = np.where(selected, 1.0, ages + 1.0)

    return {
        "method": "de_cd_whittle_cv",
        "cost": total_cost / horizon,
        "rank_loss": rank_loss / horizon,
        "max_gap": int(max_gap.max()),
        "alarms_per_10k_arm_slots": (
            10000.0 * alarms / (horizon * arms)
        ),
        "forced_slot_fraction": forced_slots / horizon,
        "exploration_blocks": int(
            schedule.blocks_started[0]
        ),
        "selected_observation_blocks": selected_observation_blocks,
        "runtime_seconds": time.perf_counter() - started,
    }


def trace_posterior_normal_bank(
    dataset: str,
    seed: int,
    horizon: int = H,
    arms: int = K,
) -> np.ndarray:
    """Dataset- and episode-specific DTS common random-number bank."""
    if dataset not in POSTERIOR_STREAM_TAGS:
        raise ValueError(f"unknown dataset key: {dataset}")
    sequence = np.random.SeedSequence(
        [
            int(seed),
            POSTERIOR_STREAM_TAGS[dataset],
            len(SCHEMA_VERSION),
        ]
    )
    rng = np.random.default_rng(sequence)
    return rng.standard_normal((horizon, 1, arms, 2))


def run_dts_trace_policy(
    physical: np.ndarray,
    gamma: float,
    posterior_standard_normals: np.ndarray,
    budget: int = N,
    n0: int = N0,
):
    """Run DTS-Whittle-CV under the frozen selected-only trace interface."""
    arms, horizon, ages = _base_state(physical, n0)
    normals = np.asarray(posterior_standard_normals, dtype=float)
    expected = (horizon, 1, arms, 2)
    if normals.shape != expected:
        raise ValueError(
            f"posterior_standard_normals must have shape {expected}"
        )
    posterior = DiscountedCVMomentPosterior(
        _initial_observations(physical, n0),
        gamma=gamma,
        variance_floor=0.01,
        variance_ceiling=1.0,
    )
    total_cost = 0.0
    rank_loss = 0.0
    selected_observation_blocks = 0
    current_gap = np.zeros(arms, dtype=int)
    max_gap = np.zeros(arms, dtype=int)
    started = time.perf_counter()

    for h in range(horizon):
        current = physical[:, n0 + h, :][None, :, :]
        slot_cost, true_index, oracle_mask = _slot_truth(
            current, ages, budget
        )
        total_cost += slot_cost

        physical_draw = posterior.draw(normals[h])
        theta_draw = physical_to_theta(physical_draw)
        selected = topn_mask(
            W_from_pack(
                ages, coeff_pack(1.0, theta_draw)
            ),
            budget,
        )
        rank_loss += float(
            np.where(oracle_mask, true_index, 0.0).sum()
            - np.where(selected, true_index, 0.0).sum()
        )
        observation = _selected_observation(current, selected)
        posterior.update(selected, observation)

        selected_observation_blocks += int(selected.sum())
        current_gap = np.where(
            selected[0], 0, current_gap + 1
        )
        max_gap = np.maximum(max_gap, current_gap)
        ages = np.where(selected, 1.0, ages + 1.0)

    clips = int(posterior.clip_count[0])
    draws = int(posterior.draw_count[0])
    return {
        "method": "dts_whittle_cv",
        "cost": total_cost / horizon,
        "rank_loss": rank_loss / horizon,
        "max_gap": int(max_gap.max()),
        "alarms_per_10k_arm_slots": 0.0,
        "posterior_clip_fraction": clips / draws,
        "final_effective_weight_mean": float(
            posterior.weight.mean()
        ),
        "selected_observation_blocks": selected_observation_blocks,
        "runtime_seconds": time.perf_counter() - started,
    }


def _summary_rng(
    dataset: str, method_index: int, metric_index: int
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                BOOTSTRAP_SEEDS[dataset],
                method_index,
                metric_index,
            ]
        )
    )


def summarize_episodes(
    dataset: str,
    episodes: list[dict],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    """Episode-bootstrap all metrics and paired excess differences."""
    if dataset not in BOOTSTRAP_SEEDS:
        raise ValueError(f"unknown dataset key: {dataset}")
    if not episodes:
        raise ValueError("at least one episode is required")

    metric_order = {
        "de_cd_whittle_cv": (
            "cost",
            "excess_pct",
            "rank_loss",
            "max_gap",
            "alarms_per_10k_arm_slots",
            "forced_slot_fraction",
            "exploration_blocks",
        ),
        "dts_whittle_cv": (
            "cost",
            "excess_pct",
            "rank_loss",
            "max_gap",
            "posterior_clip_fraction",
            "final_effective_weight_mean",
        ),
    }
    summaries = {}
    paired = {}
    for method_index, method in enumerate(METHODS):
        summaries[method] = {}
        for metric_index, metric in enumerate(
            metric_order[method]
        ):
            values = [
                float(episode["methods"][method][metric])
                for episode in episodes
            ]
            summaries[method][metric] = bootstrap_mean_ci(
                values,
                _summary_rng(
                    dataset, method_index, metric_index
                ),
                draws=bootstrap_replicates,
            )

        differences = [
            float(episode["methods"][method]["excess_pct"])
            - float(
                episode["references"]["sco_reset_ucb"][
                    "excess_pct"
                ]
            )
            for episode in episodes
        ]
        paired[method] = {
            "direction": (
                f"{method} minus SCO-reset-UCB excess-cost percentage "
                "points; positive favors SCO"
            ),
            "mean_ci95": bootstrap_mean_ci(
                differences,
                _summary_rng(dataset, method_index, 90),
                draws=bootstrap_replicates,
            ),
        }

    sco_metrics = {}
    for metric_index, metric in enumerate(
        (
            "cost",
            "excess_pct",
            "rank_loss",
            "max_gap",
            "alarms_per_10k_arm_slots",
        )
    ):
        values = [
            float(
                episode["references"]["sco_reset_ucb"][metric]
            )
            for episode in episodes
        ]
        sco_metrics[metric] = bootstrap_mean_ci(
            values,
            _summary_rng(dataset, 10, metric_index),
            draws=bootstrap_replicates,
        )

    return {
        "resampling_unit": (
            "paired computational replay episode; source flights are reused "
            "and are not 30 independent physical trials"
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "confidence_level": 0.95,
        "methods": summaries,
        "sco_reset_ucb_reference": sco_metrics,
        "paired_method_minus_sco": paired,
    }


def evaluate_dataset(
    dataset: str,
    trace_bank,
    seeds,
    de_alpha: float,
    dts_gamma: float,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict:
    """Run one frozen 30-episode public-flight dataset."""
    seeds = [int(seed) for seed in seeds]
    episodes = []
    started = time.perf_counter()

    for index, seed in enumerate(seeds, 1):
        physical, names = make_episode(
            trace_bank, seed, K=K, length=LENGTH
        )
        oracle = run_method(
            "oracle", physical, N=N, n0=N0
        )
        sco = run_method(
            "sco_reset_ucb", physical, N=N, n0=N0
        )
        sco["excess_pct"] = (
            100.0 * (sco["cost"] / oracle["cost"] - 1.0)
        )
        de = run_de_trace_policy(
            physical, alpha=de_alpha, budget=N, n0=N0
        )
        dts = run_dts_trace_policy(
            physical,
            gamma=dts_gamma,
            posterior_standard_normals=(
                trace_posterior_normal_bank(dataset, seed)
            ),
            budget=N,
            n0=N0,
        )
        methods = {
            "de_cd_whittle_cv": de,
            "dts_whittle_cv": dts,
        }
        for result in methods.values():
            result["excess_pct"] = (
                100.0
                * (result["cost"] / oracle["cost"] - 1.0)
            )
        episodes.append(
            {
                "seed": seed,
                "sequences": list(names),
                "oracle_cost": float(oracle["cost"]),
                "references": {
                    "sco_reset_ucb": sco,
                },
                "methods": methods,
            }
        )
        print(
            f"{dataset} {index:02d}/{len(seeds)} seed={seed} "
            f"DE={de['excess_pct']:.3f}% "
            f"DTS={dts['excess_pct']:.3f}%",
            flush=True,
        )

    return {
        "protocol": {
            "seeds": seeds,
            "episodes": len(seeds),
            "K": K,
            "N": N,
            "length": LENGTH,
            "n0": N0,
            "H": H,
            "episode_constructor": (
                "run_uzh_trace_replay.make_episode"
            ),
            "oracle": (
                "run_uzh_trace_replay.run_method('oracle')"
            ),
            "sco_reference": (
                "run_uzh_trace_replay.run_method('sco_reset_ucb')"
            ),
            "selected_only_update": True,
            "hyperparameters_transferred_without_trace_tuning": True,
        },
        "summary_mean_ci95": summarize_episodes(
            dataset,
            episodes,
            bootstrap_replicates=bootstrap_replicates,
        ),
        "episodes": episodes,
        "runtime_seconds": time.perf_counter() - started,
    }


def load_frozen_choices(path: Path) -> tuple[dict, dict]:
    """Read and validate alpha/gamma selected by the disjoint pilot."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != CONTROLLED_SCHEMA_VERSION:
        raise ValueError("unexpected controlled-baseline schema version")
    protocol = payload.get("protocol", {})
    if not protocol.get("formal_data_not_used_for_tuning"):
        raise ValueError("controlled pilot/evaluation separation is absent")
    choices = payload.get("pilot", {}).get("choices", {})
    alpha = float(choices["de_alpha"])
    gamma = float(choices["dts_gamma"])
    if alpha not in DE_ALPHA_CANDIDATES:
        raise ValueError("frozen alpha is outside the declared pilot grid")
    if gamma not in DTS_GAMMA_CANDIDATES:
        raise ValueError("frozen gamma is outside the declared pilot grid")
    return {"de_alpha": alpha, "dts_gamma": gamma}, payload


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_provenance(
    script_path: Path,
    controlled_path: Path,
    controlled_payload: dict,
) -> dict:
    root = script_path.resolve().parent
    sources = (
        "run_tmc_v16_baseline_expansion.py",
        "run_uzh_trace_replay.py",
        "run_m3ed_trace_replay.py",
        "core/uzh_fpv_replay_v2.py",
        "core/m3ed_pose.py",
    )
    controlled_provenance = controlled_payload.get(
        "provenance", {}
    )
    return {
        "retrospective_matched_addendum": True,
        "not_official_reproduction": True,
        "reason_retrospective": (
            "The original UZH-FPV and M3ED policy outcomes were inspected "
            "before these two baseline adapters were added."
        ),
        "algorithm_provenance": {
            "de_cd_whittle_cv": controlled_provenance.get(
                "de_cd_whittle_cv"
            ),
            "dts_whittle_cv": controlled_provenance.get(
                "dts_whittle_cv"
            ),
        },
        "hyperparameter_transfer": {
            "source": "results/tmc_v16_baseline_expansion.json",
            "source_sha256": sha256_file(controlled_path),
            "selection_seeds": list(range(309901, 309913)),
            "trace_results_used_for_selection": False,
        },
        "feedback_contract": {
            "initial_information": (
                "the same first eight per-arm physical-moment observations"
            ),
            "online_information": (
                "only selected streams' current physical moments; unselected "
                "entries are zeroed before detector/posterior update"
            ),
            "evaluation_only_information": (
                "the full current trace is used only for true cost, oracle "
                "action, and ranking-loss evaluation"
            ),
        },
        "paired_randomness": {
            "episodes_and_projections": (
                "the frozen 410001--410030 and 420001--420030 make_episode "
                "calls are unchanged"
            ),
            "dts_draws": (
                "dataset-specific NumPy SeedSequence keyed by episode seed"
            ),
        },
        "inference_boundary": (
            "Episode bootstrap describes replay variation; episodes reuse "
            "public source flights and do not represent independent trials."
        ),
        "source_file_sha256": {
            relative: sha256_file(root / relative)
            for relative in sources
        },
        "runner_sha256": sha256_file(script_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }


def build_payload(
    choices: dict,
    controlled_path: Path,
    controlled_payload: dict,
    datasets: dict,
    script_path: Path,
    runtime_seconds: float,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "retrospective_matched_trace_addendum",
        "frozen_hyperparameters": choices,
        "datasets": datasets,
        "provenance": build_provenance(
            script_path,
            controlled_path,
            controlled_payload,
        ),
        "runtime_seconds": runtime_seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controlled-json",
        type=Path,
        default=Path("results/tmc_v16_baseline_expansion.json"),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(
            "results/tmc_v16_trace_baseline_expansion.json"
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 1000:
        raise SystemExit("bootstrap-replicates must be at least 1000")

    choices, controlled_payload = load_frozen_choices(
        args.controlled_json
    )
    print(
        f"frozen alpha={choices['de_alpha']:g}, "
        f"gamma={choices['dts_gamma']:g}",
        flush=True,
    )
    started = time.perf_counter()
    uzh_bank = load_all_replay_traces(
        Path("data/uzh_fpv_gt")
    )
    m3ed_bank = load_all_falcon_pose(
        Path("data/m3ed_falcon_pose")
    )
    datasets = {
        "uzh_fpv": evaluate_dataset(
            "uzh_fpv",
            uzh_bank,
            UZH_SEEDS,
            choices["de_alpha"],
            choices["dts_gamma"],
            bootstrap_replicates=args.bootstrap_replicates,
        ),
        "m3ed_falcon": evaluate_dataset(
            "m3ed_falcon",
            m3ed_bank,
            M3ED_SEEDS,
            choices["de_alpha"],
            choices["dts_gamma"],
            bootstrap_replicates=args.bootstrap_replicates,
        ),
    }
    runtime_seconds = time.perf_counter() - started
    payload = build_payload(
        choices,
        args.controlled_json,
        controlled_payload,
        datasets,
        Path(__file__),
        runtime_seconds,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"json={args.json_out}", flush=True)
    for dataset, result in datasets.items():
        for method in METHODS:
            excess = result["summary_mean_ci95"]["methods"][
                method
            ]["excess_pct"]
            print(
                f"{dataset:12s} {method:22s} "
                f"{excess[0]:.3f}% "
                f"[{excess[1]:.3f},{excess[2]:.3f}]",
                flush=True,
            )
    print(f"runtime_seconds={runtime_seconds:.2f}", flush=True)


if __name__ == "__main__":
    main()
