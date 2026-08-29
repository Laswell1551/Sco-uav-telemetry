"""Reproduce the crossed capacity--delay recovery trajectories.

This runner extends the frozen one-dimensional in-flight scans to all twelve
capacity--delay cells.  Every policy in a seed reuses the same latent problem,
observation-indexed CV banks, and attempt-indexed channel uniforms.  Forty
genuine post-change checkpoints are retained; no interpolation is used.

The full run is compute intensive.  Fresh outputs belong in a timestamped
``runs/`` directory created by ``scripts/reproduce.py capacity-delay`` and
never overwrite ``results/frozen``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FROZEN = ROOT / "results" / "frozen"
for import_root in (ROOT, HERE):
    value = str(import_root)
    if value not in sys.path:
        sys.path.insert(0, value)

from core.change_detection_cv import TwoWindowCVMomentDetector  # noqa: E402
from core.sim import W_from_pack, coeff_pack, poly_cost, topn_mask  # noqa: E402
from run_cv_piecewise_pilot import (  # noqa: E402
    bank_observation,
    make_problem,
    round_robin_mask,
)
from run_tmc_channel_stress import make_estimator  # noqa: E402


FORMAL_SEEDS = tuple(20261000 + offset for offset in range(12))
CAPACITIES = (2, 4, 8)
DELAYS = (0, 1, 3, 5)
METHODS = (
    "cumulative_ucb_cv",
    "sco_reset_ucb",
    "ps_forced_reset_ucb",
    "inflight_sco_ucb",
)
ALL_METHODS = ("true",) + METHODS
CHECKPOINTS = tuple(range(10, 401, 10))
FROZEN_BETA = 16.0
B = 4
K = 20
H = 800
CHANGE_T = 400
N0 = 8
BLOCK_LENGTH = 64
SUCCESS_PROBABILITY = 0.9
OLD_RAW = FROZEN / "tmc_inflight_formal_raw.csv"
FROZEN_TRAJECTORY = FROZEN / "tmc_capacity_delay_trajectory_raw_v36.csv"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError(f"empty trajectory file: {path}")
    return rows


def mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    samples = np.asarray(list(values), dtype=float)
    if samples.size != len(FORMAL_SEEDS):
        raise AssertionError(
            f"expected {len(FORMAL_SEEDS)} independent seeds, got {samples.size}"
        )
    mean = float(samples.mean())
    half = 1.96 * float(samples.std(ddof=1)) / math.sqrt(samples.size)
    return mean, mean - half, mean + half


def run_policy_trajectory(
    name: str,
    theta0: np.ndarray,
    theta1: np.ndarray,
    c00: np.ndarray,
    c01: np.ndarray,
    pre_bank: np.ndarray,
    post_bank: np.ndarray,
    channel_uniform: np.ndarray,
    ages0: np.ndarray,
    capacity: int,
    delay: int,
) -> dict[str, object]:
    """Mirror the formal pipeline runner while retaining slot-level costs."""
    initial = np.transpose(pre_bank[:, :, :N0, :], (2, 0, 1, 3))
    estimator = make_estimator(name, initial)
    detector = None
    if name in ("sco_reset_ucb", "ps_forced_reset_ucb", "inflight_sco_ucb"):
        detector = TwoWindowCVMomentDetector(B, K, window=8, threshold=5.0)
        all_selected = np.ones((B, K), dtype=bool)
        for block in initial:
            detector.update(all_selected, block)

    ages = ages0.copy()
    pre_attempt = np.full((B, K), N0, dtype=int)
    post_attempt = np.zeros((B, K), dtype=int)
    channel_attempt = np.zeros((B, K), dtype=int)
    pending: list[list[tuple[np.ndarray, np.ndarray, bool, np.ndarray]]] = [
        [] for _ in range(H + delay + 1)
    ]
    inflight_count = np.zeros((B, K), dtype=int)
    exploration_index = 0
    post_slot_costs: list[np.ndarray] = []

    for t in range(H):
        post = t >= CHANGE_T
        theta = theta1 if post else theta0
        c0 = c01 if post else c00
        true_pack = coeff_pack(1.0, theta)
        true_pack[..., 0] = c0
        slot_cost = poly_cost(ages, true_pack).sum(axis=1)
        if post:
            post_slot_costs.append(slot_cost.copy())

        w_true = W_from_pack(ages, true_pack)
        forced = name == "ps_forced_reset_ucb" and t % 50 == 0
        if name == "true":
            selected = topn_mask(w_true, capacity)
        elif forced:
            selected = round_robin_mask(B, K, capacity, exploration_index)
            exploration_index += 1
        else:
            if estimator is None:
                raise AssertionError(f"missing estimator for {name}")
            _, theta_hi = estimator.effective_box
            score = W_from_pack(ages, coeff_pack(1.0, theta_hi))
            if name == "inflight_sco_ucb":
                score = score / (1.0 + FROZEN_BETA * inflight_count)
            selected = topn_mask(score, capacity)

        inflight_count += selected
        observation = (
            bank_observation(post_bank, post_attempt, selected)
            if post
            else bank_observation(pre_bank, pre_attempt, selected)
        )
        if post:
            post_attempt += selected
        else:
            pre_attempt += selected

        success = np.zeros((B, K), dtype=bool)
        for b, k in np.argwhere(selected):
            attempt = int(channel_attempt[b, k])
            if attempt >= channel_uniform.shape[2]:
                raise AssertionError("channel bank exhausted")
            success[b, k] = channel_uniform[b, k, attempt] < SUCCESS_PROBABILITY
            channel_attempt[b, k] += 1
        pending[t + delay].append((success, observation, post, selected))

        delivered = np.zeros((B, K), dtype=bool)
        for delivered_mask, delivered_observation, _, attempted_mask in pending[t]:
            inflight_count -= attempted_mask
            if np.any(inflight_count < 0):
                raise AssertionError("negative in-flight count")
            delivered |= delivered_mask
            if estimator is None or not np.any(delivered_mask):
                continue
            if detector is None:
                estimator.update(delivered_mask, delivered_observation)
            else:
                detection = detector.update(delivered_mask, delivered_observation)
                estimator.update_and_reset(
                    delivered_mask, delivered_observation, detection
                )

        ages = np.where(delivered, float(delay + 1), ages + 1.0)

    post_array = np.asarray(post_slot_costs, dtype=float)
    if post_array.shape != (H - CHANGE_T, B):
        raise AssertionError(f"unexpected trajectory shape: {post_array.shape}")
    cumulative = np.cumsum(post_array, axis=0).mean(axis=1)
    return {
        "cumulative_post_cost": cumulative,
        "post_cost": float(cumulative[-1] / (H - CHANGE_T)),
    }


def load_old_endpoints() -> dict[tuple[int, int, int, str], float]:
    endpoints: dict[tuple[int, int, int, str], float] = {}
    for row in read_csv(OLD_RAW):
        key = (
            int(row["seed"]),
            int(row["capacity"]),
            int(row["delay_slots"]),
            row["method"],
        )
        value = float(row["post_cost"])
        if key in endpoints and not math.isclose(
            endpoints[key], value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise AssertionError(f"conflicting frozen endpoint: {key}")
        endpoints[key] = value
    return endpoints


def simulate() -> list[dict[str, object]]:
    raw: list[dict[str, object]] = []
    for seed in FORMAL_SEEDS:
        (
            _,
            _,
            theta0,
            theta1,
            c00,
            c01,
            _,
            pre_bank,
            post_bank,
            ages0,
        ) = make_problem(seed, B, K, CHANGE_T, H, N0, BLOCK_LENGTH)
        channel_uniform = np.random.default_rng(seed + 50000).random(
            (B, K, H + N0)
        )
        for capacity in CAPACITIES:
            for delay in DELAYS:
                results = {
                    method: run_policy_trajectory(
                        method,
                        theta0,
                        theta1,
                        c00,
                        c01,
                        pre_bank,
                        post_bank,
                        channel_uniform,
                        ages0,
                        capacity,
                        delay,
                    )
                    for method in ALL_METHODS
                }
                reference = np.asarray(
                    results["true"]["cumulative_post_cost"], dtype=float
                )
                if delay == 0:
                    sco = np.asarray(
                        results["sco_reset_ucb"]["cumulative_post_cost"], dtype=float
                    )
                    pa_sco = np.asarray(
                        results["inflight_sco_ucb"]["cumulative_post_cost"],
                        dtype=float,
                    )
                    if not np.array_equal(sco, pa_sco):
                        raise AssertionError(
                            f"zero-delay reduction failed: seed={seed}, N={capacity}"
                        )
                for method in METHODS:
                    cumulative = np.asarray(
                        results[method]["cumulative_post_cost"], dtype=float
                    )
                    for checkpoint in CHECKPOINTS:
                        index = checkpoint - 1
                        raw.append(
                            {
                                "seed": seed,
                                "success_probability": SUCCESS_PROBABILITY,
                                "delay_slots": delay,
                                "capacity": capacity,
                                "capacity_ratio": capacity / K,
                                "method": method,
                                "post_change_slot": checkpoint,
                                "cumulative_post_cost": cumulative[index],
                                "reference_cumulative_post_cost": reference[index],
                                "cumulative_excess_pct": 100.0
                                * (cumulative[index] / reference[index] - 1.0),
                            }
                        )
                print(f"seed={seed} N/K={capacity / K:.0%} d={delay}", flush=True)
    return raw


def summarize(raw: list[dict[str, object] | dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, str, int], list[float]] = {}
    for row in raw:
        key = (
            int(row["capacity"]),
            int(row["delay_slots"]),
            str(row["method"]),
            int(row["post_change_slot"]),
        )
        groups.setdefault(key, []).append(float(row["cumulative_excess_pct"]))
    summary: list[dict[str, object]] = []
    for capacity in CAPACITIES:
        for delay in DELAYS:
            for method in METHODS:
                for checkpoint in CHECKPOINTS:
                    values = groups[(capacity, delay, method, checkpoint)]
                    mean, low, high = mean_ci(values)
                    summary.append(
                        {
                            "success_probability": SUCCESS_PROBABILITY,
                            "delay_slots": delay,
                            "capacity": capacity,
                            "capacity_ratio": capacity / K,
                            "method": method,
                            "post_change_slot": checkpoint,
                            "independent_seeds": len(values),
                            "cumulative_excess_pct_mean": mean,
                            "cumulative_excess_pct_ci_low": low,
                            "cumulative_excess_pct_ci_high": high,
                        }
                    )
    return summary


def audit_saved_raw(raw: list[dict[str, str]]) -> dict[str, object]:
    expected_rows = (
        len(FORMAL_SEEDS)
        * len(CAPACITIES)
        * len(DELAYS)
        * len(METHODS)
        * len(CHECKPOINTS)
    )
    if len(raw) != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows, found {len(raw)}")
    indexed: dict[tuple[int, int, int, str, int], dict[str, str]] = {}
    for row in raw:
        key = (
            int(row["seed"]),
            int(row["capacity"]),
            int(row["delay_slots"]),
            row["method"],
            int(row["post_change_slot"]),
        )
        if key in indexed:
            raise AssertionError(f"duplicate trajectory key: {key}")
        indexed[key] = row

    terminal = [row for row in raw if int(row["post_change_slot"]) == 400]
    policy_endpoint = {
        (
            int(row["seed"]),
            int(row["capacity"]),
            int(row["delay_slots"]),
            row["method"],
        ): float(row["cumulative_post_cost"]) / (H - CHANGE_T)
        for row in terminal
    }
    reference_endpoint: dict[tuple[int, int, int], float] = {}
    for row in terminal:
        key = (int(row["seed"]), int(row["capacity"]), int(row["delay_slots"]))
        value = float(row["reference_cumulative_post_cost"]) / (H - CHANGE_T)
        if key in reference_endpoint and not math.isclose(
            reference_endpoint[key], value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise AssertionError(f"inconsistent matched reference: {key}")
        reference_endpoint[key] = value

    errors: list[float] = []
    matches = 0
    for (seed, capacity, delay, method), frozen_value in load_old_endpoints().items():
        if method == "true":
            candidate = reference_endpoint[(seed, capacity, delay)]
        elif method in METHODS:
            candidate = policy_endpoint[(seed, capacity, delay, method)]
        else:
            continue
        errors.append(abs(candidate - frozen_value))
        matches += 1
    maximum_error = max(errors, default=0.0)
    if matches != 288 or maximum_error > 1e-9:
        raise AssertionError(
            f"endpoint audit failed: matches={matches}, max_error={maximum_error:.3e}"
        )

    zero_delay_checks = 0
    for seed in FORMAL_SEEDS:
        for capacity in CAPACITIES:
            for checkpoint in CHECKPOINTS:
                sco = float(
                    indexed[(seed, capacity, 0, "sco_reset_ucb", checkpoint)][
                        "cumulative_post_cost"
                    ]
                )
                pa_sco = float(
                    indexed[(seed, capacity, 0, "inflight_sco_ucb", checkpoint)][
                        "cumulative_post_cost"
                    ]
                )
                if sco != pa_sco:
                    raise AssertionError("saved zero-delay trajectory identity failed")
            zero_delay_checks += 1
    return {
        "rows": len(raw),
        "frozen_endpoint_matches": matches,
        "maximum_absolute_endpoint_error": maximum_error,
        "zero_delay_sco_pa_sco_exact_checks": zero_delay_checks,
        "zero_delay_identity": True,
    }


def protocol(audit: dict[str, object]) -> dict[str, object]:
    return {
        "audit": audit,
        "formal_seeds": list(FORMAL_SEEDS),
        "paired_batches_per_seed": B,
        "K": K,
        "capacities": list(CAPACITIES),
        "capacity_ratios": [capacity / K for capacity in CAPACITIES],
        "delays": list(DELAYS),
        "success_probability": SUCCESS_PROBABILITY,
        "H": H,
        "change_t": CHANGE_T,
        "post_change_slots": H - CHANGE_T,
        "checkpoints": list(CHECKPOINTS),
        "n0": N0,
        "block_length": BLOCK_LENGTH,
        "frozen_inflight_beta": FROZEN_BETA,
        "methods": list(METHODS),
        "reference": (
            "delay-unaware, immediate-reset true-model Whittle run over the "
            "same delayed channel; a matched reference, not a global optimum"
        ),
        "pairing": (
            "same latent problem, observation-indexed CV bank, and "
            "attempt-indexed channel uniforms within each formal seed"
        ),
        "no_interpolation": True,
        "no_synthetic_points": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--verify-frozen",
        action="store_true",
        help="audit the shipped trajectory rows without rerunning simulation",
    )
    args = parser.parse_args()
    if args.verify_frozen:
        audit = audit_saved_raw(read_csv(FROZEN_TRAJECTORY))
        print(json.dumps(protocol(audit), indent=2))
        return

    raw_path = args.out_dir / "tmc_capacity_delay_trajectory_raw_v36.csv"
    summary_path = args.out_dir / "tmc_capacity_delay_trajectory_summary_v36.csv"
    metadata_path = args.out_dir / "tmc_capacity_delay_trajectory_meta_v36.json"
    raw = simulate()
    write_csv(raw_path, raw)
    persisted = read_csv(raw_path)
    audit = audit_saved_raw(persisted)
    write_csv(summary_path, summarize(persisted))
    metadata_path.write_text(
        json.dumps(protocol(audit), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"raw": str(raw_path), "summary": str(summary_path), "metadata": str(metadata_path), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
