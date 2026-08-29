"""Protocol-matched Thompson-Whittle-CV baseline for the frozen TMC study.

This is a single-agent adaptation of Thompson-sampling Whittle learning,
motivated by Tong et al., IEEE TMC 2024 (FedTSWI).  It is not a reproduction:
the source paper learns finite-state Markov transitions across federated
agents, whereas this experiment samples the two physical CV moments from a
normal approximation to the selected-only online estimator.

Protocol integrity:
* episode length is selected only on disjoint pilot seeds 309901--309912;
* the selected length is then frozen on confirmatory seeds 310001--310030;
* each policy reuses the exact make_problem construction, latent instances,
  observation-indexed banks, initial ages, and true-model oracle used by
  TMC_SYNTHETIC_PROTOCOL_FROZEN_v1;
* the posterior draw stream is generated from an independent, recorded seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.online_cv_moments_stable import OnlineCVMomentEstimator
from core.sim import W_from_pack, coeff_pack, poly_cost, topn_mask
from run_cv_piecewise_pilot import bank_observation, make_problem


PROTOCOL = "TMC_SYNTHETIC_PROTOCOL_FROZEN_v1_TS_ADDENDUM"
PILOT_SEEDS = list(range(309901, 309913))
FORMAL_SEEDS = list(range(310001, 310031))
EPISODE_CANDIDATES = (1, 4, 16, 64)
B, K, N, H, N0, CHANGE_T, BLOCK_LENGTH = 3, 20, 4, 1000, 8, 500, 64


def _effective_from_physical(physical):
    from run_cv_piecewise_pilot import effective_from_physical

    theta, c0 = effective_from_physical(physical)
    pack = coeff_pack(1.0, theta)
    pack[..., 0] = c0
    return pack


def run_true(theta0, theta1, c00, c01, ages0):
    ages = ages0.copy()
    total = np.zeros(B)
    post_total = np.zeros(B)
    for t in range(H):
        theta = theta1 if t >= CHANGE_T else theta0
        c0 = c01 if t >= CHANGE_T else c00
        pack = coeff_pack(1.0, theta)
        pack[..., 0] = c0
        slot = poly_cost(ages, pack).sum(axis=1)
        total += slot
        if t >= CHANGE_T:
            post_total += slot
        selected = topn_mask(W_from_pack(ages, pack), N)
        ages = np.where(selected, 1.0, ages + 1.0)
    return total / H, post_total / (H - CHANGE_T)


def run_ts(problem, episode_length, draw_seed):
    (
        _physical0,
        _physical1,
        theta0,
        theta1,
        c00,
        c01,
        _changed,
        pre_bank,
        post_bank,
        ages0,
    ) = problem
    initial = np.transpose(pre_bank[:, :, :N0, :], (2, 0, 1, 3))
    estimator = OnlineCVMomentEstimator(
        initial,
        confidence_scale=3.0,
        variance_floor=0.01,
        variance_ceiling=1.0,
    )
    rng = np.random.default_rng(draw_seed)
    ages = ages0.copy()
    pre_seen = np.full((B, K), N0, dtype=int)
    post_seen = np.zeros((B, K), dtype=int)
    total = np.zeros(B)
    post_total = np.zeros(B)
    rank_loss = np.zeros(B)
    sampled_pack = None

    for t in range(H):
        post = t >= CHANGE_T
        theta = theta1 if post else theta0
        c0 = c01 if post else c00
        true_pack = coeff_pack(1.0, theta)
        true_pack[..., 0] = c0
        slot = poly_cost(ages, true_pack).sum(axis=1)
        total += slot
        if post:
            post_total += slot

        if t % episode_length == 0 or sampled_pack is None:
            n = estimator.count[..., None]
            sample_var = estimator.running_m2 / np.maximum(n - 1.0, 1.0)
            standard_error = np.sqrt(sample_var / n)
            physical_draw = np.clip(
                estimator.physical_mean_raw
                + rng.normal(size=estimator.physical_mean_raw.shape)
                * standard_error,
                estimator.variance_floor,
                estimator.variance_ceiling,
            )
            sampled_pack = _effective_from_physical(physical_draw)

        selected = topn_mask(W_from_pack(ages, sampled_pack), N)
        true_score = W_from_pack(ages, true_pack)
        true_selected = topn_mask(true_score, N)
        rank_loss += (
            np.where(true_selected, true_score, 0.0).sum(axis=1)
            - np.where(selected, true_score, 0.0).sum(axis=1)
        )

        if post:
            observation = bank_observation(post_bank, post_seen, selected)
            post_seen += selected
        else:
            observation = bank_observation(pre_bank, pre_seen, selected)
            pre_seen += selected
        estimator.update(selected, observation)
        ages = np.where(selected, 1.0, ages + 1.0)

    oracle_total, oracle_post = run_true(
        theta0, theta1, c00, c01, ages0
    )
    return {
        "total_ex": 100.0 * (total / H / oracle_total - 1.0),
        "post_ex": 100.0
        * (post_total / (H - CHANGE_T) / oracle_post - 1.0),
        "rank_loss": rank_loss / H,
    }


def evaluate(seeds, episode_length, draw_seed_base):
    rows = []
    for seed in seeds:
        problem = make_problem(
            seed, B, K, CHANGE_T, H, N0, BLOCK_LENGTH
        )
        row = run_ts(
            problem,
            episode_length,
            draw_seed_base + 1009 * seed + episode_length,
        )
        for batch in range(B):
            rows.append(
                {
                    "seed": seed,
                    "batch": batch,
                    "episode_length": episode_length,
                    "total_ex": float(row["total_ex"][batch]),
                    "post_ex": float(row["post_ex"][batch]),
                    "rank_loss": float(row["rank_loss"][batch]),
                }
            )
    return rows


def seed_means(rows, key):
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row[key])
    return np.array(
        [np.mean(by_seed[seed]) for seed in sorted(by_seed)], dtype=float
    )


def bootstrap_ci(values, rng, replicates=100000):
    draw = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draw].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tmc_ts_baseline_expansion.json"),
    )
    args = parser.parse_args()
    pilot = {}
    for episode_length in EPISODE_CANDIDATES:
        rows = evaluate(PILOT_SEEDS, episode_length, 20267280)
        pilot[str(episode_length)] = {
            "mean_post_ex": float(seed_means(rows, "post_ex").mean()),
            "mean_total_ex": float(seed_means(rows, "total_ex").mean()),
        }
    selected = min(
        EPISODE_CANDIDATES,
        key=lambda length: (
            pilot[str(length)]["mean_post_ex"],
            pilot[str(length)]["mean_total_ex"],
            length,
        ),
    )
    formal_rows = evaluate(FORMAL_SEEDS, selected, 20267281)
    rng = np.random.default_rng(20267282)
    summary = {}
    for key in ("total_ex", "post_ex", "rank_loss"):
        values = seed_means(formal_rows, key)
        summary[key] = {
            "mean": float(values.mean()),
            "ci95": bootstrap_ci(values, rng),
            "seed_values": [float(x) for x in values],
        }
    payload = {
        "protocol": PROTOCOL,
        "source_baseline": {
            "name": "TS-Whittle-CV",
            "inspiration": "Tong et al., IEEE TMC 2024, FedTSWI",
            "reproduction_status": "matched single-agent physical-CV adaptation",
            "difference": (
                "Samples selected-only CV physical moments rather than "
                "federated finite-state transition kernels."
            ),
        },
        "pilot_seeds": PILOT_SEEDS,
        "formal_seeds": FORMAL_SEEDS,
        "episode_candidates": list(EPISODE_CANDIDATES),
        "pilot": pilot,
        "selected_episode_length": selected,
        "formal_rows": formal_rows,
        "formal_summary": summary,
        "settings": {
            "batches_per_seed": B,
            "K": K,
            "N": N,
            "H": H,
            "n0": N0,
            "change_t": CHANGE_T,
            "block_length": BLOCK_LENGTH,
            "draw_seed_base_pilot": 20267280,
            "draw_seed_base_formal": 20267281,
            "bootstrap_seed": 20267282,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"selected_episode_length={selected}")
    for length in EPISODE_CANDIDATES:
        print(f"pilot L={length}: {pilot[str(length)]}")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
