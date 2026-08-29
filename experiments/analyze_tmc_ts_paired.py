"""Pair the retrospective TS-Whittle-CV addendum with frozen SCO seeds.

The TS addendum already stores one value per seed after averaging the three
batch instances.  This script deterministically replays SCO-reset-UCB on the
same latent problems and observation banks, then reports paired TS-minus-SCO
differences.  Positive differences favor SCO because lower cost is better.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_cv_piecewise_pilot import make_problem, run_policy


B, K, N, H, N0, CHANGE_T, BLOCK_LENGTH = 3, 20, 4, 1000, 8, 500, 64
EXPECTED_SEEDS = list(range(310001, 310031))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, reps=100000):
    draw = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[draw].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def paired_signflip_p(values: np.ndarray, rng: np.random.Generator, reps=200000):
    observed = abs(float(values.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (reps + 1))


def replay_sco():
    rows = []
    for seed in EXPECTED_SEEDS:
        (
            _physical0,
            _physical1,
            theta0,
            theta1,
            c00,
            c01,
            changed,
            pre_bank,
            post_bank,
            ages0,
        ) = make_problem(seed, B, K, CHANGE_T, H, N0, BLOCK_LENGTH)
        common = (
            theta0,
            theta1,
            c00,
            c01,
            changed,
            pre_bank,
            post_bank,
            ages0,
            N,
            N0,
            CHANGE_T,
            H,
        )
        oracle = run_policy("true", *common)
        sco = run_policy(
            "sco_reset_ucb",
            *common,
            detector_window=8,
            detector_threshold=5.0,
            explore_period=50,
        )
        total = 100.0 * (sco["avg_cost"] / oracle["avg_cost"] - 1.0)
        post = 100.0 * (sco["post_cost"] / oracle["post_cost"] - 1.0)
        rows.append(
            {
                "seed": seed,
                "total_ex_seed_mean": float(np.mean(total)),
                "post_ex_seed_mean": float(np.mean(post)),
                "rank_loss_seed_mean": float(np.mean(sco["rank_loss"])),
            }
        )
        print(f"seed={seed} total={np.mean(total):.6f} post={np.mean(post):.6f}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ts-json",
        type=Path,
        default=Path("results/tmc_ts_baseline_expansion.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tmc_ts_paired_sco_addendum.json"),
    )
    args = parser.parse_args()
    ts = json.loads(args.ts_json.read_text(encoding="utf-8-sig"))
    if ts["formal_seeds"] != EXPECTED_SEEDS:
        raise AssertionError("TS seed set does not match the frozen protocol")
    sco_rows = replay_sco()
    sco_by_seed = {row["seed"]: row for row in sco_rows}
    rng = np.random.default_rng(20267283)
    comparisons = {}
    key_map = {
        "total_ex": "total_ex_seed_mean",
        "post_ex": "post_ex_seed_mean",
        "rank_loss": "rank_loss_seed_mean",
    }
    for key, sco_key in key_map.items():
        ts_values = np.asarray(ts["formal_summary"][key]["seed_values"], dtype=float)
        sco_values = np.asarray(
            [sco_by_seed[seed][sco_key] for seed in EXPECTED_SEEDS], dtype=float
        )
        diff = ts_values - sco_values
        comparisons[key] = {
            "direction": "TS-Whittle-CV minus SCO-reset-UCB; positive favors SCO",
            "ts_mean": float(ts_values.mean()),
            "sco_mean": float(sco_values.mean()),
            "paired_difference_mean": float(diff.mean()),
            "paired_difference_ci95": bootstrap_mean_ci(diff, rng),
            "paired_signflip_p": paired_signflip_p(diff, rng),
            "paired_seed_values": [float(x) for x in diff],
        }
    payload = {
        "protocol": "TMC_TS_VS_SCO_PAIRED_ADDENDUM_v1",
        "evidence_status": "retrospective_frozen_seed_addendum",
        "comparison_scope": (
            "Matched single-agent physical-CV TS adaptation; not a FedTSWI "
            "reproduction."
        ),
        "seeds": EXPECTED_SEEDS,
        "batches_per_seed": B,
        "sco_rows": sco_rows,
        "comparisons": comparisons,
        "bootstrap_seed": 20267283,
        "bootstrap_replicates": 100000,
        "signflip_replicates": 200000,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(comparisons, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
