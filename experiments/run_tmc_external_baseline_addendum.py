"""Paired frozen-seed addendum for newly admitted low-information baselines.

The original outcomes on seeds 310001--310030 had already been inspected
before this addendum was designed.  Results must therefore be reported as a
retrospective paired addendum, not as preregistered confirmatory evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_cv_piecewise_pilot import make_problem, run_policy


def cluster_bootstrap_ci(values, rng, replicates=100_000):
    values = np.asarray(values, dtype=float)
    means = np.empty(replicates)
    batch = 5_000
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        draw = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[draw].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=310001)
    parser.add_argument("--seed-end", type=int, default=310030)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/tmc_external_baseline_addendum_v1.json"),
    )
    args = parser.parse_args()
    if args.seed_end < args.seed_start:
        raise SystemExit("seed-end must be at least seed-start")

    B, K, N, H, n0, change_t, block_length = (
        args.batches,
        20,
        4,
        1000,
        8,
        500,
        64,
    )
    per_seed = []
    for seed in range(args.seed_start, args.seed_end + 1):
        (
            _,
            _,
            theta0,
            theta1,
            c00,
            c01,
            changed,
            pre_bank,
            post_bank,
            ages0,
        ) = make_problem(seed, B, K, change_t, H, n0, block_length)
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
            n0,
            change_t,
            H,
        )
        oracle = run_policy("true", *common)
        anchor = run_policy("max_age", *common)
        total = 100.0 * (anchor["avg_cost"] / oracle["avg_cost"] - 1.0)
        post = 100.0 * (anchor["post_cost"] / oracle["post_cost"] - 1.0)
        per_seed.append(
            {
                "seed": seed,
                "total_ex_instances": total.tolist(),
                "post_ex_instances": post.tolist(),
                "total_ex_seed_mean": float(total.mean()),
                "post_ex_seed_mean": float(post.mean()),
                "rank_loss_seed_mean": float(anchor["rank_loss"].mean()),
                "changed_fraction": float(changed.mean()),
            }
        )
        print(
            f"seed={seed} max_age total={total.mean():.3f}% "
            f"post={post.mean():.3f}%"
        )

    total_seed = np.array([row["total_ex_seed_mean"] for row in per_seed])
    post_seed = np.array([row["post_ex_seed_mean"] for row in per_seed])
    rng = np.random.default_rng(310199)
    payload = {
        "protocol": "TMC_EXTERNAL_BASELINE_ADDENDUM_v1",
        "evidence_status": "retrospective_frozen_seed_addendum",
        "reason": (
            "The baseline protocol was frozen after the original outcomes on "
            "these evaluation seeds had been inspected."
        ),
        "baseline": "max_age",
        "information": "public age state only",
        "seeds": list(range(args.seed_start, args.seed_end + 1)),
        "batches_per_seed": B,
        "K": K,
        "N": N,
        "H": H,
        "change_t": change_t,
        "n0": n0,
        "block_length": block_length,
        "per_seed": per_seed,
        "summary": {
            "total_ex_mean": float(total_seed.mean()),
            "total_ex_cluster_ci": cluster_bootstrap_ci(total_seed, rng),
            "post_ex_mean": float(post_seed.mean()),
            "post_ex_cluster_ci": cluster_bootstrap_ci(post_seed, rng),
            "rank_loss_mean": float(
                np.mean([row["rank_loss_seed_mean"] for row in per_seed])
            ),
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"saved={args.json_out}")


if __name__ == "__main__":
    main()
