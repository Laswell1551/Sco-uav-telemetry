"""Run an arbitrary frozen confirmatory seed range and emit machine-readable JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_cv_piecewise_pilot import make_problem, run_policy


METHODS = [
    "cumulative_ce",
    "cumulative_ucb_cv",
    "sw_ce_32",
    "sw_ucb_cv_64",
    "sco_reset_ce",
    "sco_reset_ucb",
    "ps_forced_reset_ucb",
]


def run_range(seed_start, seed_end, batches_per_seed=3):
    seeds = list(range(seed_start, seed_end + 1))
    B, K, N, H, n0, change_t, block_length = (
        batches_per_seed,
        20,
        4,
        1000,
        8,
        500,
        64,
    )
    records = {
        name: {
            "total_ex": [],
            "post_ex": [],
            "rank_loss": [],
            "detection": [],
            "calendar_delay": [],
            "observation_delay": [],
            "pre_fa_per_10k": [],
            "post_unchanged_alarms": [],
        }
        for name in METHODS
    }
    seed_status = []

    for seed in seeds:
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
        for name in METHODS:
            row = run_policy(
                name,
                *common,
                detector_window=8,
                detector_threshold=5.0,
                explore_period=50,
            )
            records[name]["total_ex"].extend(
                (100.0 * (row["avg_cost"] / oracle["avg_cost"] - 1.0)).tolist()
            )
            records[name]["post_ex"].extend(
                (
                    100.0
                    * (row["post_cost"] / oracle["post_cost"] - 1.0)
                ).tolist()
            )
            records[name]["rank_loss"].append(float(row["rank_loss"].mean()))
            if name.startswith("sco_reset_") or name.startswith("ps_forced_"):
                records[name]["detection"].append(
                    float(row["detected_fraction"])
                )
                records[name]["calendar_delay"].append(
                    float(row["calendar_delay"])
                )
                records[name]["observation_delay"].append(
                    float(row["observation_delay"])
                )
                records[name]["pre_fa_per_10k"].append(
                    10000.0 * row["pre_alarms"] / (B * K * change_t)
                )
                records[name]["post_unchanged_alarms"].append(
                    int(row["post_unchanged_alarms"])
                )
        seed_status.append({"seed": seed, "changed_fraction": float(changed.mean())})

    return {
        "protocol": "TMC_SYNTHETIC_PROTOCOL_FROZEN_v1",
        "seed_start": seed_start,
        "seed_end": seed_end,
        "seeds": seeds,
        "batches_per_seed": B,
        "K": K,
        "N": N,
        "H": H,
        "change_t": change_t,
        "n0": n0,
        "block_length": block_length,
        "seed_status": seed_status,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.seed_end < args.seed_start:
        raise SystemExit("seed-end must be at least seed-start")
    result = run_range(args.seed_start, args.seed_end, args.batches)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"completed seeds {args.seed_start}-{args.seed_end}; "
        f"json={args.json_out}"
    )
    for name, row in result["records"].items():
        print(
            f"{name:24s} total={np.mean(row['total_ex']):7.3f}% "
            f"post={np.mean(row['post_ex']):7.3f}%"
        )


if __name__ == "__main__":
    main()
