"""Frozen M3ED Falcon trace-driven scheduling replay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.m3ed_pose import load_all_falcon_pose
from run_uzh_trace_replay import (
    METHODS,
    bootstrap_mean_ci,
    make_episode,
    run_method,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/m3ed_falcon_pose"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/traces/m3ed_trace_replay_v1.json"),
    )
    args = parser.parse_args()
    trace_bank = load_all_falcon_pose(args.data_root)
    seeds = list(range(420001, 420031))
    raw = []
    for seed in seeds:
        physical, names = make_episode(trace_bank, seed)
        rows = {name: run_method(name, physical) for name in METHODS}
        oracle = rows["oracle"]["cost"]
        for name in METHODS:
            rows[name]["excess_pct"] = 100.0 * (rows[name]["cost"] / oracle - 1.0)
        raw.append({"seed": seed, "sequences": names, "methods": rows})

    rng = np.random.default_rng(920247)
    summary = {}
    for name in METHODS:
        summary[name] = {}
        for metric in (
            "excess_pct",
            "rank_loss",
            "max_gap",
            "alarms_per_10k_arm_slots",
        ):
            values = [episode["methods"][name][metric] for episode in raw]
            summary[name][metric] = bootstrap_mean_ci(values, rng)

    natural = np.array(
        [row["methods"]["sco_reset_ucb"]["excess_pct"] for row in raw]
    )
    forced = np.array(
        [row["methods"]["forced_reset_ucb"]["excess_pct"] for row in raw]
    )
    output = {
        "protocol": {
            "dataset": "M3ED Falcon public pose ground truth",
            "seeds": seeds,
            "K": 12,
            "N": 3,
            "length": 640,
            "n0": 8,
            "H": 632,
            "methods": METHODS,
        },
        "summary_mean_ci95": summary,
        "paired_natural_minus_forced_excess_pct": bootstrap_mean_ci(
            natural - forced, rng
        ),
        "episodes": raw,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
