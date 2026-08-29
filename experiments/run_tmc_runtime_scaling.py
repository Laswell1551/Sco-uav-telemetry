"""Frozen online-path runtime microbenchmark for the TMC extension.

The benchmark excludes trace/block generation and measures only ``run_policy``
on pre-generated observation banks.  It is a systems microbenchmark, not a
new scheduling-quality experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_cv_piecewise_pilot import make_problem, run_policy  # noqa: E402


SEED = 2026072401
K_VALUES = [20, 40, 80, 160, 320]
METHODS = ["cumulative_ucb_cv", "sco_reset_ucb"]
REPETITIONS = 7
H = 400
N0 = 8
CHANGE_T = 200
BLOCK_LENGTH = 64


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "median_ms_per_slot": float(np.median(values)),
        "q1_ms_per_slot": float(np.percentile(values, 25)),
        "q3_ms_per_slot": float(np.percentile(values, 75)),
        "mean_ms_per_slot": float(np.mean(values)),
        "std_ms_per_slot": float(np.std(values, ddof=1)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/runtime"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    raw = []
    for k in K_VALUES:
        n = max(1, k // 5)
        problem = make_problem(
            SEED + k, 1, k, CHANGE_T, H, N0, BLOCK_LENGTH
        )
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
        ) = problem
        common = (
            theta0,
            theta1,
            c00,
            c01,
            changed,
            pre_bank,
            post_bank,
            ages0,
            n,
            N0,
            CHANGE_T,
            H,
        )
        for method in METHODS:
            # Untimed warm-up exercises NumPy dispatch and allocations.
            run_policy(
                method,
                *common,
                detector_window=8,
                detector_threshold=5.0,
            )
            timings = []
            for repetition in range(REPETITIONS):
                result = run_policy(
                    method,
                    *common,
                    detector_window=8,
                    detector_threshold=5.0,
                )
                value = 1000.0 * result["seconds"] / H
                timings.append(value)
                raw.append(
                    {
                        "K": k,
                        "N": n,
                        "H": H,
                        "method": method,
                        "repetition": repetition,
                        "ms_per_slot": value,
                    }
                )
            summary = summarize(timings)
            rows.append(
                {
                    "K": k,
                    "N": n,
                    "H": H,
                    "method": method,
                    **summary,
                }
            )

    csv_path = args.out_dir / "tmc_runtime_scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "seed": SEED,
        "K_values": K_VALUES,
        "capacity_rule": "N=K/5",
        "H": H,
        "n0": N0,
        "change_t": CHANGE_T,
        "block_length": BLOCK_LENGTH,
        "repetitions": REPETITIONS,
        "warmup_runs": 1,
        "timed_scope": "run_policy only; observation-bank generation excluded",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "raw": raw,
    }
    json_path = args.out_dir / "tmc_runtime_scaling_meta.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(csv_path)
    for row in rows:
        print(
            f"K={row['K']:3d} N={row['N']:2d} {row['method']:18s} "
            f"median={row['median_ms_per_slot']:.4f} "
            f"IQR=[{row['q1_ms_per_slot']:.4f},"
            f"{row['q3_ms_per_slot']:.4f}] ms/slot"
        )


if __name__ == "__main__":
    main()
