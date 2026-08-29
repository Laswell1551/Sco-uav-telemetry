"""Pilot calibration for in-flight-aware SCO under fixed delivery delay.

The pilot uses seed-disjoint, reduced-size instances to select one dimensionless
in-flight penalty before the paper-facing evaluation.  No pilot value is used
as a manuscript result.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_cv_piecewise_pilot import make_problem
from run_tmc_channel_stress import mean_ci, run_policy_channel


BETAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
DELAYS = (1, 3, 5)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    seeds = [20262000 + offset for offset in range(6)]
    B, K, N, H, n0, change_t, block_length = 2, 12, 3, 320, 8, 160, 64
    probability = 0.9
    raw = []

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
        channel_uniform = np.random.default_rng(seed + 50000).random(
            (B, K, H + n0)
        )

        for delay in DELAYS:
            common = (
                theta0,
                theta1,
                c00,
                c01,
                changed,
                pre_bank,
                post_bank,
                channel_uniform,
                ages0,
                N,
                n0,
                change_t,
                H,
                probability,
                delay,
            )
            oracle = run_policy_channel("true", *common)
            variants = [
                ("sco_reset_ucb", np.nan, {}),
                ("ps_forced_reset_ucb", np.nan, {}),
            ]
            variants.extend(
                (
                    "inflight_sco_ucb",
                    beta,
                    {"inflight_beta": beta},
                )
                for beta in BETAS
            )
            for method, beta, kwargs in variants:
                result = run_policy_channel(method, *common, **kwargs)
                raw.append(
                    {
                        "seed": seed,
                        "delay_slots": delay,
                        "method": method,
                        "inflight_beta": beta,
                        "post_excess_pct": float(
                            np.mean(
                                100
                                * (
                                    result["post_cost"]
                                    / oracle["post_cost"]
                                    - 1
                                )
                            )
                        ),
                        "total_excess_pct": float(
                            np.mean(
                                100
                                * (
                                    result["avg_cost"]
                                    / oracle["avg_cost"]
                                    - 1
                                )
                            )
                        ),
                        "redundant_attempt_rate": result[
                            "redundant_attempt_rate"
                        ],
                        "calendar_delay": result["calendar_delay"],
                        "detected_fraction": result["detected_fraction"],
                    }
                )
            print(f"pilot seed={seed} d={delay}", flush=True)

    groups = {}
    for row in raw:
        key = (
            row["delay_slots"],
            row["method"],
            row["inflight_beta"],
        )
        groups.setdefault(key, []).append(row)

    summary = []
    for (delay, method, beta), rows in groups.items():
        out = {
            "delay_slots": delay,
            "method": method,
            "inflight_beta": beta,
            "seeds": len(rows),
        }
        for metric in (
            "post_excess_pct",
            "total_excess_pct",
            "redundant_attempt_rate",
            "calendar_delay",
            "detected_fraction",
        ):
            mean, low, high = mean_ci([row[metric] for row in rows])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_ci_low"] = low
            out[f"{metric}_ci_high"] = high
        summary.append(out)

    beta_scores = []
    for beta in BETAS:
        values = [
            row["post_excess_pct"]
            for row in raw
            if row["method"] == "inflight_sco_ucb"
            and row["inflight_beta"] == beta
        ]
        beta_scores.append((float(np.mean(values)), beta))
    beta_scores.sort()
    selected_beta = beta_scores[0][1]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "tmc_inflight_pilot_raw.csv", raw)
    write_csv(args.out_dir / "tmc_inflight_pilot_summary.csv", summary)
    metadata = {
        "mode": "pilot_not_for_manuscript",
        "seeds": seeds,
        "B": B,
        "K": K,
        "N": N,
        "H": H,
        "change_t": change_t,
        "success_probability": probability,
        "delays": DELAYS,
        "candidate_betas": BETAS,
        "selection_rule": (
            "minimum mean post-change excess across all pilot seeds "
            "and fixed delays"
        ),
        "selected_beta": selected_beta,
        "seed_separation": (
            "pilot seeds 20262000--20262005 are disjoint from formal "
            "seeds 20261000--20261011"
        ),
    }
    (
        args.out_dir / "tmc_inflight_pilot_meta.json"
    ).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"selected beta={selected_beta:g}; "
        f"mean post excess={beta_scores[0][0]:.4f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
