"""Boundary-extension pilot for the in-flight penalty.

This second seed-identical pilot is triggered because the first search selected
its largest beta.  It evaluates only larger penalties, records absolute costs,
and chooses over the union of both pilot grids.  Outputs are not manuscript
evidence.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from run_cv_piecewise_pilot import make_problem
from run_tmc_channel_stress import mean_ci, run_policy_channel


EXTRA_BETAS = (8.0, 16.0, 32.0, 64.0, 1.0e6)
DELAYS = (1, 3, 5)
RESULTS = Path("results")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
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
            reference = run_policy_channel("true", *common)
            reference_post = float(np.mean(reference["post_cost"]))
            for beta in EXTRA_BETAS:
                result = run_policy_channel(
                    "inflight_sco_ucb",
                    *common,
                    inflight_beta=beta,
                )
                method_post = float(np.mean(result["post_cost"]))
                raw.append(
                    {
                        "seed": seed,
                        "delay_slots": delay,
                        "inflight_beta": beta,
                        "reference_post_cost": reference_post,
                        "method_post_cost": method_post,
                        "post_excess_pct": 100
                        * (method_post / reference_post - 1),
                        "redundant_attempt_rate": result[
                            "redundant_attempt_rate"
                        ],
                        "calendar_delay": result["calendar_delay"],
                        "detected_fraction": result["detected_fraction"],
                    }
                )
            print(f"extended pilot seed={seed} d={delay}", flush=True)

    groups = {}
    for row in raw:
        key = (row["delay_slots"], row["inflight_beta"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (delay, beta), rows in groups.items():
        out = {
            "delay_slots": delay,
            "inflight_beta": beta,
            "seeds": len(rows),
        }
        for metric in (
            "reference_post_cost",
            "method_post_cost",
            "post_excess_pct",
            "redundant_attempt_rate",
            "calendar_delay",
            "detected_fraction",
        ):
            mean, low, high = mean_ci([row[metric] for row in rows])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_ci_low"] = low
            out[f"{metric}_ci_high"] = high
        summary.append(out)

    write_csv(RESULTS / "tmc_inflight_pilot_extend_raw.csv", raw)
    write_csv(
        RESULTS / "tmc_inflight_pilot_extend_summary.csv", summary
    )

    first_raw = read_csv(RESULTS / "tmc_inflight_pilot_raw.csv")
    scores = {}
    for row in first_raw:
        if row["method"] != "inflight_sco_ucb":
            continue
        beta = float(row["inflight_beta"])
        scores.setdefault(beta, []).append(float(row["post_excess_pct"]))
    for row in raw:
        beta = float(row["inflight_beta"])
        scores.setdefault(beta, []).append(float(row["post_excess_pct"]))
    ranked = sorted(
        (float(np.mean(values)), beta) for beta, values in scores.items()
    )
    selected_beta = ranked[0][1]
    metadata = {
        "mode": "pilot_not_for_manuscript",
        "reason_for_extension": (
            "the first pilot selected the largest candidate beta=4"
        ),
        "seeds": seeds,
        "extra_candidate_betas": EXTRA_BETAS,
        "all_candidate_betas": sorted(scores),
        "selection_rule": (
            "minimum mean post-change excess across all pilot seeds "
            "and delays"
        ),
        "selected_beta": selected_beta,
        "ranked_mean_post_excess": [
            {"beta": beta, "mean_post_excess_pct": score}
            for score, beta in ranked
        ],
    }
    (
        RESULTS / "tmc_inflight_pilot_selected_meta.json"
    ).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"union-grid selected beta={selected_beta:g}; "
        f"mean post excess={ranked[0][0]:.4f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
