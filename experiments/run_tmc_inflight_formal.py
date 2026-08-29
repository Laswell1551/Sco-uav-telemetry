"""Paper-facing paired evaluation of pipeline-aware SCO.

The in-flight penalty is loaded from the seed-disjoint pilot record and frozen
before formal seeds are evaluated.  Formal results focus on fixed delay and
capacity, because zero-delay delivery settings are exactly identical to SCO.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from run_cv_piecewise_pilot import make_problem
from run_tmc_channel_stress import mean_ci, run_policy_channel


RESULTS = Path("results")
FORMAL_SEEDS = tuple(20261000 + offset for offset in range(12))
METHODS = (
    "true",
    "sco_reset_ucb",
    "ps_forced_reset_ucb",
    "inflight_sco_ucb",
)


def formal_settings():
    rows = [
        ("delay", 0.9, delay, 4) for delay in (0, 1, 3, 5)
    ]
    rows.extend(
        ("capacity", 0.9, 1, capacity) for capacity in (2, 4, 8)
    )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_frozen_beta():
    path = RESULTS / "tmc_inflight_pilot_selected_meta.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    pilot_seeds = set(metadata["seeds"])
    overlap = pilot_seeds.intersection(FORMAL_SEEDS)
    if overlap:
        raise AssertionError(f"pilot/formal seed overlap: {sorted(overlap)}")
    return float(metadata["selected_beta"]), metadata


def main():
    beta, pilot_metadata = load_frozen_beta()
    B, K, H, n0, change_t, block_length = 4, 20, 800, 8, 400, 64
    raw = []
    paired = []

    for seed in FORMAL_SEEDS:
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

        for family, probability, delay, capacity in formal_settings():
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
                capacity,
                n0,
                change_t,
                H,
                probability,
                delay,
            )
            results = {}
            for method in METHODS:
                kwargs = (
                    {"inflight_beta": beta}
                    if method == "inflight_sco_ucb"
                    else {}
                )
                results[method] = run_policy_channel(
                    method, *common, **kwargs
                )

            reference_post = float(
                np.mean(results["true"]["post_cost"])
            )
            reference_total = float(
                np.mean(results["true"]["avg_cost"])
            )
            for method in METHODS:
                result = results[method]
                method_post = float(np.mean(result["post_cost"]))
                method_total = float(np.mean(result["avg_cost"]))
                raw.append(
                    {
                        "seed": seed,
                        "family": family,
                        "success_probability": probability,
                        "delay_slots": delay,
                        "capacity": capacity,
                        "capacity_ratio": capacity / K,
                        "method": method,
                        "frozen_inflight_beta": beta,
                        "post_cost": method_post,
                        "total_cost": method_total,
                        "reference_post_cost": reference_post,
                        "reference_total_cost": reference_total,
                        "post_ratio_to_reference": (
                            method_post / reference_post
                        ),
                        "total_ratio_to_reference": (
                            method_total / reference_total
                        ),
                        "delivery_rate": result["delivery_rate"],
                        "redundant_attempt_rate": result[
                            "redundant_attempt_rate"
                        ],
                        "max_inflight_count": result[
                            "max_inflight_count"
                        ],
                        "detected_fraction": result[
                            "detected_fraction"
                        ],
                        "calendar_delay": result["calendar_delay"],
                        "observation_delay": result[
                            "observation_delay"
                        ],
                    }
                )

            sco = results["sco_reset_ucb"]
            pipeline = results["inflight_sco_ucb"]
            sco_post = float(np.mean(sco["post_cost"]))
            pipeline_post = float(np.mean(pipeline["post_cost"]))
            paired.append(
                {
                    "seed": seed,
                    "family": family,
                    "success_probability": probability,
                    "delay_slots": delay,
                    "capacity": capacity,
                    "capacity_ratio": capacity / K,
                    "frozen_inflight_beta": beta,
                    "sco_post_cost": sco_post,
                    "pipeline_post_cost": pipeline_post,
                    "pipeline_reduction_vs_sco_pct": (
                        100 * (sco_post - pipeline_post) / sco_post
                    ),
                    "sco_redundant_attempt_rate": sco[
                        "redundant_attempt_rate"
                    ],
                    "pipeline_redundant_attempt_rate": pipeline[
                        "redundant_attempt_rate"
                    ],
                    "redundant_attempt_reduction": (
                        sco["redundant_attempt_rate"]
                        - pipeline["redundant_attempt_rate"]
                    ),
                    "sco_calendar_delay": sco["calendar_delay"],
                    "pipeline_calendar_delay": pipeline[
                        "calendar_delay"
                    ],
                }
            )
            print(
                f"formal seed={seed} {family} p={probability:g} "
                f"d={delay} N={capacity}",
                flush=True,
            )

    metric_names = (
        "post_cost",
        "total_cost",
        "post_ratio_to_reference",
        "total_ratio_to_reference",
        "delivery_rate",
        "redundant_attempt_rate",
        "max_inflight_count",
        "detected_fraction",
        "calendar_delay",
        "observation_delay",
    )
    group_names = (
        "family",
        "success_probability",
        "delay_slots",
        "capacity",
        "capacity_ratio",
        "method",
        "frozen_inflight_beta",
    )
    groups = {}
    for row in raw:
        key = tuple(row[name] for name in group_names)
        groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in groups.items():
        out = dict(zip(group_names, key))
        out["seeds"] = len(rows)
        for metric in metric_names:
            mean, low, high = mean_ci(
                [row[metric] for row in rows]
            )
            out[f"{metric}_mean"] = mean
            out[f"{metric}_ci_low"] = low
            out[f"{metric}_ci_high"] = high
        summary.append(out)

    paired_metric_names = (
        "sco_post_cost",
        "pipeline_post_cost",
        "pipeline_reduction_vs_sco_pct",
        "sco_redundant_attempt_rate",
        "pipeline_redundant_attempt_rate",
        "redundant_attempt_reduction",
        "sco_calendar_delay",
        "pipeline_calendar_delay",
    )
    paired_group_names = (
        "family",
        "success_probability",
        "delay_slots",
        "capacity",
        "capacity_ratio",
        "frozen_inflight_beta",
    )
    paired_groups = {}
    for row in paired:
        key = tuple(row[name] for name in paired_group_names)
        paired_groups.setdefault(key, []).append(row)
    paired_summary = []
    for key, rows in paired_groups.items():
        out = dict(zip(paired_group_names, key))
        out["seeds"] = len(rows)
        for metric in paired_metric_names:
            mean, low, high = mean_ci(
                [row[metric] for row in rows]
            )
            out[f"{metric}_mean"] = mean
            out[f"{metric}_ci_low"] = low
            out[f"{metric}_ci_high"] = high
        paired_summary.append(out)

    write_csv(RESULTS / "tmc_inflight_formal_raw.csv", raw)
    write_csv(RESULTS / "tmc_inflight_formal_summary.csv", summary)
    write_csv(RESULTS / "tmc_inflight_formal_paired_raw.csv", paired)
    write_csv(
        RESULTS / "tmc_inflight_formal_paired_summary.csv",
        paired_summary,
    )
    metadata = {
        "mode": "paper",
        "formal_seeds": FORMAL_SEEDS,
        "pilot_seeds": pilot_metadata["seeds"],
        "frozen_inflight_beta": beta,
        "B": B,
        "K": K,
        "H": H,
        "change_t": change_t,
        "n0": n0,
        "block_length": block_length,
        "settings": [
            {
                "family": family,
                "success_probability": probability,
                "delay_slots": delay,
                "capacity": capacity,
            }
            for family, probability, delay, capacity
            in formal_settings()
        ],
        "pairing": (
            "same latent problem, observation-indexed CV bank, and "
            "attempt-indexed channel uniforms within each formal seed"
        ),
        "reference_boundary": (
            "true-model immediate-reset Whittle run over the same "
            "delayed channel; it is a reference and can be outperformed"
        ),
        "zero_delay_boundary": (
            "pipeline-aware SCO is exactly identical to SCO at delay zero"
        ),
    }
    (
        RESULTS / "tmc_inflight_formal_meta.json"
    ).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("formal in-flight evaluation complete", flush=True)


if __name__ == "__main__":
    main()
