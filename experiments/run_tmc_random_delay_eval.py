"""Frozen-parameter PA-SCO evaluation under random two-way delay."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_cv_piecewise_pilot import make_problem
from run_tmc_channel_stress import mean_ci
from run_tmc_random_delay import (
    PROFILES,
    delay_bank_summary,
    make_delay_banks,
    run_policy_random_delay,
)


RESULTS = Path("results")
METHODS = (
    ("true", "true", None),
    ("sco", "sco_reset_ucb", None),
    ("forced", "ps_forced_reset_ucb", None),
    ("pa16", "inflight_sco_ucb", 16.0),
    ("stopwait", "inflight_sco_ucb", 1e6),
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, group_names, metrics):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in group_names)
        groups.setdefault(key, []).append(row)
    out = []
    for key, group in groups.items():
        record = dict(zip(group_names, key))
        record["seeds"] = len(group)
        for metric in metrics:
            mean, low, high = mean_ci([row[metric] for row in group])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        out.append(record)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    mode = "pilot" if args.quick else "formal"
    if args.quick:
        seeds = tuple(20265000 + i for i in range(6))
        B, K, H, change_t = 2, 12, 320, 160
    else:
        seeds = tuple(20266000 + i for i in range(12))
        B, K, H, change_t = 4, 20, 800, 400
    n0, block_length, capacity, success_probability = 8, 64, 4, 0.9

    frozen = json.loads(
        (RESULTS / "tmc_inflight_formal_meta.json").read_text(
            encoding="utf-8"
        )
    )
    beta = float(frozen["frozen_inflight_beta"])
    if beta != 16.0:
        raise AssertionError(f"unexpected frozen beta: {beta}")
    if set(seeds).intersection(frozen["formal_seeds"]):
        raise AssertionError("random-delay seeds overlap fixed-delay seeds")

    raw = []
    paired = []
    profile_names = [item.name for item in PROFILES]
    for seed in seeds:
        (
            _, _, theta0, theta1, c00, c01, changed,
            pre_bank, post_bank, ages0,
        ) = make_problem(seed, B, K, change_t, H, n0, block_length)
        bank_shape = (B, K, H + n0)
        success_uniform = np.random.default_rng(seed + 50000).random(
            bank_shape
        )
        for profile in profile_names:
            forward, feedback = make_delay_banks(seed, profile, bank_shape)
            delay_stats = delay_bank_summary(forward, feedback)
            common = (
                theta0, theta1, c00, c01, changed, pre_bank, post_bank,
                success_uniform, forward, feedback, ages0, capacity, n0,
                change_t, H,
            )
            results = {}
            for label, policy, method_beta in METHODS:
                kwargs = {"success_probability": success_probability}
                if method_beta is not None:
                    kwargs["inflight_beta"] = method_beta
                results[label] = run_policy_random_delay(
                    policy, *common, **kwargs
                )

            for label, _, method_beta in METHODS:
                result = results[label]
                raw.append({
                    "seed": seed,
                    "profile": profile,
                    "method": label,
                    "inflight_beta": (
                        "" if method_beta is None else method_beta
                    ),
                    **delay_stats,
                    "post_cost": float(np.mean(result["post_cost"])),
                    "total_cost": float(np.mean(result["avg_cost"])),
                    "delivery_rate": result["delivery_rate"],
                    "ack_rate": result["ack_rate"],
                    "redundant_attempt_rate": result[
                        "redundant_attempt_rate"
                    ],
                    "stale_arrival_rate": result["stale_arrival_rate"],
                    "mean_inflight_per_slot": result[
                        "mean_inflight_per_slot"
                    ],
                    "capacity_utilization": result[
                        "capacity_utilization"
                    ],
                    "max_inflight_count": result["max_inflight_count"],
                    "learned_wait_threshold": result[
                        "learned_wait_threshold"
                    ],
                    "detected_fraction": result["detected_fraction"],
                    "calendar_delay": result["calendar_delay"],
                    "observation_delay": result["observation_delay"],
                })

            sco = results["sco"]
            pa = results["pa16"]
            stop = results["stopwait"]
            forced = results["forced"]
            sco_post = float(np.mean(sco["post_cost"]))
            pa_post = float(np.mean(pa["post_cost"]))
            stop_post = float(np.mean(stop["post_cost"]))
            forced_post = float(np.mean(forced["post_cost"]))
            paired.append({
                "seed": seed,
                "profile": profile,
                **delay_stats,
                "pa_reduction_vs_sco_pct": 100 * (sco_post-pa_post)/sco_post,
                "pa_reduction_vs_forced_pct": 100 * (forced_post-pa_post)/forced_post,
                "pa_reduction_vs_stopwait_pct": 100 * (stop_post-pa_post)/stop_post,
                "sco_redundant_attempt_rate": sco["redundant_attempt_rate"],
                "pa_redundant_attempt_rate": pa["redundant_attempt_rate"],
                "stopwait_redundant_attempt_rate": stop[
                    "redundant_attempt_rate"
                ],
                "sco_stale_arrival_rate": sco["stale_arrival_rate"],
                "pa_stale_arrival_rate": pa["stale_arrival_rate"],
                "stopwait_stale_arrival_rate": stop["stale_arrival_rate"],
                "sco_post_cost": sco_post,
                "pa_post_cost": pa_post,
                "stopwait_post_cost": stop_post,
                "forced_post_cost": forced_post,
                "pa_calendar_delay": pa["calendar_delay"],
                "pa_observation_delay": pa["observation_delay"],
            })
            print(f"{mode} seed={seed} profile={profile}", flush=True)

    raw_metrics = (
        "post_cost", "total_cost", "delivery_rate", "ack_rate",
        "redundant_attempt_rate", "stale_arrival_rate",
        "mean_inflight_per_slot", "capacity_utilization",
        "max_inflight_count", "learned_wait_threshold",
        "detected_fraction", "calendar_delay", "observation_delay",
    )
    paired_metrics = (
        "pa_reduction_vs_sco_pct", "pa_reduction_vs_forced_pct",
        "pa_reduction_vs_stopwait_pct", "sco_redundant_attempt_rate",
        "pa_redundant_attempt_rate", "stopwait_redundant_attempt_rate",
        "sco_stale_arrival_rate", "pa_stale_arrival_rate",
        "stopwait_stale_arrival_rate", "sco_post_cost", "pa_post_cost",
        "stopwait_post_cost", "forced_post_cost", "pa_calendar_delay",
        "pa_observation_delay",
    )
    raw_summary = summarize(raw, ("profile", "method"), raw_metrics)
    paired_summary = summarize(paired, ("profile",), paired_metrics)
    prefix = f"tmc_random_delay_{mode}"
    write_csv(RESULTS / f"{prefix}_raw.csv", raw)
    write_csv(RESULTS / f"{prefix}_summary.csv", raw_summary)
    write_csv(RESULTS / f"{prefix}_paired_raw.csv", paired)
    write_csv(RESULTS / f"{prefix}_paired_summary.csv", paired_summary)
    metadata = {
        "mode": mode,
        "seeds": seeds,
        "fixed_delay_formal_seeds": frozen["formal_seeds"],
        "frozen_inflight_beta": beta,
        "stopwait_beta": 1e6,
        "B": B, "K": K, "H": H, "change_t": change_t,
        "n0": n0, "block_length": block_length,
        "capacity": capacity,
        "success_probability": success_probability,
        "profiles": [item.__dict__ for item in PROFILES],
        "pairing": (
            "same latent problem, observation bank, success uniforms, "
            "and attempt-indexed forward/feedback delays within seed-profile"
        ),
        "age_semantics": (
            "receiver accepts only arrivals with a newer generation time; "
            "scheduler age and in-flight count change on ACK/NACK"
        ),
        "claim_boundary": (
            "robustness stress only; no random-delay optimality claim"
        ),
    }
    (RESULTS / f"{prefix}_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"{mode} random-delay evaluation complete", flush=True)


if __name__ == "__main__":
    main()
