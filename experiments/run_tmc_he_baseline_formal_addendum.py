"""Retrospective formal addendum for the He-style RM-ACK baseline."""
from __future__ import annotations

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


def read_existing_costs():
    path = RESULTS / "tmc_random_delay_formal_raw.csv"
    by_key = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] in {"sco", "pa16"}:
                by_key[(int(row["seed"]), row["profile"], row["method"])] = (
                    float(row["post_cost"])
                )
    return by_key


def main():
    seeds = tuple(20266000 + i for i in range(12))
    B, K, H, change_t = 4, 20, 800, 400
    n0, block_length, capacity, success_probability = 8, 64, 4, 0.9
    existing = read_existing_costs()
    raw = []
    paired = []

    for seed in seeds:
        (
            _, _, theta0, theta1, c00, c01, changed,
            pre_bank, post_bank, ages0,
        ) = make_problem(seed, B, K, change_t, H, n0, block_length)
        bank_shape = (B, K, H + n0)
        success_uniform = np.random.default_rng(seed + 50000).random(bank_shape)
        for profile_item in PROFILES:
            profile = profile_item.name
            forward, feedback = make_delay_banks(seed, profile, bank_shape)
            delay_stats = delay_bank_summary(forward, feedback)
            result = run_policy_random_delay(
                "he_rm_age",
                theta0, theta1, c00, c01, changed, pre_bank, post_bank,
                success_uniform, forward, feedback, ages0, capacity, n0,
                change_t, H, success_probability=success_probability,
            )
            post_cost = float(np.mean(result["post_cost"]))
            total_cost = float(np.mean(result["avg_cost"]))
            raw.append({
                "seed": seed,
                "profile": profile,
                "method": "he_rm",
                "evidence_status": "retrospective_addendum",
                **delay_stats,
                "post_cost": post_cost,
                "total_cost": total_cost,
                "delivery_rate": result["delivery_rate"],
                "ack_rate": result["ack_rate"],
                "redundant_attempt_rate": result["redundant_attempt_rate"],
                "stale_arrival_rate": result["stale_arrival_rate"],
                "mean_inflight_per_slot": result["mean_inflight_per_slot"],
                "capacity_utilization": result["capacity_utilization"],
                "max_inflight_count": result["max_inflight_count"],
                "learned_wait_threshold": result["learned_wait_threshold"],
                "detected_fraction": result["detected_fraction"],
                "calendar_delay": result["calendar_delay"],
                "observation_delay": result["observation_delay"],
            })
            sco = existing[(seed, profile, "sco")]
            pa = existing[(seed, profile, "pa16")]
            paired.append({
                "seed": seed,
                "profile": profile,
                "he_post_cost": post_cost,
                "sco_post_cost": sco,
                "pa_post_cost": pa,
                "pa_reduction_vs_he_pct": 100.0 * (post_cost - pa) / post_cost,
                "sco_reduction_vs_he_pct": 100.0 * (post_cost - sco) / post_cost,
            })
            print(f"formal addendum seed={seed} profile={profile}", flush=True)

    raw_metrics = (
        "post_cost", "total_cost", "delivery_rate", "ack_rate",
        "redundant_attempt_rate", "stale_arrival_rate",
        "mean_inflight_per_slot", "capacity_utilization",
        "max_inflight_count", "learned_wait_threshold",
        "detected_fraction", "calendar_delay", "observation_delay",
    )
    paired_metrics = (
        "he_post_cost", "sco_post_cost", "pa_post_cost",
        "pa_reduction_vs_he_pct", "sco_reduction_vs_he_pct",
    )
    prefix = "tmc_he_rm_formal_addendum"
    write_csv(RESULTS / f"{prefix}_raw.csv", raw)
    write_csv(
        RESULTS / f"{prefix}_summary.csv",
        summarize(raw, ("profile", "method"), raw_metrics),
    )
    write_csv(RESULTS / f"{prefix}_paired_raw.csv", paired)
    write_csv(
        RESULTS / f"{prefix}_paired_summary.csv",
        summarize(paired, ("profile",), paired_metrics),
    )
    metadata = {
        "mode": "formal_retrospective_addendum",
        "evidence_status": "retrospective",
        "seeds": seeds,
        "B": B,
        "K": K,
        "H": H,
        "change_t": change_t,
        "n0": n0,
        "block_length": block_length,
        "capacity": capacity,
        "success_probability": success_probability,
        "profiles": [item.__dict__ for item in PROFILES],
        "baseline_label": "He-style RM-ACK (matched multi-stream adaptation)",
        "source_scope": (
            "single-source optimal waiting under unknown two-way delay; "
            "the implemented multi-stream capacity-constrained adaptation "
            "is not claimed by the source paper"
        ),
        "pairing": (
            "same frozen seeds, latent problems, observation banks, success "
            "uniforms, and attempt-indexed delays as the prior formal run"
        ),
        "claim_boundary": (
            "retrospective external-baseline addendum; negative transfer "
            "does not refute the source theorem"
        ),
    }
    (RESULTS / f"{prefix}_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("formal He-style addendum complete", flush=True)


if __name__ == "__main__":
    main()
