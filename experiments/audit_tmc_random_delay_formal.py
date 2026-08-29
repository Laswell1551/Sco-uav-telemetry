"""Integrity audit for paper-facing random two-way-delay results."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_cv_piecewise_pilot import make_problem
from run_tmc_channel_stress import mean_ci
from run_tmc_random_delay import make_delay_banks, run_policy_random_delay


RESULTS = Path("results")


def read_csv(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def close(left, right, atol=1e-10):
    return bool(np.isclose(float(left), float(right), atol=atol, rtol=1e-10))


def main():
    raw = read_csv("tmc_random_delay_formal_raw.csv")
    summary = read_csv("tmc_random_delay_formal_summary.csv")
    paired = read_csv("tmc_random_delay_formal_paired_raw.csv")
    paired_summary = read_csv("tmc_random_delay_formal_paired_summary.csv")
    pilot_meta = json.loads(
        (RESULTS / "tmc_random_delay_pilot_meta.json").read_text(encoding="utf-8")
    )
    fixed_meta = json.loads(
        (RESULTS / "tmc_inflight_formal_meta.json").read_text(encoding="utf-8")
    )
    meta_path = RESULTS / "tmc_random_delay_formal_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert len(raw) == 12 * 7 * 5, len(raw)
    assert len(summary) == 7 * 5, len(summary)
    assert len(paired) == 12 * 7, len(paired)
    assert len(paired_summary) == 7, len(paired_summary)
    assert set(meta["seeds"]).isdisjoint(pilot_meta["seeds"])
    assert set(meta["seeds"]).isdisjoint(fixed_meta["formal_seeds"])
    assert float(meta["frozen_inflight_beta"]) == 16.0
    assert float(meta["stopwait_beta"]) == 1e6

    index = {}
    rate_fields = (
        "delivery_rate", "ack_rate", "redundant_attempt_rate",
        "stale_arrival_rate", "detected_fraction",
    )
    for row in raw:
        key = (int(row["seed"]), row["profile"], row["method"])
        if key in index:
            raise AssertionError(f"duplicate raw row: {key}")
        index[key] = row
        for field in ("post_cost", "total_cost", "mean_inflight_per_slot"):
            if not np.isfinite(float(row[field])) or float(row[field]) < 0:
                raise AssertionError(f"invalid {field}: {key}")
        for field in rate_fields:
            value = float(row[field])
            if np.isfinite(value) and not 0 <= value <= 1:
                raise AssertionError(f"invalid rate {field}: {key}")
        if row["method"] == "pa16" and float(row["inflight_beta"]) != 16:
            raise AssertionError(f"PA beta drift: {key}")
        if row["method"] == "stopwait" and float(row["inflight_beta"]) != 1e6:
            raise AssertionError(f"stopwait beta drift: {key}")

    paired_checks = 0
    delay_stat_checks = 0
    for row in paired:
        seed, profile = int(row["seed"]), row["profile"]
        methods = {name: index[(seed, profile, name)] for name in (
            "sco", "forced", "pa16", "stopwait"
        )}
        sco = float(methods["sco"]["post_cost"])
        forced = float(methods["forced"]["post_cost"])
        pa = float(methods["pa16"]["post_cost"])
        stop = float(methods["stopwait"]["post_cost"])
        expected = {
            "pa_reduction_vs_sco_pct": 100 * (sco-pa) / sco,
            "pa_reduction_vs_forced_pct": 100 * (forced-pa) / forced,
            "pa_reduction_vs_stopwait_pct": 100 * (stop-pa) / stop,
        }
        for field, value in expected.items():
            if not close(row[field], value):
                raise AssertionError(f"paired mismatch {seed}/{profile}/{field}")
            paired_checks += 1
        for field in (
            "forward_mean", "feedback_mean", "round_trip_mean",
            "round_trip_p95", "round_trip_p99", "round_trip_max",
        ):
            values = {float(index[(seed, profile, method)][field]) for method in (
                "true", "sco", "forced", "pa16", "stopwait"
            )}
            if len(values) != 1 or not close(row[field], values.pop()):
                raise AssertionError(f"delay bank mismatch {seed}/{profile}/{field}")
            delay_stat_checks += 1
        if profile == "fixed":
            for field, value in (
                ("forward_mean", 2), ("feedback_mean", 2),
                ("round_trip_mean", 4), ("round_trip_p99", 4),
                ("round_trip_max", 4),
            ):
                if not close(row[field], value):
                    raise AssertionError(f"fixed profile mismatch: {field}")

    summary_checks = 0
    for row in paired_summary:
        group = [item for item in paired if item["profile"] == row["profile"]]
        assert len(group) == 12
        for metric in (
            "pa_reduction_vs_sco_pct", "pa_reduction_vs_forced_pct",
            "pa_reduction_vs_stopwait_pct", "sco_redundant_attempt_rate",
            "pa_redundant_attempt_rate", "sco_stale_arrival_rate",
            "pa_stale_arrival_rate", "pa_post_cost",
        ):
            expected = mean_ci([item[metric] for item in group])
            for suffix, value in zip(("mean", "ci_low", "ci_high"), expected):
                if not close(row[f"{metric}_{suffix}"], value):
                    raise AssertionError(
                        f"summary mismatch {row['profile']}/{metric}/{suffix}"
                    )
                summary_checks += 1

    # Deterministic spot reproduction of a paper-facing heavy-tail setting.
    seed = int(meta["seeds"][0])
    B, K, H = int(meta["B"]), int(meta["K"]), int(meta["H"])
    n0, change_t, block = int(meta["n0"]), int(meta["change_t"]), int(meta["block_length"])
    (_, _, theta0, theta1, c00, c01, changed,
     pre_bank, post_bank, ages0) = make_problem(seed, B, K, change_t, H, n0, block)
    shape = (B, K, H + n0)
    uniforms = np.random.default_rng(seed + 50000).random(shape)
    forward, feedback = make_delay_banks(seed, "heavy_iid", shape)
    common = (
        theta0, theta1, c00, c01, changed, pre_bank, post_bank,
        uniforms, forward, feedback, ages0, int(meta["capacity"]), n0,
        change_t, H,
    )
    reproduced = run_policy_random_delay(
        "inflight_sco_ucb", *common,
        success_probability=float(meta["success_probability"]),
        inflight_beta=16.0,
    )
    saved = index[(seed, "heavy_iid", "pa16")]
    reproduction_checks = 0
    for field, value in (
        ("post_cost", np.mean(reproduced["post_cost"])),
        ("total_cost", np.mean(reproduced["avg_cost"])),
        ("redundant_attempt_rate", reproduced["redundant_attempt_rate"]),
        ("stale_arrival_rate", reproduced["stale_arrival_rate"]),
        ("calendar_delay", reproduced["calendar_delay"]),
        ("observation_delay", reproduced["observation_delay"]),
    ):
        if not close(saved[field], value):
            raise AssertionError(f"spot reproduction mismatch: {field}")
        reproduction_checks += 1

    files = (
        "tmc_random_delay_formal_raw.csv",
        "tmc_random_delay_formal_summary.csv",
        "tmc_random_delay_formal_paired_raw.csv",
        "tmc_random_delay_formal_paired_summary.csv",
        "tmc_random_delay_formal_meta.json",
    )
    audit = {
        "status": "PASS",
        "row_counts": {
            "raw": len(raw), "summary": len(summary),
            "paired_raw": len(paired), "paired_summary": len(paired_summary),
        },
        "seed_disjointness": "PASS",
        "frozen_beta": meta["frozen_inflight_beta"],
        "paired_formula_checks": paired_checks,
        "delay_bank_consistency_checks": delay_stat_checks,
        "paired_summary_checks": summary_checks,
        "heavy_iid_spot_reproduction_checks": reproduction_checks,
        "hashes": {name: sha256(RESULTS / name) for name in files},
    }
    (RESULTS / "tmc_random_delay_formal_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
