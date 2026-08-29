"""Integrity audit for paper-facing PA-SCO results."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


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
    raw = read_csv("tmc_inflight_formal_raw.csv")
    summary = read_csv("tmc_inflight_formal_summary.csv")
    paired = read_csv("tmc_inflight_formal_paired_raw.csv")
    paired_summary = read_csv("tmc_inflight_formal_paired_summary.csv")
    old = read_csv("tmc_channel_stress_raw.csv")
    metadata_path = RESULTS / "tmc_inflight_formal_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert len(raw) == 336, len(raw)
    assert len(summary) == 28, len(summary)
    assert len(paired) == 84, len(paired)
    assert len(paired_summary) == 7, len(paired_summary)
    assert set(metadata["formal_seeds"]).isdisjoint(metadata["pilot_seeds"])
    assert float(metadata["frozen_inflight_beta"]) == 16.0

    indexed = {}
    for row in raw:
        key = (
            int(row["seed"]),
            row["family"],
            float(row["success_probability"]),
            int(row["delay_slots"]),
            int(row["capacity"]),
            row["method"],
        )
        if key in indexed:
            raise AssertionError(f"duplicate formal row: {key}")
        indexed[key] = row

    equality_fields = (
        "post_cost",
        "total_cost",
        "post_ratio_to_reference",
        "total_ratio_to_reference",
        "delivery_rate",
        "redundant_attempt_rate",
        "detected_fraction",
        "calendar_delay",
        "observation_delay",
    )
    for seed in metadata["formal_seeds"]:
        base = (seed, "delay", 0.9, 0, 4)
        sco = indexed[base + ("sco_reset_ucb",)]
        pipeline = indexed[base + ("inflight_sco_ucb",)]
        for field in equality_fields:
            if not close(sco[field], pipeline[field]):
                raise AssertionError(
                    f"zero-delay mismatch seed={seed} field={field}"
                )

    old_index = {}
    for row in old:
        key = (
            int(row["seed"]),
            row["family"],
            float(row["success_probability"]),
            int(row["delay_slots"]),
            int(row["capacity"]),
            row["method"],
        )
        old_index[key] = row
    old_row_comparisons = 0
    old_metric_comparisons = 0
    aggregation_differences = {"post": [], "total": []}
    for key, row in indexed.items():
        if row["method"] not in (
            "sco_reset_ucb",
            "ps_forced_reset_ucb",
        ):
            continue
        old_row = old_index[key]
        # v5 uses mean of ratios; formal uses ratio of means.
        post_excess = 100 * (
            float(row["post_ratio_to_reference"]) - 1
        )
        total_excess = 100 * (
            float(row["total_ratio_to_reference"]) - 1
        )
        aggregation_differences["post"].append(
            post_excess - float(old_row["post_excess_pct"])
        )
        aggregation_differences["total"].append(
            total_excess - float(old_row["total_excess_pct"])
        )
        for new_field, old_field in (
            ("delivery_rate", "delivery_rate"),
            ("detected_fraction", "detected_fraction"),
            ("calendar_delay", "calendar_delay"),
            ("observation_delay", "observation_delay"),
        ):
            if not close(
                row[new_field], old_row[old_field], atol=1e-8
            ):
                raise AssertionError(
                    f"v5 metric mismatch {key}: {new_field}"
                )
            old_metric_comparisons += 1
        old_row_comparisons += 1
    assert old_row_comparisons == 12 * 7 * 2
    assert old_metric_comparisons == 12 * 7 * 2 * 4

    files = (
        "tmc_inflight_formal_raw.csv",
        "tmc_inflight_formal_summary.csv",
        "tmc_inflight_formal_paired_raw.csv",
        "tmc_inflight_formal_paired_summary.csv",
        "tmc_inflight_formal_meta.json",
    )
    audit = {
        "status": "PASS",
        "row_counts": {
            "raw": len(raw),
            "summary": len(summary),
            "paired_raw": len(paired),
            "paired_summary": len(paired_summary),
        },
        "frozen_beta": metadata["frozen_inflight_beta"],
        "zero_delay_exact_pairs": 12,
        "v5_regression_rows": old_row_comparisons,
        "v5_action_metric_comparisons": old_metric_comparisons,
        "ratio_aggregation_boundary": {
            "v5": "mean of per-batch policy/oracle ratios",
            "formal": "ratio of cross-batch mean costs",
            "max_abs_post_percentage_point_difference": float(
                np.max(np.abs(aggregation_differences["post"]))
            ),
            "max_abs_total_percentage_point_difference": float(
                np.max(np.abs(aggregation_differences["total"]))
            ),
        },
        "hashes": {
            name: sha256(RESULTS / name) for name in files
        },
    }
    (RESULTS / "tmc_inflight_formal_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
