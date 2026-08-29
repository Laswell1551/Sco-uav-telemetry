"""Integrity checks for the TMC round-2 formal result packages."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


RESULTS = Path("results")


def rows(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_seed_set(data, expected):
    found = {int(row["seed"]) for row in data}
    assert found == set(expected), (min(found), max(found), len(found))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope",
        choices=("boundary", "all"),
        default="all",
        help="Use boundary for a standalone boundary/calibration run.",
    )
    args = parser.parse_args()
    report = {"status": "pass", "checks": {}}

    if args.scope == "all":
        he = rows("tmc_he_rm_formal_addendum_raw.csv")
        assert len(he) == 12 * 7
        assert_seed_set(he, range(20266000, 20266012))
        assert {row["profile"] for row in he} == {
            "fixed", "light_iid", "heavy_iid", "lognormal",
            "markov_burst", "forward_heavy", "feedback_heavy",
        }
        assert all(row["method"] == "he_rm" for row in he)
        assert all(
            row["evidence_status"] == "retrospective_addendum" for row in he
        )
        assert all(float(row["redundant_attempt_rate"]) == 0.0 for row in he)
        he_pair = rows("tmc_he_rm_formal_addendum_paired_summary.csv")
        assert len(he_pair) == 7
        assert all(
            float(row["pa_reduction_vs_he_pct_ci_low"]) > 0 for row in he_pair
        )
        report["checks"]["he_addendum"] = {
            "raw_rows": len(he),
            "profiles": len(he_pair),
            "all_pa_vs_he_ci_positive": True,
            "all_he_duplicate_rates_zero": True,
        }

    ca4 = rows("tmc_ca_mismatch_formal_v2_n4_raw.csv")
    assert len(ca4) == 30 * 3 * 3
    assert_seed_set(ca4, range(20270000, 20270030))
    assert {int(row["spatial_dimension"]) for row in ca4} == {1, 2, 3}
    assert {row["method"] for row in ca4} == {
        "ca_index", "cubic_cv_surrogate", "max_age"
    }
    report["checks"]["ca_n4"] = {
        "raw_rows": len(ca4),
        "seeds": 30,
        "dimensions": 3,
        "reference_label": "ca_index_not_global_oracle",
    }

    ca1_path = RESULTS / "tmc_ca_mismatch_formal_v2_n1_raw.csv"
    if ca1_path.exists():
        ca1 = rows(ca1_path.name)
        assert len(ca1) == 30 * 3 * 3
        assert_seed_set(ca1, range(20270000, 20270030))
        report["checks"]["ca_n1"] = {
            "raw_rows": len(ca1),
            "seeds": 30,
            "dimensions": 3,
        }
    else:
        report["status"] = "partial"
        report["checks"]["ca_n1"] = "pending"

    multi_path = RESULTS / "tmc_multiaxis_formal_raw.csv"
    if multi_path.exists():
        multi = rows(multi_path.name)
        assert len(multi) == 12 * 3 * 7
        assert_seed_set(multi, range(20268000, 20268012))
        assert {int(row["dimension"]) for row in multi} == {1, 2, 3}
        report["checks"]["multiaxis"] = {
            "raw_rows": len(multi),
            "seeds": 12,
            "dimensions": 3,
            "methods": 7,
        }
    else:
        report["status"] = "partial"
        report["checks"]["multiaxis"] = "pending"

    certificate = rows("tmc_certificate_sweep_raw.csv")
    assert len(certificate) == 12 * 6
    assert_seed_set(certificate, range(20260900, 20260912))
    assert {float(row["scale"]) for row in certificate} == {
        0.5, 1.0, 2.0, 3.0, 4.0, 6.0,
    }
    report["checks"]["certificate"] = {
        "raw_rows": len(certificate),
        "seeds": 12,
        "scales": 6,
    }

    core_files = [
        "tmc_ca_mismatch_formal_v2_n4_raw.csv",
        "tmc_ca_mismatch_formal_v2_n4_summary.csv",
        "tmc_ca_mismatch_formal_v2_n1_raw.csv",
        "tmc_ca_mismatch_formal_v2_n1_summary.csv",
        "tmc_multiaxis_formal_raw.csv",
        "tmc_multiaxis_formal_summary.csv",
        "tmc_certificate_sweep_raw.csv",
        "tmc_certificate_sweep_summary.csv",
    ]
    if args.scope == "all":
        core_files.extend([
            "tmc_he_rm_formal_addendum_raw.csv",
            "tmc_he_rm_formal_addendum_summary.csv",
            "tmc_he_rm_formal_addendum_paired_summary.csv",
        ])
    report["sha256"] = {
        name: sha256(RESULTS / name)
        for name in core_files
        if (RESULTS / name).exists()
    }
    (RESULTS / "tmc_round2_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
