"""Post-hoc UZH preprocessing sensitivity driven by the existing speed guard.

The frozen primary replay uses the predeclared v2 filter (0.15 m absolute
curvature threshold, multiplier 10).  One sequence exceeds the pre-existing
50 m/s integrity guard.  This audit does not replace the primary result: it
uses the smallest tested parameter pair (0.12, 4) that satisfies that guard,
then reruns the identical 30 seeds and method set to test ordering stability.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.uzh_fpv import deterministic_projection, parse_leica
from core.uzh_fpv_protocol import OFFICIAL, crop_official_motion_window
from core.uzh_fpv_replay_v2 import despike_positions
from run_uzh_trace_replay import (
    METHODS,
    bootstrap_mean_ci,
    make_episode,
    run_method,
)


ABSOLUTE_THRESHOLD = 0.12
MULTIPLIER = 4.0
SPEED_GUARD_MPS = 50.0
SEEDS = list(range(410001, 410031))


def load_strict_bank(root: Path):
    traces = []
    diagnostics = []
    for name in sorted(OFFICIAL):
        t, xyz = parse_leica(root / name / "leica.txt")
        t, xyz = crop_official_motion_window(name, t, xyz, rate_hz=20.0)
        clean = despike_positions(
            xyz,
            absolute_threshold=ABSOLUTE_THRESHOLD,
            multiplier=MULTIPLIER,
        )
        speed = np.linalg.norm(np.diff(clean, axis=0), axis=1) / np.diff(t)
        path = float(np.linalg.norm(np.diff(clean, axis=0), axis=1).sum())
        direction = deterministic_projection(name)
        traces.append(
            {
                "name": name,
                "t": t,
                "xyz": clean,
                "position": (clean - clean[0]) @ direction,
                "projection": direction,
                "official": OFFICIAL[name],
            }
        )
        diagnostics.append(
            {
                "name": name,
                "max_speed_mps": float(speed.max()),
                "path_ratio": path / float(OFFICIAL[name][1]),
            }
        )
    assert max(row["max_speed_mps"] for row in diagnostics) < SPEED_GUARD_MPS
    return traces, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/uzh_fpv_gt"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/traces/tmc_uzh_filter_sensitivity.json"),
    )
    args = parser.parse_args()
    bank, diagnostics = load_strict_bank(args.data_root)
    episodes = []
    for seed in SEEDS:
        physical, names = make_episode(bank, seed)
        rows = {name: run_method(name, physical) for name in METHODS}
        oracle = rows["oracle"]["cost"]
        for name in METHODS:
            rows[name]["excess_pct"] = 100.0 * (
                rows[name]["cost"] / oracle - 1.0
            )
        episodes.append({"seed": seed, "sequences": names, "methods": rows})
        print(f"completed {seed}", flush=True)
    rng = np.random.default_rng(910247)
    summary = {}
    for name in METHODS:
        values = [row["methods"][name]["excess_pct"] for row in episodes]
        summary[name] = bootstrap_mean_ci(values, rng)
    contrasts = {}
    for right in (
        "cumulative_ce",
        "cumulative_ucb_cv",
        "sw_ce_32",
        "sw_ucb_cv_64",
        "forced_reset_ucb",
    ):
        diff = [
            row["methods"]["sco_reset_ucb"]["excess_pct"]
            - row["methods"][right]["excess_pct"]
            for row in episodes
        ]
        contrasts[f"sco_reset_ucb-minus-{right}"] = bootstrap_mean_ci(diff, rng)
    payload = {
        "status": "post_hoc_preprocessing_sensitivity_not_primary_evidence",
        "reason": "existing 50 m/s integrity guard",
        "filter": {
            "absolute_threshold": ABSOLUTE_THRESHOLD,
            "multiplier": MULTIPLIER,
            "speed_guard_mps": SPEED_GUARD_MPS,
        },
        "diagnostics": diagnostics,
        "summary_excess_pct": summary,
        "paired_contrasts": contrasts,
        "seeds": SEEDS,
        "primary_result_unchanged": True,
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
