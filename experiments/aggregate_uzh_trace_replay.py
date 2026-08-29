"""Audit and paired inference for frozen UZH-FPV replay JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONTRASTS = (
    ("sco_reset_ucb", "cumulative_ce"),
    ("sco_reset_ucb", "cumulative_ucb_cv"),
    ("sco_reset_ucb", "sw_ce_32"),
    ("sco_reset_ucb", "sw_ucb_cv_64"),
    ("sco_reset_ucb", "forced_reset_ucb"),
)


def bootstrap_ci(values, rng, draws=100000):
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_pvalue(values, rng, draws=200000):
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    signs = rng.choice((-1.0, 1.0), size=(draws, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (draws + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "frozen" / "uzh_trace_replay_v1.json",
    )
    args = parser.parse_args()
    path = args.input
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    episodes = payload["episodes"]
    seeds = [episode["seed"] for episode in episodes]
    expected = list(range(410001, 410031))
    methods = tuple(payload["protocol"]["methods"])
    assert seeds == expected
    assert len(seeds) == len(set(seeds)) == 30
    assert all(tuple(row["methods"]) == methods for row in episodes)
    assert all(len(row["sequences"]) == 12 for row in episodes)
    assert all(len(set(row["sequences"])) == 12 for row in episodes)

    rng = np.random.default_rng(2026072401)
    print("AUDIT PASS: 30 exact unique seeds; 10 methods; 12 unique sequences/episode")
    print("left,right,mean_difference_pp,ci95_lo,ci95_hi,sign_flip_p")
    pvalues = []
    rows = []
    for left, right in CONTRASTS:
        difference = np.array(
            [
                episode["methods"][left]["excess_pct"]
                - episode["methods"][right]["excess_pct"]
                for episode in episodes
            ]
        )
        lo, hi = bootstrap_ci(difference, rng)
        pvalue = sign_flip_pvalue(difference, rng)
        pvalues.append(pvalue)
        rows.append((left, right, float(difference.mean()), lo, hi, pvalue))

    # Holm correction across the five preregistered method contrasts.
    order = np.argsort(pvalues)
    holm = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, adjusted)
        holm[index] = running
    for row, adjusted in zip(rows, holm):
        print(
            f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},"
            f"{row[4]:.6f},{row[5]:.8f},holm={adjusted:.8f}"
        )


if __name__ == "__main__":
    main()
