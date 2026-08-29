"""Audit and paired inference for frozen M3ED replay JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aggregate_uzh_trace_replay import (
    CONTRASTS,
    bootstrap_ci,
    sign_flip_pvalue,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "frozen" / "m3ed_trace_replay_v1.json",
    )
    args = parser.parse_args()
    payload = json.loads(
        args.input.read_text(encoding="utf-8-sig")
    )
    episodes = payload["episodes"]
    seeds = [episode["seed"] for episode in episodes]
    expected = list(range(420001, 420031))
    methods = tuple(payload["protocol"]["methods"])
    assert seeds == expected
    assert len(seeds) == len(set(seeds)) == 30
    assert all(tuple(row["methods"]) == methods for row in episodes)
    assert all(len(row["sequences"]) == 12 for row in episodes)
    assert all(len(set(row["sequences"])) == 12 for row in episodes)

    rng = np.random.default_rng(2026072402)
    print("AUDIT PASS: 30 exact unique seeds; 10 methods; 12 unique sequences/episode")
    print("left,right,mean_difference_pp,ci95_lo,ci95_hi,sign_flip_p")
    pvalues, rows = [], []
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
    order = np.argsort(pvalues)
    holm = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        holm[index] = running
    for row, adjusted in zip(rows, holm):
        print(
            f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},"
            f"{row[4]:.6f},{row[5]:.8f},holm={adjusted:.8f}"
        )


if __name__ == "__main__":
    main()
