"""Audit and aggregate all 30 frozen confirmatory seeds."""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


METHODS = [
    "cumulative_ce",
    "cumulative_ucb_cv",
    "sw_ce_32",
    "sw_ucb_cv_64",
    "sco_reset_ce",
    "sco_reset_ucb",
    "ps_forced_reset_ucb",
]


def seed_cluster_means(values, batches_per_seed=3):
    """Collapse paired within-seed instances before seed-level inference."""
    values = np.asarray(values, dtype=float)
    if values.size % batches_per_seed:
        raise AssertionError("instance count is not divisible by cluster size")
    return values.reshape(-1, batches_per_seed).mean(axis=1)


def bootstrap_ci(values, rng, replicates=100000, alpha=0.05):
    values = np.asarray(values, dtype=float)
    means = np.empty(replicates)
    batch = 5000
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        draw = rng.integers(
            0, values.size, size=(stop - start, values.size)
        )
        means[start:stop] = values[draw].mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def sign_flip_pvalue(values, rng, replicates=200000):
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    exceed = 0
    batch = 5000
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        sign = rng.choice(
            np.array([-1.0, 1.0]),
            size=(stop - start, values.size),
        )
        permuted = np.abs((sign * values).mean(axis=1))
        exceed += int(np.sum(permuted >= observed))
    return (exceed + 1.0) / (replicates + 1.0)


def holm_adjust(pvalues):
    names = list(pvalues)
    ordered = sorted(names, key=lambda name: pvalues[name])
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for i, name in enumerate(ordered):
        candidate = min(1.0, (m - i) * pvalues[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def load_and_audit(paths):
    payloads = []
    for path in paths:
        with open(path, "r", encoding="utf-8-sig") as handle:
            payloads.append(json.load(handle))
    seeds = [seed for item in payloads for seed in item["seeds"]]
    expected = list(range(310001, 310031))
    if sorted(seeds) != expected or len(seeds) != len(set(seeds)):
        raise AssertionError(
            f"seed coverage mismatch: got {sorted(seeds)}, expected {expected}"
        )
    for item in payloads:
        if item["protocol"] != "TMC_SYNTHETIC_PROTOCOL_FROZEN_v1":
            raise AssertionError("protocol mismatch")
        if item["batches_per_seed"] != 3:
            raise AssertionError("batch-count mismatch")
        if set(item["records"]) != set(METHODS):
            raise AssertionError("method-set mismatch")
        n_seed = len(item["seeds"])
        for method in METHODS:
            row = item["records"][method]
            if len(row["total_ex"]) != 3 * n_seed:
                raise AssertionError(f"total length mismatch for {method}")
            if len(row["post_ex"]) != 3 * n_seed:
                raise AssertionError(f"post length mismatch for {method}")
    payloads.sort(key=lambda item: item["seed_start"])
    return payloads


def combine(payloads):
    records = {
        method: {
            key: []
            for key in payloads[0]["records"][method]
        }
        for method in METHODS
    }
    seed_status = []
    for item in payloads:
        seed_status.extend(item["seed_status"])
        for method in METHODS:
            for key, values in item["records"][method].items():
                records[method][key].extend(values)
    return records, seed_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="runs/controlled/*.json",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("runs/controlled/tmc_confirmatory_summary.csv"),
    )
    args = parser.parse_args()
    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No controlled result shards match {args.glob!r}")
    payloads = load_and_audit(paths)
    records, seed_status = combine(payloads)
    rng = np.random.default_rng(310099)

    summaries = {}
    for method in METHODS:
        total = seed_cluster_means(records[method]["total_ex"])
        post = seed_cluster_means(records[method]["post_ex"])
        summaries[method] = {
            "total_mean": float(total.mean()),
            "total_ci": bootstrap_ci(total, rng),
            "post_mean": float(post.mean()),
            "post_ci": bootstrap_ci(post, rng),
            "rank_loss": float(np.mean(records[method]["rank_loss"])),
        }
        detection = records[method]["detection"]
        if detection:
            changed_counts = np.array(
                [
                    round(item["changed_fraction"] * 3 * 20)
                    for item in seed_status
                ],
                dtype=int,
            )
            detected_counts = np.rint(
                np.asarray(detection) * changed_counts
            ).astype(int)
            summaries[method].update(
                {
                    "detection": float(
                        detected_counts.sum() / changed_counts.sum()
                    ),
                    "calendar_delay": float(
                        np.average(
                            records[method]["calendar_delay"],
                            weights=detected_counts,
                        )
                    ),
                    "observation_delay": float(
                        np.average(
                            records[method]["observation_delay"],
                            weights=detected_counts,
                        )
                    ),
                    "pre_fa_per_10k": float(
                        np.mean(records[method]["pre_fa_per_10k"])
                    ),
                    "post_unchanged_alarms": int(
                        np.sum(records[method]["post_unchanged_alarms"])
                    ),
                }
            )

    contrasts = {
        "sco_ce_vs_cumulative_ce": ("sco_reset_ce", "cumulative_ce"),
        "sco_ce_vs_sw_ce": ("sco_reset_ce", "sw_ce_32"),
        "sco_ce_vs_sw_ucb": ("sco_reset_ce", "sw_ucb_cv_64"),
        "natural_ucb_vs_forced_ucb": (
            "sco_reset_ucb",
            "ps_forced_reset_ucb",
        ),
    }
    effects = {}
    raw_p = {}
    for contrast, (left, right) in contrasts.items():
        for endpoint, key in (("total", "total_ex"), ("post", "post_ex")):
            name = f"{contrast}:{endpoint}"
            diff = seed_cluster_means(
                np.asarray(records[left][key])
                - np.asarray(records[right][key])
            )
            effects[name] = {
                "mean": float(diff.mean()),
                "ci": bootstrap_ci(diff, rng),
            }
            raw_p[name] = sign_flip_pvalue(diff, rng)
    adjusted = holm_adjust(raw_p)
    for name in effects:
        effects[name]["raw_p"] = raw_p[name]
        effects[name]["holm_p"] = adjusted[name]

    fieldnames = [
        "method",
        "total_excess_mean",
        "total_excess_ci_low",
        "total_excess_ci_high",
        "post_excess_mean",
        "post_excess_ci_low",
        "post_excess_ci_high",
        "ranking_loss",
        "detection",
        "pre_false_alarms_per_10k",
        "calendar_delay",
        "observation_delay",
    ]
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHODS:
            row = summaries[method]
            writer.writerow(
                {
                    "method": method,
                    "total_excess_mean": row["total_mean"],
                    "total_excess_ci_low": row["total_ci"][0],
                    "total_excess_ci_high": row["total_ci"][1],
                    "post_excess_mean": row["post_mean"],
                    "post_excess_ci_low": row["post_ci"][0],
                    "post_excess_ci_high": row["post_ci"][1],
                    "ranking_loss": row["rank_loss"],
                    "detection": row.get("detection", ""),
                    "pre_false_alarms_per_10k": row.get(
                        "pre_fa_per_10k", ""
                    ),
                    "calendar_delay": row.get("calendar_delay", ""),
                    "observation_delay": row.get("observation_delay", ""),
                }
            )

    print("FINAL 30-SEED FROZEN SYNTHETIC AGGREGATION")
    print(
        f"files={len(paths)} seeds=30 paired_instances="
        f"{len(records['sco_reset_ce']['total_ex'])}"
    )
    print(
        "method                   total_ex [95% CI]          "
        "post_ex [95% CI]           rank detect FA/10k calDelay obsDelay"
    )
    for method in METHODS:
        row = summaries[method]
        print(
            f"{method:24s} "
            f"{row['total_mean']:7.3f}% "
            f"[{row['total_ci'][0]:7.3f},{row['total_ci'][1]:7.3f}] "
            f"{row['post_mean']:7.3f}% "
            f"[{row['post_ci'][0]:7.3f},{row['post_ci'][1]:7.3f}] "
            f"{row['rank_loss']:7.3f} "
            f"{row.get('detection', np.nan):6.3f} "
            f"{row.get('pre_fa_per_10k', np.nan):7.3f} "
            f"{row.get('calendar_delay', np.nan):8.2f} "
            f"{row.get('observation_delay', np.nan):8.2f}"
        )
    print("PRIMARY PAIRED CONTRASTS: left minus right")
    print("contrast:endpoint                 effect [95% CI] raw_p holm_p")
    for name, row in effects.items():
        print(
            f"{name:36s} {row['mean']:7.3f} "
            f"[{row['ci'][0]:7.3f},{row['ci'][1]:7.3f}] "
            f"{row['raw_p']:.6g} {row['holm_p']:.6g}"
        )
    print(
        "AUDIT PASS: exact seeds 310001-310030, no duplicates, "
        "protocol/method/batch lengths consistent"
    )
    print(f"summary={args.summary_out}")


if __name__ == "__main__":
    main()
