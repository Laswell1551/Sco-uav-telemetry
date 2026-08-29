"""Paired 1D/2D/3D CV telemetry pilot.

Each scheduled packet carries every spatial axis of one UAV.  The packet age
is therefore shared across axes, while estimation costs and Whittle priorities
add across axis-specific CV models.  This tests whether the SCO result depends
on reducing a public trajectory to one projected coordinate.

Pilot results are development evidence only.  They are written with explicit
seed and dimension metadata and are not promoted to manuscript claims until a
separate formal protocol is frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from core.change_detection_cv import TwoWindowCVMomentDetector
from core.sim import W_from_pack, coeff_pack, poly_cost, topn_mask
from run_cv_piecewise_pilot import (
    effective_from_physical,
    generate_bank,
    make_estimator,
)
from run_tmc_channel_stress import mean_ci


def make_multiaxis_problem(
    seed,
    B,
    K,
    dimension,
    change_t,
    H,
    n0,
    block_length,
):
    rng = np.random.default_rng(seed)
    physical0 = np.exp(
        rng.uniform(np.log(0.01), 0.0, size=(B, K, dimension, 2))
    )
    changed = rng.random((B, K)) < 0.4
    changed[np.arange(B), rng.integers(0, K, B)] = True
    multiplier = np.where(physical0 <= 0.1, 4.0, 0.25)
    physical1 = np.where(
        changed[:, :, None, None],
        np.clip(physical0 * multiplier, 0.01, 1.0),
        physical0,
    )

    flat0 = physical0.reshape(B, K * dimension, 2)
    flat1 = physical1.reshape(B, K * dimension, 2)
    theta0_flat, c00_flat = effective_from_physical(flat0)
    theta1_flat, c01_flat = effective_from_physical(flat1)
    theta0 = theta0_flat.reshape(B, K, dimension, 3).sum(axis=2)
    theta1 = theta1_flat.reshape(B, K, dimension, 3).sum(axis=2)
    c00 = c00_flat.reshape(B, K, dimension).sum(axis=2)
    c01 = c01_flat.reshape(B, K, dimension).sum(axis=2)

    pre_flat = generate_bank(flat0, n0 + change_t, block_length, rng)
    post_flat = generate_bank(flat1, H - change_t, block_length, rng)
    pre_bank = pre_flat.reshape(
        B, K, dimension, n0 + change_t, 2
    )
    post_bank = post_flat.reshape(
        B, K, dimension, H - change_t, 2
    )
    ages0 = rng.integers(1, 8, size=(B, K)).astype(float)
    return (
        theta0,
        theta1,
        c00,
        c01,
        changed,
        pre_bank,
        post_bank,
        ages0,
    )


def multiaxis_observation(bank, seen, selected):
    B, K = selected.shape
    dimension = bank.shape[2]
    observation = np.zeros((B, K, dimension, 2), dtype=float)
    for b, k in np.argwhere(selected):
        index = int(seen[b, k])
        if index >= bank.shape[3]:
            raise AssertionError("multiaxis observation bank exhausted")
        observation[b, k] = bank[b, k, :, index]
    return observation


def aggregate_theta(flat_theta, B, K, dimension):
    return flat_theta.reshape(B, K, dimension, 3).sum(axis=2)


def run_multiaxis_policy(
    name,
    theta0,
    theta1,
    c00,
    c01,
    changed,
    pre_bank,
    post_bank,
    ages0,
    N,
    n0,
    change_t,
    H,
    detector_window=8,
    detector_threshold=5.0,
):
    B, K, dimension, _, _ = pre_bank.shape
    initial = np.transpose(
        pre_bank[:, :, :, :n0, :], (3, 0, 1, 2, 4)
    ).reshape(n0, B, K * dimension, 2)
    estimator = make_estimator(name, initial)
    detector = None
    if name.startswith("sco_reset_"):
        detector = TwoWindowCVMomentDetector(
            B,
            K * dimension,
            window=detector_window,
            threshold=detector_threshold,
        )
        all_selected = np.ones((B, K * dimension), dtype=bool)
        for initial_block in initial:
            detector.update(all_selected, initial_block)

    ages = ages0.copy()
    pre_seen = np.full((B, K), n0, dtype=int)
    post_seen = np.zeros((B, K), dtype=int)
    total = np.zeros(B)
    post_cost = np.zeros(B)
    rank_loss = np.zeros(B)
    alarms = 0

    for t in range(H):
        post = t >= change_t
        theta = theta1 if post else theta0
        c0 = c01 if post else c00
        true_pack = coeff_pack(1.0, theta)
        true_pack[..., 0] = c0
        slot_cost = poly_cost(ages, true_pack).sum(axis=1)
        total += slot_cost
        if post:
            post_cost += slot_cost
        true_score = W_from_pack(ages, true_pack)
        true_mask = topn_mask(true_score, N)

        if name == "true":
            selected = true_mask
        elif name == "max_age":
            selected = topn_mask(ages, N)
        else:
            use_ucb = (
                name == "cumulative_ucb_cv"
                or name.startswith("sw_whittle_cv_")
                or name.endswith("_ucb")
            )
            flat_theta = (
                estimator.effective_box[1] if use_ucb else estimator.mean
            )
            theta_score = aggregate_theta(
                flat_theta, B, K, dimension
            )
            selected = topn_mask(
                W_from_pack(ages, coeff_pack(1.0, theta_score)), N
            )

        rank_loss += (
            np.where(true_mask, true_score, 0.0).sum(axis=1)
            - np.where(selected, true_score, 0.0).sum(axis=1)
        )

        if estimator is not None:
            bank = post_bank if post else pre_bank
            seen = post_seen if post else pre_seen
            observation = multiaxis_observation(bank, seen, selected)
            seen += selected
            flat_selected = np.repeat(selected, dimension, axis=1)
            flat_observation = observation.reshape(
                B, K * dimension, 2
            )
            if detector is None:
                estimator.update(flat_selected, flat_observation)
            else:
                detection = detector.update(
                    flat_selected, flat_observation
                )
                alarms += int(detection["alarms"].sum())
                estimator.update_and_reset(
                    flat_selected, flat_observation, detection
                )

        ages = np.where(selected, 1.0, ages + 1.0)

    axis_count_spread = 0.0
    if estimator is not None:
        counts = estimator.count.reshape(B, K, dimension)
        axis_count_spread = float(
            np.max(counts.max(axis=2) - counts.min(axis=2))
        )
    return {
        "name": name,
        "avg_cost": total / H,
        "post_cost": post_cost / (H - change_t),
        "rank_loss": rank_loss / H,
        "alarms": alarms,
        "axis_count_spread": axis_count_spread,
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results")
    )
    args = parser.parse_args()
    if args.quick:
        seeds = tuple(20267000 + i for i in range(4))
        B, K, N, H, change_t = 2, 12, 3, 320, 160
    else:
        seeds = tuple(20268000 + i for i in range(12))
        B, K, N, H, change_t = 4, 20, 4, 800, 400
    n0, block_length = 8, 64
    dimensions = (1, 2, 3)
    methods = (
        "true",
        "max_age",
        "cumulative_ce",
        "cumulative_ucb_cv",
        "sw_whittle_cv_64",
        "sco_reset_ce",
        "sco_reset_ucb",
    )

    raw = []
    for seed in seeds:
        for dimension in dimensions:
            problem = make_multiaxis_problem(
                seed,
                B,
                K,
                dimension,
                change_t,
                H,
                n0,
                block_length,
            )
            common = (*problem, N, n0, change_t, H)
            oracle = run_multiaxis_policy("true", *common)
            for method in methods:
                result = (
                    oracle
                    if method == "true"
                    else run_multiaxis_policy(method, *common)
                )
                total_ex = 100.0 * (
                    result["avg_cost"] / oracle["avg_cost"] - 1.0
                )
                post_ex = 100.0 * (
                    result["post_cost"] / oracle["post_cost"] - 1.0
                )
                raw.append(
                    {
                        "seed": seed,
                        "dimension": dimension,
                        "method": method,
                        "total_excess_mean": float(total_ex.mean()),
                        "post_excess_mean": float(post_ex.mean()),
                        "rank_loss_mean": float(
                            result["rank_loss"].mean()
                        ),
                        "alarms": result["alarms"],
                        "axis_count_spread": result[
                            "axis_count_spread"
                        ],
                    }
                )
            print(
                f"seed={seed} dimension={dimension} complete",
                flush=True,
            )

    summary = []
    for dimension in dimensions:
        for method in methods:
            group = [
                row
                for row in raw
                if row["dimension"] == dimension
                and row["method"] == method
            ]
            record = {"dimension": dimension, "method": method}
            for metric in (
                "total_excess_mean",
                "post_excess_mean",
                "rank_loss_mean",
            ):
                mean, low, high = mean_ci(
                    [row[metric] for row in group]
                )
                record[metric] = mean
                record[f"{metric}_ci_low"] = low
                record[f"{metric}_ci_high"] = high
            record["max_axis_count_spread"] = max(
                row["axis_count_spread"] for row in group
            )
            summary.append(record)

    mode = "quick" if args.quick else "formal"
    write_csv(
        args.out_dir / f"tmc_multiaxis_{mode}_raw.csv", raw
    )
    write_csv(
        args.out_dir / f"tmc_multiaxis_{mode}_summary.csv", summary
    )
    metadata = {
        "mode": mode,
        "evidence_status": (
            "development_pilot"
            if args.quick
            else "formal_after_protocol_freeze"
        ),
        "seeds": seeds,
        "dimensions": dimensions,
        "B": B,
        "K": K,
        "N": N,
        "H": H,
        "change_t": change_t,
        "n0": n0,
        "block_length": block_length,
        "packet_semantics": (
            "one selected UAV packet reveals all coordinate axes; "
            "age is shared and CV costs/indices add across axes"
        ),
        "pairing": (
            "same seed, latent per-axis parameters, change mask, "
            "and selected-observation banks within each dimension"
        ),
        "claim_boundary": (
            "independent-axis CV generalization; correlated-axis and "
            "constant-acceleration mismatch require separate tests"
        ),
    }
    (
        args.out_dir / f"tmc_multiaxis_{mode}_meta.json"
    ).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{mode} multiaxis evaluation complete", flush=True)


if __name__ == "__main__":
    main()
