"""Paper-facing certificate-width sweep for the Q/TMC extension.

The experiment keeps the latent CV problem and observation-indexed banks
paired across confidence scales.  It reports joint physical/index coverage,
decision-certificate firing and error rates, and top-N mismatch rates.

This is an empirical calibration study of the proxy radius.  It does not turn
the proxy into a finite-sample confidence sequence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from core.instances import pbar_batch
from core.online_cv_moments_stable import OnlineCVMomentEstimator
from core.sim import W_from_pack, coeff_pack, topn_mask
from run_cv_piecewise_pilot import bank_observation, make_problem


SCALES = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
METRICS = (
    "physical_joint_coverage",
    "index_joint_coverage",
    "certificate_rate",
    "certificate_error_rate",
    "mismatch_rate",
    "mean_normalized_interval_width",
)


def certificate_for_mask(mask: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    selected_lo = np.where(mask, lo, np.inf).min(axis=1)
    unselected_hi = np.where(~mask, hi, -np.inf).max(axis=1)
    return selected_lo > unselected_hi


def run_one(seed: int, scale: float, B: int, K: int, N: int, H: int,
            n0: int, block_length: int):
    # The shared piecewise generator requires at least two post-change blocks.
    # Generate two unused tail slots while keeping the first H slots stationary.
    problem = make_problem(seed, B, K, H, H + 2, n0, block_length)
    physical, _, theta_true, _, c0, _, _, pre_bank, _, ages0 = problem
    initial = np.transpose(pre_bank[:, :, :n0, :], (2, 0, 1, 3))
    estimator = OnlineCVMomentEstimator(
        initial,
        confidence_scale=scale,
        variance_floor=0.01,
        variance_ceiling=1.0,
    )
    ages = ages0.copy()
    seen = np.full((B, K), n0, dtype=int)
    true_pack = coeff_pack(1.0, theta_true)
    true_pack[..., 0] = c0

    physical_joint = 0
    index_joint = 0
    certified = 0
    wrong_certified = 0
    wrong_certified_on_index_cover = 0
    mismatch_total = 0
    normalized_width_sum = 0.0

    for _ in range(H):
        physical_lo, physical_hi = estimator.physical_box
        theta_lo, theta_hi = estimator.effective_box
        w_lo = W_from_pack(ages, coeff_pack(1.0, theta_lo))
        w_hi = W_from_pack(ages, coeff_pack(1.0, theta_hi))
        w_true = W_from_pack(ages, true_pack)

        selected = topn_mask(w_hi, N)
        true_selected = topn_mask(w_true, N)
        cert = certificate_for_mask(selected, w_lo, w_hi)
        mismatch = np.any(selected != true_selected, axis=1)
        physical_cover = np.all(
            (physical >= physical_lo) & (physical <= physical_hi),
            axis=(1, 2),
        )
        index_cover = np.all(
            (w_true >= w_lo) & (w_true <= w_hi),
            axis=1,
        )

        physical_joint += int(physical_cover.sum())
        index_joint += int(index_cover.sum())
        certified += int(cert.sum())
        wrong_certified += int((cert & mismatch).sum())
        wrong_certified_on_index_cover += int(
            (cert & mismatch & index_cover).sum()
        )
        mismatch_total += int(mismatch.sum())
        normalized_width_sum += float(
            np.mean((w_hi - w_lo) / np.maximum(np.abs(w_true), 1e-9))
        )

        observation = bank_observation(pre_bank, seen, selected)
        seen += selected
        estimator.update(selected, observation)
        ages = np.where(selected, 1.0, ages + 1.0)

    if wrong_certified_on_index_cover:
        raise AssertionError(
            "a wrong certificate occurred despite simultaneous index coverage"
        )
    batch_slots = B * H
    return {
        "seed": seed,
        "scale": scale,
        "B": B,
        "K": K,
        "N": N,
        "H": H,
        "n0": n0,
        "block_length": block_length,
        "physical_joint_coverage": physical_joint / batch_slots,
        "index_joint_coverage": index_joint / batch_slots,
        "certificate_rate": certified / batch_slots,
        "certificate_error_rate": (
            wrong_certified / certified if certified else np.nan
        ),
        "mismatch_rate": mismatch_total / batch_slots,
        "mean_normalized_interval_width": normalized_width_sum / H,
        "certified_count": certified,
        "wrong_certified_count": wrong_certified,
        "wrong_certified_on_index_coverage": wrong_certified_on_index_cover,
    }


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    half = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.quick:
        seeds, B, K, N, H, n0, block_length = range(2), 2, 12, 3, 200, 8, 64
    else:
        seeds, B, K, N, H, n0, block_length = (
            range(12), 4, 20, 4, 800, 8, 64
        )

    raw = []
    for seed_offset in seeds:
        seed = 20260900 + seed_offset
        for scale in SCALES:
            row = run_one(
                seed, scale, B, K, N, H, n0, block_length
            )
            raw.append(row)
            print(
                f"seed={seed} scale={scale:g} "
                f"joint={row['index_joint_coverage']:.3f} "
                f"cert={row['certificate_rate']:.3f} "
                f"wrong={row['wrong_certified_count']}",
                flush=True,
            )

    summary = []
    for scale in SCALES:
        selected = [row for row in raw if row["scale"] == scale]
        out = {"scale": scale, "seeds": len(selected)}
        for metric in METRICS:
            mean, low, high = mean_ci([row[metric] for row in selected])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_ci_low"] = low
            out[f"{metric}_ci_high"] = high
        out["certified_count"] = sum(row["certified_count"] for row in selected)
        out["wrong_certified_count"] = sum(
            row["wrong_certified_count"] for row in selected
        )
        out["wrong_certified_on_index_coverage"] = sum(
            row["wrong_certified_on_index_coverage"] for row in selected
        )
        summary.append(out)

    suffix = "_quick" if args.quick else ""
    write_csv(args.out_dir / f"tmc_certificate_sweep_raw{suffix}.csv", raw)
    write_csv(
        args.out_dir / f"tmc_certificate_sweep_summary{suffix}.csv", summary
    )
    metadata = {
        "mode": "quick" if args.quick else "paper",
        "seeds": [20260900 + s for s in seeds],
        "B": B,
        "K": K,
        "N": N,
        "H": H,
        "n0": n0,
        "block_length": block_length,
        "scales": list(SCALES),
        "paired_design": (
            "same latent problem and observation-indexed bank across scales"
        ),
        "claim_boundary": (
            "empirical calibration of a proxy radius; no finite-sample "
            "confidence-sequence claim"
        ),
    }
    (args.out_dir / f"tmc_certificate_sweep_meta{suffix}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("certificate sweep complete", flush=True)


if __name__ == "__main__":
    main()
