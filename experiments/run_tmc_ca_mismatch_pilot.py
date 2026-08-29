"""Constant-acceleration model-class mismatch pilot.

Truth uses a three-state constant-acceleration (CA) Kalman model with white
jerk and position measurements.  The comparator is a deliberately favorable
CV-class surrogate: for every arm, a nonnegative cubic age-cost polynomial is
least-squares fitted to the exact CA cost over young ages 1..8.  Scheduling
then uses the closed-form cubic Whittle index.  Divergence at larger ages is
therefore model-class extrapolation error rather than online estimation noise.

This is a failure-boundary pilot, not evidence that SCO handles CA dynamics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from core.sim import W_from_pack, topn_mask
from run_tmc_channel_stress import mean_ci


def ca_matrices(jerk_variance, T=1.0):
    F = np.array(
        [
            [1.0, T, 0.5 * T**2],
            [0.0, 1.0, T],
            [0.0, 0.0, 1.0],
        ]
    )
    Q0 = np.array(
        [
            [T**5 / 20.0, T**4 / 8.0, T**3 / 6.0],
            [T**4 / 8.0, T**3 / 3.0, T**2 / 2.0],
            [T**3 / 6.0, T**2 / 2.0, T],
        ]
    )
    return F, float(jerk_variance) * Q0


def steady_posterior_ca(jerk_variance, measurement_variance, T=1.0):
    F, Q = ca_matrices(jerk_variance, T)
    H = np.array([[1.0, 0.0, 0.0]])
    R = float(measurement_variance)
    P = np.eye(3)
    for _ in range(20_000):
        prior = F @ P @ F.T + Q
        S = float((H @ prior @ H.T)[0, 0] + R)
        gain = prior @ H.T / S
        updated = prior - gain @ H @ prior
        if np.max(np.abs(updated - P)) < 1e-13:
            return updated, F, Q
        P = updated
    raise RuntimeError("CA Riccati iteration did not converge")


def ca_position_costs(
    jerk_variance,
    measurement_variance,
    max_age,
    T=1.0,
):
    P, F, Q = steady_posterior_ca(
        jerk_variance, measurement_variance, T
    )
    costs = np.zeros(max_age + 2)
    covariance = P.copy()
    for age in range(1, max_age + 2):
        covariance = F @ covariance @ F.T + Q
        costs[age] = covariance[0, 0]
    return costs


def general_whittle_from_cost(cost):
    """Index for deterministic age increment and active reset to age one."""
    cost = np.asarray(cost, dtype=float)
    cumulative = np.cumsum(cost)
    index = np.zeros_like(cost)
    ages = np.arange(1, cost.size - 1)
    index[ages] = (
        ages * cost[ages + 1] - (cumulative[ages] - cost[0])
    )
    return index


def fit_cubic_cv_surrogate(cost, fit_max_age=8):
    ages = np.arange(1, fit_max_age + 1, dtype=float)
    design = np.stack(
        [np.ones_like(ages), ages, ages**2, ages**3], axis=1
    )
    coeffs, _ = nnls(design, cost[1 : fit_max_age + 1])
    return coeffs


def cubic_cost(age, coeffs):
    age = np.asarray(age, dtype=float)
    return (
        coeffs[..., 0]
        + coeffs[..., 1] * age
        + coeffs[..., 2] * age**2
        + coeffs[..., 3] * age**3
    )


def build_instance(seed, K, spatial_dimension, max_age):
    rng = np.random.default_rng(seed)
    jerk = np.exp(
        rng.uniform(
            np.log(1e-4),
            np.log(5e-2),
            size=(K, spatial_dimension),
        )
    )
    measurement = np.exp(
        rng.uniform(
            np.log(1e-2),
            np.log(1.0),
            size=(K, spatial_dimension),
        )
    )
    exact_cost = np.zeros((K, max_age + 2))
    surrogate_coeffs = np.zeros((K, 4))
    for k in range(K):
        for d in range(spatial_dimension):
            axis_cost = ca_position_costs(
                jerk[k, d], measurement[k, d], max_age
            )
            exact_cost[k] += axis_cost
        surrogate_coeffs[k] = fit_cubic_cv_surrogate(exact_cost[k])
    exact_index = np.stack(
        [general_whittle_from_cost(exact_cost[k]) for k in range(K)]
    )
    return exact_cost, exact_index, surrogate_coeffs


def run_policy(name, exact_cost, exact_index, surrogate_coeffs, N, H, ages0):
    K = exact_cost.shape[0]
    ages = ages0.copy()
    total = 0.0
    oracle_disagreement = 0
    surrogate_pack = surrogate_coeffs[None, :, :]
    for _ in range(H):
        integer_age = ages.astype(int)
        total += float(
            exact_cost[np.arange(K), integer_age].sum()
        )
        oracle_score = exact_index[np.arange(K), integer_age]
        oracle = topn_mask(oracle_score[None], N)[0]
        if name == "ca_index":
            selected = oracle
        elif name == "cubic_cv_surrogate":
            selected = topn_mask(
                W_from_pack(ages[None], surrogate_pack), N
            )[0]
        elif name == "max_age":
            selected = topn_mask(ages[None], N)[0]
        else:
            raise ValueError(f"unknown policy: {name}")
        oracle_disagreement += int(np.sum(selected != oracle) // 2)
        ages = np.where(selected, 1.0, ages + 1.0)
        if int(ages.max()) >= exact_cost.shape[1] - 1:
            raise AssertionError("CA cost lookup exhausted")
    return {
        "average_cost": total / H,
        "action_disagreement_rate": oracle_disagreement / (H * N),
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
    parser.add_argument("--capacity", type=int)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results")
    )
    args = parser.parse_args()
    if args.quick:
        seeds = tuple(20269000 + i for i in range(8))
        K, N, H = 12, 3, 600
    else:
        seeds = tuple(20270000 + i for i in range(30))
        K, N, H = 20, 4, 2000
    if args.capacity is not None:
        if not 1 <= args.capacity < K:
            raise SystemExit("capacity must lie in [1,K)")
        N = args.capacity
    dimensions = (1, 2, 3)
    methods = ("ca_index", "cubic_cv_surrogate", "max_age")
    max_age = H + 16

    raw = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 1000)
        ages0 = rng.integers(1, 8, size=K).astype(float)
        for dimension in dimensions:
            exact_cost, exact_index, coeffs = build_instance(
                seed, K, dimension, max_age
            )
            results = {
                method: run_policy(
                    method,
                    exact_cost,
                    exact_index,
                    coeffs,
                    N,
                    H,
                    ages0,
                )
                for method in methods
            }
            reference_cost = results["ca_index"]["average_cost"]
            for method in methods:
                diagnostic = {}
                for age in (8, 16, 32, 64):
                    true_value = exact_cost[:, age]
                    fitted_value = cubic_cost(age, coeffs)
                    diagnostic[f"cost_ratio_age_{age}"] = float(
                        np.mean(fitted_value / true_value)
                    )
                raw.append(
                    {
                        "seed": seed,
                        "spatial_dimension": dimension,
                        "method": method,
                        "average_cost": results[method][
                            "average_cost"
                        ],
                        "relative_vs_ca_index_pct": 100.0
                        * (
                            results[method]["average_cost"]
                            / reference_cost
                            - 1.0
                        ),
                        "action_disagreement_rate": results[method][
                            "action_disagreement_rate"
                        ],
                        **diagnostic,
                    }
                )
            print(
                f"seed={seed} CA dimension={dimension} complete",
                flush=True,
            )

    summary = []
    for dimension in dimensions:
        for method in methods:
            group = [
                row
                for row in raw
                if row["spatial_dimension"] == dimension
                and row["method"] == method
            ]
            record = {
                "spatial_dimension": dimension,
                "method": method,
            }
            for metric in (
                "relative_vs_ca_index_pct",
                "action_disagreement_rate",
                "cost_ratio_age_8",
                "cost_ratio_age_16",
                "cost_ratio_age_32",
                "cost_ratio_age_64",
            ):
                mean, low, high = mean_ci(
                    [row[metric] for row in group]
                )
                record[f"{metric}_mean"] = mean
                record[f"{metric}_ci_low"] = low
                record[f"{metric}_ci_high"] = high
            summary.append(record)

    mode = "quick" if args.quick else "formal"
    prefix = f"tmc_ca_mismatch_{mode}_v2_n{N}"
    write_csv(args.out_dir / f"{prefix}_raw.csv", raw)
    write_csv(args.out_dir / f"{prefix}_summary.csv", summary)
    metadata = {
        "mode": mode,
        "evidence_status": (
            "development_failure_boundary"
            if args.quick
            else "formal_after_protocol_freeze"
        ),
        "seeds": seeds,
        "K": K,
        "N": N,
        "H": H,
        "spatial_dimensions": dimensions,
        "truth": (
            "independent spatial axes, each a three-state CA model with "
            "white jerk and position measurement"
        ),
        "surrogate": (
            "per-arm nonnegative cubic age-cost fit on exact CA ages 1..8; "
            "no online estimation noise"
        ),
        "claim_boundary": (
            "tests CV model-class extrapolation under CA truth; it does not "
            "claim SCO learns CA dynamics or that the exact-CA Whittle "
            "index is globally optimal for the finite coupled system"
        ),
    }
    (
        args.out_dir / f"{prefix}_meta.json"
    ).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"{mode} CA mismatch evaluation complete", flush=True)


if __name__ == "__main__":
    main()
