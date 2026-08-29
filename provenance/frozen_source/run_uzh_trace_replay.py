"""Frozen UZH-FPV trace-driven scheduling replay (protocol v1)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.change_detection_cv import (
    ResettableOnlineCVMomentEstimator,
    TwoWindowCVMomentDetector,
)
from core.instances import pbar_batch
from core.online_cv_moments_stable import OnlineCVMomentEstimator
from core.sim import W_from_pack, coeff_pack, poly_cost, topn_mask
from core.sliding_cv_moments import SlidingWindowCVMomentEstimator
from core.uzh_fpv_replay_v2 import load_all_replay_traces


METHODS = (
    "oracle",
    "cumulative_ce",
    "cumulative_ucb_cv",
    "sw_ce_32",
    "sw_ucb_cv_64",
    "sco_reset_ce",
    "sco_reset_ucb",
    "forced_reset_ucb",
    "aoi",
    "round_robin",
)


def resample(values, length):
    old = np.linspace(0.0, 1.0, len(values))
    new = np.linspace(0.0, 1.0, length)
    return np.interp(new, old, values)


def trailing_energy(position, width=16):
    d2 = np.diff(position, n=2)
    square = d2 * d2
    cumulative = np.concatenate(([0.0], np.cumsum(square)))
    out = np.empty(position.size)
    for t in range(position.size):
        end = min(max(t - 1, 0), square.size)
        start = max(0, end - width)
        out[t] = (
            (cumulative[end] - cumulative[start]) / max(end - start, 1)
            if end
            else square[0]
        )
    return out


def physical_to_theta(physical):
    q = physical[..., 0]
    r = physical[..., 1]
    _, p12, p22 = pbar_batch(1.0, q.ravel(), r.ravel())
    shape = q.shape
    return np.stack(
        [q, p12.reshape(shape), p22.reshape(shape)], axis=-1
    )


def make_episode(trace_bank, seed, K=12, length=640):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(trace_bank), size=K, replace=False)
    energy = np.empty((K, length))
    names = []
    for k, index in enumerate(chosen):
        trace = trace_bank[int(index)]
        xyz = np.column_stack(
            [resample(trace["xyz"][:, axis], length) for axis in range(3)]
        )
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        position = (xyz - xyz[0]) @ direction
        energy[k] = trailing_energy(position)
        names.append(trace["name"])
    scale = max(float(np.quantile(energy, 0.95)), 1e-12)
    q = np.clip(0.01 + 0.99 * energy / scale, 0.01, 1.0)
    physical = np.stack([q, np.full_like(q, 0.05)], axis=-1)
    return physical, names


def make_estimator(name, initial):
    common = dict(
        confidence_scale=3.0,
        variance_floor=0.01,
        variance_ceiling=1.0,
    )
    if name in ("cumulative_ce", "cumulative_ucb_cv"):
        return OnlineCVMomentEstimator(initial, **common)
    if name == "sw_ce_32":
        return SlidingWindowCVMomentEstimator(initial, window=32, **common)
    if name == "sw_ucb_cv_64":
        return SlidingWindowCVMomentEstimator(initial, window=64, **common)
    if name in ("sco_reset_ce", "sco_reset_ucb", "forced_reset_ucb"):
        return ResettableOnlineCVMomentEstimator(initial, **common)
    return None


def round_robin_mask(K, N, slot):
    mask = np.zeros((1, K), dtype=bool)
    start = (slot * N) % K
    mask[0, (start + np.arange(N)) % K] = True
    return mask


def run_method(name, physical, N=3, n0=8):
    K, length, _ = physical.shape
    H = length - n0
    initial = np.transpose(physical[:, :n0, :], (1, 0, 2))[:, None, :, :]
    estimator = make_estimator(name, initial)
    detector = None
    if name in ("sco_reset_ce", "sco_reset_ucb", "forced_reset_ucb"):
        detector = TwoWindowCVMomentDetector(1, K, window=8, threshold=5.0)
        all_selected = np.ones((1, K), dtype=bool)
        for block in initial:
            detector.update(all_selected, block)

    ages = np.ones((1, K))
    total = 0.0
    rank_loss = 0.0
    alarms = 0
    max_gap = np.zeros(K, dtype=int)
    current_gap = np.zeros(K, dtype=int)
    rr_index = 0

    for h in range(H):
        current = physical[:, n0 + h, :][None, :, :]
        theta_true = physical_to_theta(current)
        C_true = coeff_pack(1.0, theta_true)
        W_true = W_from_pack(ages, C_true)
        oracle_mask = topn_mask(W_true, N)
        total += float(poly_cost(ages, C_true).sum())

        if name == "oracle":
            selected = oracle_mask
        elif name == "aoi":
            selected = topn_mask(ages, N)
        elif name == "round_robin":
            selected = round_robin_mask(K, N, rr_index)
            rr_index += 1
        elif name == "forced_reset_ucb" and h % 50 == 0:
            selected = round_robin_mask(K, N, rr_index)
            rr_index += 1
        else:
            use_ucb = name in (
                "cumulative_ucb_cv",
                "sw_ucb_cv_64",
                "sco_reset_ucb",
                "forced_reset_ucb",
            )
            theta_score = estimator.effective_box[1] if use_ucb else estimator.mean
            selected = topn_mask(
                W_from_pack(ages, coeff_pack(1.0, theta_score)), N
            )

        rank_loss += float(
            np.where(oracle_mask, W_true, 0.0).sum()
            - np.where(selected, W_true, 0.0).sum()
        )

        if estimator is not None:
            if detector is None:
                estimator.update(selected, current)
            else:
                result = detector.update(selected, current)
                alarms += int(result["alarms"].sum())
                estimator.update_and_reset(selected, current, result)

        current_gap = np.where(selected[0], 0, current_gap + 1)
        max_gap = np.maximum(max_gap, current_gap)
        ages = np.where(selected, 1.0, ages + 1.0)

    return {
        "cost": total / H,
        "rank_loss": rank_loss / H,
        "max_gap": int(max_gap.max()),
        "alarms_per_10k_arm_slots": 10000.0 * alarms / (H * K),
    }


def bootstrap_mean_ci(values, rng, draws=20000):
    values = np.asarray(values, dtype=float)
    index = rng.integers(0, values.size, size=(draws, values.size))
    means = values[index].mean(axis=1)
    return [
        float(values.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def main():
    trace_bank = load_all_replay_traces(Path("data/uzh_fpv_gt"))
    seeds = list(range(410001, 410031))
    raw = []
    for seed in seeds:
        physical, names = make_episode(trace_bank, seed)
        rows = {name: run_method(name, physical) for name in METHODS}
        oracle = rows["oracle"]["cost"]
        for name in METHODS:
            rows[name]["excess_pct"] = 100.0 * (rows[name]["cost"] / oracle - 1.0)
        raw.append({"seed": seed, "sequences": names, "methods": rows})

    rng = np.random.default_rng(910247)
    summary = {}
    for name in METHODS:
        summary[name] = {}
        for metric in (
            "excess_pct",
            "rank_loss",
            "max_gap",
            "alarms_per_10k_arm_slots",
        ):
            values = [episode["methods"][name][metric] for episode in raw]
            summary[name][metric] = bootstrap_mean_ci(values, rng)

    natural = np.array(
        [row["methods"]["sco_reset_ucb"]["excess_pct"] for row in raw]
    )
    forced = np.array(
        [row["methods"]["forced_reset_ucb"]["excess_pct"] for row in raw]
    )
    output = {
        "protocol": {
            "dataset": "UZH-FPV public Leica ground truth",
            "seeds": seeds,
            "K": 12,
            "N": 3,
            "length": 640,
            "n0": 8,
            "H": 632,
            "methods": METHODS,
        },
        "summary_mean_ci95": summary,
        "paired_natural_minus_forced_excess_pct": bootstrap_mean_ci(
            natural - forced, rng
        ),
        "episodes": raw,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

