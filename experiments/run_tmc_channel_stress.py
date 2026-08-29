"""Packet-delivery, fixed-delay, and capacity stress tests for SCO-reset-UCB.

Every policy sees the same latent problem, observation-indexed CV blocks, and
attempt-indexed channel uniforms in a setting.  A selected stream updates the
remote age and online estimator only after successful delivery.  With delay d,
a delivered packet resets the next-slot age to d+1.

The channel-matched true-model Whittle scheduler is an oracle reference, not
the globally optimal policy under delayed/lossy delivery.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from core.change_detection_cv import (
    ResettableOnlineCVMomentEstimator,
    TwoWindowCVMomentDetector,
)
from core.online_cv_moments_stable import OnlineCVMomentEstimator
from core.sim import W_from_pack, coeff_pack, poly_cost, topn_mask
from run_cv_piecewise_pilot import (
    bank_observation,
    make_problem,
    round_robin_mask,
)


METHODS = (
    "cumulative_ucb_cv",
    "sco_reset_ucb",
    "ps_forced_reset_ucb",
)


def make_estimator(name, initial):
    common = dict(
        confidence_scale=3.0,
        variance_floor=0.01,
        variance_ceiling=1.0,
    )
    if name == "cumulative_ucb_cv":
        return OnlineCVMomentEstimator(initial, **common)
    if name in (
        "sco_reset_ucb",
        "ps_forced_reset_ucb",
        "inflight_sco_ucb",
    ):
        return ResettableOnlineCVMomentEstimator(initial, **common)
    return None


def run_policy_channel(
    name,
    theta0,
    theta1,
    c00,
    c01,
    changed,
    pre_bank,
    post_bank,
    channel_uniform,
    ages0,
    N,
    n0,
    change_t,
    H,
    success_probability,
    delay,
    detector_window=8,
    detector_threshold=5.0,
    explore_period=50,
    inflight_beta=1.0,
):
    B, K, _ = theta0.shape
    initial = np.transpose(pre_bank[:, :, :n0, :], (2, 0, 1, 3))
    estimator = make_estimator(name, initial)
    detector = None
    if name in (
        "sco_reset_ucb",
        "ps_forced_reset_ucb",
        "inflight_sco_ucb",
    ):
        detector = TwoWindowCVMomentDetector(
            B, K, window=detector_window, threshold=detector_threshold
        )
        all_selected = np.ones((B, K), dtype=bool)
        for block in initial:
            detector.update(all_selected, block)

    ages = ages0.copy()
    pre_attempt = np.full((B, K), n0, dtype=int)
    post_attempt = np.zeros((B, K), dtype=int)
    channel_attempt = np.zeros((B, K), dtype=int)
    post_delivered = np.zeros((B, K), dtype=int)
    total = np.zeros(B)
    pre_cost = np.zeros(B)
    post_cost = np.zeros(B)
    delivered_total = 0
    attempted_total = 0
    pre_alarms = np.zeros(B, dtype=int)
    post_unchanged_alarms = np.zeros(B, dtype=int)
    first_calendar_delay = np.full((B, K), -1, dtype=int)
    first_observation_delay = np.full((B, K), -1, dtype=int)
    pending = [[] for _ in range(H + delay + 1)]
    inflight_count = np.zeros((B, K), dtype=int)
    redundant_attempted_total = 0
    max_inflight_count = 0
    exploration_index = 0

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
        else:
            pre_cost += slot_cost

        w_true = W_from_pack(ages, true_pack)
        forced = (
            name == "ps_forced_reset_ucb"
            and explore_period
            and t % explore_period == 0
        )
        if name == "true":
            selected = topn_mask(w_true, N)
        elif forced:
            selected = round_robin_mask(B, K, N, exploration_index)
            exploration_index += 1
        else:
            _, theta_hi = estimator.effective_box
            score = W_from_pack(
                ages, coeff_pack(1.0, theta_hi)
            )
            if name == "inflight_sco_ucb":
                if inflight_beta < 0:
                    raise ValueError("inflight_beta must be nonnegative")
                score = score / (
                    1.0 + inflight_beta * inflight_count
                )
            selected = topn_mask(score, N)

        attempted_total += int(selected.sum())
        redundant_attempted_total += int(
            (selected & (inflight_count > 0)).sum()
        )
        inflight_count += selected
        max_inflight_count = max(
            max_inflight_count, int(inflight_count.max())
        )
        observation = (
            bank_observation(post_bank, post_attempt, selected)
            if post
            else bank_observation(pre_bank, pre_attempt, selected)
        )
        if post:
            post_attempt += selected
        else:
            pre_attempt += selected

        success = np.zeros((B, K), dtype=bool)
        for b, k in np.argwhere(selected):
            attempt = int(channel_attempt[b, k])
            if attempt >= channel_uniform.shape[2]:
                raise AssertionError("channel bank exhausted")
            success[b, k] = (
                channel_uniform[b, k, attempt] < success_probability
            )
            channel_attempt[b, k] += 1
        pending[t + delay].append(
            (success, observation, post, selected)
        )

        delivered = np.zeros((B, K), dtype=bool)
        for (
            delivered_mask,
            delivered_observation,
            generated_post,
            attempted_mask,
        ) in pending[t]:
            inflight_count -= attempted_mask
            if np.any(inflight_count < 0):
                raise AssertionError("negative in-flight count")
            delivered |= delivered_mask
            delivered_total += int(delivered_mask.sum())
            if estimator is None or not np.any(delivered_mask):
                continue
            if generated_post:
                post_delivered += delivered_mask
            if detector is None:
                estimator.update(delivered_mask, delivered_observation)
            else:
                detection = detector.update(
                    delivered_mask, delivered_observation
                )
                alarms = detection["alarms"]
                estimator.update_and_reset(
                    delivered_mask, delivered_observation, detection
                )
                if t < change_t or not generated_post:
                    # A pre-change packet can arrive after the change under
                    # fixed delay.  It is not evidence of the new regime.
                    pre_alarms += alarms.sum(axis=1)
                else:
                    post_unchanged_alarms += (alarms & ~changed).sum(axis=1)
                    first = (
                        alarms & changed & (first_calendar_delay < 0)
                    )
                    first_calendar_delay[first] = t - change_t + 1
                    first_observation_delay[first] = post_delivered[first]

        ages = np.where(delivered, float(delay + 1), ages + 1.0)

    changed_total = int(changed.sum())
    detected = (first_calendar_delay >= 0) & changed
    return {
        "name": name,
        "avg_cost": total / H,
        "pre_cost": pre_cost / change_t,
        "post_cost": post_cost / (H - change_t),
        "delivery_rate": delivered_total / attempted_total,
        "redundant_attempt_rate": (
            redundant_attempted_total / attempted_total
        ),
        "max_inflight_count": max_inflight_count,
        "pre_alarms": int(pre_alarms.sum()),
        "post_unchanged_alarms": int(post_unchanged_alarms.sum()),
        "detected_fraction": (
            float(detected.sum() / changed_total) if detector else np.nan
        ),
        "calendar_delay": (
            float(first_calendar_delay[detected].mean())
            if np.any(detected) else np.nan
        ),
        "observation_delay": (
            float(first_observation_delay[detected].mean())
            if np.any(detected) else np.nan
        ),
    }


def settings():
    rows = []
    for probability in (1.0, 0.9, 0.8, 0.7):
        rows.append(("delivery", probability, 0, 4))
    for delay in (0, 1, 3, 5):
        rows.append(("delay", 0.9, delay, 4))
    for capacity in (2, 4, 8):
        rows.append(("capacity", 0.9, 1, capacity))
    unique = []
    for row in rows:
        if row not in unique:
            unique.append(row)
    return unique


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
        seed_offsets, B, K, H, n0, change_t, block_length = (
            range(2), 2, 12, 240, 8, 120, 64
        )
    else:
        seed_offsets, B, K, H, n0, change_t, block_length = (
            range(12), 4, 20, 800, 8, 400, 64
        )

    raw = []
    for seed_offset in seed_offsets:
        seed = 20261000 + seed_offset
        problem = make_problem(
            seed, B, K, change_t, H, n0, block_length
        )
        (
            _, _, theta0, theta1, c00, c01, changed,
            pre_bank, post_bank, ages0,
        ) = problem
        channel_uniform = np.random.default_rng(seed + 50000).random(
            (B, K, H + n0)
        )
        for family, probability, delay, capacity in settings():
            policy_rows = {}
            for name in ("true",) + METHODS:
                policy_rows[name] = run_policy_channel(
                    name,
                    theta0,
                    theta1,
                    c00,
                    c01,
                    changed,
                    pre_bank,
                    post_bank,
                    channel_uniform,
                    ages0,
                    capacity,
                    n0,
                    change_t,
                    H,
                    probability,
                    delay,
                )
            oracle = policy_rows["true"]
            for name in METHODS:
                result = policy_rows[name]
                raw.append(
                    {
                        "seed": seed,
                        "family": family,
                        "success_probability": probability,
                        "delay_slots": delay,
                        "capacity": capacity,
                        "capacity_ratio": capacity / K,
                        "method": name,
                        "total_excess_pct": float(np.mean(
                            100 * (result["avg_cost"] / oracle["avg_cost"] - 1)
                        )),
                        "pre_excess_pct": float(np.mean(
                            100 * (result["pre_cost"] / oracle["pre_cost"] - 1)
                        )),
                        "post_excess_pct": float(np.mean(
                            100 * (result["post_cost"] / oracle["post_cost"] - 1)
                        )),
                        "delivery_rate": result["delivery_rate"],
                        "detected_fraction": result["detected_fraction"],
                        "calendar_delay": result["calendar_delay"],
                        "observation_delay": result["observation_delay"],
                        "pre_alarms": result["pre_alarms"],
                        "post_unchanged_alarms": result[
                            "post_unchanged_alarms"
                        ],
                    }
                )
            print(
                f"seed={seed} {family} p={probability:g} d={delay} "
                f"N={capacity}",
                flush=True,
            )

    summary = []
    keys = (
        "total_excess_pct",
        "pre_excess_pct",
        "post_excess_pct",
        "delivery_rate",
        "detected_fraction",
        "calendar_delay",
        "observation_delay",
        "pre_alarms",
        "post_unchanged_alarms",
    )
    group_keys = (
        "family", "success_probability", "delay_slots",
        "capacity", "capacity_ratio", "method",
    )
    groups = {}
    for row in raw:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    for key, rows in groups.items():
        out = dict(zip(group_keys, key))
        out["seeds"] = len(rows)
        for metric in keys:
            mean, low, high = mean_ci([row[metric] for row in rows])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_ci_low"] = low
            out[f"{metric}_ci_high"] = high
        summary.append(out)

    suffix = "_quick" if args.quick else ""
    write_csv(args.out_dir / f"tmc_channel_stress_raw{suffix}.csv", raw)
    write_csv(
        args.out_dir / f"tmc_channel_stress_summary{suffix}.csv", summary
    )
    metadata = {
        "mode": "quick" if args.quick else "paper",
        "seeds": [20261000 + s for s in seed_offsets],
        "B": B,
        "K": K,
        "H": H,
        "change_t": change_t,
        "n0": n0,
        "block_length": block_length,
        "settings": [
            {
                "family": family,
                "success_probability": probability,
                "delay_slots": delay,
                "capacity": capacity,
            }
            for family, probability, delay, capacity in settings()
        ],
        "pairing": (
            "same latent problem, observation-indexed CV bank, and "
            "attempt-indexed channel uniforms within each seed"
        ),
        "oracle_boundary": (
            "true-model Whittle under the same channel sample path; not a "
            "global optimum for delayed/lossy control"
        ),
    }
    (args.out_dir / f"tmc_channel_stress_meta{suffix}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("channel stress complete", flush=True)


if __name__ == "__main__":
    main()
