"""Paired piecewise-stationary real-CV baseline pilot.

Pre- and post-change observation banks are generated once.  Within each
regime, every policy receives the same r-th raw CV moment block when it makes
its r-th observation of an arm.

``sw_whittle_cv_*`` is a matched CV adaptation of the sliding-window
optimistic Whittle mechanism, not an official SW-Whittle reproduction.
Detector/reset rows are PS-RMAB-inspired wrappers, not reproductions.
"""
from __future__ import annotations

import time

import numpy as np

from core.change_detection_cv import (
    ResettableOnlineCVMomentEstimator,
    TwoWindowCVMomentDetector,
)
from core.cv_sequential_calibration import quadratic_block_sequences
from core.instances import make_flows, pbar_batch
from core.online_cv_moments_stable import OnlineCVMomentEstimator
from core.sim import W_from_pack, coeff_pack, poly_cost, topn_mask
from core.sliding_cv_moments import SlidingWindowCVMomentEstimator


def effective_from_physical(physical):
    sw2, sv2 = physical[..., 0], physical[..., 1]
    p11, p12, p22 = pbar_batch(1.0, sw2.ravel(), sv2.ravel())
    shape = sw2.shape
    theta = np.stack(
        [sw2, p12.reshape(shape), p22.reshape(shape)], axis=-1
    )
    c0 = p11.reshape(shape) + p22.reshape(shape)
    return theta, c0


def generate_bank(physical, n_blocks, block_length, rng):
    B, K, _ = physical.shape
    bank = np.empty((B, K, n_blocks, 2))
    for b in range(B):
        for k in range(K):
            bank[b, k] = quadratic_block_sequences(
                physical[b, k, 0],
                physical[b, k, 1],
                n_slots=block_length,
                n_sequences=1,
                n_blocks=n_blocks,
                rng=rng,
            )[0]
    return bank


def make_problem(seed, B, K, change_t, H, n0, block_length):
    rng = np.random.default_rng(seed)
    physical0 = np.empty((B, K, 2))
    for b in range(B):
        flows = make_flows(K, heterogeneous=True, rng=rng)
        physical0[b, :, 0] = flows["sw2"]
        physical0[b, :, 1] = flows["sv2"]
    changed = rng.random((B, K)) < 0.4
    # Guarantee at least one change in every batch.
    changed[np.arange(B), rng.integers(0, K, B)] = True
    multiplier = np.where(physical0 <= 0.1, 4.0, 0.25)
    physical1 = np.where(
        changed[..., None],
        np.clip(physical0 * multiplier, 0.01, 1.0),
        physical0,
    )
    theta0, c00 = effective_from_physical(physical0)
    theta1, c01 = effective_from_physical(physical1)
    pre_bank = generate_bank(
        physical0, n0 + change_t, block_length, rng
    )
    post_bank = generate_bank(
        physical1, H - change_t, block_length, rng
    )
    ages0 = rng.integers(1, 8, size=(B, K)).astype(float)
    return (
        physical0,
        physical1,
        theta0,
        theta1,
        c00,
        c01,
        changed,
        pre_bank,
        post_bank,
        ages0,
    )


def make_estimator(name, initial):
    common = dict(
        confidence_scale=3.0,
        variance_floor=0.01,
        variance_ceiling=1.0,
    )
    if name in ("cumulative_ce", "cumulative_ucb_cv"):
        return OnlineCVMomentEstimator(initial, **common)
    if (
        name.startswith("sw_ce_")
        or name.startswith("sw_ucb_cv_")
        or name.startswith("sw_whittle_cv_")
    ):
        window = int(name.rsplit("_", 1)[1])
        return SlidingWindowCVMomentEstimator(
            initial, window=window, **common
        )
    if name.startswith("sco_reset_") or name.startswith("ps_forced_"):
        return ResettableOnlineCVMomentEstimator(initial, **common)
    return None


def bank_observation(bank, seen, selected):
    B, K = selected.shape
    observation = np.zeros((B, K, 2))
    for b, k in np.argwhere(selected):
        index = int(seen[b, k])
        if index >= bank.shape[2]:
            raise AssertionError("regime observation bank exhausted")
        observation[b, k] = bank[b, k, index]
    return observation


def round_robin_mask(B, K, N, exploration_index):
    selected = np.zeros((B, K), dtype=bool)
    start = (exploration_index * N) % K
    arms = (start + np.arange(N)) % K
    selected[:, arms] = True
    return selected


def run_policy(
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
    explore_period=50,
):
    B, K, _ = theta0.shape
    initial = np.transpose(pre_bank[:, :, :n0, :], (2, 0, 1, 3))
    estimator = make_estimator(name, initial)
    detector = None
    if name.startswith("sco_reset_") or name.startswith("ps_forced_"):
        detector = TwoWindowCVMomentDetector(
            B,
            K,
            window=detector_window,
            threshold=detector_threshold,
        )
        all_selected = np.ones((B, K), dtype=bool)
        for initial_block in initial:
            detector.update(all_selected, initial_block)

    ages = ages0.copy()
    pre_seen = np.full((B, K), n0, dtype=int)
    post_seen = np.zeros((B, K), dtype=int)
    total = np.zeros(B)
    pre_cost = np.zeros(B)
    post_cost = np.zeros(B)
    rank_loss = np.zeros(B)
    pre_alarms = np.zeros(B, dtype=int)
    post_changed_alarms = np.zeros(B, dtype=int)
    post_unchanged_alarms = np.zeros(B, dtype=int)
    first_calendar_delay = np.full((B, K), -1, dtype=int)
    first_observation_delay = np.full((B, K), -1, dtype=int)
    exploration_index = 0
    started = time.perf_counter()

    for t in range(H):
        post = t >= change_t
        theta = theta1 if post else theta0
        c0 = c01 if post else c00
        C_true = coeff_pack(1.0, theta)
        C_true[..., 0] = c0
        slot_cost = poly_cost(ages, C_true).sum(axis=1)
        total += slot_cost
        if post:
            post_cost += slot_cost
        else:
            pre_cost += slot_cost
        W_true = W_from_pack(ages, C_true)
        true_mask = topn_mask(W_true, N)

        forced = (
            name.startswith("ps_forced_")
            and explore_period
            and t % explore_period == 0
        )
        if name == "true":
            selected = true_mask
        elif name == "max_age":
            # Low-information domain anchor.  It receives only the public age
            # state and has no estimator, drift label, or model parameter.
            selected = topn_mask(ages, N)
        elif forced:
            selected = round_robin_mask(B, K, N, exploration_index)
            exploration_index += 1
        else:
            theta_hat = estimator.mean
            use_ucb = (
                name == "cumulative_ucb_cv"
                or name.startswith("sw_ucb_cv_")
                or name.startswith("sw_whittle_cv_")
                or name.endswith("_ucb")
            )
            if use_ucb:
                _, theta_score = estimator.effective_box
            else:
                theta_score = theta_hat
            selected = topn_mask(
                W_from_pack(ages, coeff_pack(1.0, theta_score)), N
            )

        rank_loss += (
            np.where(true_mask, W_true, 0.0).sum(axis=1)
            - np.where(selected, W_true, 0.0).sum(axis=1)
        )

        if estimator is not None:
            if post:
                observation = bank_observation(
                    post_bank, post_seen, selected
                )
                post_seen += selected
            else:
                observation = bank_observation(
                    pre_bank, pre_seen, selected
                )
                pre_seen += selected

            if detector is None:
                estimator.update(selected, observation)
            else:
                detection = detector.update(selected, observation)
                alarms = detection["alarms"]
                estimator.update_and_reset(selected, observation, detection)
                if post:
                    post_changed_alarms += (alarms & changed).sum(axis=1)
                    post_unchanged_alarms += (alarms & ~changed).sum(axis=1)
                    first = alarms & changed & (first_calendar_delay < 0)
                    first_calendar_delay[first] = t - change_t + 1
                    first_observation_delay[first] = post_seen[first]
                else:
                    pre_alarms += alarms.sum(axis=1)

        ages = np.where(selected, 1.0, ages + 1.0)

    elapsed = time.perf_counter() - started
    changed_total = int(changed.sum())
    detected = (first_calendar_delay >= 0) & changed
    return {
        "name": name,
        "avg_cost": total / H,
        "pre_cost": pre_cost / change_t,
        "post_cost": post_cost / (H - change_t),
        "rank_loss": rank_loss / H,
        "pre_alarms": int(pre_alarms.sum()),
        "post_changed_alarms": int(post_changed_alarms.sum()),
        "post_unchanged_alarms": int(post_unchanged_alarms.sum()),
        "detected_fraction": (
            float(detected.sum() / changed_total) if detector else np.nan
        ),
        "calendar_delay": (
            float(first_calendar_delay[detected].mean())
            if np.any(detected)
            else np.nan
        ),
        "observation_delay": (
            float(first_observation_delay[detected].mean())
            if np.any(detected)
            else np.nan
        ),
        "seconds": elapsed,
    }


def main():
    seed = 20260801
    B, K, N, H, n0, change_t, block_length = 6, 20, 4, 1000, 8, 500, 64
    problem = make_problem(seed, B, K, change_t, H, n0, block_length)
    (
        _,
        _,
        theta0,
        theta1,
        c00,
        c01,
        changed,
        pre_bank,
        post_bank,
        ages0,
    ) = problem
    names = [
        "true",
        "max_age",
        "cumulative_ce",
        "cumulative_ucb_cv",
        "sw_ce_32",
        "sw_whittle_cv_32",
        "sw_ce_128",
        "sw_whittle_cv_128",
        "sco_reset_ce",
        "sco_reset_ucb",
        "ps_forced_reset_ucb",
    ]
    results = [
        run_policy(
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
        )
        for name in names
    ]
    oracle = results[0]

    print("PAIRED PIECEWISE REAL-CV PILOT -- NOT PAPER EVIDENCE")
    print(
        f"seed={seed} B={B} K={K} N={N} H={H} change={change_t} "
        f"n0={n0} block_length={block_length} changed={changed.mean():.3f}"
    )
    print(
        "policy                   total_ex  pre_ex post_ex rank_loss "
        "preFA postFA detect calDelay obsDelay sec"
    )
    for row in results:
        total_ex = np.mean(100 * (row["avg_cost"] / oracle["avg_cost"] - 1))
        pre_ex = np.mean(100 * (row["pre_cost"] / oracle["pre_cost"] - 1))
        post_ex = np.mean(100 * (row["post_cost"] / oracle["post_cost"] - 1))
        print(
            f"{row['name']:24s} {total_ex:8.3f}% {pre_ex:7.3f}% "
            f"{post_ex:7.3f}% {row['rank_loss'].mean():9.3f} "
            f"{row['pre_alarms']:5d} {row['post_unchanged_alarms']:6d} "
            f"{row['detected_fraction']:6.3f} "
            f"{row['calendar_delay']:8.2f} {row['observation_delay']:8.2f} "
            f"{row['seconds']:5.2f}"
        )


if __name__ == "__main__":
    main()
