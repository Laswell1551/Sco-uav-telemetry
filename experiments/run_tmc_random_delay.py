"""Event-driven random two-way-delay stress runner for PA-SCO.

Forward arrival and ACK/NACK feedback are distinct events.  The receiver keeps
the freshest generation timestamp, so a late stale packet cannot decrease its
age.  The scheduler ranks its acknowledged age and removes an outstanding
attempt only when feedback returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.change_detection_cv import (
    ResettableOnlineCVMomentEstimator,
    TwoWindowCVMomentDetector,
)
from core.sim import W_from_pack, coeff_pack, poly_cost, topn_mask
from run_cv_piecewise_pilot import bank_observation, round_robin_mask
from run_tmc_channel_stress import make_estimator


@dataclass(frozen=True)
class DelayProfile:
    name: str
    description: str


PROFILES = (
    DelayProfile("fixed", "forward=2 and feedback=2 slots"),
    DelayProfile("light_iid", "iid 1/3-slot links with equal probability"),
    DelayProfile("heavy_iid", "iid 1/11-slot links with probabilities .9/.1"),
    DelayProfile("lognormal", "mean-calibrated discretized log-normal links"),
    DelayProfile("markov_burst", "correlated 1/6-slot Markov links"),
    DelayProfile("forward_heavy", "heavy-tailed forward and fixed feedback"),
    DelayProfile("feedback_heavy", "fixed forward and heavy-tailed feedback"),
)


def _heavy(rng, shape):
    return np.where(rng.random(shape) < 0.1, 11, 1).astype(int)


def _light(rng, shape):
    return np.where(rng.random(shape) < 0.5, 3, 1).astype(int)


def _lognormal(rng, shape):
    sigma = 1.5
    mu = np.log(2.0) - 0.5 * sigma**2
    values = np.rint(rng.lognormal(mu, sigma, size=shape))
    return np.clip(values, 1, 64).astype(int)


def _markov(rng, shape):
    B, K, A = shape
    high = rng.random((B, K)) < 0.2
    out = np.empty(shape, dtype=int)
    for a in range(A):
        out[:, :, a] = np.where(high, 6, 1)
        u = rng.random((B, K))
        high = np.where(high, u >= 0.2, u < 0.05)
    return out


def make_delay_banks(seed, profile, shape):
    """Create attempt-indexed forward and feedback delays."""
    names = {item.name for item in PROFILES}
    if profile not in names:
        raise ValueError(f"unknown delay profile: {profile}")
    forward_rng = np.random.default_rng(seed + 61000)
    feedback_rng = np.random.default_rng(seed + 62000)
    if profile == "fixed":
        forward = np.full(shape, 2, dtype=int)
        feedback = np.full(shape, 2, dtype=int)
    elif profile == "light_iid":
        forward = _light(forward_rng, shape)
        feedback = _light(feedback_rng, shape)
    elif profile == "heavy_iid":
        forward = _heavy(forward_rng, shape)
        feedback = _heavy(feedback_rng, shape)
    elif profile == "lognormal":
        forward = _lognormal(forward_rng, shape)
        feedback = _lognormal(feedback_rng, shape)
    elif profile == "markov_burst":
        forward = _markov(forward_rng, shape)
        feedback = _markov(feedback_rng, shape)
    elif profile == "forward_heavy":
        forward = _heavy(forward_rng, shape)
        feedback = np.full(shape, 2, dtype=int)
    else:
        forward = np.full(shape, 2, dtype=int)
        feedback = _heavy(feedback_rng, shape)
    if np.any(forward < 0) or np.any(feedback < 0):
        raise AssertionError("delay banks must be nonnegative")
    return forward, feedback


def delay_bank_summary(forward, feedback):
    round_trip = forward + feedback
    return {
        "forward_mean": float(np.mean(forward)),
        "feedback_mean": float(np.mean(feedback)),
        "round_trip_mean": float(np.mean(round_trip)),
        "round_trip_p95": float(np.quantile(round_trip, 0.95)),
        "round_trip_p99": float(np.quantile(round_trip, 0.99)),
        "round_trip_max": int(np.max(round_trip)),
    }


def _feedback_batches(events):
    """Partition same-slot feedback so each batch has at most one event/arm."""
    remaining = list(events)
    while remaining:
        used = set()
        batch = []
        later = []
        for event in remaining:
            key = event[:2]
            if key in used:
                later.append(event)
            else:
                used.add(key)
                batch.append(event)
        yield batch
        remaining = later


def run_policy_random_delay(
    name,
    theta0,
    theta1,
    c00,
    c01,
    changed,
    pre_bank,
    post_bank,
    success_uniform,
    forward_delays,
    feedback_delays,
    ages0,
    N,
    n0,
    change_t,
    H,
    success_probability=0.9,
    detector_window=8,
    detector_threshold=5.0,
    explore_period=50,
    inflight_beta=16.0,
):
    B, K, _ = theta0.shape
    expected_shape = success_uniform.shape
    if forward_delays.shape != expected_shape or feedback_delays.shape != expected_shape:
        raise ValueError("success and delay banks must have identical shapes")
    if inflight_beta < 0:
        raise ValueError("inflight_beta must be nonnegative")

    initial = np.transpose(pre_bank[:, :, :n0, :], (2, 0, 1, 3))
    estimator = make_estimator(name, initial)
    detector = None
    if name in ("sco_reset_ucb", "ps_forced_reset_ucb", "inflight_sco_ucb"):
        detector = TwoWindowCVMomentDetector(
            B, K, window=detector_window, threshold=detector_threshold
        )
        all_selected = np.ones((B, K), dtype=bool)
        for block in initial:
            detector.update(all_selected, block)

    receiver_age = ages0.copy()
    known_age = ages0.copy()
    receiver_latest = 1.0 - ages0
    known_latest = 1.0 - ages0
    pre_attempt = np.full((B, K), n0, dtype=int)
    post_attempt = np.zeros((B, K), dtype=int)
    channel_attempt = np.zeros((B, K), dtype=int)
    post_observed = np.zeros((B, K), dtype=int)
    inflight = np.zeros((B, K), dtype=int)
    he_eligible_time = np.zeros((B, K), dtype=int)
    he_gamma = np.zeros((B, K), dtype=float)
    he_virtual_mean = np.zeros((B, K), dtype=float)
    he_virtual_second = np.zeros((B, K), dtype=float)
    he_epochs = np.zeros((B, K), dtype=int)
    he_attempts = np.zeros((B, K), dtype=int)
    he_first_rtt = np.zeros((B, K), dtype=float)
    he_virtual_rtt = np.zeros((B, K), dtype=float)

    max_event_time = H + int(np.max(forward_delays + feedback_delays)) + 1
    arrivals = [[] for _ in range(max_event_time)]
    feedback = [[] for _ in range(max_event_time)]

    total = np.zeros(B)
    pre_cost = np.zeros(B)
    post_cost = np.zeros(B)
    attempted_total = 0
    delivered_total = 0
    acknowledged_total = 0
    redundant_total = 0
    stale_arrivals = 0
    successful_arrivals = 0
    max_inflight = 0
    inflight_integral = 0
    pre_alarms = np.zeros(B, dtype=int)
    post_unchanged_alarms = np.zeros(B, dtype=int)
    first_calendar_delay = np.full((B, K), -1, dtype=int)
    first_observation_delay = np.full((B, K), -1, dtype=int)
    exploration_index = 0

    for t in range(H):
        post = t >= change_t
        theta = theta1 if post else theta0
        c0 = c01 if post else c00
        true_pack = coeff_pack(1.0, theta)
        true_pack[..., 0] = c0
        slot_cost = poly_cost(receiver_age, true_pack).sum(axis=1)
        total += slot_cost
        (post_cost if post else pre_cost)[:] += slot_cost

        forced = (
            name == "ps_forced_reset_ucb"
            and explore_period
            and t % explore_period == 0
        )
        if name == "true":
            selected = topn_mask(W_from_pack(receiver_age, true_pack), N)
        elif name == "he_rm_age":
            # Multi-stream matched adaptation of He et al.'s single-source
            # Robbins--Monro sampler: at most one unacknowledged packet per
            # arm, then an ACK-learned waiting threshold.  Shared capacity is
            # allocated by acknowledged age among currently eligible arms.
            eligible = (inflight == 0) & (t >= he_eligible_time)
            age_score = np.where(eligible, known_age, -np.inf)
            selected = topn_mask(age_score, N) & eligible
        elif forced:
            selected = round_robin_mask(B, K, N, exploration_index)
            exploration_index += 1
        else:
            _, theta_hi = estimator.effective_box
            score = W_from_pack(known_age, coeff_pack(1.0, theta_hi))
            if name == "inflight_sco_ucb":
                score = score / (1.0 + inflight_beta * inflight)
            selected = topn_mask(score, N)

        attempted_total += int(selected.sum())
        redundant_total += int((selected & (inflight > 0)).sum())
        inflight += selected
        inflight_integral += int(inflight.sum())
        max_inflight = max(max_inflight, int(inflight.max()))

        observation = (
            bank_observation(post_bank, post_attempt, selected)
            if post
            else bank_observation(pre_bank, pre_attempt, selected)
        )
        if post:
            post_attempt += selected
        else:
            pre_attempt += selected

        for b, k in np.argwhere(selected):
            attempt = int(channel_attempt[b, k])
            if attempt >= success_uniform.shape[2]:
                raise AssertionError("channel bank exhausted")
            success = bool(
                success_uniform[b, k, attempt] < success_probability
            )
            df = int(forward_delays[b, k, attempt])
            db = int(feedback_delays[b, k, attempt])
            rtt = df + db
            epoch_attempt = int(he_attempts[b, k]) + 1
            if name == "he_rm_age":
                he_attempts[b, k] = epoch_attempt
            obs = observation[b, k].copy()
            arrival_t = t + df
            feedback_t = arrival_t + db
            if success and arrival_t < len(arrivals):
                arrivals[arrival_t].append((int(b), int(k), t))
            if feedback_t < len(feedback):
                feedback[feedback_t].append(
                    (
                        int(b), int(k), success, obs, post, t,
                        rtt, epoch_attempt,
                    )
                )
            channel_attempt[b, k] += 1

        receiver_next = receiver_age + 1.0
        newest_arrival = {}
        for b, k, generation_t in arrivals[t]:
            successful_arrivals += 1
            key = (b, k)
            newest_arrival[key] = max(newest_arrival.get(key, -10**9), generation_t)
        for (b, k), generation_t in newest_arrival.items():
            if generation_t > receiver_latest[b, k]:
                receiver_latest[b, k] = generation_t
                receiver_next[b, k] = t - generation_t + 1.0
                delivered_total += 1
            else:
                stale_arrivals += 1

        known_next = known_age + 1.0
        slot_feedback = feedback[t]
        for b, k, *_ in slot_feedback:
            inflight[b, k] -= 1
            if inflight[b, k] < 0:
                raise AssertionError("negative in-flight count")
            acknowledged_total += 1

        for batch in _feedback_batches(slot_feedback):
            mask = np.zeros((B, K), dtype=bool)
            obs_batch = np.zeros((B, K, 2), dtype=float)
            generated_post = np.zeros((B, K), dtype=bool)
            for (
                b, k, success, obs, event_post, generation_t,
                rtt, epoch_attempt,
            ) in batch:
                if name == "he_rm_age":
                    if epoch_attempt == 1:
                        he_first_rtt[b, k] = rtt
                    else:
                        he_virtual_rtt[b, k] += rtt
                    if success:
                        he_epochs[b, k] += 1
                        epoch = he_epochs[b, k]
                        virtual = he_virtual_rtt[b, k]
                        he_virtual_mean[b, k] += (
                            virtual - he_virtual_mean[b, k]
                        ) / epoch
                        he_virtual_second[b, k] += (
                            virtual**2 - he_virtual_second[b, k]
                        ) / epoch
                        nuisance = 0.5 * max(
                            he_virtual_second[b, k]
                            - he_virtual_mean[b, k] ** 2,
                            0.0,
                        )
                        gamma = he_gamma[b, k]
                        actual = he_first_rtt[b, k]
                        clipped = max(actual, gamma)
                        score = (
                            0.5 * clipped**2
                            - gamma * (clipped + virtual)
                            + nuisance
                        )
                        delay_lower_bound = 2.0
                        eta = (
                            1.0 / (2.0 * delay_lower_bound)
                            if epoch == 1
                            else 1.0 / ((epoch + 2.0) * delay_lower_bound)
                        )
                        he_gamma[b, k] = np.clip(
                            gamma + eta * score, 0.0, 128.0
                        )
                        wait = int(np.ceil(max(
                            he_gamma[b, k] - actual, 0.0
                        )))
                        he_eligible_time[b, k] = t + 1 + wait
                        he_attempts[b, k] = 0
                        he_first_rtt[b, k] = 0.0
                        he_virtual_rtt[b, k] = 0.0
                    else:
                        # The source paper retransmits immediately after NACK;
                        # in slots, the earliest feasible retry is next slot.
                        he_eligible_time[b, k] = t + 1
                if not success:
                    continue
                mask[b, k] = True
                obs_batch[b, k] = obs
                generated_post[b, k] = event_post
                if generation_t > known_latest[b, k]:
                    known_latest[b, k] = generation_t
                    known_next[b, k] = t - generation_t + 1.0
            if estimator is None or not np.any(mask):
                continue
            post_observed += mask & generated_post
            if detector is None:
                estimator.update(mask, obs_batch)
                continue
            detection = detector.update(mask, obs_batch)
            alarms = detection["alarms"]
            estimator.update_and_reset(mask, obs_batch, detection)
            pre_generated = mask & ~generated_post
            if t < change_t:
                pre_alarms += alarms.sum(axis=1)
            else:
                pre_alarms += (alarms & pre_generated).sum(axis=1)
                post_unchanged_alarms += (
                    alarms & generated_post & ~changed
                ).sum(axis=1)
                first = (
                    alarms & generated_post & changed
                    & (first_calendar_delay < 0)
                )
                first_calendar_delay[first] = t - change_t + 1
                first_observation_delay[first] = post_observed[first]

        receiver_age = receiver_next
        known_age = known_next

    changed_total = int(changed.sum())
    detected = (first_calendar_delay >= 0) & changed
    return {
        "name": name,
        "avg_cost": total / H,
        "pre_cost": pre_cost / change_t,
        "post_cost": post_cost / (H - change_t),
        "delivery_rate": delivered_total / attempted_total,
        "ack_rate": acknowledged_total / attempted_total,
        "redundant_attempt_rate": redundant_total / attempted_total,
        "stale_arrival_rate": (
            stale_arrivals / successful_arrivals
            if successful_arrivals else 0.0
        ),
        "mean_inflight_per_slot": inflight_integral / (B * H),
        "capacity_utilization": attempted_total / (B * H * N),
        "max_inflight_count": max_inflight,
        "learned_wait_threshold": (
            float(np.mean(he_gamma)) if name == "he_rm_age" else np.nan
        ),
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
