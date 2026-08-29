"""Retrospective matched-baseline addendum for the frozen TMC protocol.

This standalone runner adds two baselines without changing the frozen problem
generator, oracle, core estimators, or earlier result artifacts:

* DE-CD-Whittle-CV: the diminishing-exploration schedule of Li et al.
  (AISTATS 2026) wrapped around the existing selected-only CV detector and
  resettable UCB-CV Whittle learner.
* DTS-Whittle-CV: a continuous-observation matched adaptation of discounted
  Thompson sampling using discounted CV-moment sufficient statistics and a
  per-slot Gaussian posterior draw.

Hyperparameters are selected only on seeds 309901--309912.  Formal evaluation
uses exactly seeds 310001--310030 with three batches per seed.  Confidence
intervals resample seeds as clusters, never the within-seed batches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.change_detection_cv import (
    ResettableOnlineCVMomentEstimator,
    TwoWindowCVMomentDetector,
)
from core.sim import W_from_pack, coeff_pack, poly_cost, topn_mask
from run_cv_piecewise_pilot import (
    bank_observation,
    effective_from_physical,
    make_problem,
    run_policy,
)


SCHEMA_VERSION = "TMC_V16_BASELINE_EXPANSION_v1"
BASE_PROTOCOL = "TMC_SYNTHETIC_PROTOCOL_FROZEN_v1"
PILOT_SEEDS = tuple(range(309901, 309913))
FORMAL_SEEDS = tuple(range(310001, 310031))
DE_ALPHA_CANDIDATES = (0.5, 1.0, 2.0, 4.0)
DTS_GAMMA_CANDIDATES = (0.95, 0.98, 0.99, 0.995)

BATCHES_PER_SEED = 3
K = 20
N = 4
H = 1000
N0 = 8
CHANGE_T = 500
BLOCK_LENGTH = 64
DETECTOR_WINDOW = 8
DETECTOR_THRESHOLD = 5.0

BOOTSTRAP_SEED = 316099
BOOTSTRAP_REPLICATES = 100_000
POSTERIOR_STREAM_TAG = 0xD75


class DiminishingExplorationSchedule:
    """Per-batch AISTATS-2026 diminishing-exploration cursor.

    The source algorithm pulls K arms in K one-arm slots.  For the paper's
    N>1 action constraint, one exploration block contains ceil(K/N) slots.
    Each slot activates exactly N cyclically consecutive arms, so all K arms
    are covered and only the final slot can repeat an already covered arm.

    Time is one-based.  At initialization,

        u = ceil((alpha - K / (4 alpha))**2),

    and after a completed block,

        u <- ceil(u + K sqrt(u) / alpha + K**2 / (4 alpha**2)).

    Following Algorithm 1, an alarm at round t resets the affected batch to
    segment origin t and u=1, making the next round the first exploration
    slot.  Batches reset independently because they are independent paired
    instances in the vectorized simulator.
    """

    def __init__(self, batches: int, arms: int, budget: int, alpha: float):
        if batches < 1 or arms < 1:
            raise ValueError("batches and arms must be positive")
        if not 1 <= budget <= arms:
            raise ValueError("budget must lie in [1, arms]")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.batches = int(batches)
        self.arms = int(arms)
        self.budget = int(budget)
        self.alpha = float(alpha)
        self.block_slots = int(math.ceil(self.arms / self.budget))
        initial = math.ceil(
            (self.alpha - self.arms / (4.0 * self.alpha)) ** 2
        )
        self.initial_cursor = max(1, int(initial))
        self.segment_start = np.zeros(self.batches, dtype=np.int64)
        self.cursor = np.full(
            self.batches, self.initial_cursor, dtype=np.int64
        )
        self.blocks_started = np.zeros(self.batches, dtype=np.int64)

    def _next_cursor(self, cursor: int) -> int:
        value = (
            cursor
            + self.arms * math.sqrt(cursor) / self.alpha
            + self.arms**2 / (4.0 * self.alpha**2)
        )
        return int(math.ceil(value))

    def mask(self, round_index: int) -> np.ndarray:
        """Return the forced N-arm mask for a sequential one-based round."""
        if round_index < 1:
            raise ValueError("round_index must be one-based and positive")
        elapsed = round_index - self.segment_start
        selected = np.zeros((self.batches, self.arms), dtype=bool)

        for batch in range(self.batches):
            # Sequential use advances once; the loop also makes the helper
            # robust to tests or trace drivers that skip non-exploration time.
            while elapsed[batch] >= (
                self.cursor[batch] + self.block_slots
            ):
                self.cursor[batch] = self._next_cursor(
                    int(self.cursor[batch])
                )
            offset = int(elapsed[batch] - self.cursor[batch])
            if 0 <= offset < self.block_slots:
                arms = (
                    offset * self.budget + np.arange(self.budget)
                ) % self.arms
                selected[batch, arms] = True
                if offset == 0:
                    self.blocks_started[batch] += 1
        return selected

    def reset(self, batch_alarm: np.ndarray, round_index: int) -> None:
        """Reset each alarmed batch so exploration restarts next round."""
        alarm = np.asarray(batch_alarm, dtype=bool)
        if alarm.shape != (self.batches,):
            raise ValueError("batch_alarm must have shape (batches,)")
        if round_index < 1:
            raise ValueError("round_index must be one-based and positive")
        self.segment_start[alarm] = int(round_index)
        self.cursor[alarm] = 1


class DiscountedCVMomentPosterior:
    """Discounted diagonal-Gaussian posterior surrogate for CV moments.

    Uniform discounting leaves the weighted mean unchanged between samples,
    but reduces effective weight and second central moment.  A selected
    observation is then incorporated with a weighted Welford update.  The
    posterior draw uses the estimated observation variance divided by the
    discounted effective weight and is projected to the frozen physical box.

    This is a matched continuous-observation adaptation; it is not claimed to
    reproduce the Bernoulli posterior in Qi, Wang, and Zhu (2023).
    """

    def __init__(
        self,
        initial_observations: np.ndarray,
        gamma: float,
        variance_floor: float = 0.01,
        variance_ceiling: float = 1.0,
        posterior_variance_floor: float = 1e-10,
        weight_floor: float = 1e-8,
    ):
        observations = np.asarray(initial_observations, dtype=float)
        if observations.ndim != 4 or observations.shape[-1] != 2:
            raise ValueError(
                "initial_observations must have shape (R,B,K,2)"
            )
        if observations.shape[0] < 2:
            raise ValueError("at least two initial blocks are required")
        if not np.all(np.isfinite(observations)):
            raise ValueError("initial observations must be finite")
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must lie strictly between zero and one")
        if not 0.0 < variance_floor < variance_ceiling:
            raise ValueError("physical projection bounds are invalid")
        if posterior_variance_floor <= 0.0 or weight_floor <= 0.0:
            raise ValueError("posterior floors must be positive")

        self.gamma = float(gamma)
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)
        self.posterior_variance_floor = float(posterior_variance_floor)
        self.weight_floor = float(weight_floor)
        self.running_mean = observations.mean(axis=0)
        centered = observations - self.running_mean[None, ...]
        self.running_m2 = np.square(centered).sum(axis=0)
        self.weight = np.full(
            observations.shape[1:3],
            float(observations.shape[0]),
            dtype=float,
        )
        self.clip_count = np.zeros(observations.shape[1], dtype=np.int64)
        self.draw_count = np.zeros(observations.shape[1], dtype=np.int64)

    @property
    def physical_mean(self) -> np.ndarray:
        return np.clip(
            self.running_mean, self.variance_floor, self.variance_ceiling
        )

    @property
    def posterior_scale(self) -> np.ndarray:
        weight = np.maximum(self.weight[..., None], self.weight_floor)
        observation_variance = np.maximum(
            self.running_m2 / weight, self.posterior_variance_floor
        )
        return np.sqrt(observation_variance / weight)

    def draw(self, standard_normal: np.ndarray) -> np.ndarray:
        standard_normal = np.asarray(standard_normal, dtype=float)
        if standard_normal.shape != self.running_mean.shape:
            raise ValueError(
                "standard_normal must have shape (B,K,2)"
            )
        raw = self.running_mean + self.posterior_scale * standard_normal
        clipped = np.clip(
            raw, self.variance_floor, self.variance_ceiling
        )
        clipped_entry = (raw < self.variance_floor) | (
            raw > self.variance_ceiling
        )
        self.clip_count += clipped_entry.sum(axis=(1, 2))
        self.draw_count += np.prod(raw.shape[1:])
        return clipped

    def update(
        self, selected: np.ndarray, observation: np.ndarray
    ) -> None:
        selected = np.asarray(selected, dtype=bool)
        observation = np.asarray(observation, dtype=float)
        if selected.shape != self.weight.shape:
            raise ValueError("selected must have shape (B,K)")
        if observation.shape != self.running_mean.shape:
            raise ValueError("observation must have shape (B,K,2)")
        if not np.all(np.isfinite(observation[selected])):
            raise ValueError("selected observations must be finite")

        # Passive forgetting is calendar-time based and therefore applies to
        # every arm, including arms not selected in this round.
        self.weight *= self.gamma
        self.running_m2 *= self.gamma

        keep = selected[..., None]
        new_weight = self.weight + selected
        delta = observation - self.running_mean
        candidate_mean = (
            self.running_mean + delta / new_weight[..., None]
        )
        delta2 = observation - candidate_mean
        candidate_m2 = self.running_m2 + delta * delta2
        self.running_mean = np.where(
            keep, candidate_mean, self.running_mean
        )
        self.running_m2 = np.where(
            keep, candidate_m2, self.running_m2
        )
        self.weight = new_weight


def _initial_blocks(pre_bank: np.ndarray, n0: int) -> np.ndarray:
    return np.transpose(pre_bank[:, :, :n0, :], (2, 0, 1, 3))


def _true_slot_state(
    t: int,
    theta0: np.ndarray,
    theta1: np.ndarray,
    c00: np.ndarray,
    c01: np.ndarray,
    ages: np.ndarray,
    change_t: int,
):
    post = t >= change_t
    theta = theta1 if post else theta0
    c0 = c01 if post else c00
    true_coefficients = coeff_pack(1.0, theta)
    true_coefficients[..., 0] = c0
    slot_cost = poly_cost(ages, true_coefficients).sum(axis=1)
    true_index = W_from_pack(ages, true_coefficients)
    return post, slot_cost, true_index


def run_de_policy(
    theta0: np.ndarray,
    theta1: np.ndarray,
    c00: np.ndarray,
    c01: np.ndarray,
    changed: np.ndarray,
    pre_bank: np.ndarray,
    post_bank: np.ndarray,
    ages0: np.ndarray,
    budget: int,
    n0: int,
    change_t: int,
    horizon: int,
    alpha: float,
    detector_window: int = DETECTOR_WINDOW,
    detector_threshold: float = DETECTOR_THRESHOLD,
):
    """Run the DE-CD-Whittle-CV matched addendum on paired moment banks."""
    batches, arms, _ = theta0.shape
    initial = _initial_blocks(pre_bank, n0)
    estimator = ResettableOnlineCVMomentEstimator(
        initial,
        confidence_scale=3.0,
        variance_floor=0.01,
        variance_ceiling=1.0,
    )
    detector = TwoWindowCVMomentDetector(
        batches,
        arms,
        window=detector_window,
        threshold=detector_threshold,
    )
    all_selected = np.ones((batches, arms), dtype=bool)
    for initial_block in initial:
        detector.update(all_selected, initial_block)

    schedule = DiminishingExplorationSchedule(
        batches, arms, budget, alpha
    )
    ages = ages0.copy()
    pre_seen = np.full((batches, arms), n0, dtype=int)
    post_seen = np.zeros((batches, arms), dtype=int)
    total_cost = np.zeros(batches)
    pre_cost = np.zeros(batches)
    post_cost = np.zeros(batches)
    rank_loss = np.zeros(batches)
    pre_alarms = np.zeros(batches, dtype=int)
    post_changed_alarms = np.zeros(batches, dtype=int)
    post_unchanged_alarms = np.zeros(batches, dtype=int)
    first_calendar_delay = np.full((batches, arms), -1, dtype=int)
    first_observation_delay = np.full((batches, arms), -1, dtype=int)
    forced_slots = np.zeros(batches, dtype=int)
    forced_activations = np.zeros(batches, dtype=int)
    started = time.perf_counter()

    for t in range(horizon):
        post, slot_cost, true_index = _true_slot_state(
            t, theta0, theta1, c00, c01, ages, change_t
        )
        total_cost += slot_cost
        if post:
            post_cost += slot_cost
        else:
            pre_cost += slot_cost
        true_mask = topn_mask(true_index, budget)

        _, optimistic_theta = estimator.effective_box
        base_selected = topn_mask(
            W_from_pack(
                ages, coeff_pack(1.0, optimistic_theta)
            ),
            budget,
        )
        forced_mask = schedule.mask(t + 1)
        forced_batch = forced_mask.any(axis=1)
        selected = np.where(
            forced_batch[:, None], forced_mask, base_selected
        )
        forced_slots += forced_batch
        forced_activations += forced_mask.sum(axis=1)

        rank_loss += (
            np.where(true_mask, true_index, 0.0).sum(axis=1)
            - np.where(selected, true_index, 0.0).sum(axis=1)
        )

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

        # One arm-level alarm resets the exploration clock for that independent
        # batch; the existing detector/estimator retain their arm-level resets.
        schedule.reset(alarms.any(axis=1), t + 1)
        ages = np.where(selected, 1.0, ages + 1.0)

    detected = (first_calendar_delay >= 0) & changed
    detected_count = detected.sum(axis=1)
    changed_count = changed.sum(axis=1)
    calendar_sum = np.where(
        detected, first_calendar_delay, 0
    ).sum(axis=1)
    observation_sum = np.where(
        detected, first_observation_delay, 0
    ).sum(axis=1)
    elapsed = time.perf_counter() - started
    return {
        "method": "de_cd_whittle_cv",
        "avg_cost": total_cost / horizon,
        "pre_cost": pre_cost / change_t,
        "post_cost": post_cost / (horizon - change_t),
        "rank_loss": rank_loss / horizon,
        "pre_alarms": pre_alarms,
        "post_changed_alarms": post_changed_alarms,
        "post_unchanged_alarms": post_unchanged_alarms,
        "changed_arms": changed_count,
        "detected_changed_arms": detected_count,
        "calendar_delay_sum": calendar_sum,
        "observation_delay_sum": observation_sum,
        "forced_slots": forced_slots,
        "forced_activations": forced_activations,
        "exploration_blocks": schedule.blocks_started.copy(),
        "seconds": elapsed,
    }


def run_dts_policy(
    theta0: np.ndarray,
    theta1: np.ndarray,
    c00: np.ndarray,
    c01: np.ndarray,
    changed: np.ndarray,
    pre_bank: np.ndarray,
    post_bank: np.ndarray,
    ages0: np.ndarray,
    budget: int,
    n0: int,
    change_t: int,
    horizon: int,
    gamma: float,
    posterior_standard_normals: np.ndarray,
):
    """Run discounted TS with selected-only CV-moment observations."""
    del changed  # Kept in the common signature for paired protocol symmetry.
    batches, arms, _ = theta0.shape
    normals = np.asarray(posterior_standard_normals, dtype=float)
    expected_shape = (horizon, batches, arms, 2)
    if normals.shape != expected_shape:
        raise ValueError(
            f"posterior_standard_normals must have shape {expected_shape}"
        )
    posterior = DiscountedCVMomentPosterior(
        _initial_blocks(pre_bank, n0),
        gamma=gamma,
        variance_floor=0.01,
        variance_ceiling=1.0,
    )
    ages = ages0.copy()
    pre_seen = np.full((batches, arms), n0, dtype=int)
    post_seen = np.zeros((batches, arms), dtype=int)
    total_cost = np.zeros(batches)
    pre_cost = np.zeros(batches)
    post_cost = np.zeros(batches)
    rank_loss = np.zeros(batches)
    started = time.perf_counter()

    for t in range(horizon):
        post, slot_cost, true_index = _true_slot_state(
            t, theta0, theta1, c00, c01, ages, change_t
        )
        total_cost += slot_cost
        if post:
            post_cost += slot_cost
        else:
            pre_cost += slot_cost
        true_mask = topn_mask(true_index, budget)

        physical_draw = posterior.draw(normals[t])
        theta_draw, _ = effective_from_physical(physical_draw)
        selected = topn_mask(
            W_from_pack(ages, coeff_pack(1.0, theta_draw)), budget
        )
        rank_loss += (
            np.where(true_mask, true_index, 0.0).sum(axis=1)
            - np.where(selected, true_index, 0.0).sum(axis=1)
        )

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
        posterior.update(selected, observation)
        ages = np.where(selected, 1.0, ages + 1.0)

    elapsed = time.perf_counter() - started
    return {
        "method": "dts_whittle_cv",
        "avg_cost": total_cost / horizon,
        "pre_cost": pre_cost / change_t,
        "post_cost": post_cost / (horizon - change_t),
        "rank_loss": rank_loss / horizon,
        "posterior_clip_count": posterior.clip_count.copy(),
        "posterior_draw_count": posterior.draw_count.copy(),
        "final_effective_weight_mean": posterior.weight.mean(axis=1),
        "seconds": elapsed,
    }


def _problem(seed: int):
    values = make_problem(
        seed,
        BATCHES_PER_SEED,
        K,
        CHANGE_T,
        H,
        N0,
        BLOCK_LENGTH,
    )
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
    ) = values
    common = (
        theta0,
        theta1,
        c00,
        c01,
        changed,
        pre_bank,
        post_bank,
        ages0,
        N,
        N0,
        CHANGE_T,
        H,
    )
    oracle = run_policy("true", *common)
    return common, oracle


def posterior_normal_bank(seed: int) -> np.ndarray:
    """Gamma-independent common random numbers for all DTS candidates."""
    sequence = np.random.SeedSequence(
        [int(seed), POSTERIOR_STREAM_TAG, SCHEMA_VERSION.__len__()]
    )
    rng = np.random.default_rng(sequence)
    return rng.standard_normal((H, BATCHES_PER_SEED, K, 2))


def _safe_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def rows_from_result(
    method: str,
    seed: int,
    result: dict,
    oracle: dict,
    alpha: float | None = None,
    gamma: float | None = None,
):
    rows = []
    for batch in range(BATCHES_PER_SEED):
        row = {
            "method": method,
            "seed": int(seed),
            "batch": batch,
            "retrospective_matched_addendum": True,
            "alpha": None if alpha is None else float(alpha),
            "gamma": None if gamma is None else float(gamma),
            "avg_cost": float(result["avg_cost"][batch]),
            "oracle_avg_cost": float(oracle["avg_cost"][batch]),
            "pre_cost": float(result["pre_cost"][batch]),
            "oracle_pre_cost": float(oracle["pre_cost"][batch]),
            "post_cost": float(result["post_cost"][batch]),
            "oracle_post_cost": float(oracle["post_cost"][batch]),
            "total_excess_cost_pct": float(
                100.0
                * (
                    result["avg_cost"][batch]
                    / oracle["avg_cost"][batch]
                    - 1.0
                )
            ),
            "pre_excess_cost_pct": float(
                100.0
                * (
                    result["pre_cost"][batch]
                    / oracle["pre_cost"][batch]
                    - 1.0
                )
            ),
            "post_excess_cost_pct": float(
                100.0
                * (
                    result["post_cost"][batch]
                    / oracle["post_cost"][batch]
                    - 1.0
                )
            ),
            "rank_loss": float(result["rank_loss"][batch]),
            "policy_runtime_seconds": float(result["seconds"]),
        }
        if method == "de_cd_whittle_cv":
            detected = int(result["detected_changed_arms"][batch])
            changed = int(result["changed_arms"][batch])
            row.update(
                {
                    "changed_arms": changed,
                    "detected_changed_arms": detected,
                    "detection_fraction": detected / changed,
                    "calendar_delay_sum": int(
                        result["calendar_delay_sum"][batch]
                    ),
                    "observation_delay_sum": int(
                        result["observation_delay_sum"][batch]
                    ),
                    "mean_calendar_delay": (
                        result["calendar_delay_sum"][batch] / detected
                        if detected
                        else None
                    ),
                    "mean_observation_delay": (
                        result["observation_delay_sum"][batch] / detected
                        if detected
                        else None
                    ),
                    "pre_alarms": int(result["pre_alarms"][batch]),
                    "post_changed_alarms": int(
                        result["post_changed_alarms"][batch]
                    ),
                    "post_unchanged_alarms": int(
                        result["post_unchanged_alarms"][batch]
                    ),
                    "pre_false_alarms_per_10k_arm_slots": float(
                        10000.0
                        * result["pre_alarms"][batch]
                        / (K * CHANGE_T)
                    ),
                    "forced_slots": int(result["forced_slots"][batch]),
                    "forced_slot_fraction": float(
                        result["forced_slots"][batch] / H
                    ),
                    "forced_activations": int(
                        result["forced_activations"][batch]
                    ),
                    "exploration_blocks": int(
                        result["exploration_blocks"][batch]
                    ),
                }
            )
        else:
            clips = int(result["posterior_clip_count"][batch])
            draws = int(result["posterior_draw_count"][batch])
            row.update(
                {
                    "posterior_clip_count": clips,
                    "posterior_draw_count": draws,
                    "posterior_clip_fraction": clips / draws,
                    "final_effective_weight_mean": float(
                        result["final_effective_weight_mean"][batch]
                    ),
                }
            )
        rows.append(row)
    return rows


def _seed_means(rows: list[dict], metric: str):
    grouped: dict[int, list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is not None:
            grouped.setdefault(int(row["seed"]), []).append(float(value))
    seeds = sorted(grouped)
    if not seeds:
        raise ValueError(f"no finite observations for metric {metric}")
    values = np.asarray(
        [np.mean(grouped[seed]) for seed in seeds], dtype=float
    )
    return seeds, values


def seed_cluster_bootstrap_summary(
    rows: list[dict],
    metric: str,
    rng: np.random.Generator,
    replicates: int = BOOTSTRAP_REPLICATES,
):
    """Percentile CI after reducing each seed cluster to one mean."""
    seeds, seed_values = _seed_means(rows, metric)
    means = np.empty(replicates)
    chunk = 5000
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        draw = rng.integers(
            0, len(seeds), size=(stop - start, len(seeds))
        )
        means[start:stop] = seed_values[draw].mean(axis=1)
    ci = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(seed_values.mean()),
        "ci95": [float(ci[0]), float(ci[1])],
        "n_seed_clusters": len(seeds),
        "cluster_statistic": "within-seed batch mean",
    }


def seed_cluster_ratio_summary(
    rows: list[dict],
    numerator: str,
    denominator: str,
    rng: np.random.Generator,
    replicates: int = BOOTSTRAP_REPLICATES,
):
    """Cluster bootstrap a ratio of sums, preserving its natural weights."""
    grouped: dict[int, list[float]] = {}
    for row in rows:
        seed = int(row["seed"])
        grouped.setdefault(seed, [0.0, 0.0])
        grouped[seed][0] += float(row[numerator])
        grouped[seed][1] += float(row[denominator])
    seeds = sorted(grouped)
    num = np.asarray([grouped[s][0] for s in seeds], dtype=float)
    den = np.asarray([grouped[s][1] for s in seeds], dtype=float)
    if den.sum() <= 0:
        return {
            "mean": None,
            "ci95": [None, None],
            "n_seed_clusters": len(seeds),
            "cluster_statistic": "ratio of cluster sums",
        }
    ratios = np.empty(replicates)
    chunk = 5000
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        draw = rng.integers(
            0, len(seeds), size=(stop - start, len(seeds))
        )
        sampled_num = num[draw].sum(axis=1)
        sampled_den = den[draw].sum(axis=1)
        ratios[start:stop] = np.divide(
            sampled_num,
            sampled_den,
            out=np.full(stop - start, np.nan),
            where=sampled_den > 0,
        )
    finite = ratios[np.isfinite(ratios)]
    ci = np.quantile(finite, [0.025, 0.975])
    return {
        "mean": float(num.sum() / den.sum()),
        "ci95": [float(ci[0]), float(ci[1])],
        "n_seed_clusters": len(seeds),
        "cluster_statistic": "ratio of sums within resampled seed clusters",
    }


def candidate_summaries(
    rows: list[dict], method: str, parameter: str
):
    values = sorted(
        {
            float(row[parameter])
            for row in rows
            if row["method"] == method
        }
    )
    summaries = []
    for value in values:
        subset = [
            row
            for row in rows
            if row["method"] == method
            and float(row[parameter]) == value
        ]
        _, post = _seed_means(subset, "post_excess_cost_pct")
        _, total = _seed_means(subset, "total_excess_cost_pct")
        summaries.append(
            {
                parameter: value,
                "mean_seed_cluster_post_excess_cost_pct": float(
                    post.mean()
                ),
                "mean_seed_cluster_total_excess_cost_pct": float(
                    total.mean()
                ),
                "n_seed_clusters": len(PILOT_SEEDS),
                "n_batch_rows": len(subset),
            }
        )
    return summaries


def select_candidate(
    summaries: list[dict], parameter: str
) -> float:
    """Post-change excess is primary; total excess and value break ties."""
    selected = min(
        summaries,
        key=lambda row: (
            row["mean_seed_cluster_post_excess_cost_pct"],
            row["mean_seed_cluster_total_excess_cost_pct"],
            row[parameter],
        ),
    )
    return float(selected[parameter])


def run_pilot():
    rows: list[dict] = []
    started = time.perf_counter()
    for index, seed in enumerate(PILOT_SEEDS, 1):
        common, oracle = _problem(seed)
        normals = posterior_normal_bank(seed)
        for alpha in DE_ALPHA_CANDIDATES:
            result = run_de_policy(*common, alpha=alpha)
            rows.extend(
                rows_from_result(
                    "de_cd_whittle_cv",
                    seed,
                    result,
                    oracle,
                    alpha=alpha,
                )
            )
        for gamma in DTS_GAMMA_CANDIDATES:
            result = run_dts_policy(
                *common,
                gamma=gamma,
                posterior_standard_normals=normals,
            )
            rows.extend(
                rows_from_result(
                    "dts_whittle_cv",
                    seed,
                    result,
                    oracle,
                    gamma=gamma,
                )
            )
        print(
            f"pilot {index:02d}/{len(PILOT_SEEDS)} seed={seed}",
            flush=True,
        )

    de_summary = candidate_summaries(
        rows, "de_cd_whittle_cv", "alpha"
    )
    dts_summary = candidate_summaries(
        rows, "dts_whittle_cv", "gamma"
    )
    choices = {
        "de_alpha": select_candidate(de_summary, "alpha"),
        "dts_gamma": select_candidate(dts_summary, "gamma"),
    }
    return {
        "seeds": list(PILOT_SEEDS),
        "selection_rule": (
            "minimize the mean of per-seed mean post-change excess-cost "
            "percent; break exact ties by total excess then smaller parameter"
        ),
        "candidate_summaries": {
            "de_cd_whittle_cv": de_summary,
            "dts_whittle_cv": dts_summary,
        },
        "choices": choices,
        "raw_seed_batch_rows": rows,
        "runtime_seconds": time.perf_counter() - started,
    }


def formal_summaries(
    rows: list[dict], replicates: int
) -> dict:
    summaries = {}
    for method_index, method in enumerate(
        ("de_cd_whittle_cv", "dts_whittle_cv")
    ):
        subset = [row for row in rows if row["method"] == method]
        method_summary = {}
        for metric_index, metric in enumerate(
            (
                "avg_cost",
                "total_excess_cost_pct",
                "pre_excess_cost_pct",
                "post_excess_cost_pct",
                "rank_loss",
            )
        ):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [BOOTSTRAP_SEED, method_index, metric_index]
                )
            )
            method_summary[metric] = seed_cluster_bootstrap_summary(
                subset, metric, rng, replicates
            )
        if method == "de_cd_whittle_cv":
            for offset, metric in enumerate(
                (
                    "pre_false_alarms_per_10k_arm_slots",
                    "forced_slot_fraction",
                ),
                start=20,
            ):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [BOOTSTRAP_SEED, method_index, offset]
                    )
                )
                method_summary[metric] = seed_cluster_bootstrap_summary(
                    subset, metric, rng, replicates
                )
            ratios = (
                (
                    "detection_fraction",
                    "detected_changed_arms",
                    "changed_arms",
                ),
                (
                    "mean_calendar_delay",
                    "calendar_delay_sum",
                    "detected_changed_arms",
                ),
                (
                    "mean_observation_delay",
                    "observation_delay_sum",
                    "detected_changed_arms",
                ),
            )
            for offset, (label, numerator, denominator) in enumerate(
                ratios, start=30
            ):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [BOOTSTRAP_SEED, method_index, offset]
                    )
                )
                method_summary[label] = seed_cluster_ratio_summary(
                    subset,
                    numerator,
                    denominator,
                    rng,
                    replicates,
                )
        else:
            for offset, metric in enumerate(
                (
                    "posterior_clip_fraction",
                    "final_effective_weight_mean",
                ),
                start=40,
            ):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [BOOTSTRAP_SEED, method_index, offset]
                    )
                )
                method_summary[metric] = seed_cluster_bootstrap_summary(
                    subset, metric, rng, replicates
                )
        summaries[method] = method_summary
    return summaries


def run_formal(
    de_alpha: float,
    dts_gamma: float,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
):
    rows: list[dict] = []
    per_seed_runtime = []
    started = time.perf_counter()
    for index, seed in enumerate(FORMAL_SEEDS, 1):
        seed_started = time.perf_counter()
        common, oracle = _problem(seed)
        de_result = run_de_policy(*common, alpha=de_alpha)
        rows.extend(
            rows_from_result(
                "de_cd_whittle_cv",
                seed,
                de_result,
                oracle,
                alpha=de_alpha,
            )
        )
        dts_result = run_dts_policy(
            *common,
            gamma=dts_gamma,
            posterior_standard_normals=posterior_normal_bank(seed),
        )
        rows.extend(
            rows_from_result(
                "dts_whittle_cv",
                seed,
                dts_result,
                oracle,
                gamma=dts_gamma,
            )
        )
        seconds = time.perf_counter() - seed_started
        per_seed_runtime.append({"seed": seed, "seconds": seconds})
        print(
            f"formal {index:02d}/{len(FORMAL_SEEDS)} seed={seed} "
            f"seconds={seconds:.2f}",
            flush=True,
        )

    summaries = formal_summaries(rows, bootstrap_replicates)
    return {
        "seeds": list(FORMAL_SEEDS),
        "raw_seed_batch_rows": rows,
        "seed_cluster_bootstrap": {
            "replicates": bootstrap_replicates,
            "confidence_level": 0.95,
            "seed": BOOTSTRAP_SEED,
            "resampling_unit": (
                "formal seed; all three within-seed batches stay together"
            ),
            "summaries": summaries,
        },
        "per_seed_runtime": per_seed_runtime,
        "runtime_seconds": time.perf_counter() - started,
    }


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(script_path: Path) -> dict:
    root = script_path.resolve().parent
    source_files = (
        "run_cv_piecewise_pilot.py",
        "core/change_detection_cv.py",
        "core/online_cv_moments_stable.py",
        "core/cv_sequential_calibration.py",
    )
    return {
        "retrospective_matched_addendum": True,
        "not_official_reproduction": True,
        "de_cd_whittle_cv": {
            "public_source": {
                "title": (
                    "A Modularized Framework for Piecewise-Stationary "
                    "Restless Bandits"
                ),
                "authors": (
                    "Kuan-Ta Li; Chia-Chun Lin; Ping-Chun Hsieh; "
                    "Yu-Chih Huang"
                ),
                "venue": "AISTATS 2026 Spotlight",
                "arxiv": "2604.10177v1",
                "url": "https://arxiv.org/abs/2604.10177",
            },
            "source_schedule": {
                "initial_cursor": (
                    "ceil((alpha - K/(4*alpha))^2), lower-bounded at 1"
                ),
                "recurrence": (
                    "ceil(u + K*sqrt(u)/alpha + K^2/(4*alpha^2))"
                ),
                "alarm_reset": "segment origin becomes alarm round; u=1",
            },
            "matched_adaptations": [
                (
                    "N>1: each uniform-exploration block uses ceil(K/N) "
                    "cyclic N-arm slots instead of K one-arm slots"
                ),
                (
                    "independent simulator batches maintain independent "
                    "exploration cursors and reset on any arm alarm"
                ),
                (
                    "the existing selected-only TwoWindowCVMomentDetector "
                    "and per-arm ResettableOnlineCVMomentEstimator replace "
                    "the source reward detector and global base-solver reset"
                ),
                (
                    "forced-exploration CV moments are shared with both the "
                    "detector and UCB-CV base learner"
                ),
            ],
        },
        "dts_whittle_cv": {
            "public_source": {
                "title": (
                    "Discounted Thompson Sampling for Non-Stationary "
                    "Bandit Problems"
                ),
                "authors": "Han Qi; Yue Wang; Li Zhu",
                "year": 2023,
                "arxiv": "2305.10718",
                "url": "https://arxiv.org/abs/2305.10718",
            },
            "matched_adaptations": [
                (
                    "calendar-time gamma discount is applied to every arm's "
                    "CV-moment effective weight and centered second moment"
                ),
                (
                    "selected arms alone receive a new physical-moment block"
                ),
                (
                    "a diagonal Gaussian plug-in posterior replaces the "
                    "source bandit's reward posterior and is redrawn per slot"
                ),
                (
                    "draws are projected to the frozen physical variance "
                    "box [0.01,1.0] before the shared Riccati/Whittle map"
                ),
            ],
        },
        "paired_randomness": {
            "problem_and_observation_banks": (
                "make_problem(seed, B=3, K=20, change_t=500, H=1000, "
                "n0=8, block_length=64)"
            ),
            "oracle": (
                "run_cv_piecewise_pilot.run_policy('true', same common data)"
            ),
            "dts_normal_stream": (
                "NumPy SeedSequence([seed, 0xD75, len(schema_version)]); "
                "same normal bank for every gamma candidate"
            ),
        },
        "source_file_sha256": {
            relative: sha256_file(root / relative)
            for relative in source_files
        },
        "runner_sha256": sha256_file(script_path),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }


def build_payload(
    pilot: dict,
    formal: dict,
    script_path: Path,
    total_runtime: float,
):
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "retrospective_matched_addendum": True,
        "protocol": {
            "base_protocol": BASE_PROTOCOL,
            "problem_generator": (
                "run_cv_piecewise_pilot.make_problem"
            ),
            "oracle": "run_cv_piecewise_pilot.run_policy('true')",
            "batches_per_seed": BATCHES_PER_SEED,
            "batch_indexing": "zero-based within each seed",
            "K": K,
            "N": N,
            "H": H,
            "n0": N0,
            "change_t": CHANGE_T,
            "block_length": BLOCK_LENGTH,
            "detector_window": DETECTOR_WINDOW,
            "detector_threshold": DETECTOR_THRESHOLD,
            "pilot_seeds": list(PILOT_SEEDS),
            "formal_seeds": list(FORMAL_SEEDS),
            "formal_data_not_used_for_tuning": True,
        },
        "pilot": pilot,
        "formal": formal,
        "provenance": provenance(script_path),
        "runtime_seconds": total_runtime,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/tmc_v16_baseline_expansion.json"),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    args = parser.parse_args()
    if args.bootstrap_replicates < 1000:
        raise SystemExit("bootstrap-replicates must be at least 1000")

    started = time.perf_counter()
    pilot = run_pilot()
    choices = pilot["choices"]
    print(
        f"pilot locked alpha={choices['de_alpha']:g}, "
        f"gamma={choices['dts_gamma']:g}",
        flush=True,
    )
    formal = run_formal(
        choices["de_alpha"],
        choices["dts_gamma"],
        bootstrap_replicates=args.bootstrap_replicates,
    )
    total_runtime = time.perf_counter() - started
    payload = build_payload(
        pilot, formal, Path(__file__), total_runtime
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"json={args.json_out}", flush=True)
    for method, summary in formal["seed_cluster_bootstrap"][
        "summaries"
    ].items():
        total = summary["total_excess_cost_pct"]
        post = summary["post_excess_cost_pct"]
        print(
            f"{method:24s} total={total['mean']:.3f}% "
            f"[{total['ci95'][0]:.3f},{total['ci95'][1]:.3f}] "
            f"post={post['mean']:.3f}% "
            f"[{post['ci95'][0]:.3f},{post['ci95'][1]:.3f}]",
            flush=True,
        )
    print(f"runtime_seconds={total_runtime:.2f}", flush=True)


if __name__ == "__main__":
    main()
