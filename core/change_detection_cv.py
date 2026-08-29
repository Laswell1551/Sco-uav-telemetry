"""Two-window change detection and per-arm reset for raw CV moments.

This is a transparent PS-RMAB-inspired component, not a reproduction of the
PS-RMAB detector.  Raw CV moment blocks are heavy-tailed quadratic forms, so
the Welch-style statistic implemented here is a tuning baseline rather than a
finite-sample guarantee.
"""
from __future__ import annotations

import numpy as np

from .online_cv_moments_stable import OnlineCVMomentEstimator


class TwoWindowCVMomentDetector:
    """Compare the latest two observation-indexed windows for each arm."""

    def __init__(self, B, K, window=8, threshold=5.0, se_floor=1e-10):
        if window < 2:
            raise ValueError("window must be at least two")
        if threshold <= 0 or se_floor <= 0:
            raise ValueError("threshold and se_floor must be positive")
        self.window = int(window)
        self.capacity = 2 * self.window
        self.threshold = float(threshold)
        self.se_floor = float(se_floor)
        self.buffer = np.zeros((B, K, self.capacity, 2), dtype=float)
        self.count = np.zeros((B, K), dtype=int)
        self.pointer = np.zeros((B, K), dtype=int)

    def _ordered(self, b, k):
        n = self.count[b, k]
        if n < self.capacity:
            return self.buffer[b, k, :n]
        p = self.pointer[b, k]
        return np.concatenate(
            [self.buffer[b, k, p:], self.buffer[b, k, :p]], axis=0
        )

    def update(self, selected, observation):
        selected = np.asarray(selected, dtype=bool)
        observation = np.asarray(observation, dtype=float)
        if observation.shape != selected.shape + (2,):
            raise ValueError("observation must have shape selected.shape + (2,)")
        B, K = selected.shape
        alarms = np.zeros((B, K), dtype=bool)
        statistic = np.zeros((B, K, 2), dtype=float)
        reset_mean = np.zeros((B, K, 2), dtype=float)
        reset_m2 = np.zeros((B, K, 2), dtype=float)
        reset_count = np.zeros((B, K), dtype=int)

        for b, k in np.argwhere(selected):
            p = self.pointer[b, k]
            self.buffer[b, k, p] = observation[b, k]
            self.pointer[b, k] = (p + 1) % self.capacity
            self.count[b, k] = min(self.count[b, k] + 1, self.capacity)
            if self.count[b, k] < self.capacity:
                continue

            values = self._ordered(b, k)
            old = values[: self.window]
            new = values[self.window :]
            old_var = old.var(axis=0, ddof=1)
            new_var = new.var(axis=0, ddof=1)
            se = np.sqrt((old_var + new_var) / self.window)
            stat = np.abs(new.mean(axis=0) - old.mean(axis=0)) / np.maximum(
                se, self.se_floor
            )
            statistic[b, k] = stat
            if np.any(stat > self.threshold):
                alarms[b, k] = True
                mean = new.mean(axis=0)
                reset_mean[b, k] = mean
                reset_m2[b, k] = np.square(new - mean).sum(axis=0)
                reset_count[b, k] = self.window

                self.buffer[b, k] = 0.0
                self.buffer[b, k, : self.window] = new
                self.count[b, k] = self.window
                self.pointer[b, k] = self.window
        return {
            "alarms": alarms,
            "statistic": statistic,
            "reset_mean": reset_mean,
            "reset_m2": reset_m2,
            "reset_count": reset_count,
        }


class ResettableOnlineCVMomentEstimator(OnlineCVMomentEstimator):
    """Stable cumulative estimator with detector-provided per-arm resets."""

    def update_and_reset(self, selected, observation, detector_result):
        alarms = np.asarray(detector_result["alarms"], dtype=bool)
        if alarms.shape != self.count.shape:
            raise ValueError("alarm shape differs from estimator")
        self.update(selected, observation)
        reset_mean = np.asarray(detector_result["reset_mean"], dtype=float)
        reset_m2 = np.asarray(detector_result["reset_m2"], dtype=float)
        reset_count = np.asarray(detector_result["reset_count"], dtype=int)
        if reset_mean.shape != self.running_mean.shape:
            raise ValueError("reset state shape differs from estimator")
        self.running_mean = np.where(
            alarms[..., None], reset_mean, self.running_mean
        )
        self.running_m2 = np.where(
            alarms[..., None], reset_m2, self.running_m2
        )
        self.count = np.where(alarms, reset_count, self.count)
