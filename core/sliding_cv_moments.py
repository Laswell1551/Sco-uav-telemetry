"""Sliding-window CV physical-moment estimator for SW-Whittle-CV baselines."""
from __future__ import annotations

import numpy as np

from .online_cv_moments import _effective_theta, effective_corner_box


class SlidingWindowCVMomentEstimator:
    """Per-arm circular buffer of raw (sw2, sv2) block estimates."""

    def __init__(
        self,
        initial_observations,
        window,
        confidence_scale=3.0,
        variance_floor=1e-6,
        variance_ceiling=10.0,
        T=1.0,
    ):
        observations = np.asarray(initial_observations, dtype=float)
        if observations.ndim != 4 or observations.shape[-1] != 2:
            raise ValueError("initial_observations must have shape (R,B,K,2)")
        if window < 2:
            raise ValueError("window must be at least two")
        if observations.shape[0] < 2:
            raise ValueError("at least two initial blocks are required")
        if not np.all(np.isfinite(observations)):
            raise ValueError("initial observations must be finite")
        if not 0 < variance_floor < variance_ceiling:
            raise ValueError("projection bounds must be positive and ordered")

        keep = observations[-window:]
        _, B, K, _ = keep.shape
        self.window = int(window)
        self.buffer = np.zeros((B, K, self.window, 2), dtype=float)
        self.buffer[:, :, : keep.shape[0], :] = np.transpose(keep, (1, 2, 0, 3))
        self.count = np.full((B, K), keep.shape[0], dtype=int)
        self.total_count = np.full((B, K), observations.shape[0], dtype=int)
        self.pointer = np.full((B, K), keep.shape[0] % self.window, dtype=int)
        self.confidence_scale = float(confidence_scale)
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)
        self.T = float(T)

    def _mean_var(self):
        slots = np.arange(self.window)[None, None, :, None]
        valid = slots < self.count[:, :, None, None]
        total = np.where(valid, self.buffer, 0.0).sum(axis=2)
        mean = total / self.count[..., None]
        centered = np.where(valid, self.buffer - mean[:, :, None, :], 0.0)
        var = np.square(centered).sum(axis=2) / np.maximum(
            self.count[..., None] - 1, 1
        )
        return mean, var

    @property
    def physical_mean_raw(self):
        return self._mean_var()[0]

    @property
    def physical_mean(self):
        return np.clip(
            self.physical_mean_raw, self.variance_floor, self.variance_ceiling
        )

    @property
    def physical_radius_proxy(self):
        _, var = self._mean_var()
        return self.confidence_scale * np.sqrt(var / self.count[..., None])

    @property
    def physical_box(self):
        mean = self.physical_mean_raw
        radius = self.physical_radius_proxy
        lo = np.clip(
            mean - radius, self.variance_floor, self.variance_ceiling
        )
        hi = np.clip(
            mean + radius, self.variance_floor, self.variance_ceiling
        )
        return lo, np.maximum(hi, lo)

    @property
    def mean(self):
        physical = self.physical_mean
        return _effective_theta(physical[..., 0], physical[..., 1], self.T)

    @property
    def effective_box(self):
        return effective_corner_box(*self.physical_box, T=self.T)

    @property
    def radius(self):
        lo, hi = self.effective_box
        center = self.mean
        return np.maximum(center - lo, hi - center)

    def update(self, selected, observation):
        selected = np.asarray(selected, dtype=bool)
        observation = np.asarray(observation, dtype=float)
        if selected.shape != self.count.shape:
            raise ValueError("selected must have shape (B,K)")
        if observation.shape != self.count.shape + (2,):
            raise ValueError("observation must have shape (B,K,2)")
        if not np.all(np.isfinite(observation[selected])):
            raise ValueError("selected observations must be finite")

        for b, k in np.argwhere(selected):
            self.buffer[b, k, self.pointer[b, k]] = observation[b, k]
            self.pointer[b, k] = (self.pointer[b, k] + 1) % self.window
            self.count[b, k] = min(self.count[b, k] + 1, self.window)
            self.total_count[b, k] += 1
