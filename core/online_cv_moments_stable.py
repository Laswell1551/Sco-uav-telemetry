"""Numerically stable online CV moments.

This supersedes ``OnlineCVMomentEstimator`` in ``online_cv_moments.py``.
The earlier prototype stores raw sums and squared sums; this implementation
uses masked Welford updates to avoid catastrophic cancellation in the sample
variance.  Shared block-generation and DARE-box helpers remain imported from
the original module.
"""
from __future__ import annotations

import numpy as np

from .online_cv_moments import (
    _effective_theta,
    cv_block_physical_observations,
    effective_corner_box,
)


class OnlineCVMomentEstimator:
    """Masked Welford estimator for per-arm physical CV parameters."""

    def __init__(
        self,
        initial_observations,
        confidence_scale=3.0,
        variance_floor=1e-6,
        variance_ceiling=10.0,
        T=1.0,
    ):
        observations = np.asarray(initial_observations, dtype=float)
        if observations.ndim != 4 or observations.shape[-1] != 2:
            raise ValueError("initial_observations must have shape (R, B, K, 2)")
        if observations.shape[0] < 2:
            raise ValueError("at least two initial blocks are required")
        if not np.all(np.isfinite(observations)):
            raise ValueError("initial observations must be finite")
        if not 0 < variance_floor < variance_ceiling:
            raise ValueError("projection bounds must be positive and ordered")

        self.running_mean = observations.mean(axis=0)
        centered = observations - self.running_mean[None, ...]
        self.running_m2 = np.square(centered).sum(axis=0)
        self.count = np.full(
            observations.shape[1:3], observations.shape[0], dtype=float
        )
        self.confidence_scale = float(confidence_scale)
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)
        self.T = float(T)

    @property
    def physical_mean_raw(self):
        return self.running_mean

    @property
    def physical_mean(self):
        return np.clip(
            self.physical_mean_raw,
            self.variance_floor,
            self.variance_ceiling,
        )

    @property
    def physical_radius_proxy(self):
        n = self.count[..., None]
        sample_var = self.running_m2 / np.maximum(n - 1.0, 1.0)
        return self.confidence_scale * np.sqrt(sample_var / n)

    @property
    def physical_box(self):
        mean = self.physical_mean_raw
        radius = self.physical_radius_proxy
        lo = np.clip(
            mean - radius,
            self.variance_floor,
            self.variance_ceiling,
        )
        hi = np.clip(
            mean + radius,
            self.variance_floor,
            self.variance_ceiling,
        )
        hi = np.maximum(hi, lo)
        return lo, hi

    @property
    def mean(self):
        physical = self.physical_mean
        return _effective_theta(physical[..., 0], physical[..., 1], self.T)

    @property
    def effective_box(self):
        lo, hi = self.physical_box
        return effective_corner_box(lo, hi, self.T)

    @property
    def radius(self):
        lo, hi = self.effective_box
        center = self.mean
        return np.maximum(center - lo, hi - center)

    def update(self, selected, observation):
        selected = np.asarray(selected, dtype=bool)
        observation = np.asarray(observation, dtype=float)
        if selected.shape != self.count.shape:
            raise ValueError("selected must have shape (B, K)")
        if observation.shape != self.running_mean.shape:
            raise ValueError("observation must have shape (B, K, 2)")
        if not np.all(np.isfinite(observation[selected])):
            raise ValueError("selected observations must be finite")

        keep = selected[..., None]
        new_count = self.count + selected
        delta = observation - self.running_mean
        candidate_mean = self.running_mean + delta / new_count[..., None]
        delta2 = observation - candidate_mean
        self.running_mean = np.where(keep, candidate_mean, self.running_mean)
        self.running_m2 = np.where(
            keep,
            self.running_m2 + delta * delta2,
            self.running_m2,
        )
        self.count = new_count


__all__ = [
    "OnlineCVMomentEstimator",
    "cv_block_physical_observations",
    "effective_corner_box",
]
