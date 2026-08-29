"""Consistent online CV-moment estimator under action-dependent feedback.

Each served arm provides a fresh constant-velocity position window.  The
window's second-difference moments give an *unclipped* estimate of the two
physical variance parameters (sw2, sv2).  The online estimator averages these
linear moment estimates before applying positivity projection and the
nonlinear DARE map.

This ordering matters: directly clipping and averaging blockwise
(sw2, P12, P22) estimates generally creates a finite-window bias that does not
vanish with the number of service blocks.

The empirical standard-error radius below is a diagnostic proxy, not a
time-uniform confidence sequence.  The final TMC method must replace it with
the declared sub-exponential/robust sequential bound.
"""
from __future__ import annotations

import itertools

import numpy as np

from .instances import mom_estimate_batch, pbar_batch, simulate_position_windows


def cv_block_physical_observations(
    sw2_true,
    sv2_true,
    selected,
    n_slots,
    rng,
    T=1.0,
):
    """Return raw per-block estimates of (sw2, sv2) for selected arms.

    Unselected entries are zero and must be ignored through the selection
    mask.  Estimates are intentionally not clipped.
    """
    sw2_true = np.asarray(sw2_true, dtype=float)
    sv2_true = np.asarray(sv2_true, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    if sw2_true.shape != sv2_true.shape or selected.shape != sw2_true.shape:
        raise ValueError("sw2_true, sv2_true, and selected must share shape (B, K)")
    if np.any(sw2_true <= 0) or np.any(sv2_true <= 0):
        raise ValueError("true variances must be positive")
    if n_slots < 4:
        raise ValueError("n_slots must be at least four")

    observation = np.zeros(sw2_true.shape + (2,), dtype=float)
    for b, k in np.argwhere(selected):
        positions = simulate_position_windows(
            T,
            float(sw2_true[b, k]),
            float(sv2_true[b, k]),
            n_win=1,
            n_slots=n_slots,
            rng=rng,
        )
        sw2_hat, sv2_hat = mom_estimate_batch(positions, T)
        observation[b, k] = (sw2_hat[0], sv2_hat[0])
    return observation


def _effective_theta(sw2, sv2, T):
    shape = np.broadcast_shapes(np.shape(sw2), np.shape(sv2))
    sw2 = np.broadcast_to(np.asarray(sw2, dtype=float), shape)
    sv2 = np.broadcast_to(np.asarray(sv2, dtype=float), shape)
    _, p12, p22 = pbar_batch(T, sw2.ravel(), sv2.ravel())
    return np.stack(
        [sw2, p12.reshape(shape), p22.reshape(shape)],
        axis=-1,
    )


def effective_corner_box(physical_lo, physical_hi, T=1.0):
    """Corner envelope for the DARE effective-parameter map.

    The function evaluates all four corners of each (sw2, sv2) rectangle.
    Numerical audits over the declared parameter domain indicate coordinatewise
    monotonicity, but a four-corner envelope is a certified interval only after
    that monotonicity is proved (or an interval optimizer replaces it).
    """
    physical_lo = np.asarray(physical_lo, dtype=float)
    physical_hi = np.asarray(physical_hi, dtype=float)
    if physical_lo.shape != physical_hi.shape or physical_lo.shape[-1] != 2:
        raise ValueError("physical boxes must share shape (..., 2)")
    if np.any(physical_lo <= 0) or np.any(physical_hi < physical_lo):
        raise ValueError("physical boxes must be positive and ordered")

    corner_values = []
    for use_hi_sw, use_hi_sv in itertools.product((False, True), repeat=2):
        sw2 = physical_hi[..., 0] if use_hi_sw else physical_lo[..., 0]
        sv2 = physical_hi[..., 1] if use_hi_sv else physical_lo[..., 1]
        corner_values.append(_effective_theta(sw2, sv2, T))
    stacked = np.stack(corner_values, axis=0)
    return stacked.min(axis=0), stacked.max(axis=0)


class OnlineCVMomentEstimator:
    """Masked running moments for per-arm physical CV parameters."""

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

        self.total = observations.sum(axis=0)
        self.total_sq = np.square(observations).sum(axis=0)
        self.count = np.full(observations.shape[1:3], observations.shape[0], dtype=float)
        self.confidence_scale = float(confidence_scale)
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)
        self.T = float(T)

    @property
    def physical_mean_raw(self):
        return self.total / self.count[..., None]

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
        centered_ss = np.maximum(self.total_sq - np.square(self.total) / n, 0.0)
        sample_var = centered_ss / np.maximum(n - 1.0, 1.0)
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
        if observation.shape != self.total.shape:
            raise ValueError("observation must have shape (B, K, 2)")
        if not np.all(np.isfinite(observation[selected])):
            raise ValueError("selected observations must be finite")
        keep = selected[..., None]
        self.total += np.where(keep, observation, 0.0)
        self.total_sq += np.where(keep, np.square(observation), 0.0)
        self.count += selected
