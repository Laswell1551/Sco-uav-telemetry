"""Empirical calibration of practical finite-horizon CV radii.

Cal-SCO uses the maximum studentized running-mean error over all sample counts
in a calibration sequence.  A split-conformal upper quantile turns this into
a finite-horizon multiplier for exchangeable calibration/deployment
sequences.  Taking the worst calibrated multiplier over a physical-parameter
grid improves stress coverage but does not prove coverage between grid points.

This module is intentionally separate from the analytic Safe-SCO bound.
"""
from __future__ import annotations

import math

import numpy as np

from .cv_moment_confidence import (
    physical_quadratic_matrices,
    second_difference_covariance_components,
)


def conformal_upper_quantile(scores, alpha):
    """Finite-sample split-conformal upper quantile.

    Returns infinity when the calibration size cannot resolve the requested
    tail probability.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size < 1 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a finite nonempty vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    rank = int(math.ceil((scores.size + 1) * (1.0 - alpha)))
    if rank > scores.size:
        return math.inf
    return float(np.partition(scores, rank - 1)[rank - 1])


def studentized_max_scores(observations, truth, n_start=4, se_floor=1e-12):
    """Maximum running-mean t score for each calibration sequence.

    ``observations`` has shape (R, n_max, 2), and ``truth`` broadcasts to
    (R, 1, 2).
    """
    observations = np.asarray(observations, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if observations.ndim != 3 or observations.shape[-1] != 2:
        raise ValueError("observations must have shape (R, n_max, 2)")
    if not 2 <= n_start <= observations.shape[1]:
        raise ValueError("n_start must lie between 2 and n_max")
    if se_floor <= 0:
        raise ValueError("se_floor must be positive")

    n = np.arange(1, observations.shape[1] + 1, dtype=float)[None, :, None]
    cumulative = np.cumsum(observations, axis=1)
    cumulative_sq = np.cumsum(np.square(observations), axis=1)
    mean = cumulative / n
    centered_ss = np.maximum(cumulative_sq - np.square(cumulative) / n, 0.0)
    sample_var = centered_ss / np.maximum(n - 1.0, 1.0)
    se = np.sqrt(sample_var / n)
    score = np.abs(mean - truth) / np.maximum(se, se_floor)
    return np.max(score[:, n_start - 1 :, :], axis=(1, 2))


def quadratic_block_sequences(
    sw2,
    sv2,
    n_slots,
    n_sequences,
    n_blocks,
    rng,
):
    """Directly simulate raw physical block estimators as quadratic forms."""
    if sw2 <= 0 or sv2 <= 0:
        raise ValueError("physical variances must be positive")
    if n_sequences < 1 or n_blocks < 2:
        raise ValueError("need positive sequences and at least two blocks")
    gamma_sw, gamma_sv = second_difference_covariance_components(n_slots)
    gamma = sw2 * gamma_sw + sv2 * gamma_sv
    eigval, eigvec = np.linalg.eigh(gamma)
    root = (eigvec * np.sqrt(np.maximum(eigval, 0.0))) @ eigvec.T
    standard = rng.standard_normal((n_sequences, n_blocks, gamma.shape[0]))
    d = standard @ root.T
    matrices = physical_quadratic_matrices(n_slots)
    return np.stack(
        [np.einsum("rni,ij,rnj->rn", d, A, d) for A in matrices],
        axis=-1,
    )


def calibrate_studentized_multiplier(
    parameter_grid,
    n_slots,
    n_sequences,
    n_blocks,
    alpha,
    rng,
    n_start=4,
    batch_size=64,
):
    """Calibrate per-grid and worst-grid sequential t multipliers."""
    grid = np.asarray(parameter_grid, dtype=float)
    if grid.ndim != 2 or grid.shape[1] != 2 or np.any(grid <= 0):
        raise ValueError("parameter_grid must have shape (G,2) and be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    grid_quantiles = np.empty(grid.shape[0])
    for g, (sw2, sv2) in enumerate(grid):
        pieces = []
        remaining = n_sequences
        while remaining:
            take = min(batch_size, remaining)
            obs = quadratic_block_sequences(
                sw2,
                sv2,
                n_slots,
                take,
                n_blocks,
                rng,
            )
            pieces.append(
                studentized_max_scores(
                    obs,
                    np.array([sw2, sv2])[None, None, :],
                    n_start=n_start,
                )
            )
            remaining -= take
        scores = np.concatenate(pieces)
        grid_quantiles[g] = conformal_upper_quantile(scores, alpha)
    return {
        "multiplier": float(np.max(grid_quantiles)),
        "grid_quantiles": grid_quantiles,
        "parameter_grid": grid,
        "alpha": float(alpha),
        "n_sequences": int(n_sequences),
        "n_blocks": int(n_blocks),
        "n_start": int(n_start),
    }


def studentized_radius(running_m2, count, multiplier, se_floor=1e-12):
    """Apply a calibrated multiplier to masked Welford state."""
    running_m2 = np.asarray(running_m2, dtype=float)
    count = np.asarray(count, dtype=float)
    if running_m2.shape != count.shape + (2,):
        raise ValueError("running_m2 must have shape count.shape + (2,)")
    if np.any(count < 2) or multiplier < 0:
        raise ValueError("counts must be at least two and multiplier non-negative")
    sample_var = running_m2 / (count[..., None] - 1.0)
    se = np.sqrt(np.maximum(sample_var, 0.0) / count[..., None])
    return multiplier * np.maximum(se, se_floor)
