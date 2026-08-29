"""Finite-horizon simultaneous bounds for CV second-difference moments.

For one position window, the raw physical estimates (sw2_hat, sv2_hat) are
Gaussian quadratic forms d' A_m d, where d is the vector of second
differences.  Independent service windows give independent copies.

For a centered Gaussian quadratic form with B = Gamma^(1/2) A Gamma^(1/2),

    P(|Q - E Q| >= 2 ||B||_F sqrt(x) + 2 ||B||_op x) <= 2 exp(-x).

For the mean of n independent windows the Frobenius and operator terms scale
as n^(-1/2) and n^(-1), respectively.  We upper-bound both B norms using a
declared physical-variance box, then union-bound over K arms, two physical
coordinates, and all sample counts up to n_max.

The bound is intentionally conservative but is a genuine simultaneous
finite-horizon guarantee under the declared independent Gaussian CV-window
model.  It is not valid for arbitrary real trajectories without an additional
model-misspecification argument.
"""
from __future__ import annotations

import math

import numpy as np


def second_difference_covariance_components(n_slots, T=1.0):
    """Return Gamma_sw and Gamma_sv for a length-n_slots CV position window."""
    if n_slots < 4:
        raise ValueError("n_slots must be at least four")
    if T <= 0:
        raise ValueError("T must be positive")

    # Position samples 0,...,W-1 depend on process increments 0,...,W-2.
    W = int(n_slots)
    n_process = W - 1
    mapping = np.zeros((W, 2 * n_process))
    for j in range(1, W):
        for r in range(j):
            mapping[j, 2 * r] = 1.0
            if r <= j - 2:
                mapping[j, 2 * r + 1] = T * (j - r - 1)

    q_unit = np.array(
        [[T**3 / 3.0, T**2 / 2.0], [T**2 / 2.0, T]],
        dtype=float,
    )
    process_cov = np.kron(np.eye(n_process), q_unit)
    position_process_cov = mapping @ process_cov @ mapping.T

    D = np.zeros((W - 2, W))
    rows = np.arange(W - 2)
    D[rows, rows] = 1.0
    D[rows, rows + 1] = -2.0
    D[rows, rows + 2] = 1.0
    gamma_sw = D @ position_process_cov @ D.T
    gamma_sv = D @ D.T
    return gamma_sw, gamma_sv


def physical_quadratic_matrices(n_slots, T=1.0):
    """Matrices A_sw and A_sv whose quadratic forms are the raw estimators."""
    if n_slots < 4:
        raise ValueError("n_slots must be at least four")
    d = n_slots - 2
    A0 = np.eye(d) / d
    A1 = np.zeros((d, d))
    off = 1.0 / (2.0 * (d - 1))
    idx = np.arange(d - 1)
    A1[idx, idx + 1] = off
    A1[idx + 1, idx] = off

    A_sw = (12.0 / 11.0) * (A0 + 1.5 * A1) / T**3
    A_sv = (A0 - (2.0 / 3.0) * T**3 * A_sw) / 6.0
    return A_sw, A_sv


def verify_moment_identities(n_slots, T=1.0, atol=1e-10):
    """Numerically verify E[hat theta] = theta for the constructed matrices."""
    gamma_sw, gamma_sv = second_difference_covariance_components(n_slots, T)
    A_sw, A_sv = physical_quadratic_matrices(n_slots, T)
    matrix = np.array(
        [
            [np.trace(A_sw @ gamma_sw), np.trace(A_sw @ gamma_sv)],
            [np.trace(A_sv @ gamma_sw), np.trace(A_sv @ gamma_sv)],
        ]
    )
    if not np.allclose(matrix, np.eye(2), atol=atol, rtol=0):
        raise AssertionError(f"moment identity failed:\n{matrix}")
    return matrix


def uniform_quadratic_norm_bounds(
    n_slots,
    sw2_upper,
    sv2_upper,
    T=1.0,
):
    """Return conservative (Frobenius, operator) bounds for both estimators."""
    if sw2_upper <= 0 or sv2_upper <= 0:
        raise ValueError("variance upper bounds must be positive")
    gamma_sw, gamma_sv = second_difference_covariance_components(n_slots, T)
    gamma_op = (
        sw2_upper * np.linalg.norm(gamma_sw, ord=2)
        + sv2_upper * np.linalg.norm(gamma_sv, ord=2)
    )
    out = np.empty((2, 2))
    for m, A in enumerate(physical_quadratic_matrices(n_slots, T)):
        out[m, 0] = gamma_op * np.linalg.norm(A, ord="fro")
        out[m, 1] = gamma_op * np.linalg.norm(A, ord=2)
    return out


def finite_horizon_physical_radius(
    count,
    n_slots,
    K,
    n_max,
    delta,
    sw2_upper,
    sv2_upper,
    T=1.0,
):
    """Simultaneous radius for every arm, coordinate, and n <= n_max.

    ``count`` may be a scalar or an array.  The result appends a final
    coordinate dimension of size two, ordered as (sw2, sv2).
    """
    count = np.asarray(count, dtype=float)
    if np.any(count < 1):
        raise ValueError("count must be at least one")
    if K < 1 or n_max < 1 or not 0 < delta < 1:
        raise ValueError("invalid K, n_max, or delta")
    x = math.log(4.0 * K * n_max / delta)
    norms = uniform_quadratic_norm_bounds(
        n_slots, sw2_upper, sv2_upper, T
    )
    flat = (
        2.0 * norms[:, 0] / np.sqrt(count[..., None]) * math.sqrt(x)
        + 2.0 * norms[:, 1] / count[..., None] * x
    )
    return flat
