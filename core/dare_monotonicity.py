"""Dimensionless CV-DARE formulas used by the primitive-box certificate.

Let r = sigma_v^2, q = sigma_w^2, and

    eta = sqrt(q T^3 / r),  z = y / sqrt(r),  s = z + 1/z.

For the stabilizing (largest-y) solution of the CV quartic,

    eta = 3 s - sqrt(3) sqrt(s^2 + 8),  s > 2.

The filtered covariance entries are

    P12 = (r/T) eta/z,
    P22 = (r/T^2) (eta sqrt(s^2-4) - eta^2/2).

These formulas make the componentwise monotonicity in (q,r) explicit.
This module is intentionally small: the production covariance evaluator remains
``core.instances.pbar_batch`` and this file supports proof/audit tests.
"""
from __future__ import annotations

import numpy as np


def eta_of_s(s):
    """Physical/stabilizing CV-DARE branch eta(s), for s > 2."""
    s = np.asarray(s, dtype=float)
    return 3.0 * s - np.sqrt(3.0) * np.sqrt(s * s + 8.0)


def eta_prime_of_s(s):
    """Derivative d eta / d s on the stabilizing branch."""
    s = np.asarray(s, dtype=float)
    return 3.0 - np.sqrt(3.0) * s / np.sqrt(s * s + 8.0)


def normalized_covariances(s):
    """Return g,h such that P12=(r/T)g and P22=(r/T^2)h."""
    s = np.asarray(s, dtype=float)
    x = np.sqrt(s * s - 4.0)
    z = 0.5 * (s + x)
    eta = eta_of_s(s)
    g = eta / z
    h = eta * x - 0.5 * eta * eta
    return g, h


def derivative_witnesses(s):
    """Positive witnesses for all four primitive covariance derivatives.

    Returned quantities have the same signs as
      dP12/dq, dP12/dr, dP22/dq, dP22/dr,
    after removing strictly positive scale factors.
    """
    s = np.asarray(s, dtype=float)
    if np.any(s <= 2.0):
        raise ValueError("The physical stabilizing branch requires s > 2.")
    x = np.sqrt(s * s - 4.0)
    z = 0.5 * (s + x)
    eta = eta_of_s(s)
    etap = eta_prime_of_s(s)
    zs = z / x

    # g_s = (eta' z - eta z_s)/z^2.
    p12_q = etap - eta / x

    # 2 eta' [g - eta g_eta/2]
    p12_r = eta * etap / z + eta * eta * zs / (z * z)

    # h_s = eta'(x-eta) + eta s/x.
    p22_q = etap * (x - eta) + eta * s / x

    # Sign-equivalent form for h - eta h_eta/2.
    p22_r = etap * (s - eta / 6.0) - s
    return p12_q, p12_r, p22_q, p22_r

