"""Single-arm CV-model core: cost polynomial, closed-form Whittle index,
sensitivities, error bars, DARE. Shared by numeric_checks.py (A0) and the
Part-B simulator.

Conventions used by the SCO artifact:
  - age a in {1,2,...}; selected => a=1 next slot, else a+1
  - both actions pay holding cost c(a) in the current slot
  - effective parameter vector theta = (sigma_w2, P12, P22); P11 only shifts c0
  - subsidy lambda is paid for PASSIVE slots
"""
import numpy as np
from scipy.linalg import solve_discrete_are


# ---------------------------------------------------------------- CV model
def cv_matrices(T):
    A = np.array([[1.0, T], [0.0, 1.0]])
    Qt = np.array([[T**3 / 3.0, T**2 / 2.0], [T**2 / 2.0, T]])
    return A, Qt


def dare_filtered_cov(T, sigma_w2, sigma_v2):
    """Steady-state *filtered* covariance P̄ for the CV model with position
    measurement C=(1,0) and measurement noise sigma_v2.
    Returns (Pbar 2x2, Sigma 2x2 predicted)."""
    A, Qt = cv_matrices(T)
    C = np.array([[1.0, 0.0]])
    # solve_discrete_are solves the *predicted*-covariance DARE for
    # Sigma = A Sigma A' - A Sigma C'(C Sigma C'+R)^-1 C Sigma A' + Q
    Sigma = solve_discrete_are(A.T, C.T, sigma_w2 * Qt, np.array([[sigma_v2]]))
    S = float(Sigma[0, 0] + sigma_v2)
    K = Sigma @ C.T / S                       # 2x1 Kalman gain
    Pbar = Sigma - K @ (C @ Sigma)
    Pbar = 0.5 * (Pbar + Pbar.T)
    return Pbar, Sigma


def cost_coeffs(T, sigma_w2, P11, P12, P22):
    """(F1): c(a) = c0 + c1 a + c2 a^2 + c3 a^3."""
    c0 = P11 + P22
    c1 = 2.0 * T * P12 + T * sigma_w2
    c2 = T**2 * P22
    c3 = (T**3 / 3.0) * sigma_w2
    return c0, c1, c2, c3


def cost_of_age(a, coeffs):
    c0, c1, c2, c3 = coeffs
    a = np.asarray(a, dtype=float)
    return c0 + c1 * a + c2 * a**2 + c3 * a**3


def cov_of_age(a, T, sigma_w2, Pbar):
    """Matrix form (eq:Pa): P(a) = A^a Pbar A^a' + sigma_w2 * sum_{i<a} A^i Qt A^i'."""
    A, Qt = cv_matrices(T)
    P = np.linalg.matrix_power(A, a) @ Pbar @ np.linalg.matrix_power(A, a).T
    for i in range(a):
        Ai = np.linalg.matrix_power(A, i)
        P = P + sigma_w2 * (Ai @ Qt @ Ai.T)
    return P


# ---------------------------------------------------------------- Whittle index
def whittle_generic(a_max, cost_seq):
    """(F2) generic: W(a) = a*c(a+1) - sum_{j<=a} c(j) for a=1..a_max.
    cost_seq must cover ages 1..a_max+1 (index 0 -> age 1)."""
    a = np.arange(1, a_max + 1, dtype=float)
    csum = np.cumsum(cost_seq[:a_max])
    return a * cost_seq[1:a_max + 1] - csum


def whittle_cv(a, c1, c2, c3):
    """(F2) CV closed form, affine in theta; c0 drops out.
    Vectorized over a (and over leading batch dims of c1,c2,c3 if broadcastable)."""
    a = np.asarray(a, dtype=float)
    return (c1 * a * (a + 1) / 2.0
            + c2 * a * (a + 1) * (4 * a + 5) / 6.0
            + c3 * a * (a + 1)**2 * (3 * a + 4) / 4.0)


def dW_dtheta(a, T):
    """(F3) closed-form sensitivities w.r.t. (sigma_w2, P12, P22); all > 0."""
    a = np.asarray(a, dtype=float)
    dsig = T * a * (a + 1) / 2.0 + (T**3 / 12.0) * a * (a + 1)**2 * (3 * a + 4)
    dP12 = T * a * (a + 1)
    dP22 = (T**2 / 6.0) * a * (a + 1) * (4 * a + 5)
    return dsig, dP12, dP22


def eps_bar(a, T, qhat_sig, qhat_12, qhat_22):
    """(F4) tight affine error bar eps(a) = <|grad W|, qhat>."""
    dsig, dP12, dP22 = dW_dtheta(a, T)
    return dsig * qhat_sig + dP12 * qhat_12 + dP22 * qhat_22


def indexability_margin(T, qhat_sig, qhat_12, qhat_22):
    """(F6) certified increment bar beta; Delta c(1) affine weights."""
    return (T + 7.0 * T**3 / 3.0) * qhat_sig + 2.0 * T * qhat_12 + 3.0 * T**2 * qhat_22


def delta_c1(T, sigma_w2, P12, P22):
    """Delta c(1) = c1 + 3 c2 + 7 c3 (minimal cost increment)."""
    _, c1, c2, c3 = cost_coeffs(T, sigma_w2, 0.0, P12, P22)
    return c1 + 3.0 * c2 + 7.0 * c3


# ---------------------------------------------------------------- single-arm averages
def g_threshold(h, lam, coeffs):
    """Average cost of threshold-h policy under subsidy lam:
    g(h;lam) = [sum_{j<=h} c(j) - lam*(h-1)] / h.  Vectorized over h."""
    h = np.asarray(h, dtype=float)
    c0, c1, c2, c3 = coeffs
    # sum_{j=1}^{h} c(j) closed form
    s1 = h * (h + 1) / 2.0
    s2 = h * (h + 1) * (2 * h + 1) / 6.0
    s3 = (h * (h + 1) / 2.0)**2
    C = c0 * h + c1 * s1 + c2 * s2 + c3 * s3
    return (C - lam * (h - 1)) / h


def g_star(lam, coeffs, h_max=100000):
    """min_h g(h;lam) and argmin, via the index rule h* = min{a: W(a) >= lam}
    cross-checked against a local grid (W strictly increasing => renewal g is
    unimodal in h; verified numerically in A0 check 5)."""
    c0, c1, c2, c3 = coeffs
    # find h* by bisection on W(a) >= lam
    lo, hi = 1, 2
    while whittle_cv(hi, c1, c2, c3) < lam and hi < h_max:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if whittle_cv(mid, c1, c2, c3) >= lam:
            hi = mid
        else:
            lo = mid + 1
    hstar = lo
    # guard: evaluate g on a window around hstar
    window = np.arange(max(1, hstar - 2), hstar + 3)
    gs = g_threshold(window, lam, coeffs)
    j = int(np.argmin(gs))
    return float(gs[j]), int(window[j])
