"""Flow-instance generation, error injection, and certificates.
All heavy paths vectorized (no scipy DARE inside calibration loops: the
quartic characterization from dare_symbolic.py is used batched instead)."""
import numpy as np
from .model import dare_filtered_cov


def pbar_batch(T, sw2, sv2):
    """Batched steady-state filtered covariance entries via the quartic
    y^4 - bT y^3 - (2r - b^2T^2/6) y^2 - bT r y + r^2 = 0,  b=sqrt(sw2*T), r=sv2.
    Verified against scipy DARE in dare_symbolic.py (rel err ~1e-12).
    Returns (P11, P12, P22) arrays."""
    sw2 = np.atleast_1d(np.asarray(sw2, dtype=float))
    sv2 = np.atleast_1d(np.asarray(sv2, dtype=float))
    n = sw2.size
    b = np.sqrt(sw2 * T)
    r = sv2
    # companion matrices of the monic quartic
    Cm = np.zeros((n, 4, 4))
    Cm[:, 1, 0] = Cm[:, 2, 1] = Cm[:, 3, 2] = 1.0
    Cm[:, 0, 3] = -(r**2)
    Cm[:, 1, 3] = b * T * r
    Cm[:, 2, 3] = (2 * r - b**2 * T**2 / 6.0)
    Cm[:, 3, 3] = b * T
    ev = np.linalg.eigvals(Cm)                       # (n,4) complex
    real = np.where(np.abs(ev.imag) < 1e-7, ev.real, -np.inf)
    real = np.where(real > np.sqrt(r)[:, None] * (1 + 1e-12), real, -np.inf)
    y = real.max(axis=1)
    x = y**2 - r
    m = b * y
    P12 = b * r / y
    P11 = x * r / (x + r)
    P22 = (m * x / (x + r) - sw2 * T**2 / 2.0) / T
    return P11, P12, P22


def make_flows(K, heterogeneous, rng, T=1.0):
    """True per-flow parameters. sigma ranges per manual B1."""
    if heterogeneous:
        sw2 = np.exp(rng.uniform(np.log(0.01), np.log(1.0), K))
        sv2 = np.exp(rng.uniform(np.log(0.01), np.log(1.0), K))
    else:
        sw2 = np.full(K, 0.1)
        sv2 = np.full(K, 0.1)
    P11, P12, P22 = pbar_batch(T, sw2, sv2)
    return dict(T=T, sw2=sw2, sv2=sv2, P11=P11, P12=P12, P22=P22, K=K)


def inject_error_paired(flows, q_rel, u):
    """Paired design for g(q) sweeps: eta = q_rel * u with a FIXED direction
    u in [-1,1]^{K x 3} shared across q levels (per seed), so per-seed excess
    curves are smooth in q and slopes are estimated cleanly."""
    th = np.stack([flows["sw2"], flows["P12"], flows["P22"]], axis=1)
    th_hat = th * (1.0 + q_rel * u)
    return th_hat, np.abs(th_hat - th)


def inject_error(flows, q_rel, rng):
    """theta_hat = theta * (1 + eta), eta ~ U(-q_rel, q_rel) i.i.d. per component.
    Effective components: (sw2, P12, P22). Returns (theta_hat (K,3), qhat_oracle (K,3))."""
    K = flows["K"]
    th = np.stack([flows["sw2"], flows["P12"], flows["P22"]], axis=1)
    eta = rng.uniform(-q_rel, q_rel, (K, 3))
    th_hat = th * (1.0 + eta)
    qhat = np.abs(th_hat - th)
    return th_hat, qhat


# ------------------------------------------------------------------ realistic
def simulate_position_windows(T, sw2, sv2, n_win, n_slots, rng):
    """(n_win, n_slots) position measurements of i.i.d. CV ground-truth windows.
    pos_{t+1} = pos_t + T vel_t + w1_t ; vel_{t+1} = vel_t + w2_t ;
    (w1,w2) ~ N(0, sw2*Qt), z_t = pos_t + N(0, sv2)."""
    Qt = sw2 * np.array([[T**3 / 3, T**2 / 2], [T**2 / 2, T]])
    L = np.linalg.cholesky(Qt + 1e-18 * np.eye(2))
    Z = rng.standard_normal((n_win, n_slots, 2))
    w = Z @ L.T                                       # (n_win, n_slots, 2)
    vel0 = rng.normal(0, 1, (n_win, 1))
    vel = vel0 + np.concatenate([np.zeros((n_win, 1)), np.cumsum(w[:, :-1, 1], axis=1)], axis=1)
    incr = T * vel + w[:, :, 0]                       # pos_{t+1}-pos_t at index t
    pos = np.concatenate([np.zeros((n_win, 1)), np.cumsum(incr[:, :-1], axis=1)], axis=1)
    return pos + rng.normal(0, np.sqrt(sv2), (n_win, n_slots))


def mom_estimate_batch(Z, T):
    """Vectorized method-of-moments (sw2, sv2) from second differences:
      M0 = E d^2 = (2/3) T^3 sw2 + 6 sv2 ;  M1 = E d_t d_{t+1} = (1/6) T^3 sw2 - 4 sv2
    => sw2 = (12/11)(M0 + 1.5 M1)/T^3 ; sv2 = (M0 - (2/3) T^3 sw2)/6."""
    d = np.diff(Z, n=2, axis=1)
    M0 = np.mean(d * d, axis=1)
    M1 = np.mean(d[:, :-1] * d[:, 1:], axis=1)
    sw2 = (12.0 / 11.0) * (M0 + 1.5 * M1) / T**3
    sv2 = (M0 - (2.0 / 3.0) * T**3 * sw2) / 6.0
    return sw2, sv2


def realistic_estimates(flows, W_cal, rng, n_cal=2000, delta=0.1, fwer=True,
                        levels=None):
    """E3 'realistic q' tier. Per flow: one deployment window -> theta_hat via
    MoM + DARE(quartic); n_cal calibration windows with known ground truth ->
    split-conformal certificates on |theta_hat_m - theta_m| per component.
    Certificate level: 1 - delta/(3K) per component (union bound over 3K tests,
    FWER-controlled calibration event) if fwer, else 1 - delta marginal.
    If levels is given (list), returns dict level->qhat instead of single qhat.
    Returns (th_hat (K,3), qhat, clip_frac, residuals (K, n_cal, 3));
    residuals are SIGNED (est - true): quantiles use their absolute values,
    and the signed bank doubles as a bootstrap posterior surrogate for the
    Thompson-sampling baseline (theta_true ~ th_hat - residual_draw)."""
    K, T = flows["K"], flows["T"]
    th_hat = np.empty((K, 3))
    res = np.empty((K, n_cal, 3))
    clip = 0
    for k in range(K):
        sw2, sv2 = flows["sw2"][k], flows["sv2"][k]
        th_true = np.array([sw2, flows["P12"][k], flows["P22"][k]])
        Z = simulate_position_windows(T, sw2, sv2, n_cal + 1, W_cal, rng)
        e_w, e_v = mom_estimate_batch(Z, T)
        clip += int(np.sum((e_w <= 1e-6) | (e_v <= 1e-6)))
        e_w, e_v = np.maximum(e_w, 1e-6), np.maximum(e_v, 1e-6)
        _, p12, p22 = pbar_batch(T, e_w, e_v)
        est = np.stack([e_w, p12, p22], axis=1)       # (n_cal+1, 3)
        th_hat[k] = est[0]
        res[k] = est[1:] - th_true
    clip_frac = clip / (K * (n_cal + 1))
    R_sorted = np.sort(np.abs(res), axis=1)           # (K, n_cal, 3)

    def q_at(level):
        idx = min(int(np.ceil((n_cal + 1) * level)), n_cal) - 1
        return R_sorted[:, idx, :]

    if levels is not None:
        return th_hat, {lv: q_at(lv) for lv in levels}, clip_frac, res
    lvl = 1.0 - (delta / (3 * K) if fwer else delta)
    return th_hat, q_at(lvl), clip_frac, res
