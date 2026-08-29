"""Vectorized analytic-age-mode simulator (manual B1 mode (a)).

State: ages (B, K). Per slot: pay sum_k c_k(a_k) under TRUE coefficients,
select N flows per policy, ages: selected -> 1, else +1 (delivery abstraction;
Bernoulli variant in kalman_sim/b6). All selections use STABLE descending sort
so tie-breaking is the fixed deterministic rule required by Thm 2.

Coefficient packs per batch element:
  CT (B,K,4): true (c0,c1,c2,c3);  CH (B,K,4): plug-in;  CE (B,K,3): the
  eps-bar polynomial coefficients (c1e,c2e,c3e) built from qhat -- by affine
  exactness eps(a) is itself a Whittle-form polynomial with
  c1e = T*qhat_sig + 2T*qhat_12, c2e = T^2*qhat_22, c3e = T^3/3*qhat_sig.
"""
import numpy as np


# ------------------------------------------------------------- coefficient packs
def coeff_pack(T, th, ages_dtype=float):
    """th (...,3) = (sw2, P12, P22) -> (...,4) cost coeffs with c0 dropped
    (c0 is added separately from true P11+P22 only for absolute cost)."""
    c1 = 2 * T * th[..., 1] + T * th[..., 0]
    c2 = T**2 * th[..., 2]
    c3 = (T**3 / 3.0) * th[..., 0]
    c0 = np.zeros_like(c1)
    return np.stack([c0, c1, c2, c3], axis=-1)


def eps_pack(T, qhat):
    """qhat (...,3) -> eps-polynomial coeffs (c1e,c2e,c3e)."""
    c1e = T * qhat[..., 0] + 2 * T * qhat[..., 1]
    c2e = T**2 * qhat[..., 2]
    c3e = (T**3 / 3.0) * qhat[..., 0]
    return np.stack([c1e, c2e, c3e], axis=-1)


def poly_cost(a, C):
    """C (...,4); a (...) ages."""
    return C[..., 0] + C[..., 1] * a + C[..., 2] * a * a + C[..., 3] * a * a * a


def whittle_of(a, c1, c2, c3):
    u = a * (a + 1.0)
    return c1 * u / 2.0 + c2 * u * (4 * a + 5.0) / 6.0 + c3 * u * (a + 1.0) * (3 * a + 4.0) / 4.0


def W_from_pack(a, C):
    return whittle_of(a, C[..., 1], C[..., 2], C[..., 3])


def eps_from_pack(a, E):
    return whittle_of(a, E[..., 0], E[..., 1], E[..., 2])


def topn_mask(score, N):
    """(B,K) scores -> boolean mask of N largest per row; stable order
    (ties -> lower flow index), matching the paper's deterministic rule."""
    order = np.argsort(-score, axis=1, kind="stable")
    B, K = score.shape
    mask = np.zeros((B, K), dtype=bool)
    np.put_along_axis(mask, order[:, :N], True, axis=1)
    return mask


def trust_test(W_hat, eps, mask):
    """(F5): min_S (W-eps) > max_notS (W+eps), rowwise."""
    lo = np.where(mask, W_hat - eps, np.inf).min(axis=1)
    hi = np.where(mask, -np.inf, W_hat + eps).max(axis=1)
    return lo > hi


# ------------------------------------------------------------- policy library
class Policy:
    """Stateless vectorized policies. kind in:
      true, ce, trust_r1, trust_r2, trust_r3, maxmin, optimistic,
      myopic, aoi, rr"""

    def __init__(self, kind, N, ctx):
        self.kind, self.N, self.ctx = kind, N, ctx
        K = ctx["K"]
        if kind == "rr":
            period = int(np.ceil(K / N))
            sched = np.zeros((period, K), dtype=bool)
            for g in range(period):
                sched[g, g * N:(g + 1) * N] = True
            extra = period * N - K
            if extra > 0:                     # wrap the last group
                sched[period - 1, :extra] = True
            self.sched = sched

    def select(self, ages, t):
        c = self.ctx
        N = self.N
        if self.kind == "rr":
            m = self.sched[t % self.sched.shape[0]]
            return np.broadcast_to(m, ages.shape).copy(), None
        if self.kind == "aoi":
            return topn_mask(ages.astype(float), N), None
        if self.kind == "true":
            return topn_mask(W_from_pack(ages, c["CT"]), N), None
        if self.kind == "ce":
            return topn_mask(W_from_pack(ages, c["CH"]), N), None
        if self.kind == "myopic":
            s = poly_cost(ages + 1.0, c["CH"]) - poly_cost(np.ones_like(ages), c["CH"])
            return topn_mask(s, N), None
        W_hat = W_from_pack(ages, c["CH"])
        eps = eps_from_pack(ages, c["CE"])
        if self.kind == "maxmin":
            return topn_mask(W_from_pack(ages, c["CLO"]), N), None
        if self.kind == "optimistic":
            return topn_mask(W_from_pack(ages, c["CHI"]), N), None
        # trust_* : gated
        mask = topn_mask(W_hat, N)
        cert = trust_test(W_hat, eps, mask)
        if self.kind == "trust_r1":
            fb = topn_mask(W_from_pack(ages, c["CLO"]), N)
        elif self.kind == "trust_r2":
            fb = topn_mask(W_from_pack(ages, c["CHI"]), N)
        elif self.kind == "trust_r3":         # 1-step lookahead == myopic rank
            s = poly_cost(ages + 1.0, c["CH"]) - poly_cost(np.ones_like(ages), c["CH"])
            fb = topn_mask(s, N)
        else:
            raise ValueError(self.kind)
        out = np.where(cert[:, None], mask, fb)
        return out, cert


def make_ctx(T, th_true, th_hat, qhat, c0_true=None, corner_floor=0.05):
    """th_* (B,K,3); qhat (B,K,3). c0_true (B,K) optional absolute offset.

    CLO/CHI: coefficient packs of the max--min / optimistic corners. The lower
    corner is PROJECTED onto the certified-indexable set (Prop. margin /
    no-starvation lemma): componentwise max(th_hat - qhat, corner_floor *
    th_hat). The projected point stays inside the certificate box, so the
    score error keeps the 2*eps bound, and its coefficients stay positive, so
    corner scores remain strictly increasing in age (bounded ages)."""
    lo = np.maximum(th_hat - qhat, corner_floor * np.abs(th_hat))
    hi = th_hat + qhat
    ctx = dict(K=th_true.shape[1], T=T,
               CT=coeff_pack(T, th_true), CH=coeff_pack(T, th_hat),
               CE=eps_pack(T, qhat),
               CLO=coeff_pack(T, lo), CHI=coeff_pack(T, hi))
    if c0_true is not None:
        ctx["CT"][..., 0] = c0_true
    return ctx


# ------------------------------------------------------------- runners
def run_policy(policy, ages0, horizon, collect_hist=False, age_cap=4096,
               hist_stride=4):
    """Single-policy batched run. Returns metrics dict."""
    ctx = policy.ctx
    ages = ages0.astype(float).copy()
    B, K = ages.shape
    tot = np.zeros(B)
    perflow = np.zeros((B, K))
    cert_cnt = np.zeros(B)
    hist = np.zeros((B, age_cap), dtype=np.int64) if collect_hist else None
    offs = (np.arange(B) * age_cap)[:, None] if collect_hist else None
    for t in range(horizon):
        cst = poly_cost(ages, ctx["CT"])
        tot += cst.sum(axis=1)
        perflow += cst
        if collect_hist and t % hist_stride == 0:
            idx = (offs + np.minimum(ages, age_cap - 1).astype(np.int64)).ravel()
            hist.ravel()[:] += np.bincount(idx, minlength=B * age_cap)
        mask, cert = policy.select(ages, t)
        if cert is not None:
            cert_cnt += cert
        ages = np.where(mask, 1.0, ages + 1.0)
    out = dict(avg_cost=tot / horizon, perflow_avg=perflow / horizon,
               cert_frac=cert_cnt / horizon)
    if collect_hist:
        cdf = np.cumsum(hist, axis=1)
        p99 = np.argmax(cdf >= 0.99 * cdf[:, [-1]], axis=1)
        out["p99_age"] = p99.astype(float)
        pf = out["perflow_avg"]
        out["jain"] = pf.sum(1)**2 / (K * (pf**2).sum(1))
    return out


def run_trust_vs_true(ctx, N, ages0, horizon, rule="trust_r1",
                      margin_bins=None):
    """Coupled run of TrustWhittle vs True-Whittle from identical ages.
    Tracks: excess cost, uncertified slots, Thm-3 machine assertions,
    and (optionally) the true-index N-vs-(N+1) margin histogram."""
    trust = Policy(rule, N, ctx)
    true_p = Policy("true", N, ctx)
    a_tr = ages0.astype(float).copy()
    a_te = ages0.astype(float).copy()
    B, K = a_tr.shape
    exc = np.zeros(B)
    cost_true_tot = np.zeros(B)
    n_uncert = np.zeros(B, dtype=int)
    all_cert = np.ones(B, dtype=bool)
    mismatch_cert = np.zeros(B, dtype=int)          # must stay 0 (Thm 3)
    exc_before_uncert = np.zeros(B)                 # must stay 0
    first_unc = np.full(B, -1)
    mhist = np.zeros(len(margin_bins) - 1, dtype=np.int64) if margin_bins is not None else None
    for t in range(horizon):
        c_tr = poly_cost(a_tr, ctx["CT"]).sum(1)
        c_te = poly_cost(a_te, ctx["CT"]).sum(1)
        exc += c_tr - c_te
        cost_true_tot += c_te
        exc_before_uncert += np.where(all_cert, c_tr - c_te, 0.0)
        m_tr, cert = trust.select(a_tr, t)
        m_te, _ = true_p.select(a_te, t)
        if margin_bins is not None:
            Wt = W_from_pack(a_te, ctx["CT"])
            part = -np.partition(-Wt, [N - 1, N], axis=1)   # both kth guaranteed
            gap = part[:, N - 1] - part[:, N]
            mhist += np.histogram(gap, bins=margin_bins)[0]
        same_traj = np.all(a_tr == a_te, axis=1)
        mism = np.any(m_tr != m_te, axis=1)
        mismatch_cert += (all_cert & cert & same_traj & mism)
        first_unc = np.where((first_unc < 0) & ~cert, t, first_unc)
        n_uncert += ~cert
        all_cert &= cert
        a_tr = np.where(m_tr, 1.0, a_tr + 1.0)
        a_te = np.where(m_te, 1.0, a_te + 1.0)
    return dict(excess=exc / horizon, cost_true=cost_true_tot / horizon,
                n_uncert=n_uncert, all_cert=all_cert,
                mismatch_while_cert=mismatch_cert,
                exc_before_uncert=exc_before_uncert, first_uncert=first_unc,
                margin_hist=mhist)
