"""Deterministic Top-N exchange decomposition used in the TMC proof."""
from __future__ import annotations

import numpy as np


def stable_topn(values, n):
    """Indices of the n largest entries; lower index wins exact ties."""
    values = np.asarray(values, dtype=float)
    return np.argsort(-values, kind="stable")[:n]


def topn_exchange_regret(true_index, score, radius, n):
    """Return exchange data and verify the score-error premise.

    Parameters
    ----------
    true_index : (K,) array
        True Whittle indices.
    score : (K,) array
        Scores used by the scheduler.
    radius : (K,) array
        Deterministic coordinatewise error bounds.
    n : int
        Per-slot service budget.

    Returns a dict containing the true and selected sets, paired missed/extra
    arms in descending true-index order, ranking regret, boundary margin, and
    the heterogeneous and max-radius envelopes from the theorem.
    """
    w = np.asarray(true_index, dtype=float)
    s = np.asarray(score, dtype=float)
    e = np.asarray(radius, dtype=float)
    if w.ndim != 1 or s.shape != w.shape or e.shape != w.shape:
        raise ValueError("true_index, score, and radius must be same-length vectors")
    if not 0 < n < w.size:
        raise ValueError("n must lie in {1,...,K-1}")
    if np.any(e < 0) or np.any(np.abs(s - w) > e + 1e-12):
        raise ValueError("score-error premise |score-W| <= radius is violated")

    true_order = stable_topn(w, n)
    score_order = stable_topn(s, n)
    true_set = set(map(int, true_order))
    score_set = set(map(int, score_order))
    missed = sorted(true_set - score_set, key=lambda k: (-w[k], k))
    extra = sorted(score_set - true_set, key=lambda k: (-w[k], k))
    pairs = list(zip(missed, extra))

    regret = float(sum(w[i] - w[j] for i, j in pairs))
    heterogeneous_envelope = float(sum(e[i] + e[j] for i, j in pairs))
    max_radius_envelope = float(2.0 * len(pairs) * np.max(e))
    sorted_w = np.sort(w)[::-1]
    boundary_margin = float(sorted_w[n - 1] - sorted_w[n])
    return {
        "true_set": tuple(sorted(true_set)),
        "selected_set": tuple(sorted(score_set)),
        "pairs": tuple(pairs),
        "exchange_count": len(pairs),
        "regret": regret,
        "boundary_margin": boundary_margin,
        "heterogeneous_envelope": heterogeneous_envelope,
        "max_radius_envelope": max_radius_envelope,
    }

