import numpy as np

from core.sim import W_from_pack, coeff_pack
from run_tmc_multiaxis_pilot import (
    aggregate_theta,
    make_multiaxis_problem,
    run_multiaxis_policy,
)


def test_aggregate_index_equals_sum_of_axis_indices():
    rng = np.random.default_rng(20260727)
    B, K, dimension = 2, 5, 3
    flat = rng.uniform(0.01, 1.0, size=(B, K * dimension, 3))
    ages = rng.integers(1, 20, size=(B, K)).astype(float)
    aggregate = aggregate_theta(flat, B, K, dimension)
    left = W_from_pack(ages, coeff_pack(1.0, aggregate))
    axis = flat.reshape(B, K, dimension, 3)
    right = np.zeros((B, K))
    for d in range(dimension):
        right += W_from_pack(ages, coeff_pack(1.0, axis[:, :, d]))
    np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)


def test_selected_packet_updates_every_axis_once():
    B, K, dimension = 1, 6, 3
    N, n0, H, change_t, block = 2, 4, 40, 20, 16
    problem = make_multiaxis_problem(
        20260728, B, K, dimension, change_t, H, n0, block
    )
    result = run_multiaxis_policy(
        "cumulative_ce", *problem, N, n0, change_t, H
    )
    assert result["axis_count_spread"] == 0.0
    assert np.all(np.isfinite(result["avg_cost"]))
    assert np.all(np.isfinite(result["post_cost"]))


def test_three_axis_sco_smoke_is_finite():
    B, K, dimension = 1, 6, 3
    N, n0, H, change_t, block = 2, 4, 48, 24, 16
    problem = make_multiaxis_problem(
        20260729, B, K, dimension, change_t, H, n0, block
    )
    result = run_multiaxis_policy(
        "sco_reset_ucb", *problem, N, n0, change_t, H
    )
    assert result["axis_count_spread"] == 0.0
    assert np.all(np.isfinite(result["avg_cost"]))
    assert np.all(np.isfinite(result["rank_loss"]))
