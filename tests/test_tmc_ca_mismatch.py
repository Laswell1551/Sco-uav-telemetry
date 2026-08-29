import numpy as np

from core.sim import W_from_pack
from run_tmc_ca_mismatch_pilot import (
    ca_position_costs,
    general_whittle_from_cost,
)


def test_general_index_matches_cubic_closed_form():
    ages = np.arange(0, 66, dtype=float)
    coeffs = np.array([2.0, 0.3, 0.07, 0.004])
    cost = (
        coeffs[0]
        + coeffs[1] * ages
        + coeffs[2] * ages**2
        + coeffs[3] * ages**3
    )
    index = general_whittle_from_cost(cost)
    closed = W_from_pack(ages[None], coeffs[None, None, :])[0]
    np.testing.assert_allclose(index[1:-1], closed[1:-1], rtol=1e-11)


def test_ca_position_cost_is_finite_and_monotone():
    cost = ca_position_costs(1e-2, 0.1, 128)
    assert np.all(np.isfinite(cost))
    assert np.all(np.diff(cost[1:]) > 0)
    index = general_whittle_from_cost(cost)
    assert np.all(np.diff(index[1:-1]) > 0)
