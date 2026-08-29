import numpy as np

from run_cv_piecewise_pilot import make_problem, run_policy


def _small_problem():
    seed = 20260727
    B, K, N, H, n0, change_t, block_length = 2, 8, 2, 80, 4, 40, 16
    problem = make_problem(seed, B, K, change_t, H, n0, block_length)
    (
        _,
        _,
        theta0,
        theta1,
        c00,
        c01,
        changed,
        pre_bank,
        post_bank,
        ages0,
    ) = problem
    args = (
        theta0,
        theta1,
        c00,
        c01,
        changed,
        pre_bank,
        post_bank,
        ages0,
        N,
        n0,
        change_t,
        H,
    )
    return args


def test_sw_whittle_name_is_numerically_identical_to_legacy_alias():
    args = _small_problem()
    legacy = run_policy("sw_ucb_cv_32", *args)
    matched = run_policy("sw_whittle_cv_32", *args)

    for key in ("avg_cost", "pre_cost", "post_cost", "rank_loss"):
        np.testing.assert_array_equal(legacy[key], matched[key])


def test_max_age_is_a_finite_estimator_free_anchor():
    args = _small_problem()
    result = run_policy("max_age", *args)

    assert np.all(np.isfinite(result["avg_cost"]))
    assert np.all(np.isfinite(result["rank_loss"]))
    assert result["pre_alarms"] == 0
    assert result["post_changed_alarms"] == 0
    assert result["post_unchanged_alarms"] == 0
    assert np.isnan(result["detected_fraction"])
