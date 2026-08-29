import numpy as np

from run_tmc_ts_baseline_expansion import (
    B,
    BLOCK_LENGTH,
    CHANGE_T,
    H,
    K,
    N0,
    run_ts,
)
from run_cv_piecewise_pilot import make_problem


def test_ts_baseline_is_deterministic_and_finite():
    problem = make_problem(309901, B, K, CHANGE_T, H, N0, BLOCK_LENGTH)
    first = run_ts(problem, episode_length=4, draw_seed=12345)
    second = run_ts(problem, episode_length=4, draw_seed=12345)
    for key in ("total_ex", "post_ex", "rank_loss"):
        assert first[key].shape == (B,)
        assert np.all(np.isfinite(first[key]))
        np.testing.assert_allclose(first[key], second[key], rtol=0, atol=0)


def test_no_negative_ranking_loss_beyond_roundoff():
    problem = make_problem(309902, B, K, CHANGE_T, H, N0, BLOCK_LENGTH)
    row = run_ts(problem, episode_length=1, draw_seed=9876)
    assert np.min(row["rank_loss"]) >= -1e-9
