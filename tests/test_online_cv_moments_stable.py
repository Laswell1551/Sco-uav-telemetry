import unittest

import numpy as np

from core.online_cv_moments_stable import OnlineCVMomentEstimator


class StableOnlineCVMomentTests(unittest.TestCase):
    def test_identical_blocks_have_exact_zero_radius(self):
        initial = np.broadcast_to(
            np.array([0.2, 0.3]), (5, 2, 3, 2)
        ).copy()
        estimator = OnlineCVMomentEstimator(initial)
        np.testing.assert_array_equal(estimator.physical_radius_proxy, 0.0)
        np.testing.assert_array_equal(estimator.radius, 0.0)

    def test_masked_welford_matches_numpy(self):
        initial = np.array(
            [
                [[[1.0, 2.0], [3.0, 4.0]]],
                [[[2.0, 4.0], [5.0, 8.0]]],
            ]
        )
        estimator = OnlineCVMomentEstimator(
            initial, variance_floor=1e-4, variance_ceiling=20.0
        )
        selected = np.array([[True, False]])
        new = np.array([[[6.0, 9.0], [100.0, 100.0]]])
        estimator.update(selected, new)

        direct_first = np.array([[1.0, 2.0], [2.0, 4.0], [6.0, 9.0]])
        direct_second = np.array([[3.0, 4.0], [5.0, 8.0]])
        np.testing.assert_allclose(
            estimator.physical_mean_raw[0, 0], direct_first.mean(axis=0)
        )
        np.testing.assert_allclose(
            estimator.physical_mean_raw[0, 1], direct_second.mean(axis=0)
        )
        expected_var = np.stack(
            [
                direct_first.var(axis=0, ddof=1),
                direct_second.var(axis=0, ddof=1),
            ]
        )
        actual_var = estimator.running_m2[0] / (estimator.count[0, :, None] - 1)
        np.testing.assert_allclose(actual_var, expected_var)

    def test_unselected_state_is_bitwise_unchanged(self):
        rng = np.random.default_rng(5)
        initial = rng.normal(size=(4, 1, 3, 2))
        estimator = OnlineCVMomentEstimator(
            initial, variance_floor=1e-4, variance_ceiling=20.0
        )
        mean_before = estimator.running_mean.copy()
        m2_before = estimator.running_m2.copy()
        selected = np.array([[False, True, False]])
        estimator.update(selected, rng.normal(size=(1, 3, 2)))
        np.testing.assert_array_equal(
            estimator.running_mean[0, [0, 2]], mean_before[0, [0, 2]]
        )
        np.testing.assert_array_equal(
            estimator.running_m2[0, [0, 2]], m2_before[0, [0, 2]]
        )


if __name__ == "__main__":
    unittest.main()
