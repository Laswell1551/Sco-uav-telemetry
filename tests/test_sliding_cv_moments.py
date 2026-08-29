import unittest

import numpy as np

from core.sliding_cv_moments import SlidingWindowCVMomentEstimator


class SlidingCVMomentTests(unittest.TestCase):
    def test_window_drops_oldest_selected_block(self):
        initial = np.array(
            [
                [[[1.0, 10.0], [2.0, 20.0]]],
                [[[3.0, 30.0], [4.0, 40.0]]],
            ]
        )
        estimator = SlidingWindowCVMomentEstimator(
            initial, window=2, variance_floor=1e-4, variance_ceiling=100.0
        )
        estimator.update(
            np.array([[True, False]]),
            np.array([[[5.0, 50.0], [999.0, 999.0]]]),
        )
        np.testing.assert_allclose(estimator.physical_mean_raw[0, 0], [4.0, 40.0])
        np.testing.assert_allclose(estimator.physical_mean_raw[0, 1], [3.0, 30.0])
        np.testing.assert_array_equal(estimator.total_count, [[3, 2]])

    def test_partial_window_statistics(self):
        initial = np.array(
            [
                [[[1.0, 2.0]]],
                [[[3.0, 6.0]]],
            ]
        )
        estimator = SlidingWindowCVMomentEstimator(
            initial, window=4, variance_floor=1e-4, variance_ceiling=100.0
        )
        np.testing.assert_allclose(estimator.physical_mean_raw[0, 0], [2.0, 4.0])
        expected_var = np.array([2.0, 8.0])
        actual_var = (
            estimator.physical_radius_proxy[0, 0]
            / estimator.confidence_scale
        ) ** 2 * estimator.count[0, 0]
        np.testing.assert_allclose(actual_var, expected_var)

    def test_effective_outputs_are_ordered_and_positive(self):
        rng = np.random.default_rng(3)
        initial = np.abs(rng.normal(size=(5, 2, 3, 2))) + 0.01
        estimator = SlidingWindowCVMomentEstimator(initial, window=4)
        lo, hi = estimator.effective_box
        self.assertTrue(np.all(lo > 0))
        self.assertTrue(np.all(hi >= lo))
        self.assertTrue(np.all(estimator.radius >= 0))


if __name__ == "__main__":
    unittest.main()
