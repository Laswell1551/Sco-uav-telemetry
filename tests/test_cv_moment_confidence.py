import unittest

import numpy as np

from core.cv_moment_confidence import (
    finite_horizon_physical_radius,
    second_difference_covariance_components,
    uniform_quadratic_norm_bounds,
    verify_moment_identities,
)


class CVMomentConfidenceTests(unittest.TestCase):
    def test_second_difference_covariances_are_psd(self):
        for W in (8, 32, 64):
            gamma_sw, gamma_sv = second_difference_covariance_components(W)
            self.assertGreaterEqual(np.linalg.eigvalsh(gamma_sw).min(), -1e-10)
            self.assertGreaterEqual(np.linalg.eigvalsh(gamma_sv).min(), -1e-10)

    def test_raw_moment_estimators_are_unbiased(self):
        for W in (8, 16, 64):
            np.testing.assert_allclose(
                verify_moment_identities(W), np.eye(2), atol=1e-10
            )

    def test_radius_is_positive_and_decreases_with_count(self):
        count = np.array([1.0, 10.0, 100.0, 1000.0])
        radius = finite_horizon_physical_radius(
            count,
            n_slots=64,
            K=20,
            n_max=2000,
            delta=0.05,
            sw2_upper=1.0,
            sv2_upper=1.0,
        )
        self.assertEqual(radius.shape, (4, 2))
        self.assertTrue(np.all(radius > 0))
        self.assertTrue(np.all(np.diff(radius, axis=0) < 0))

    def test_uniform_norm_bounds_are_finite(self):
        bounds = uniform_quadratic_norm_bounds(64, 1.0, 1.0)
        self.assertEqual(bounds.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(bounds)))
        self.assertTrue(np.all(bounds > 0))


if __name__ == "__main__":
    unittest.main()
