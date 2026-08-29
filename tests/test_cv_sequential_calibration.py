import math
import unittest

import numpy as np

from core.cv_sequential_calibration import (
    calibrate_studentized_multiplier,
    conformal_upper_quantile,
    quadratic_block_sequences,
    studentized_max_scores,
    studentized_radius,
)


class CVSequentialCalibrationTests(unittest.TestCase):
    def test_conformal_quantile_and_resolution_limit(self):
        scores = np.arange(1.0, 101.0)
        self.assertEqual(conformal_upper_quantile(scores, 0.1), 91.0)
        self.assertTrue(math.isinf(conformal_upper_quantile(scores, 0.001)))

    def test_quadratic_sequences_have_correct_shape_and_approximate_mean(self):
        obs = quadratic_block_sequences(
            0.2,
            0.3,
            n_slots=16,
            n_sequences=1000,
            n_blocks=8,
            rng=np.random.default_rng(2),
        )
        self.assertEqual(obs.shape, (1000, 8, 2))
        np.testing.assert_allclose(obs.mean(axis=(0, 1)), [0.2, 0.3], rtol=0.12)

    def test_studentized_scores_are_finite(self):
        obs = quadratic_block_sequences(
            0.1,
            0.2,
            n_slots=12,
            n_sequences=20,
            n_blocks=10,
            rng=np.random.default_rng(3),
        )
        score = studentized_max_scores(
            obs, np.array([0.1, 0.2])[None, None, :], n_start=4
        )
        self.assertEqual(score.shape, (20,))
        self.assertTrue(np.all(np.isfinite(score)))
        self.assertTrue(np.all(score >= 0))

    def test_small_grid_calibration_and_radius(self):
        result = calibrate_studentized_multiplier(
            [[0.1, 0.1], [0.5, 0.5]],
            n_slots=8,
            n_sequences=50,
            n_blocks=10,
            alpha=0.1,
            rng=np.random.default_rng(4),
            n_start=4,
            batch_size=10,
        )
        self.assertTrue(np.isfinite(result["multiplier"]))
        self.assertGreater(result["multiplier"], 0)
        radius = studentized_radius(
            np.ones((1, 2, 2)),
            np.array([[5.0, 10.0]]),
            result["multiplier"],
        )
        self.assertEqual(radius.shape, (1, 2, 2))
        self.assertTrue(np.all(radius > 0))


if __name__ == "__main__":
    unittest.main()
