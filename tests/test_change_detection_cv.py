import unittest

import numpy as np

from core.change_detection_cv import (
    ResettableOnlineCVMomentEstimator,
    TwoWindowCVMomentDetector,
)


class CVChangeDetectionTests(unittest.TestCase):
    def test_constant_stream_does_not_alarm(self):
        detector = TwoWindowCVMomentDetector(1, 1, window=4, threshold=4.0)
        selected = np.ones((1, 1), dtype=bool)
        alarms = []
        for _ in range(20):
            result = detector.update(
                selected, np.array([[[0.2, 0.3]]])
            )
            alarms.append(result["alarms"][0, 0])
        self.assertFalse(any(alarms))

    def test_large_step_alarms_and_supplies_recent_state(self):
        detector = TwoWindowCVMomentDetector(1, 1, window=4, threshold=3.0)
        selected = np.ones((1, 1), dtype=bool)
        rng = np.random.default_rng(2)
        alarm_result = None
        for value in np.r_[
            rng.normal(0.1, 0.005, 8),
            rng.normal(1.0, 0.005, 12),
        ]:
            result = detector.update(
                selected, np.array([[[value, 2.0 * value]]])
            )
            if result["alarms"][0, 0]:
                alarm_result = result
                break
        self.assertIsNotNone(alarm_result)
        self.assertEqual(alarm_result["reset_count"][0, 0], 4)
        self.assertGreater(alarm_result["reset_mean"][0, 0, 0], 0.5)

    def test_estimator_uses_detector_reset_window(self):
        initial = np.array(
            [
                [[[0.1, 0.2]]],
                [[[0.2, 0.4]]],
                [[[0.3, 0.6]]],
                [[[0.4, 0.8]]],
            ]
        )
        estimator = ResettableOnlineCVMomentEstimator(
            initial, variance_floor=1e-4, variance_ceiling=10.0
        )
        result = {
            "alarms": np.array([[True]]),
            "reset_mean": np.array([[[2.0, 3.0]]]),
            "reset_m2": np.array([[[0.5, 0.7]]]),
            "reset_count": np.array([[4]]),
        }
        estimator.update_and_reset(
            np.array([[True]]), np.array([[[9.0, 9.0]]]), result
        )
        np.testing.assert_array_equal(estimator.running_mean, [[[2.0, 3.0]]])
        np.testing.assert_array_equal(estimator.running_m2, [[[0.5, 0.7]]])
        np.testing.assert_array_equal(estimator.count, [[4.0]])


if __name__ == "__main__":
    unittest.main()
