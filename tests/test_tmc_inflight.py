"""Regression and mechanism checks for in-flight-aware SCO."""
from __future__ import annotations

import unittest

import numpy as np

from run_cv_piecewise_pilot import make_problem
from run_tmc_channel_stress import run_policy_channel


class InflightAwareSCOTests(unittest.TestCase):
    @staticmethod
    def fixture(seed=20262101):
        B, K, N, H, n0, change_t, block_length = 2, 8, 2, 120, 8, 60, 64
        problem = make_problem(
            seed, B, K, change_t, H, n0, block_length
        )
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
        channel_uniform = np.random.default_rng(seed + 50000).random(
            (B, K, H + n0)
        )
        common = (
            theta0,
            theta1,
            c00,
            c01,
            changed,
            pre_bank,
            post_bank,
            channel_uniform,
            ages0,
            N,
            n0,
            change_t,
            H,
        )
        return common

    def assert_policy_outputs_equal(self, left, right):
        for key in (
            "avg_cost",
            "pre_cost",
            "post_cost",
            "delivery_rate",
            "redundant_attempt_rate",
            "pre_alarms",
            "post_unchanged_alarms",
            "detected_fraction",
            "calendar_delay",
            "observation_delay",
        ):
            self.assertTrue(
                np.allclose(
                    left[key],
                    right[key],
                    equal_nan=True,
                ),
                msg=key,
            )

    def test_zero_delay_matches_sco_for_any_penalty(self):
        common = self.fixture()
        sco = run_policy_channel(
            "sco_reset_ucb",
            *common,
            success_probability=0.9,
            delay=0,
        )
        pipeline = run_policy_channel(
            "inflight_sco_ucb",
            *common,
            success_probability=0.9,
            delay=0,
            inflight_beta=64.0,
        )
        self.assert_policy_outputs_equal(sco, pipeline)
        self.assertEqual(pipeline["max_inflight_count"], 1)

    def test_zero_penalty_matches_sco_with_delay(self):
        common = self.fixture()
        sco = run_policy_channel(
            "sco_reset_ucb",
            *common,
            success_probability=0.9,
            delay=3,
        )
        pipeline = run_policy_channel(
            "inflight_sco_ucb",
            *common,
            success_probability=0.9,
            delay=3,
            inflight_beta=0.0,
        )
        self.assert_policy_outputs_equal(sco, pipeline)

    def test_positive_penalty_reduces_redundant_attempts(self):
        common = self.fixture()
        sco = run_policy_channel(
            "sco_reset_ucb",
            *common,
            success_probability=0.9,
            delay=3,
        )
        pipeline = run_policy_channel(
            "inflight_sco_ucb",
            *common,
            success_probability=0.9,
            delay=3,
            inflight_beta=16.0,
        )
        self.assertLess(
            pipeline["redundant_attempt_rate"],
            sco["redundant_attempt_rate"],
        )

    def test_negative_penalty_is_rejected(self):
        common = self.fixture()
        with self.assertRaises(ValueError):
            run_policy_channel(
                "inflight_sco_ucb",
                *common,
                success_probability=0.9,
                delay=1,
                inflight_beta=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
