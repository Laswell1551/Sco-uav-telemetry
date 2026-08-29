"""Regression checks for the paper-facing channel stress layer."""
from __future__ import annotations

import unittest

import numpy as np

from run_cv_piecewise_pilot import make_problem, run_policy
from run_tmc_channel_stress import run_policy_channel


class ChannelStressRegressionTests(unittest.TestCase):
    def test_reliable_zero_delay_matches_original_policy(self):
        seed = 20261101
        B, K, N, H, n0, change_t, block_length = 2, 8, 2, 100, 8, 50, 64
        problem = make_problem(
            seed, B, K, change_t, H, n0, block_length
        )
        (
            _, _, theta0, theta1, c00, c01, changed,
            pre_bank, post_bank, ages0,
        ) = problem
        channel_uniform = np.zeros((B, K, H + n0), dtype=float)

        for name in (
            "cumulative_ucb_cv",
            "sco_reset_ucb",
            "ps_forced_reset_ucb",
        ):
            original = run_policy(
                name,
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
            stressed = run_policy_channel(
                name,
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
                success_probability=1.0,
                delay=0,
            )
            self.assertTrue(
                np.allclose(original["avg_cost"], stressed["avg_cost"])
            )
            self.assertTrue(
                np.allclose(original["pre_cost"], stressed["pre_cost"])
            )
            self.assertTrue(
                np.allclose(original["post_cost"], stressed["post_cost"])
            )
            self.assertEqual(stressed["delivery_rate"], 1.0)
            if name != "cumulative_ucb_cv":
                self.assertEqual(
                    original["detected_fraction"],
                    stressed["detected_fraction"],
                )
                self.assertEqual(
                    original["calendar_delay"],
                    stressed["calendar_delay"],
                )
                self.assertEqual(
                    original["observation_delay"],
                    stressed["observation_delay"],
                )


if __name__ == "__main__":
    unittest.main()
