import unittest

import numpy as np

from run_cv_piecewise_pilot import make_problem
from run_tmc_channel_stress import run_policy_channel
from run_tmc_random_delay import (
    delay_bank_summary,
    make_delay_banks,
    run_policy_random_delay,
)


class RandomDelayRunnerTests(unittest.TestCase):
    def fixture(self, seed=20264001, H=160):
        B, K, n0, change_t, block = 2, 10, 8, H // 2, 64
        problem = make_problem(seed, B, K, change_t, H, n0, block)
        (_, _, theta0, theta1, c00, c01, changed,
         pre_bank, post_bank, ages0) = problem
        uniforms = np.random.default_rng(seed + 50000).random(
            (B, K, H + n0)
        )
        common = (
            theta0, theta1, c00, c01, changed, pre_bank, post_bank,
            uniforms, ages0, 3, n0, change_t, H, 0.9,
        )
        return common, uniforms.shape

    def assert_result_close(self, left, right):
        for field in (
            "avg_cost", "pre_cost", "post_cost", "delivery_rate",
            "redundant_attempt_rate", "detected_fraction",
            "calendar_delay", "observation_delay",
        ):
            np.testing.assert_allclose(
                left[field], right[field], rtol=1e-11, atol=1e-11,
                equal_nan=True,
            )

    def test_zero_two_way_delay_matches_fixed_runner(self):
        common, shape = self.fixture()
        old = run_policy_channel("sco_reset_ucb", *common, delay=0)
        zero = np.zeros(shape, dtype=int)
        args = common[:8] + (zero, zero) + common[8:-1]
        new = run_policy_random_delay(
            "sco_reset_ucb", *args,
            success_probability=common[-1],
        )
        self.assert_result_close(old, new)

    def test_fixed_forward_delay_matches_fixed_runner(self):
        common, shape = self.fixture(seed=20264002)
        old = run_policy_channel("sco_reset_ucb", *common, delay=3)
        forward = np.full(shape, 3, dtype=int)
        feedback = np.zeros(shape, dtype=int)
        args = common[:8] + (forward, feedback) + common[8:-1]
        new = run_policy_random_delay(
            "sco_reset_ucb", *args,
            success_probability=common[-1],
        )
        self.assert_result_close(old, new)

    def test_delay_banks_are_deterministic_and_tail_ordered(self):
        shape = (4, 20, 2000)
        light = make_delay_banks(19, "light_iid", shape)
        heavy = make_delay_banks(19, "heavy_iid", shape)
        lognormal = make_delay_banks(19, "lognormal", shape)
        for profile, banks in (
            ("light_iid", light),
            ("heavy_iid", heavy),
            ("lognormal", lognormal),
        ):
            again = make_delay_banks(19, profile, shape)
            np.testing.assert_array_equal(banks[0], again[0])
            np.testing.assert_array_equal(banks[1], again[1])
        ls = delay_bank_summary(*light)
        hs = delay_bank_summary(*heavy)
        gs = delay_bank_summary(*lognormal)
        self.assertAlmostEqual(ls["round_trip_mean"], 4.0, delta=0.03)
        self.assertAlmostEqual(hs["round_trip_mean"], 4.0, delta=0.08)
        self.assertGreater(hs["round_trip_p99"], ls["round_trip_p99"])
        self.assertGreater(gs["round_trip_p99"], ls["round_trip_p99"])

    def test_heavy_delay_produces_stale_out_of_order_arrivals(self):
        common, shape = self.fixture(seed=20264003, H=240)
        forward, feedback = make_delay_banks(
            20264003, "forward_heavy", shape
        )
        args = common[:8] + (forward, feedback) + common[8:-1]
        result = run_policy_random_delay(
            "sco_reset_ucb", *args,
            success_probability=common[-1],
        )
        self.assertGreater(result["stale_arrival_rate"], 0.0)
        self.assertTrue(np.all(np.isfinite(result["post_cost"])))

    def test_pipeline_penalty_reduces_random_delay_duplicates(self):
        common, shape = self.fixture(seed=20264004, H=240)
        forward, feedback = make_delay_banks(
            20264004, "heavy_iid", shape
        )
        args = common[:8] + (forward, feedback) + common[8:-1]
        sco = run_policy_random_delay(
            "sco_reset_ucb", *args,
            success_probability=common[-1],
        )
        pa = run_policy_random_delay(
            "inflight_sco_ucb", *args,
            success_probability=common[-1], inflight_beta=16.0,
        )
        self.assertLess(
            pa["redundant_attempt_rate"], sco["redundant_attempt_rate"]
        )

    def test_he_rm_age_is_ack_gated_and_learns_finite_threshold(self):
        common, shape = self.fixture(seed=20264005, H=320)
        forward, feedback = make_delay_banks(
            20264005, "heavy_iid", shape
        )
        args = common[:8] + (forward, feedback) + common[8:-1]
        result = run_policy_random_delay(
            "he_rm_age", *args,
            success_probability=common[-1],
        )
        self.assertEqual(result["redundant_attempt_rate"], 0.0)
        self.assertLessEqual(result["max_inflight_count"], 1)
        self.assertGreater(result["learned_wait_threshold"], 0.0)
        self.assertTrue(np.isfinite(result["learned_wait_threshold"]))
        self.assertGreater(result["capacity_utilization"], 0.0)
        self.assertLessEqual(result["capacity_utilization"], 1.0)
        self.assertTrue(np.all(np.isfinite(result["post_cost"])))


if __name__ == "__main__":
    unittest.main()
