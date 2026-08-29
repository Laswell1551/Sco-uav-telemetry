"""Minimal deterministic tests for the retrospective v16 baseline runner."""
from __future__ import annotations

import unittest

import numpy as np

from run_tmc_v16_baseline_expansion import (
    DiminishingExplorationSchedule,
    DiscountedCVMomentPosterior,
    seed_cluster_bootstrap_summary,
)


class DiminishingExplorationScheduleTests(unittest.TestCase):
    def test_n_greater_than_one_block_coverage_and_reset(self):
        schedule = DiminishingExplorationSchedule(
            batches=2, arms=5, budget=2, alpha=1.0
        )
        self.assertEqual(schedule.initial_cursor, 1)
        self.assertEqual(schedule.block_slots, 3)

        first_block = [schedule.mask(t) for t in (1, 2, 3)]
        for mask in first_block:
            self.assertTrue(np.all(mask.sum(axis=1) == 2))
        covered = np.logical_or.reduce(
            [mask[0] for mask in first_block]
        )
        self.assertTrue(np.all(covered))

        # u=1 advances to ceil(1 + 5 + 25/4)=13 after the block.
        self.assertFalse(schedule.mask(4).any())
        self.assertTrue(np.array_equal(schedule.cursor, [13, 13]))

        schedule.reset(np.array([True, False]), round_index=4)
        after_reset = schedule.mask(5)
        self.assertTrue(after_reset[0].any())
        self.assertFalse(after_reset[1].any())
        self.assertEqual(int(schedule.cursor[0]), 1)
        self.assertEqual(int(schedule.cursor[1]), 13)


class DiscountedPosteriorTests(unittest.TestCase):
    def test_selected_only_update_with_calendar_discount(self):
        initial = np.array(
            [
                [[[0.1, 0.2], [0.3, 0.4]]],
                [[[0.2, 0.4], [0.5, 0.6]]],
            ],
            dtype=float,
        )
        posterior = DiscountedCVMomentPosterior(initial, gamma=0.5)
        old_mean = posterior.running_mean.copy()
        selected = np.array([[True, False]])
        observation = np.array([[[0.9, 0.8], [0.0, 0.0]]])
        posterior.update(selected, observation)

        self.assertAlmostEqual(float(posterior.weight[0, 0]), 2.0)
        self.assertAlmostEqual(float(posterior.weight[0, 1]), 1.0)
        expected_selected = old_mean[0, 0] + (
            observation[0, 0] - old_mean[0, 0]
        ) / 2.0
        np.testing.assert_allclose(
            posterior.running_mean[0, 0], expected_selected
        )
        np.testing.assert_allclose(
            posterior.running_mean[0, 1], old_mean[0, 1]
        )

        draw = posterior.draw(np.zeros_like(posterior.running_mean))
        np.testing.assert_allclose(draw, posterior.physical_mean)
        self.assertTrue(np.all((draw >= 0.01) & (draw <= 1.0)))


class ClusterBootstrapTests(unittest.TestCase):
    def test_seed_is_resampling_unit(self):
        rows = [
            {"seed": 10, "metric": value}
            for value in (1.0, 1.0, 1.0)
        ] + [
            {"seed": 11, "metric": value}
            for value in (3.0, 3.0, 3.0)
        ]
        summary = seed_cluster_bootstrap_summary(
            rows,
            "metric",
            np.random.default_rng(7),
            replicates=2000,
        )
        self.assertEqual(summary["n_seed_clusters"], 2)
        self.assertAlmostEqual(summary["mean"], 2.0)
        self.assertEqual(
            summary["cluster_statistic"], "within-seed batch mean"
        )


if __name__ == "__main__":
    unittest.main()
