import itertools
import unittest

import numpy as np

from core.topn_loss import topn_exchange_regret


class TopNLossTests(unittest.TestCase):
    def test_exhaustive_small_integer_perturbations(self):
        # Exhausts 5^4 true vectors, 3^4 signed perturbations, and N=1..3.
        values = range(5)
        perturbations = (-0.45, 0.0, 0.45)
        radius = np.full(4, 0.45)
        for w_tuple in itertools.product(values, repeat=4):
            w = np.asarray(w_tuple, dtype=float)
            for d_tuple in itertools.product(perturbations, repeat=4):
                score = w + np.asarray(d_tuple)
                for n in (1, 2, 3):
                    out = topn_exchange_regret(w, score, radius, n)
                    self.assertGreaterEqual(out["regret"], -1e-12)
                    self.assertLessEqual(
                        out["regret"], out["heterogeneous_envelope"] + 1e-12
                    )
                    self.assertLessEqual(
                        out["regret"], out["max_radius_envelope"] + 1e-12
                    )
                    if out["exchange_count"]:
                        self.assertLessEqual(
                            out["boundary_margin"], 2.0 * radius.max() + 1e-12
                        )

    def test_random_heterogeneous_radii(self):
        rng = np.random.default_rng(20260724)
        for _ in range(5000):
            k = int(rng.integers(3, 25))
            n = int(rng.integers(1, k))
            w = rng.normal(size=k)
            radius = rng.uniform(0.0, 0.8, size=k)
            score = w + rng.uniform(-1.0, 1.0, size=k) * radius
            out = topn_exchange_regret(w, score, radius, n)
            self.assertLessEqual(
                out["regret"], out["heterogeneous_envelope"] + 1e-11
            )
            self.assertLessEqual(
                out["exchange_count"], min(n, k - n)
            )

    def test_invalid_error_envelope_is_rejected(self):
        with self.assertRaises(ValueError):
            topn_exchange_regret(
                np.array([2.0, 1.0]),
                np.array([0.0, 3.0]),
                np.array([0.1, 0.1]),
                1,
            )


if __name__ == "__main__":
    unittest.main()

