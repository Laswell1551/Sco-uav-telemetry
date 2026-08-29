import unittest

import numpy as np
import sympy as sp

from core.dare_monotonicity import derivative_witnesses, eta_of_s
from core.instances import pbar_batch
from core.sim import W_from_pack, coeff_pack


class DareMonotonicityTests(unittest.TestCase):
    def test_symbolic_identities(self):
        s = sp.symbols("s", positive=True)
        d = sp.sqrt(s**2 + 8)
        x = sp.sqrt(s**2 - 4)
        eta = 3 * s - sp.sqrt(3) * d
        etap = sp.diff(eta, s)
        self.assertEqual(sp.simplify(x**2 - eta * s + eta**2 / 6), 0)
        self.assertEqual(
            sp.simplify(etap * (s - eta / 6) - s - 4 * sp.sqrt(3) / d),
            0,
        )

    def test_all_derivative_witnesses_are_positive(self):
        s = 2.0 + np.logspace(-7, 5, 2000)
        self.assertTrue(np.all(eta_of_s(s) > 0.0))
        for witness in derivative_witnesses(s):
            self.assertTrue(np.all(np.isfinite(witness)))
            self.assertTrue(np.all(witness > 0.0))

    def test_random_box_corners_bound_whittle_index(self):
        rng = np.random.default_rng(81427)
        for _ in range(80):
            T = float(np.exp(rng.uniform(np.log(0.1), np.log(3.0))))
            q0, q1 = np.sort(np.exp(rng.uniform(np.log(1e-4), np.log(3.0), 2)))
            r0, r1 = np.sort(np.exp(rng.uniform(np.log(1e-4), np.log(3.0), 2)))
            q = np.geomspace(q0, q1, 11)
            r = np.geomspace(r0, r1, 11)
            qq, rr = np.meshgrid(q, r, indexing="ij")
            _, p12, p22 = pbar_batch(T, qq.ravel(), rr.ravel())
            theta = np.stack([qq.ravel(), p12, p22], axis=1)

            _, p12_lo, p22_lo = pbar_batch(T, np.array([q0]), np.array([r0]))
            _, p12_hi, p22_hi = pbar_batch(T, np.array([q1]), np.array([r1]))
            lo = np.array([[q0, p12_lo[0], p22_lo[0]]])
            hi = np.array([[q1, p12_hi[0], p22_hi[0]]])
            for age in (1, 3, 10, 50):
                dense = W_from_pack(np.full(qq.size, age), coeff_pack(T, theta))
                lower = W_from_pack(np.array([age]), coeff_pack(T, lo))[0]
                upper = W_from_pack(np.array([age]), coeff_pack(T, hi))[0]
                self.assertGreaterEqual(float(dense.min()), float(lower) - 1e-9)
                self.assertLessEqual(float(dense.max()), float(upper) + 1e-7)


if __name__ == "__main__":
    unittest.main()

