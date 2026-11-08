import unittest

import numpy as np
from run_validation import WEIGHTS, Q, dataset, metrics, offsets, oracle, precision_terms


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
        self.g = np.repeat(np.arange(3), 4)
        self.s = np.array([2, 2, 1, 0, 3, 1, 0, 0, 2, 1, 3, 3], float)

    def test_direct_threshold_ap_and_weighting(self):
        for w in WEIGHTS:
            direct = np.array(
                [
                    self.y[self.s >= v].sum()
                    / (self.y[self.s >= v].sum() + w * (~self.y[self.s >= v]).sum())
                    for v in self.s[self.y]
                ]
            )
            np.testing.assert_allclose(precision_terms(self.y, self.s, w), direct)
            values = metrics(self.y, self.g, self.s, w)
            self.assertAlmostEqual(values[0][0], values[2].sum())

    def test_rank_direct_pairwise_ties(self):
        bg = self.s[~self.y]
        expected = [
            np.mean(
                [(v > bg).mean() + 0.5 * (v == bg).mean() for v in self.s[self.y & (self.g == k)]]
            )
            for k in range(3)
        ]
        np.testing.assert_allclose(metrics(self.y, self.g, self.s, 1)[3], expected)

    def test_within_shift_invariance(self):
        shifted = self.s + np.array([-0.5, 0.3, 2])[self.g]
        np.testing.assert_allclose(
            metrics(self.y, self.g, self.s, 1)[1], metrics(self.y, self.g, shifted, 1)[1]
        )

    def test_population_permutation_and_convergence(self):
        for q in Q:
            for a in [0, 0.25, 0.5, 1]:
                for w in WEIGHTS:
                    t = offsets("redistribution", a)
                    first = oracle(q, t[0], w)
                    last = oracle(q, t[2], w)
                    np.testing.assert_allclose(first[0], last[0], atol=1e-12)
                    np.testing.assert_allclose(first[1], last[1], atol=1e-12)
                    for x, z in zip(first, oracle(q, t[0], w, 256)):
                        np.testing.assert_allclose(x, z, atol=1e-7, rtol=0)

    def test_seed_disjointness_and_determinism(self):
        keys = [(0, q, 0, 0, r) for q in range(2) for r in range(300)]
        keys += [
            (1, q, s, a, r)
            for q in range(2)
            for s in range(3)
            for a in range(4)
            for r in range(200)
        ]
        self.assertEqual(len(keys), len(set(keys)))
        a = dataset(0, 0, 0, 0, 0)[2]
        np.testing.assert_array_equal(a, dataset(0, 0, 0, 0, 0)[2])
        self.assertFalse(np.array_equal(a, dataset(0, 1, 0, 0, 0)[2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
