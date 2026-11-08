"""Independent threshold-count and pairwise-background tests, including ties."""

import unittest

import numpy as np
from analyze_local_references import positive_statistics


class LocalReferenceTests(unittest.TestCase):
    def test_threshold_count_and_pairwise_background(self) -> None:
        y = np.array([1, 0, 1, 0, 1, 0])
        for scores in (np.array([1, 1, 2, 3, 2, 0]), np.ones(6), np.arange(6)):
            actual = positive_statistics(y, scores)
            p = scores[y == 1]
            b = scores[y == 0]
            expected_ap = [((p >= v).sum() / (scores >= v).sum()) / len(p) for v in p]
            expected_rank = [((b < v).sum() + 0.5 * (b == v).sum()) / len(b) for v in p]
            np.testing.assert_allclose(actual[:, 0], expected_ap)
            np.testing.assert_allclose(actual[:, 2], expected_rank)

    def test_paired_identity_and_additivity(self) -> None:
        y = np.array([1, 0, 1, 0, 1, 0])
        scores = [np.array([1, 1, 3, 2, 0, 4]), np.arange(6), np.arange(6)[::-1]]
        own, adj, d8 = [positive_statistics(y, s) for s in scores]
        np.testing.assert_allclose((d8 - own) - (d8 - adj), adj - own, atol=1e-15)
        groups = np.array([0, 0, 2])
        for ref in (own, adj):
            delta = d8 - ref
            parts = [delta[groups == g, 0].sum() for g in range(3)]
            self.assertAlmostEqual(sum(parts), d8[:, 0].sum() - ref[:, 0].sum())
            self.assertEqual(parts[1], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
