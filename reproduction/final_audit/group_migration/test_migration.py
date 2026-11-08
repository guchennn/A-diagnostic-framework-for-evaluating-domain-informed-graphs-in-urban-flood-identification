"""Independent small-graph checks for the structural migration audit."""

import itertools
import unittest

import numpy as np
import pandas as pd
from audit_migration import EDGE_KEYS, assert_acyclic, classify, summarize


class MigrationTests(unittest.TestCase):
    def test_chain_and_isolate(self) -> None:
        edges = pd.DataFrame([("a", "b"), ("b", "c"), ("c", "d")], columns=EDGE_KEYS)
        np.testing.assert_array_equal(classify(list("abcde"), edges), ["0", "1", "2", "2", "0"])
        assert_acyclic(list("abcde"), edges)

    def test_branching_is_not_multihop(self) -> None:
        edges = pd.DataFrame([("a", "c"), ("b", "c")], columns=EDGE_KEYS)
        np.testing.assert_array_equal(classify(list("abc"), edges), ["0", "0", "1"])

    def test_every_edge_subset_cannot_raise_group(self) -> None:
        original = [("a", "b"), ("b", "c"), ("c", "d")]
        before = classify(list("abcde"), pd.DataFrame(original, columns=EDGE_KEYS)).astype(int)
        for bits in itertools.product([False, True], repeat=3):
            after = pd.DataFrame(
                [e for e, keep in zip(original, bits, strict=True) if keep], columns=EDGE_KEYS
            )
            self.assertTrue((classify(list("abcde"), after).astype(int) <= before).all())

    def test_cycles_are_rejected(self) -> None:
        for edge_pairs in [[("a", "a")], [("a", "b"), ("b", "a")]]:
            with self.assertRaises(ValueError):
                assert_acyclic(list("ab"), pd.DataFrame(edge_pairs, columns=EDGE_KEYS))

    def test_transition_denominators_and_pooling(self) -> None:
        frame = pd.DataFrame(
            {
                "city": ["toy"] * 4,
                "outer_fold": [0, 0, 1, 1],
                "full_group": ["0", "1", "2", "2"],
                "heldout_group": ["0", "0", "1", "2"],
                "positive": [True, False, True, False],
                "indegree_full": [0, 1, 1, 1],
                "indegree_test": [0, 0, 1, 1],
                "K_full": [0, 1, 2, 2],
                "K_test": [0, 0, 1, 2],
            }
        )
        matrices, summary = summarize(frame)
        pooled = matrices.loc[matrices.outer_fold.eq("POOLED") & matrices.population.eq("all")]
        self.assertEqual(pooled["count"].sum(), 4)
        self.assertAlmostEqual(pooled.population_fraction.sum(), 1)
        self.assertTrue(pooled.groupby("full_group").row_fraction.sum().eq(1).all())
        s = summary.loc[summary.outer_fold.eq("POOLED") & summary.population.eq("positive")].iloc[0]
        self.assertEqual(s["n"], 2)
        self.assertEqual(s.group_changed_n, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
