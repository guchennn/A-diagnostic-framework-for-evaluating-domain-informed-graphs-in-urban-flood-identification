"""Synthetic spatial isolation, preprocessing, degree and tie invariants."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from floodgnn_ceus import synthetic_diagnostic_validation as synthetic


@pytest.fixture
def config():
    return yaml.safe_load(
        (Path(__file__).parents[1] / "configs/synthetic_diagnostic_validation_v1.yaml").read_text(
            "utf-8"
        )
    )


def test_generated_edges_are_isolated_and_rewiring_preserves_degrees(config):
    data = synthetic.generate(config, config["generation_seeds"][0])
    split = data["split"]
    n = len(split)
    for edges in data["edges"].values():
        assert np.all(split[edges[:, 0]] == split[edges[:, 1]])
        assert np.all(split[edges[:, 0]] >= 0)
        assert np.all(edges[:, 0] != edges[:, 1])
        assert len(set(map(tuple, edges))) == len(edges)
    correct, wrong = data["edges"]["useful"], data["edges"]["degree_preserving_wrong"]
    for column in (0, 1):
        np.testing.assert_array_equal(
            np.bincount(correct[:, column], minlength=n), np.bincount(wrong[:, column], minlength=n)
        )
    assert len(set(map(tuple, correct)) & set(map(tuple, wrong))) < 0.1 * len(correct)
    assert np.min(data["row"][split == 1]) - np.max(data["row"][split == 0]) == 5
    assert np.min(data["row"][split == 2]) - np.max(data["row"][split == 1]) == 5


def test_preprocessing_uses_only_training_scope():
    x = np.array([[0.0, 2.0], [2.0, 2.0], [1e9, -1e9]])
    train = np.array([True, True, False])
    transformed, fitted = synthetic.train_standardize(x, train)
    np.testing.assert_array_equal(fitted["mean"], [1, 2])
    np.testing.assert_array_equal(fitted["scale"], [1, 1])
    np.testing.assert_array_equal(transformed[:2], [[-1, 0], [1, 0]])


def test_half_ties_ap_and_pairwise_auc():
    y, score = np.array([1, 0, 1, 0, 0]), np.array([2, 2, 1, 1, 0])
    stats = synthetic.midpoint_statistics(y, score)
    pairwise = [float(p > b) + 0.5 * float(p == b) for p in score[y == 1] for b in score[y == 0]]
    assert stats[:, 2].mean() == pytest.approx(np.mean(pairwise))
    assert stats[:, 0].sum() == pytest.approx(0.5)
    np.testing.assert_array_equal(synthetic.midpoint_statistics(y, np.zeros(5))[:, 1:], 0.5)


def test_labels_do_not_change_edges_and_boundary_has_cut_group(config):
    a = synthetic.generate(config, config["generation_seeds"][0])
    modified = dict(config, label_intercept=0.0, upstream_coefficient=-2.0)
    b = synthetic.generate(modified, config["generation_seeds"][0])
    for condition in a["edges"]:
        np.testing.assert_array_equal(a["edges"][condition], b["edges"][condition])
    assert not np.array_equal(a["labels"], b["labels"])
    groups = synthetic.structural_groups(
        a["edges"]["boundary_truncated"], a["full"], len(a["split"])
    )
    assert set(groups["support"][a["split"] == 2]) == {"root", "cut", "retained"}


def test_test_labels_do_not_enter_selection_or_fit(config):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(240, 3))
    y = (rng.random(240) < 0.3).astype(int)
    split = np.repeat([0, 1, 2], 80)
    first, audit = synthetic.fit_scores(x, y, split, config)
    y[split == 2] = 1 - y[split == 2]
    second, other = synthetic.fit_scores(x, y, split, config)
    np.testing.assert_array_equal(first, second)
    assert audit == other
