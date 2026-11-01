from __future__ import annotations

import numpy as np
import pytest

from floodgnn_ceus.models import predict_tree, train_neural


def test_xgboost_is_deterministic_and_returns_probabilities() -> None:
    pytest.importorskip("xgboost")
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    weights = np.full(4, 0.25, dtype=np.float64)
    params = {
        "max_depth": 2,
        "learning_rate": 0.1,
        "n_estimators": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    first = predict_tree("XGBoost", params, x, labels, weights, x, seed=7)
    repeated = predict_tree("XGBoost", params, x, labels, weights, x, seed=7)

    np.testing.assert_array_equal(first, repeated)
    assert np.isfinite(first).all()
    assert ((0 <= first) & (first <= 1)).all()


@pytest.mark.parametrize("model", ["MLP", "GraphSAGE-G2"])
def test_neural_model_completes_one_fixed_cuda_epoch(model: str) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    if not torch.cuda.is_available():
        pytest.skip("CUDA smoke test runs in the configured local CUDA environment")

    x = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    edges = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    result = train_neural(
        model,
        {"hidden": [4, 4], "dropout": 0.0, "learning_rate": 0.001, "weight_decay": 0.0},
        x,
        edges if model.startswith("GraphSAGE") else np.empty((2, 0), dtype=np.int64),
        np.arange(4),
        np.asarray([1.0, 1.0, 0.0, 0.0]),
        np.full(4, 0.25),
        None,
        seed=7,
        max_epochs=1,
        patience=1,
        fixed_epochs=1,
    )

    assert result.best_epoch == 1
    assert result.checkpoint_bytes > 0
    assert result.scores.shape == (4,)
    assert np.isfinite(result.scores).all()


def test_tree_models_force_min_child_weight_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: XGBoost and Neighbor-XGB must not use the default min_child_weight=1.

    With balanced sample weights summing to 1.0 (0.5 positive / 0.5 unlabeled), the
    root-node Hessian is bounded by 0.25 < 1.0, so the default min_child_weight=1
    forbids every split and locks the model into a constant, prevalence-only
    prediction (PB-AUPRC = 1/11). The frozen fix is min_child_weight=0; this test
    pins it so it cannot silently regress back to the broken default.
    """
    pytest.importorskip("xgboost")

    seen: list[dict[str, object]] = []

    class _ProbeClassifier:
        def __init__(self, **kwargs: object) -> None:
            seen.append(kwargs)

        def fit(self, x, y, sample_weight=None):
            return self

        def predict_proba(self, x):
            return np.full((len(x), 2), 0.5, dtype=np.float64)

    monkeypatch.setattr("xgboost.XGBClassifier", _ProbeClassifier)

    x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
    weights = np.full(4, 0.25, dtype=np.float64)
    params = {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 4}

    predict_tree("XGBoost", params, x, labels, weights, x, seed=7)
    predict_tree("Neighbor-XGB", params, x, labels, weights, x, seed=7)

    assert len(seen) == 2
    for kwargs in seen:
        assert kwargs.get("min_child_weight") == 0


def test_graphsage_uses_edges_and_repeats_the_same_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    if not torch.cuda.is_available():
        pytest.skip("CUDA audit runs in the configured local CUDA environment")

    x = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    edges = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    arguments = (
        "GraphSAGE-G2",
        {"hidden": [4, 4], "dropout": 0.0, "learning_rate": 0.001, "weight_decay": 0.0},
        x,
    )
    keywords = {
        "train_indices": np.arange(4),
        "train_labels": np.asarray([1.0, 1.0, 0.0, 0.0]),
        "train_weights": np.full(4, 0.25),
        "validation_banks": None,
        "max_epochs": 2,
        "patience": 1,
        "fixed_epochs": 2,
    }

    gradient_norms = []
    original_step = torch.optim.AdamW.step

    def audited_step(optimizer, *args, **kwargs):
        gradients = [
            parameter.grad.detach()
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        gradient_norms.append(max(float(torch.linalg.vector_norm(item)) for item in gradients))
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", audited_step)
    first = train_neural(*arguments, edges, seed=7, **keywords)
    repeated = train_neural(*arguments, edges, seed=7, **keywords)
    different_seed = train_neural(*arguments, edges, seed=8, **keywords)
    no_edges = train_neural(*arguments, np.empty((2, 0), dtype=np.int64), seed=7, **keywords)

    np.testing.assert_array_equal(first.scores, repeated.scores)
    assert not np.array_equal(first.scores, different_seed.scores)
    assert not np.array_equal(first.scores, no_edges.scores)
    assert gradient_norms and min(gradient_norms) > 0
