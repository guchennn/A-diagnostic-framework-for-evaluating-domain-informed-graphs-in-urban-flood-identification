"""Minimal implementations of the seven frozen FloodGNN model families."""

from __future__ import annotations

import copy
import io
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def _edge_array(edges: np.ndarray) -> np.ndarray:
    values = np.asarray(edges, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("edge array must have shape [2, E]")
    return values


def fixed_filter_features(x: np.ndarray, edges: np.ndarray, *, steps: int = 1) -> np.ndarray:
    """Append one or two incoming-neighbor plus self averages, preserving X."""
    if type(steps) is not int or steps not in (1, 2):
        raise ValueError("fixed-filter steps must be 1 or 2")
    values = np.asarray(x, dtype=np.float64)
    edge_index = _edge_array(edges)
    filtered = values
    for _ in range(steps):
        total = filtered.copy()
        counts = np.ones((len(values), 1), dtype=np.float64)
        for source, target in edge_index.T:
            total[target] += filtered[source]
            counts[target] += 1
        filtered = total / counts
    return np.concatenate([values, filtered], axis=1)


def neighbor_features(raw: np.ndarray, imputed: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Append partition-local grid8 covariate summaries without labels."""
    raw_values = np.asarray(raw, dtype=np.float64)
    center = np.asarray(imputed, dtype=np.float64)
    if raw_values.shape != center.shape:
        raise ValueError("raw and imputed feature matrices must align")
    edge_index = _edge_array(edges)
    neighbors: list[list[int]] = [[] for _ in range(len(center))]
    for source, target in edge_index.T:
        neighbors[target].append(int(source))
    means = center.copy()
    deviations = np.zeros_like(center)
    counts = np.zeros_like(center)
    has_neighbor = np.zeros((len(center), 1), dtype=np.float64)
    for target, sources in enumerate(neighbors):
        if sources:
            has_neighbor[target] = 1.0
        for feature in range(center.shape[1]):
            valid = raw_values[sources, feature]
            valid = valid[np.isfinite(valid)]
            if len(valid):
                means[target, feature] = valid.mean()
                deviations[target, feature] = valid.std(ddof=0)
                counts[target, feature] = len(valid)
    differences = center - means
    return np.concatenate([center, means, deviations, differences, counts, has_neighbor], axis=1)


def predict_kde(train_coordinates: np.ndarray, prediction_coordinates: np.ndarray) -> np.ndarray:
    """Score coordinates with the frozen Gaussian KDE and Scott bandwidth."""
    train = np.asarray(train_coordinates, dtype=np.float64)
    predict = np.asarray(prediction_coordinates, dtype=np.float64)
    if train.ndim != 2 or train.shape[1] != 2 or predict.ndim != 2 or predict.shape[1] != 2:
        raise ValueError("KDE coordinates must have shape [N, 2]")
    estimator = gaussian_kde(train.T, bw_method="scott")
    return estimator.logpdf(predict.T)


def predict_tree(
    model: str,
    params: Mapping[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_predict: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Fit one deterministic single-thread tree model and return positive scores."""
    if model == "XGBoost" or model == "Neighbor-XGB":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            **params,
            n_jobs=1,
            random_state=seed % (2**31),
            objective="binary:logistic",
            min_child_weight=0,
        )
    elif model == "LightGBM":
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            **params,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            random_state=seed % (2**31),
            verbosity=-1,
        )
    else:
        raise ValueError(f"unsupported tree model: {model}")
    estimator.fit(x_train, y_train, sample_weight=weights)
    return estimator.predict_proba(x_predict)[:, 1]


def mean_bank_auprc(scores: np.ndarray, banks: pd.DataFrame) -> float:
    """Average PB-AUPRC across three fixed banks without pooling them."""
    from sklearn.metrics import average_precision_score

    required = {"bank_id", "node_index", "sample_role"}
    if not required.issubset(banks.columns) or set(banks["bank_id"]) != {1, 2, 3}:
        raise ValueError("validation banks are incomplete")
    values = []
    for _, bank in banks.groupby("bank_id", sort=True):
        labels = bank["sample_role"].eq("positive").to_numpy(dtype=np.int8)
        values.append(float(average_precision_score(labels, scores[bank["node_index"]])))
    return float(np.mean(values))


@dataclass(frozen=True)
class NeuralResult:
    scores: np.ndarray
    best_epoch: int
    checkpoint_bytes: int


def train_neural(
    model: str,
    params: Mapping[str, object],
    x: np.ndarray,
    edge_index: np.ndarray,
    train_indices: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    validation_banks: pd.DataFrame | None,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
    fixed_epochs: int | None = None,
    prediction_x: np.ndarray | None = None,
    prediction_edge_index: np.ndarray | None = None,
    aggregator: str = "mean",
    layers: int = 2,
) -> NeuralResult:
    """Fit the frozen MLP or GraphSAGE (two layers by default)."""
    if type(layers) is not int or layers not in (1, 2):
        raise ValueError("GraphSAGE layers must be 1 or 2")
    import torch
    from torch import nn
    from torch_geometric.nn import SAGEConv

    torch.manual_seed(seed % (2**63))
    torch.cuda.manual_seed_all(seed % (2**63))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    hidden = int(params["hidden"][0])
    dropout = float(params["dropout"])

    class GraphSAGE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = SAGEConv(x.shape[1], hidden, aggr=aggregator, root_weight=True)
            if layers == 2:
                self.second = SAGEConv(hidden, hidden, aggr=aggregator, root_weight=True)
            self.output = nn.Linear(hidden, 1)

        def forward(self, features: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
            values = nn.functional.dropout(
                nn.functional.relu(self.first(features, edges)), dropout, self.training
            )
            if layers == 2:
                values = nn.functional.dropout(
                    nn.functional.relu(self.second(values, edges)), dropout, self.training
                )
            return self.output(values).squeeze(1)

    if model in {"MLP", "Fixed-filter"}:
        network: nn.Module = nn.Sequential(
            nn.Linear(x.shape[1], hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Flatten(0),
        )
    elif model.startswith("GraphSAGE"):
        network = GraphSAGE()
    else:
        raise ValueError(f"unsupported neural model: {model}")
    network = network.to(device)
    features = torch.as_tensor(x, dtype=torch.float32, device=device)
    edges = torch.as_tensor(_edge_array(edge_index), dtype=torch.long, device=device)
    predict_features = torch.as_tensor(
        x if prediction_x is None else prediction_x,
        dtype=torch.float32,
        device=device,
    )
    predict_edges = torch.as_tensor(
        _edge_array(edge_index if prediction_edge_index is None else prediction_edge_index),
        dtype=torch.long,
        device=device,
    )
    indices = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    labels = torch.as_tensor(train_labels, dtype=torch.float32, device=device)
    weights = torch.as_tensor(train_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    limit = fixed_epochs or max_epochs
    best_score = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, limit + 1):
        network.train()
        optimizer.zero_grad(set_to_none=True)
        logits = network(features, edges) if model.startswith("GraphSAGE") else network(features)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits[indices], labels, weight=weights, reduction="sum"
        )
        loss.backward()
        optimizer.step()
        if fixed_epochs is not None:
            best_epoch = epoch
            continue
        if validation_banks is None:
            raise ValueError("inner neural training requires validation banks")
        network.eval()
        with torch.no_grad():
            logits = (
                network(predict_features, predict_edges)
                if model.startswith("GraphSAGE")
                else network(predict_features)
            )
            validation_score = mean_bank_auprc(
                torch.sigmoid(logits).cpu().numpy(), validation_banks
            )
        if validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            best_state = copy.deepcopy(network.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if fixed_epochs is None:
        if best_state is None:
            raise ValueError("neural training produced no valid epoch")
        network.load_state_dict(best_state)
    network.eval()
    with torch.no_grad():
        logits = (
            network(predict_features, predict_edges)
            if model.startswith("GraphSAGE")
            else network(predict_features)
        )
        scores = torch.sigmoid(logits).cpu().numpy()
    checkpoint = io.BytesIO()
    torch.save(network.state_dict(), checkpoint)
    return NeuralResult(
        scores=scores,
        best_epoch=best_epoch,
        checkpoint_bytes=checkpoint.tell(),
    )
