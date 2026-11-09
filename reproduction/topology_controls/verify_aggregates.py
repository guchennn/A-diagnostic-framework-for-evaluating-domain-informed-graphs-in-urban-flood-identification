"""Verify published topology aggregates using only the Python standard library."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results/topology_controls"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(name):
    with (RESULTS / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    require(math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), (a, b))


manifest = json.loads((RESULTS / "manifest.json").read_text("utf-8"))
require(manifest["status"] == "complete", "incomplete analysis")
require(
    [manifest[k] for k in ("control_fits", "reference_fits", "accepted_graphs")] == [900, 300, 960],
    "incorrect run counts",
)
for name, expected in manifest["outputs"].items():
    require(Path(name).name == name, "invalid output name")
    require(
        hashlib.sha256((RESULTS / f"{name}.csv").read_bytes()).hexdigest() == expected,
        f"aggregate hash differs: {name}",
    )

cities = ("beijing", "zhengzhou")
predictors = ("Fixed-filter", "GraphSAGE-G2")
metrics = ("sampled_ap", "full_ap", "weighted_ap", "recall_10")
keys = ("city", "outer_fold", "draw", "seed", "predictor", "representation", "realization")
records = read("training_records")
observed = {tuple(r[k] for k in keys) for r in records}
expected = set()
for unit in product(
    cities,
    map(str, range(5)),
    map(str, range(1, 6)),
    ("NEURAL-1", "NEURAL-2", "NEURAL-3"),
    predictors,
):
    expected.add((*unit, "D8", "0"))
    expected.update((*unit, "G7", str(r)) for r in (1, 2, 3))
require(len(records) == len(observed) == 1200 and observed == expected, "fit identities differ")
for row in records:
    require(row["status"] == "success", "unsuccessful fit")
    epochs = sorted(map(int, row["best_inner_epochs"].split(";")))
    require(len(epochs) == 3 and epochs[0] > 0, "invalid inner epochs")
    require(epochs[1] == int(row["scheduled_refit_epochs"]), "duration is not median")
    require(int(row["seed_uint128"]) % (2**63) == int(row["torch_seed"]), "seed differs")

graphs = read("graph_quality")
gkeys = ("city", "outer_fold", "scope", "draw", "inner_fold", "partition", "realization")
actual = {tuple(r[k] for k in gkeys) for r in graphs}
expected = set()
for city, fold, part, realization in product(
    cities, map(str, range(5)), ("train", "test"), ("1", "2", "3")
):
    expected.add((city, fold, "outer", "NA", "-1", part, realization))
    for draw, inner in product(map(str, range(1, 6)), map(str, range(3))):
        expected.add((city, fold, "inner", draw, inner, part, realization))
require(len(graphs) == len(actual) == 960 and actual == expected, "graph identities differ")
for row in graphs:
    require(
        row["exact_in_out_degree"] == row["distance_counts_preserved"] == "True",
        "structural constraint failed",
    )
    require(float(row["changed_fraction"]) >= 0.075, "insufficient perturbation")
    close(row["changed_fraction"], int(row["changed_edges"]) / int(row["edges"]))

folds = {
    (r["city"], r["predictor"], r["metric"], r["outer_fold"]): float(r["value"])
    for r in read("paired_folds")
}
require(
    set(folds) == set(product(cities, predictors, metrics, map(str, range(5)))),
    "paired fold coverage differs",
)
absolute = {
    (r["city"], r["predictor"], r["metric"], r["outer_fold"], r["representation"]): float(
        r["value"]
    )
    for r in read("absolute_folds")
}
realizations = defaultdict(list)
for row in read("graph_realization_folds"):
    key = tuple(row[k] for k in ("city", "predictor", "metric", "outer_fold"))
    realizations[key].append((row["realization"], float(row["value"])))
for key, value in folds.items():
    close(value, absolute[*key, "D8"] - absolute[*key, "G7"])
    require(sorted(r for r, _ in realizations[key]) == ["1", "2", "3"], "realizations differ")
    close(value, sum(v for _, v in realizations[key]) / 3)
for table in ("paired_summary", "absolute_summary", "graph_realization_contrasts"):
    for row in read(table):
        close(row["mean"], sum(float(row[f"fold{i}"]) for i in range(5)) / 5)
        require(int(row["n_folds"]) == 5, "incorrect independent summary unit")
        if "mean_pp" in row:
            for source, target in (
                ("mean", "mean_pp"),
                ("range_low", "range_low_pp"),
                ("range_high", "range_high_pp"),
            ):
                close(float(row[source]) * 100, row[target])

selection = json.loads((HERE / "inner_selection_verification.json").read_text("utf-8"))
require(selection["status"] == "complete" and not selection["errors"], "selection audit failed")
require(
    [
        selection[k]
        for k in ("verified_candidate_tasks", "verified_inner_fold_fits", "verified_selections")
    ]
    == [600, 1800, 300],
    "selection counts differ",
)
replay = json.loads((HERE / "reference_replay_verification.json").read_text("utf-8"))
require(len(replay["fits"]) == 4 and replay["all_scores_exactly_equal"], "replay audit differs")
require(all(r["max_absolute_score_difference"] == 0 for r in replay["fits"]), "replay mismatch")
print(
    json.dumps(
        {
            "status": "PASS",
            "hashed_tables": len(manifest["outputs"]),
            "control_fits": 900,
            "reference_fits": 300,
            "graphs": 960,
            "paired_fold_identities": len(folds),
            "independent_selections": 300,
            "reference_replay_fits": 4,
            "all_fit_numerical_replay_claimed": False,
        }
    )
)
