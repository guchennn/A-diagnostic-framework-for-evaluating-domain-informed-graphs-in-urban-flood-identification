"""Check recovered fit/bank aggregates without reconstructing missing training records."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGIMES = {"full": "F18", "noD8": "F16-noD8", "noNeigh": "F14-noNeigh"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify() -> dict:
    inventory = json.loads((HERE / "record_inventory.json").read_text(encoding="utf-8"))
    for name, expected in inventory["published_file_hashes"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name

    rows = read_csv(HERE / "full_budget_fit_bank_metrics.csv")
    assert len(rows) == 450
    identities = [
        (r["outer_fold"], r["regime"], r["method"], r["seed"], r["bank_id"]) for r in rows
    ]
    expected = {
        (str(fold), regime, method, seed, str(bank))
        for fold in range(5)
        for regime in REGIMES
        for method in ("XGB", "G0", "G1", "G2")
        for seed in (("NEURAL-1",) if method == "XGB" else ("NEURAL-1", "NEURAL-2", "NEURAL-3"))
        for bank in (1, 2, 3)
    }
    assert len(set(identities)) == len(rows) and set(identities) == expected
    groups = defaultdict(list)
    for row in rows:
        groups[row["outer_fold"], REGIMES[row["regime"]], row["method"]].append(
            float(row["pb_auprc"])
        )
    published = [
        row
        for row in read_csv(ROOT / "results/performance/ablation_absolute_folds.csv")
        if row["budget"] == "FULL"
    ]
    assert len(published) == 60
    error = max(
        abs(
            fmean(groups[row["outer_fold"], row["feature_set"], row["method"]])
            - float(row["sampled_pb_ap"])
        )
        for row in published
    )
    assert error < 1e-12

    choices = {}
    for row in read_csv(HERE / "full_budget_inner_selection.csv"):
        if row["family"] == "GraphSAGE-G2":
            for fold, config in enumerate(json.loads(row["configs_by_fold"])):
                choices[str(fold), REGIMES[row["regime"]]] = config
    settings = [
        row
        for row in read_csv(ROOT / "results/settings/ablation_selections.csv")
        if row["budget"] == "FULL" and row["model"] == "GraphSAGE-G2"
    ]
    assert len(settings) == len(choices) == 15
    for row in settings:
        feature = REGIMES.get(row["feature_set"], row["feature_set"])
        assert choices[row["outer_fold"], feature] == row["config"]
        assert row["refit_epochs"] == "" and row["best_inner_epochs"] == ""

    return {
        "status": "PASS",
        "full_budget_fit_bank_rows": len(rows),
        "verified_fold_method_feature_means": len(published),
        "maximum_absolute_ap_error": error,
        "verified_neural_selection_units": len(settings),
        "missing_refit_epoch_units": 15,
        "neural_fits_without_integer_seed_provenance": 135,
        "fresh_city_refit_numerical_replay_verified": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
