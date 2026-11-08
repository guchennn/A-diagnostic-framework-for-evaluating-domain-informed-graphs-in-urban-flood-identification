"""Verify the accompanying aggregate evidence without third-party packages."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def verify(root: Path) -> dict:
    manifest = json.loads((root / "package_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name

    def rows(name):
        with (root / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    transitions = rows("group_migration/transition_matrix.csv")
    summaries = rows("group_migration/migration_summary.csv")
    for summary in summaries:
        selected = [
            r
            for r in transitions
            if all(r[k] == summary[k] for k in ("city", "outer_fold", "population"))
        ]
        assert sum(int(r["count"]) for r in selected) == int(summary["n"])
        assert sum(
            int(r["count"]) for r in selected if r["full_group"] != r["heldout_group"]
        ) == int(summary["group_changed_n"])
        assert all(
            int(r["count"]) == 0 for r in selected if int(r["heldout_group"]) > int(r["full_group"])
        )
        for group in ("0", "1", "2"):
            subset = [r for r in selected if r["full_group"] == group]
            assert len(subset) == 3
            assert sum(int(r["count"]) for r in subset) == int(subset[0]["row_total"])

    def close(a, b):
        assert abs(a - b) < 1e-11, (a, b)

    fits = rows("local_references/fit_contributions.csv")
    indexed = defaultdict(dict)
    for row in fits:
        key = tuple(row[k] for k in ("city", "outer_fold", "trajectory", "seed", "group"))
        indexed[key][row["contrast"]] = float(row["ap_contribution"])
    for values in indexed.values():
        close(values["D8 - own"] - values["D8 - adjacency"], values["adjacency - own"])
    assert len({key[:4] for key in indexed}) == 150
    fold = rows("local_references/fold_contrasts.csv")
    summary_rows = rows("local_references/summary.csv")
    for row in summary_rows:
        selected = [
            r
            for r in fold
            if all(r[k] == row[k] for k in ("city", "condition", "contrast", "group", "metric"))
        ]
        assert len(selected) == 5
        close(sum(float(r["value"]) for r in selected) / 5, float(row["mean"]))
        close(float(row["mean"]) * 100, float(row["mean_pp"]))
    for city, n in (("zhengzhou", 791), ("beijing", 296)):
        for contrast in ("D8 - own", "D8 - adjacency", "adjacency - own"):
            selected = [
                r
                for r in summary_rows
                if r["city"] == city
                and r["contrast"] == contrast
                and r["metric"] == "ap_contribution"
            ]
            groups = [r for r in selected if r["group"] != "all"]
            total = next(r for r in selected if r["group"] == "all")
            assert sum(int(r["positive_cells_total"]) for r in groups) == n
            close(sum(float(r["mean"]) for r in groups), float(total["mean"]))
    sensitivity = rows("local_references/group_definition_sensitivity.csv")
    for row in sensitivity:
        if row["group"] == "all" and row["metric"] == "ap_contribution":
            close(float(row["mean_full_group"]), float(row["mean_heldout_group"]))
    return {
        "status": "PASS",
        "hashed_files": len(manifest["files"]),
        "paired_fit_units": 150,
        "cities": 2,
        "checks": [
            "file integrity",
            "transition conservation and downgrade",
            "paired reference identity",
            "equal-fold means and AP additivity",
            "positive-cell counts",
            "relabelled total AP invariance",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent.parent), indent=2))
