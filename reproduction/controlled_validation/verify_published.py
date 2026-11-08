"""Verify published hashes and synthetic aggregate accounting using only stdlib."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify() -> dict:
    manifest = json.loads((ROOT / "publication_manifest.json").read_text())
    for name, expected in manifest["publication_files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    units = defaultdict(dict)
    datasets = set()
    with (ROOT / "controlled_ranking/replicates.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(
                row[k]
                for k in [
                    "composition",
                    "scenario",
                    "offset",
                    "replicate",
                    "weight_id",
                    "reference_id",
                ]
            )
            datasets.add(key[:4])
            units[key][row["metric"], row["group"]] = float(row["estimate"])
    assert len(datasets) == 4800
    residual = max(
        abs(sum(v["allocation", str(g)] for g in range(3)) - v["global_ap", "0"])
        for v in units.values()
    )
    assert residual < 1e-12
    with (ROOT / "controlled_ranking/calibration.csv").open(newline="") as stream:
        calibration = {(r["composition"], r["replicate"]) for r in csv.DictReader(stream)}
    assert len(calibration) == 600
    return {
        "status": "PASS",
        "files": len(manifest["publication_files"]),
        "test_datasets": len(datasets),
        "calibration_datasets": len(calibration),
        "allocation_residual": residual,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
