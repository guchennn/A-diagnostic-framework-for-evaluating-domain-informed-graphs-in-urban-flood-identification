"""Verify public seed aggregates independently of prediction-analysis helpers.

Adapted from manuscript evidence/seed_only/verify_aggregate_arithmetic.py.
NumPy PCG64 regenerates randomized references from exported unit effects.
"""

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
CITIES = ["zhengzhou", "beijing"]
GROUPS = ["0", "1", "2", "all"]
SLOTS = ["NEURAL-1", "NEURAL-2", "NEURAL-3"]


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"Aggregate mismatch: {a} != {b}")


def verify(data, *, check_archive=True):
    def read(name):
        with (data / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def inventory(name, columns, dimensions):
        rows = read(name)
        keys = [tuple(row[c] for c in columns) for row in rows]
        if len(keys) != len(set(keys)) or set(keys) != set(itertools.product(*dimensions)):
            raise ValueError(f"{name}: duplicate, missing or unexpected aggregate keys")
        return rows

    unit_keys = ["city", "outer_fold", "trajectory", "group"]
    unit_dims = [CITIES, list("01234"), list("12345"), GROUPS]
    rows = inventory(
        "fit_group_aggregates.csv",
        unit_keys + ["variant", "seed_slot"],
        unit_dims + [["FF-G0-1", "FF-G2-1"], SLOTS],
    )
    fits = {
        tuple(r[k] for k in unit_keys + ["variant", "seed_slot"]): float(r["C_g_pp"]) for r in rows
    }
    for r in rows:
        close(r["C_g_pp"], 100 * float(r["C_g"]))
    for city, fold, trajectory, variant, slot in itertools.product(
        CITIES, list("01234"), list("12345"), ["FF-G0-1", "FF-G2-1"], SLOTS
    ):
        close(
            sum(fits[city, fold, trajectory, g, variant, slot] for g in GROUPS[:3]),
            fits[city, fold, trajectory, "all", variant, slot],
        )
    tables = {}
    for prefix, extra, values in [
        ("seed_pair", "pair_id", list("012345")),
        ("graph_delta", "seed_slot", SLOTS + ["mean3"]),
        ("graph_leave_one_slot_out", "omitted_slot", SLOTS),
    ]:
        units = inventory(prefix + "_units.csv", unit_keys + [extra], unit_dims + [values])
        for r in units:
            key = tuple(r[k] for k in unit_keys)
            if prefix == "seed_pair":
                pairs = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
                i, j = pairs[int(r["pair_id"])]
                if (r["left_slot"], r["right_slot"]) != (SLOTS[i], SLOTS[j]):
                    raise ValueError("Ordered seed-pair identity changed")
                value = fits[*key, "FF-G0-1", SLOTS[i]] - fits[*key, "FF-G0-1", SLOTS[j]]
            else:
                use = SLOTS if prefix == "graph_delta" and r[extra] == "mean3" else [r[extra]]
                if prefix == "graph_leave_one_slot_out":
                    use = [s for s in SLOTS if s != r[extra]]
                value = math.fsum(
                    fits[*key, "FF-G2-1", s] - fits[*key, "FF-G0-1", s] for s in use
                ) / len(use)
            close(r["effect_pp"], value)
        folds = inventory(
            prefix + "_folds.csv",
            ["city", "outer_fold", "group", extra],
            [CITIES, list("01234"), GROUPS, values],
        )
        cities = inventory(prefix + "_city.csv", ["city", "group", extra], [CITIES, GROUPS, values])
        for source, target, keys in [
            (units, folds, ["city", "outer_fold", "group", extra]),
            (folds, cities, ["city", "group", extra]),
        ]:
            accum = defaultdict(list)
            for r in source:
                accum[tuple(r[k] for k in keys)].append(float(r["effect_pp"]))
            for r in target:
                v = accum[tuple(r[k] for k in keys)]
                close(len(v), 5)
                close(r["effect_pp"], math.fsum(v) / 5)
                if "sign" in r:
                    close(r["sign"], np.sign(float(r["effect_pp"])))
                else:
                    close(r["positive_folds"], sum(x > 0 for x in v))
                    close(r["negative_folds"], sum(x < 0 for x in v))
                    close(r["zero_folds"], sum(x == 0 for x in v))
        tables[prefix] = (units, folds, cities)
    summaries = inventory("observed_vs_seed_references.csv", ["city", "group"], [CITIES, GROUPS])
    references = {(r["city"], r["group"]): r for r in summaries}
    pair_units, _, pair_cities = tables["seed_pair"]
    _, graph_folds, graph_cities = tables["graph_delta"]
    observed = {
        (r["city"], r["group"]): r["effect_pp"] for r in graph_cities if r["seed_slot"] == "mean3"
    }
    rng = np.random.default_rng(20261029)
    for city in CITIES:
        choices = rng.integers(0, 6, size=(10000, 5, 5))
        for group in GROUPS:
            indexed = {
                (int(r["outer_fold"]), int(r["trajectory"]), int(r["pair_id"])): float(
                    r["effect_pp"]
                )
                for r in pair_units
                if r["city"] == city and r["group"] == group
            }
            cube = np.array(
                [indexed[f, t, p] for f in range(5) for t in range(1, 6) for p in range(6)]
            ).reshape(5, 5, 6)
            draws = (
                cube[np.arange(5)[None, :, None], np.arange(5)[None, None, :], choices]
                .mean(axis=2)
                .mean(axis=1)
            )
            fixed = [
                float(r["effect_pp"])
                for r in pair_cities
                if r["city"] == city and r["group"] == group
            ]
            row = references[city, group]
            close(row["observed_graph_effect_pp"], observed[city, group])
            for name, value in {
                "fixed_six_min_pp": min(fixed),
                "fixed_six_max_pp": max(fixed),
                "random_q025_pp": np.quantile(draws, 0.025),
                "random_q975_pp": np.quantile(draws, 0.975),
                "random_min_pp": draws.min(),
                "random_max_pp": draws.max(),
                "random_mean_pp": draws.mean(),
                "random_sd_pp": draws.std(ddof=1),
                "draws": 10000,
                "folds": 5,
                "revelations_per_fold": 5,
            }.items():
                close(row[name], value)
            for column, low, high in [
                ("observed_vs_fixed_envelope", min(fixed), max(fixed)),
                (
                    "observed_vs_random_central_range",
                    np.quantile(draws, 0.025),
                    np.quantile(draws, 0.975),
                ),
                ("observed_vs_random_full_range", draws.min(), draws.max()),
            ]:
                effect = float(observed[city, group])
                location = "below" if effect < low else "above" if effect > high else "inside"
                if row[column] != location:
                    raise ValueError(f"Incorrect reference location in {column}")
    archived = []
    if check_archive:
        archived = inventory(
            "archived_graph_validation.csv",
            ["city", "outer_fold", "group"],
            [CITIES, list("01234"), GROUPS],
        )
        graph = {
            (r["city"], r["outer_fold"], r["group"]): r["effect_pp"]
            for r in graph_folds
            if r["seed_slot"] == "mean3"
        }
        for r in archived:
            close(r["effect_pp"], r["archived_pp"])
            close(r["effect_pp"], graph[r["city"], r["outer_fold"], r["group"]])
    return {
        "status": "passed",
        "fit_group_aggregates": len(fits),
        "ordered_unit_pair_rows": len(pair_units),
        "reference_rows": len(summaries),
        "archived_graph_folds": len(archived),
        "random_draws_per_city_group": 10000,
        "tolerance_pp": 1e-10,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, help="Optional verification JSON file.")
    parser.add_argument(
        "--skip-archive-check",
        action="store_true",
        help="Verify fresh-run aggregates without a historical graph comparison.",
    )
    args = parser.parse_args()
    result = verify(args.data_dir, check_archive=not args.skip_archive_check)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
