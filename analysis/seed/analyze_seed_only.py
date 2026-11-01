"""Aggregate-only seed sensitivity from frozen existing predictions; never trains.

Adapted from manuscript evidence/seed_only/analyze_seed_only.py.
The original ranking, pairing, seed and aggregation algorithms are retained."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parents[1]
sys.path[:0] = [str(PROJECT / "scripts"), str(PROJECT / "src")]

from analyze_final_review_evaluation import positive_statistics
from analyze_reviewer_training_structure import structure_inputs
from summarize_reviewer_training import (
    check_identity,
    logical_run_path,
    metrics_and_signature,
    require,
    validate_selections,
    validated_seed_frames,
    verify_provenance,
)

from floodgnn_ceus.io import sha256_file
from floodgnn_ceus.training import seed_from_name

ROOT: Path
OUT: Path
PLAN: Path
SEED_ARCHIVE: Path | None
STRUCTURE_ARCHIVE: Path | None

CITIES = ["zhengzhou", "beijing"]
VARIANTS = ["FF-G0-1", "FF-G2-1"]
SLOTS = ["NEURAL-1", "NEURAL-2", "NEURAL-3"]
GROUPS = ["0", "1", "2", "all"]
PAIRS = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
UNIT = ["city", "outer_fold", "trajectory"]


def save_json(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def group_terms(terms, membership):
    require(len(terms) == len(membership), "misaligned positive membership")
    require(set(membership).issubset({"0", "1", "2"}), "unexpected held-out L group")
    values = np.array([terms[membership == g].sum() for g in GROUPS[:3]] + [terms.sum()])
    require(np.isclose(values[:3].sum(), values[3], atol=1e-12, rtol=0), "additivity")
    return values


def synthetic_tests():
    checks = []

    def passed(name):
        checks.append(name)

    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.9, 0.3, 0.1])
    member = np.array(["0", "2"])
    a = positive_statistics(y, s)
    np.testing.assert_allclose(a[:, 0], [0.25, 1 / 3], rtol=0, atol=1e-15)
    np.testing.assert_allclose(a[:, 0].sum(), 7 / 12, rtol=0, atol=1e-15)
    passed("inclusive exact mixed-class tie thresholds and independent hand AP")
    c = positive_statistics(y, np.ones(4))
    np.testing.assert_allclose(c, [[0.25, 0.5, 0.5], [0.25, 0.5, 0.5]], rtol=0, atol=0)
    passed("constant ranking AP=prevalence; midpoint half credit")
    original = group_terms(a[:, 0], member)
    require(original[1] == 0, "empty group must be zero")
    np.testing.assert_allclose(original[:3].sum(), original[-1], atol=1e-15)
    passed("all-positive normalization, empty group, complete-domain additivity")
    perm = np.array([2, 1, 0, 3])
    b = positive_statistics(y[perm], s[perm])
    np.testing.assert_allclose(group_terms(b[:, 0], member[::-1]), original, atol=1e-15)
    passed("candidate permutation leaves group outcome invariant")
    np.testing.assert_array_equal(original - original, np.zeros(4))
    delta = original - group_terms(c[:, 0], member)
    np.testing.assert_array_equal(delta, -(group_terms(c[:, 0], member) - original))
    passed("identity and ordered-pair antisymmetry")
    for bad_y, bad_s in [(y, [1, 2]), (y, [1, 2, np.nan, 4]), ([1] * 4, [1] * 4)]:
        try:
            positive_statistics(np.asarray(bad_y), np.asarray(bad_s))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    try:
        group_terms(a[:, 0], np.array(["0", "unknown"]))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid group accepted")
    passed("misalignment, nonfinite scores, absent class, unknown group rejected")
    return checks


def analyze(run, grid, outer, structure, hashes):
    plan_hash = sha256_file(PLAN)
    require(PLAN.with_suffix(".sha256").read_text().split()[0] == plan_hash, "frozen plan changed")
    checks = synthetic_tests()
    lookup = structure.set_index(["city", "outer_fold", "grid_id"])
    require(lookup.index.is_unique, "duplicate structural rows")
    targets = [(t, p) for t, p in outer if t.realization_id in VARIANTS]
    require(len(targets) == 300, "expected 300 target fits")
    signatures, positives, domains, audits = {}, {}, {}, {}
    fit_rows, seed_rows, tie_rows = [], [], []
    max_ap_error = 0.0
    seed_archive = None
    if SEED_ARCHIVE is not None:
        seed_archive = pd.read_csv(SEED_ARCHIVE, dtype=str)
        hashes[str(SEED_ARCHIVE)] = sha256_file(SEED_ARCHIVE)
    for index, (task, p) in enumerate(targets):
        frame = pd.read_parquet(p)
        require(not frame.isna().any().any(), "missing prediction fields")
        check_identity(frame, task)
        receipt = json.loads(p.with_name("status.json").read_text())
        unit = (task.city, int(task.outer_fold), task.trajectory)
        fit_key = (*unit, task.realization_id)
        audit = receipt["fit_audit"]
        # These receipts certify identical fit rows, preprocessing, feature matrices and edges.
        require(audits.setdefault(fit_key, audit) == audit, "seed-only fit inputs changed")
        for slot, rows in validated_seed_frames(frame, False, task.seed):
            metrics, signature, positive, domain = metrics_and_signature(rows)
            require(signatures.setdefault(unit, signature) == signature, "unpaired banks/domain")
            require(positives.setdefault(unit[:2], positive) == positive, "positives changed")
            require(domains.setdefault(unit[:2], domain) == domain, "candidate domain changed")
            full = rows.loc[rows.bank_id.eq("NA")].sort_values("grid_id")
            labels = full.grid_id.isin(positive).to_numpy(np.int8)
            stats = positive_statistics(labels, full.score.to_numpy())
            err = abs(stats[:, 0].sum() - metrics["full_background_ap"])
            max_ap_error = max(max_ap_error, float(err))
            require(err < 1e-12, "AP does not match independent ranking helper")
            all_member = lookup.loc[[(unit[0], unit[1], node) for node in full.grid_id], "L"]
            member = all_member.to_numpy()[labels.astype(bool)]
            terms = group_terms(stats[:, 0], member)
            for g, v in zip(GROUPS, terms, strict=True):
                fit_rows.append(
                    dict(zip(UNIT, unit))
                    | {
                        "variant": task.realization_id,
                        "seed_slot": slot,
                        "group": g,
                        "C_g": float(v),
                        "C_g_pp": float(v * 100),
                        "positive_cells": len(member) if g == "all" else int((member == g).sum()),
                        "candidate_cells": len(labels),
                        "total_positive_cells": int(labels.sum()),
                    }
                )
            unique, counts = np.unique(full.score.to_numpy(), return_counts=True)
            tie_rows.append(
                dict(zip(UNIT, unit))
                | {
                    "variant": task.realization_id,
                    "seed_slot": slot,
                    "candidate_cells": len(labels),
                    "unique_scores": len(unique),
                    "exact_tie_blocks": int((counts > 1).sum()),
                    "candidates_in_exact_ties": int(counts[counts > 1].sum()),
                    "max_tie_block": int(counts.max()),
                    "constant_ranking": bool(len(unique) == 1),
                }
            )
            namespace = grid["seeds"]["namespace"].format(
                protocol_id=task.protocol_id,
                experiment_id="E1",
                city=task.city,
                outer_fold=task.outer_fold,
                budget=task.budget,
                trajectory_id=task.trajectory,
                model=task.model,
                config=task.config,
                seed_id=task.seed,
            )
            seed = seed_from_name(namespace)
            expected_seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:16], "big")
            require(seed == expected_seed, "seed derivation differs")
            if seed_archive is not None:
                prior = seed_archive.loc[
                    seed_archive.city.eq(task.city)
                    & seed_archive.outer_fold.eq(task.outer_fold)
                    & seed_archive.trajectory.eq(task.trajectory)
                    & seed_archive.variant.eq(task.realization_id)
                    & seed_archive.seed_id.eq(slot)
                ]
                require(len(prior) == 1, "missing archived actual seed")
                require(
                    prior.iloc[0].seed_uint128 == str(seed)
                    and prior.iloc[0].torch_mod_2_63 == str(seed % 2**63),
                    "actual seed mismatch",
                )
            saved = json.loads(
                (
                    run
                    / "selections"
                    / f"{task.city}-f{task.outer_fold}-b100-t{task.trajectory}.json"
                ).read_text()
            )
            chosen = saved["variants"][task.realization_id]
            require(chosen["config"] == task.config, "selected config changed")
            seed_rows.append(
                dict(zip(UNIT, unit))
                | {
                    "variant": task.realization_id,
                    "seed_slot": slot,
                    "selected_config": task.config,
                    "refit_epochs": chosen["refit_epochs"],
                    "namespace": namespace,
                    "seed_uint128": str(seed),
                    "torch_mod_2_63": str(seed % 2**63),
                    "fit_row_count": audit["fit_row_count"],
                    "fit_rows_sha256": audit["fit_rows_sha256"],
                    "preprocessing_sha256": audit["preprocessing_sha256"],
                    "matrix_invariants_sha256": hashlib.sha256(
                        json.dumps(audit["matrices"], sort_keys=True).encode()
                    ).hexdigest(),
                    "candidate_signature_sha256": signature,
                }
            )
        if index % 50 == 49:
            print(f"Loaded and validated {index + 1}/300 target fits", flush=True)
    for city, expected in [("zhengzhou", 791), ("beijing", 296)]:
        psets = [positives[city, f] for f in range(5)]
        require(
            sum(map(len, psets)) == len(set().union(*psets)) == expected,
            "positive folds overlap or counts differ",
        )
        dsets = [domains[city, f] for f in range(5)]
        require(sum(map(len, dsets)) == len(set().union(*dsets)), "candidate folds overlap")
    for unit in signatures:
        a, b = [audits[(*unit, v)] for v in VARIANTS]
        require(
            a["fit_rows_sha256"] == b["fit_rows_sha256"], "graph comparison training sample differs"
        )
        require(a["preprocessing_sha256"] == b["preprocessing_sha256"], "preprocessing differs")
    fits, seeds, ties = map(pd.DataFrame, (fit_rows, seed_rows, tie_rows))
    require(
        not fits.duplicated(UNIT + ["variant", "seed_slot", "group"]).any(), "duplicate fit groups"
    )
    for _, part in seeds.groupby(UNIT + ["variant"]):
        require(sorted(part.seed_slot) == SLOTS, "seed inventory incomplete")
        require(part.torch_mod_2_63.nunique() == 3, "actual seeds collide")
        for col in [
            "selected_config",
            "refit_epochs",
            "fit_rows_sha256",
            "preprocessing_sha256",
            "matrix_invariants_sha256",
        ]:
            require(part[col].nunique() == 1, f"seed-only invariant changed: {col}")
    seed_join = seeds.loc[seeds.variant.eq(VARIANTS[0])].merge(
        seeds.loc[seeds.variant.eq(VARIANTS[1])],
        on=UNIT + ["seed_slot"],
        validate="1:1",
        suffixes=("_own", "_D8"),
    )
    seed_join["same_actual_seed"] = seed_join.torch_mod_2_63_own.eq(seed_join.torch_mod_2_63_D8)
    seed_join["same_selected_config"] = seed_join.selected_config_own.eq(
        seed_join.selected_config_D8
    )
    require(
        (seed_join.same_actual_seed == seed_join.same_selected_config).all(),
        "cross-variant seed namespace consistency failed",
    )
    pairs, graphs, loo = [], [], []
    for unit, part in fits.groupby(UNIT, sort=True):
        meta = dict(zip(UNIT, unit))
        own = (
            part.loc[part.variant.eq(VARIANTS[0])]
            .pivot(index="seed_slot", columns="group", values="C_g")
            .reindex(index=SLOTS, columns=GROUPS)
            .to_numpy()
        )
        d8 = (
            part.loc[part.variant.eq(VARIANTS[1])]
            .pivot(index="seed_slot", columns="group", values="C_g")
            .reindex(index=SLOTS, columns=GROUPS)
            .to_numpy()
        )
        require(np.isfinite(own).all() and np.isfinite(d8).all(), "incomplete matrix")
        for pair_id, (i, j) in enumerate(PAIRS):
            for gi, g in enumerate(GROUPS):
                pairs.append(
                    meta
                    | {
                        "group": g,
                        "pair_id": pair_id,
                        "left_slot": SLOTS[i],
                        "right_slot": SLOTS[j],
                        "orientation": "unordered_canonical" if pair_id < 3 else "reverse",
                        "effect_pp": float(100 * (own[i, gi] - own[j, gi])),
                    }
                )
        delta = 100 * (d8 - own)
        for gi, g in enumerate(GROUPS):
            for si, slot in enumerate(SLOTS):
                graphs.append(
                    meta | {"group": g, "seed_slot": slot, "effect_pp": float(delta[si, gi])}
                )
            graphs.append(
                meta | {"group": g, "seed_slot": "mean3", "effect_pp": float(delta[:, gi].mean())}
            )
            for omitted, slot in enumerate(SLOTS):
                loo.append(
                    meta
                    | {
                        "group": g,
                        "omitted_slot": slot,
                        "effect_pp": float(np.delete(delta[:, gi], omitted).mean()),
                    }
                )
    pairs, graphs, loo = map(pd.DataFrame, (pairs, graphs, loo))

    def hierarchy(df, extra):
        folds = df.groupby(["city", "outer_fold", "group"] + extra, as_index=False).effect_pp.mean()
        city = folds.groupby(["city", "group"] + extra, as_index=False).agg(
            effect_pp=("effect_pp", "mean"),
            positive_folds=("effect_pp", lambda v: int((v > 0).sum())),
            negative_folds=("effect_pp", lambda v: int((v < 0).sum())),
            zero_folds=("effect_pp", lambda v: int((v == 0).sum())),
        )
        folds["sign"] = np.sign(folds.effect_pp).astype(int)
        return folds, city

    pair_folds, pair_city = hierarchy(pairs, ["pair_id", "left_slot", "right_slot", "orientation"])
    graph_folds, graph_city = hierarchy(graphs, ["seed_slot"])
    loo_folds, loo_city = hierarchy(loo, ["omitted_slot"])
    for df, keys in [
        (pairs, UNIT + ["pair_id"]),
        (pair_folds, ["city", "outer_fold", "pair_id"]),
        (graphs, UNIT + ["seed_slot"]),
        (graph_folds, ["city", "outer_fold", "seed_slot"]),
        (loo, UNIT + ["omitted_slot"]),
        (loo_folds, ["city", "outer_fold", "omitted_slot"]),
    ]:
        sums = df.pivot(index=keys, columns="group", values="effect_pp")
        np.testing.assert_allclose(
            sums[["0", "1", "2"]].sum(axis=1), sums["all"], atol=1e-10, rtol=0
        )
    for _, part in pairs.groupby(UNIT + ["group"]):
        vals = part.set_index("pair_id").effect_pp
        np.testing.assert_allclose(vals.loc[0] + vals.loc[2], vals.loc[1], atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            vals.loc[[0, 1, 2]].to_numpy(), -vals.loc[[3, 4, 5]].to_numpy(), atol=0, rtol=0
        )
    check = None
    if STRUCTURE_ARCHIVE is not None:
        archive = STRUCTURE_ARCHIVE
        hashes[str(archive)] = sha256_file(archive)
        old = pd.read_csv(archive, dtype={"group": str})
        old = old.loc[
            old.contrast.eq("FF-G2-1 - FF-G0-1")
            & old.metric.eq("ap_contribution")
            & old.analysis.isin(["L", "overall"])
            & old.condition.eq("full_domain")
        ]
        check = graph_folds.loc[graph_folds.seed_slot.eq("mean3")].merge(
            old[["city", "outer_fold", "group", "value"]],
            on=["city", "outer_fold", "group"],
            validate="1:1",
        )
        require(len(check) == 40, "archive matrix incomplete")
        check["archived_pp"] = 100 * check.value
        check["absolute_error_pp"] = abs(check.effect_pp - check.archived_pp)
        require(check.absolute_error_pp.max() < 1e-10, "archived graph effect not reproduced")
    obs = graph_city.loc[graph_city.seed_slot.eq("mean3")].set_index(["city", "group"]).effect_pp
    rng = np.random.default_rng(20261029)
    random_rows, random_folds, comparison, dispersion = [], [], [], []
    for city in CITIES:
        choices = rng.integers(0, 6, size=(10000, 5, 5))
        for g in GROUPS:
            cube = (
                pairs.loc[pairs.city.eq(city) & pairs.group.eq(g)]
                .pivot(index=["outer_fold", "trajectory"], columns="pair_id", values="effect_pp")
                .sort_index()
                .to_numpy()
                .reshape(5, 5, 6)
            )
            require(cube.shape == (5, 5, 6), "random-reference units incomplete")
            selected = np.take_along_axis(
                np.broadcast_to(cube, (10000, 5, 5, 6)), choices[..., None], axis=3
            )[..., 0]
            fold_draws = selected.mean(axis=2)
            draws = fold_draws.mean(axis=1)
            for draw, value in enumerate(draws):
                random_rows.append(
                    {"city": city, "group": g, "draw": draw, "effect_pp": float(value)}
                )
                for fold in range(5):
                    random_folds.append(
                        {
                            "city": city,
                            "group": g,
                            "draw": draw,
                            "outer_fold": fold,
                            "effect_pp": float(fold_draws[draw, fold]),
                        }
                    )
            fixed = pair_city.loc[
                pair_city.city.eq(city) & pair_city.group.eq(g)
            ].effect_pp.to_numpy()
            q025, q975 = np.quantile(draws, [0.025, 0.975])
            observed = float(obs.loc[city, g])
            location = lambda low, high, observed=observed: (
                "below" if observed < low else "above" if observed > high else "inside"
            )
            comparison.append(
                {
                    "city": city,
                    "group": g,
                    "observed_graph_effect_pp": observed,
                    "fixed_six_min_pp": float(fixed.min()),
                    "fixed_six_max_pp": float(fixed.max()),
                    "observed_vs_fixed_envelope": location(fixed.min(), fixed.max()),
                    "random_q025_pp": float(q025),
                    "random_q975_pp": float(q975),
                    "observed_vs_random_central_range": location(q025, q975),
                    "random_min_pp": float(draws.min()),
                    "random_max_pp": float(draws.max()),
                    "observed_vs_random_full_range": location(draws.min(), draws.max()),
                    "random_mean_pp": float(draws.mean()),
                    "random_sd_pp": float(draws.std(ddof=1)),
                    "draws": 10000,
                    "folds": 5,
                    "revelations_per_fold": 5,
                }
            )
            for level, values in [
                ("raw_unit_unordered75", cube[:, :, :3].ravel()),
                ("raw_unit_signed150", cube.ravel()),
                ("fold_unordered15", cube[:, :, :3].mean(axis=1).ravel()),
                ("city_unordered3", cube[:, :, :3].mean(axis=(0, 1))),
                ("city_signed6", fixed),
            ]:
                dispersion.append(
                    {
                        "city": city,
                        "group": g,
                        "level": level,
                        "dependent_n": len(values),
                        "mean_pp": float(values.mean()),
                        "sd_pp": float(values.std(ddof=1)),
                        "rms_pp": float(np.sqrt(np.mean(values**2))),
                        "min_pp": float(values.min()),
                        "max_pp": float(values.max()),
                        "q025_pp": float(np.quantile(values, 0.025)),
                        "q975_pp": float(np.quantile(values, 0.975)),
                    }
                )
    random_rows, random_folds, comparison, dispersion = map(
        pd.DataFrame, (random_rows, random_folds, comparison, dispersion)
    )
    summed = random_rows.pivot(index=["city", "draw"], columns="group", values="effect_pp")
    np.testing.assert_allclose(
        summed[["0", "1", "2"]].sum(axis=1), summed["all"], atol=1e-10, rtol=0
    )
    tables = {
        "fit_group_aggregates.csv": fits,
        "seed_identity_and_invariants.csv": seeds,
        "cross_variant_seed_consistency.csv": seed_join,
        "exact_ties_by_fit.csv": ties,
        "seed_pair_units.csv": pairs,
        "seed_pair_folds.csv": pair_folds,
        "seed_pair_city.csv": pair_city,
        "graph_delta_units.csv": graphs,
        "graph_delta_folds.csv": graph_folds,
        "graph_delta_city.csv": graph_city,
        "graph_leave_one_slot_out_units.csv": loo,
        "graph_leave_one_slot_out_folds.csv": loo_folds,
        "graph_leave_one_slot_out_city.csv": loo_city,
        "randomized_pair_city_draws.csv": random_rows,
        "randomized_pair_fold_draws.csv": random_folds,
        "observed_vs_seed_references.csv": comparison,
        "dispersion_by_aggregation_level.csv": dispersion,
    }
    if check is not None:
        tables["archived_graph_validation.csv"] = check
    for name, table in tables.items():
        require(
            not any(
                c in table for c in ["grid_id", "score", "sample_role", "center_x", "center_y"]
            ),
            "individual record field in export",
        )
        table.to_csv(OUT / name, index=False)
    validation = {
        "synthetic_tests_passed": checks,
        "target_fits": 300,
        "units": 50,
        "source_outer_receipts_verified": 1550,
        "source_inner_receipts_verified": 1250,
        "max_AP_identity_error": max_ap_error,
        "archived_graph_max_error_pp": (
            float(check.absolute_error_pp.max()) if check is not None else None
        ),
        "historical_seed_inventory_checked": seed_archive is not None,
        "all_aggregation_additivity": True,
        "pair_antisymmetry_and_cycle": True,
        "seed_invariants_all_passed": True,
        "cross_variant_same_actual_seed_slots": int(seed_join.same_actual_seed.sum()),
        "cross_variant_total_seed_slots": len(seed_join),
        "frozen_plan_sha256": plan_hash,
        "constant_rankings": int(ties.constant_ranking.sum()),
        "source_hashes": hashes,
        "table_rows": {n: len(t) for n, t in tables.items()},
    }
    save_json("validation.json", validation)
    print(comparison.to_string(index=False), flush=True)
    return tables, validation


def main():
    global ROOT, OUT, PLAN, SEED_ARCHIVE, STRUCTURE_ARCHIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Prepared project root containing configs and artifacts.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Existing verified run, absolute or relative to --root.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        default=BASE / "analysis_plan.json",
        help="Analysis plan with sibling .sha256 file; defaults to the portable release plan.",
    )
    parser.add_argument(
        "--seed-archive",
        type=Path,
        help="Optional historical seed inventory, e.g. provenance/primary_refit_seeds.csv.",
    )
    parser.add_argument(
        "--structure-archive",
        type=Path,
        help="Optional historical graph-fold CSV to reproduce, e.g. plots/data/fig4_structure_folds.csv.",
    )
    args = parser.parse_args()
    ROOT, OUT = args.root.resolve(), args.output.resolve()
    PLAN = args.plan.resolve()
    SEED_ARCHIVE = args.seed_archive.resolve() if args.seed_archive else None
    STRUCTURE_ARCHIVE = args.structure_archive.resolve() if args.structure_archive else None
    OUT.mkdir(parents=True, exist_ok=True)
    run = logical_run_path(
        ROOT, args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    )
    hashes = {}
    config, grid = verify_provenance(ROOT, run, hashes)
    _, outer, _ = validate_selections(ROOT, run, config, grid, hashes)
    structure, _ = structure_inputs(ROOT, hashes)
    return analyze(run, grid, outer, structure, hashes)


if __name__ == "__main__":
    main()
