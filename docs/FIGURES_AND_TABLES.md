# Figures and tables

## Figure 1

The portable renderer reads a separately prepared project and stages its map
assets beneath the explicit output directory.

```sh
python plots/make_study_area.py --root ../prepared-project --output reproduced/figure1
```

For each city, `--root` must contain `data/interim/<city>/` with
`final_scorable_grid.parquet`, `terrain_features.parquet`,
`evaluation_design/outer_partitions.parquet`, `d8_edges.parquet` and
`admin_aoi.parquet`. The renderer reads selected grid, elevation, partition,
edge and geometry columns. It checks cell inventories, grid alignment, fold
coverage, graph endpoints and aggregate support shares before exporting.

Only `plots/data/study_area_depth_shares.csv` is bundled for this figure.
`--support-table` can override it. Whole-domain all-cell support shares are
recomputed from D8 edges. Recorded-positive shares are read from the supplied
aggregate table. Generated NPZ arrays, GeoJSON boundaries and a manifest with
source-relative paths and hashes stay in `<output>/data/`. Keep these local
assets under the ignored `reproduced/` directory. Figure outputs are
`fig1_study_area.pdf`, `.svg` and `.png` in `<output>/`.

The six panels retain the v8 study-area design and calculations. The source is
`figure_code/make_study_area.py` in
`flood_graph_manuscript_EN_portrait_v8_20261030`. The release renderer has no
external panel-audit helper dependency. Relative command-line paths resolve
against the current working directory, and the bundled aggregate resolves
relative to the script.

## Figure 2

The framework diagram is supplied as the static editable PowerPoint source
`plots/framework/Figure2_editable.pptx` and its PDF export
`plots/framework/Figure2.pdf`. Open the PPTX to edit the diagram and export an
updated PDF from the presentation application.

## Figure 3

Run from the release root after installing the `figures` dependency group.

```sh
python plots/make_result_figures.py --output reproduced/figures
```

The CLI resolves bundled inputs relative to its own file, so it also runs from
another working directory. `--data-dir` can select another directory with the same
four CSV schemas. The output directory receives PDF, SVG and 600-dpi PNG files
for Figures 3 and 4, plus `results_point_verification.json`.

`plots/data/fig3_fold_contrasts.csv` supplies all 50 outer-fold observations.
`fig3_matched_contrasts.csv` supplies ten summaries covering five contrasts for
Zhengzhou and Beijing. The metric is `sampled_pb_ap`. Differences are multiplied
by 100 to obtain percentage points. The five contrasts compare one-step fixed
D8 propagation with own attributes and adjacency, two-layer GraphSAGE D8 with
own attributes and adjacency, and fixed propagation with GraphSAGE at one hop.
The renderer verifies fold membership, percentage-point conversions and means.

## Figure 4

`plots/data/fig4_structure_folds.csv` contains 80 aggregate values and
`fig4_structure_summary.csv` contains 16 summaries. Both use the full-domain
`FF-G2-1 - FF-G0-1` contrast. The panels show signed AP contributions, their
fold-level variability and background-rank changes for the same support groups.
Sixty group-level fold points are plotted. Twenty overall values support checks.

Group `0` means no incoming support, `1` means incoming support without a
 two-edge upstream path, and `2` means a two-edge upstream path exists in the
held-out D8 graph. Group `all` denotes the full domain. AP contributions use
all positives as their common denominator and sum to the overall AP difference
within each fold. Mean background-rank changes use the original 0-to-1 rank scale
and are multiplied by 100 for plotting. Whiskers retain the supplied descriptive
fold-resampling percentiles.

## Tables 2 and 3

```sh
python scripts/reproduce_tables.py --output reproduced/tables
```

The default inputs are under `results/absolute_performance/`; `--source` can
select an alternative directory with the same aggregate schemas.

| Manuscript table | Generated file | Input and mapping |
| --- | --- | --- |
| Table 2 | `table2_absolute.csv` | `main_units.csv`, grouped by city and variant. Average three refit slots within each trajectory, five trajectories within each fold, then five folds. `label` supplies the model display name. |
| Table 3 | `table3_ablation.csv` | `ablation_absolute_folds.csv`, averaged over five folds by city, budget, method and feature set. `G0` maps to `own_AP`, `G2` to `D8_AP`, and their difference to `D8_minus_own`. |

Table 2 retains `sampled_pb_ap`, `full_background_ap`,
`full_weighted_1to10_ap` and `recall_top_10pct`. Table 3 uses `sampled_pb_ap`.
The generated CSVs retain the original 0-to-1 metric scale. Multiply AP
differences by 100 when presenting percentage points. The script verifies
recomputed values against the released fold and summary tables and also writes
`main_fold_metrics.csv` and `ablation_all_models.csv`.

## Table 4 and seed verification

```sh
python analysis/seed/verify_aggregate_arithmetic.py --output reproduced/seed/verification.json
```

The verifier requires Python and NumPy and reads only the bundled aggregate
CSVs. `--data-dir` selects an alternative aggregate directory. Without `--output`,
the verification result is printed to standard output.

Table 4 comes from `analysis/seed/data/observed_vs_seed_references.csv`, using
Beijing then Zhengzhou and groups `0`, `1`, `2` within each city. Format numbers
to three decimal places and include a sign for positive effects and endpoints.

| Table 4 column | CSV field or transformation |
| --- | --- |
| City | `city`, capitalized |
| Support | `group` mapped to No upstream, One hop, Multi-hop |
| D8 minus own | `observed_graph_effect_pp` |
| Fixed-pair range | `fixed_six_min_pp`, `fixed_six_max_pp` |
| Randomized range | `random_q025_pp`, `random_q975_pp` |

The `all` rows provide total-AP checks. The graph contrast averages three refits
per model. Fixed references span six signed own-model seed-pair city means.
Randomized references use 10,000 draws from NumPy PCG64 with seed 20261029,
independently choosing an ordered pair per fold-revelation unit. City order is
Zhengzhou then Beijing, unit order is ascending fold then trajectory, and pair
order is 1−2, 1−3, 2−3, 2−1, 3−1, 3−2. Choices are shared across groups. Each
reference averages five revelations within each fold and then five folds equally.
The ranges describe empirical seed sensitivity. The graph and reference
estimators use different seed averaging.

The independent verifier checks complete unique key inventories, per-fit group
additivity, ordered pair identities, graph and leave-one-slot-out arithmetic,
two-stage equal-weight means, fold signs, all eight reference summaries and
40 archived graph-fold effects. It regenerates randomized draws in memory from
unit aggregates. Numeric comparisons use an absolute tolerance of 1e-10 pp.

`fit_group_aggregates.csv` supplies 1,200 per-fit group outcomes. The
`seed_pair_*`, `graph_delta_*` and `graph_leave_one_slot_out_*` CSVs retain unit,
fold and city aggregates. `archived_graph_validation.csv` links the recomputed
graph effects to the archived decomposition. These files contain group-level
outcomes and counts. Location-level labels, predictions and geometry are excluded.

## Optional analysis of existing predictions

The retained analysis entry point accepts a privately prepared project and run.
The command uses the published plan by default.

```sh
python analysis/seed/analyze_seed_only.py --root ../prepared-project --run-root artifacts/runs/reviewer-revision-v1-0d119f16fcb9 --output reproduced/seed-analysis
```

`--run-root` is resolved against `--root` when relative. Other explicit relative
paths resolve against the current working directory. Shared helpers come from
this release's `scripts/` and `src/`. The default plan is
`analysis/seed/analysis_plan.json`, verified against `analysis_plan.sha256`.
It preserves the original scientific settings and freeze metadata, replaces
machine-specific operational restrictions, and records the original file's
SHA-256 and the changes in `portable_release_provenance`.

Verify fresh-run aggregates with the historical comparison explicitly disabled.

```sh
python analysis/seed/verify_aggregate_arithmetic.py --data-dir reproduced/seed-analysis --skip-archive-check
```

Historical comparisons are opt-in. For the original archived run, add the
published inventories to require exact seed and graph-effect agreement.

```sh
python analysis/seed/analyze_seed_only.py --root ../prepared-project --run-root artifacts/runs/reviewer-revision-v1-0d119f16fcb9 --output reproduced/seed-archive-check --seed-archive provenance/primary_refit_seeds.csv --structure-archive plots/data/fig4_structure_folds.csv
```

Fresh runs retain internal receipt, selection, seed-derivation, pairing and
structural checks. Their effects can vary with training. Historical comparisons
run only when their corresponding archive argument is supplied. Validation
records whether the seed inventory was checked and uses a null archive error
when graph comparison was omitted. `--plan` accepts another plan with a sibling
`.sha256` file. The ranking, tie handling, pairing, randomization and aggregation
algorithms are preserved. Full prediction analysis requires the prepared inputs
and prediction receipts. Release tests cover synthetic invariants, CLI defaults,
the portable plan digest and optional archive arguments.

## Source attribution and validation

The figure renderer is adapted from `figure_code/make_result_figures.py` in
`flood_graph_manuscript_EN_portrait_v8_20261030`. Its four CSVs are copied from
that package's `figure_data/`. Seed scripts and selected CSVs originate in the
same package's `evidence/seed_only/`. Attribution is also retained in script
docstrings. Public-release changes parameterize destinations, remove source
refresh and private panel-audit dependencies, and strengthen aggregate checks.

```sh
python -m pytest tests/test_release_seed.py -q
ruff check plots analysis/seed tests/test_release_seed.py
ruff format --check plots analysis/seed tests/test_release_seed.py
```

Tests include corrupted, missing and duplicate aggregate rows, synthetic ranking
invariants and figure generation from a different working directory. Font fallback
can produce small visual differences across platforms.
