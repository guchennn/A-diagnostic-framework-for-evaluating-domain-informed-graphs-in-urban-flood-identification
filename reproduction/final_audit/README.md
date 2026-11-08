# Final operational and structural audit

This directory publishes the final manuscript's operation ledger, dual-reference local
increments (Table 5), full-to-held-out group migration (Table 6), aggregate source data
and analysis code. It contains no individual records, coordinates, scores or licensed
source datasets. The original audit used 150 paired city/fold/draw/refit units.

## Check the included evidence

From the repository root, using any standard Python 3 interpreter:

```sh
python -I -S reproduction/final_audit/scripts/verify_aggregates.py
```

No third-party package or original data is needed. This validates file hashes, transition
conservation, no upward migration, paired-reference identities, equal-fold means, additive
AP accounting and total-AP invariance under group relabelling. It verifies supplied
aggregate evidence, not reconstruction of predictions from those aggregates.

## Recompute from prepared local inputs

Install the repository's Python 3.12 requirements first. Use a separate output directory;
these commands never fit models or download data.

```sh
python -m pip install -r requirements.txt
python reproduction/final_audit/group_migration/test_migration.py
python reproduction/final_audit/local_references/test_local_references.py
python reproduction/final_audit/group_migration/audit_migration.py --project-root . --output-dir reproduced/final_audit/group_migration
python reproduction/final_audit/local_references/analyze_local_references.py --project-root . --run-root artifacts/runs/reviewer-revision-v1-RUN_ID --output-dir reproduced/final_audit/local_references
```

Replace RUN_ID with the namespace produced by the repository's training workflow, as
explained in [REPRODUCE.md](../../REPRODUCE.md). Migration requires each city's prepared
`final_scorable_grid.parquet`, `final_label_ledger.parquet`, `graphs/base_edges.parquet`,
`graphs/outer_edges.parquet` and `evaluation_design/outer_partitions.parquet`, plus its
protocol lock. Local increments additionally require a complete frozen native run,
`plan.json`, `local_environment_lock.json`, performance-blind probe, original input/source
files, successful task receipts and OOF predictions. Existing repository helpers verify
provenance, selection and pairing before calculating contrasts. `--project-root` selects
input artifacts; code imports always come from this checkout. Outputs contain aggregates.

AP contributions retain the complete test ranking and the same held-out D8 groups for
both own and adjacency references. Five draws and three refits are averaged within each
fold, then the five folds equally. Fold-resampling intervals are descriptive.

## Historical evidence versus a new run

The supplied CSVs are the audited manuscript results. Publication scripts reuse this
repository's native `artifacts/runs` protocol; historical execution used a distinct
`artifacts/remote_runs` backup protocol. Those protocols are not interchangeable.
The historical input and executed-analysis source hashes are preserved under
`provenance/`; `project_snapshot/` is an inspected-source snapshot (Python files use `.py.txt` to prevent accidental execution), not an installable
historical checkout. Its hash identity is separate from the publication code recorded
in `package_manifest.json`.

Exact replay of historical local increments requires the original research checkout,
frozen source/environment locks and original prediction products identified by those
manifests. These private artifacts are not distributed here. A new fit is not expected
to reproduce the historical means exactly.

For an explicit historical comparison, add `--verify-historical-baseline`. Migration
then requires and checks the original membership archive at
`artifacts/runtime/supp-mechanism-ab-20261007-r4/test_node_structure.parquet`.
Local-reference analysis also compares own-reference means to the included
`published_own_summary.csv`; use this only with matching historical inputs under a
compatible verified run protocol. The ordinary path always checks edge pruning, DAG
structure, group monotonicity, matched candidates, AP additivity and reference identities.
It does not require an archive or force newly fitted means to equal old results.

## Post-DSM preparation

These cumulative commands rebuild preparation in a working copy with existing DSM and
final-grid inputs. They were not executed by the final audit:

```sh
floodgnn prepare --city all --through terrain
floodgnn prepare --city all --through features
floodgnn prepare --city all --through design
```

Restore required source products through their providers first; do not rerun downloads
or labels for post-DSM work, and preserve the existing protocol lock. Preparation is
separate from frozen-prediction replay and from training.

## Files

- [Operational specification](Operational_specification.md) and
  [18-attribute ledger](feature_operations.csv): domains, units, windows and rules.
- `local_references/summary.csv`: dual-reference estimates and descriptive intervals.
- `local_references/group_definition_sensitivity.csv`: same held-out predictions,
  changed full-city versus held-out group labels; not an unpruned-model experiment.
- `group_migration/transition_matrix.csv`: each cell counted once in its own test fold;
  pooled counts are cell weighted, whereas performance estimates weight folds equally.
- `provenance/`: input/executed-source identities. Hashes do not distribute those inputs
  or establish that they are available in a public checkout.

Group stability does not imply unchanged upstream neighbourhoods or establish a causal
boundary effect. The adjacent controlled-validation package covers the earlier
synthetic diagnostic experiments.
