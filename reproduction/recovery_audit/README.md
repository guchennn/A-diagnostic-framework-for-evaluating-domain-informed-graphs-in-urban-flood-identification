# Recovery of historical training and replay records

This audit searched the local frozen artifacts and the `guchennn/gnn` and
`guchennn/ceus` repository branches on 8 November 2026. It adds recovered execution
metadata and FULL-budget fit/bank aggregates, while keeping missing records explicit.
All artifact paths below are relative to the original research checkout.

## FULL-positive attribute experiment

The manuscript's Zhengzhou FULL, 18/16/14-attribute comparison comes from
`artifacts/runtime/feature-ablation-reselect-20260925/`. The recovered records are:

- `full_budget_manifest.json`: experiment identity, attributes, graph controls and
  the three `NEURAL-*` slot names.
- `full_budget_inner_selection.csv`: six rows recording selected configurations
  across five folds for three attribute sets and two model families.
- `full_budget_fit_bank_metrics.csv`: 450 fit/bank AP summaries, with no individual
  observations or predictions. These recover all 60 fold/model/attribute means in
  the public FULL-budget table.

The 15 GraphSAGE selection units (five folds times three attribute sets) have no
saved best-inner epochs or outer-refit durations. Integer optimizer seeds or their
verified original construction are also absent for the 135 neural fits (15 units
times three graphs times three seed slots). A slot name is not an optimizer seed.
The separate earlier non-reselected ablation and formal E1 archives are different
experiments and cannot fill these fields. FULL exact-training replay therefore
remains blocked; the aggregate evidence is available for numerical verification.

## What the city records verify

`main_city_execution.json` extracts non-sensitive metadata from the frozen main
run `artifacts/remote_runs/reviewer-revision-v1-0d119f16fcb9/`. It records 50 outer
units, 1,250 successful inner tasks and 1,550 successful outer tasks, with no failed
tasks. The original main run used Python 3.12.11, PyTorch 2.8.0+cu128 and PyG 2.7.0.
The original ablation's Python 3.13.5/CUDA 12.9 environment is a different lineage.

| Verification level | Evidence |
| --- | --- |
| Original main city fits | Completed; execution counts, environment and executed-source digest recovered. |
| FULL aggregate recalculation | Recovered fit/bank AP values reproduce all 60 published fold/model/attribute means. |
| Frozen city-prediction metrics | The final audit reevaluated 150 paired units; maximum manuscript mean difference was 5.56e-15 pp. |
| Published graph-migration code | Three aggregate tables reproduced byte-for-byte from the original prepared graphs. |
| Checkpoint inference replay | No main-run weight files were found; the inspected run contains JSON and Parquet artifacts. |
| Fresh fits using the reorganized public code | No completed numerical-matching replay was found. Unit tests and metric reanalysis do not establish this. |

## Search scope and provenance

`record_inventory.json` records original input hashes and inspected repository
commits. The latest `gnn` research branch matches the local research HEAD. Its GPU
branch has the same six-row ablation selection summary without epoch or integer
seed records. The `ceus` main branch contains the earlier 250 m data-preparation
project, rather than this model-training lineage. Local searches included the
identified runtime directory, relevant transfers and snapshots, Git history and
matching September session records. No original FULL reselect generator or detailed
training log was recovered within that scope; this is not a claim to have searched
every unrelated local file.

Check the recovered records from the repository root, without third-party packages:

```sh
python -I -S reproduction/recovery_audit/verify_records.py
```

The check preserves the missing-epoch fields and the unverified fresh-refit status.
It does not invent training records or execute training.
