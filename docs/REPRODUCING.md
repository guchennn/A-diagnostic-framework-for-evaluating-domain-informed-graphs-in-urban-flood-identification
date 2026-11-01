# Reproducing the final-v8 workflow

Run commands from the repository root. All output paths below are repository-relative.
Python 3.12 and `uv.lock` define the environment. Keep this checkout fixed once a fresh
protocol and local environment lock have been created.

## Install and verify

```powershell
uv sync --frozen
uv run --no-sync ruff check src/floodgnn_ceus/cli.py src/floodgnn_ceus/reviewer_revision.py tests/test_release_workflow.py
uv run --no-sync pytest -q --basetemp reproduced/pytest-workflow
uv run --no-sync python scripts/reproduce_tables.py --output reproduced/tables
```

The table command rebuilds tables from the supplied aggregate results. Fresh model
reproduction additionally requires authorized source access, both cities' prepared
inputs, and a compatible local NVIDIA CUDA environment. Restricted records and
raw sources belong in the ignored `data/` tree and must stay outside Git.

## Prepare both cities from fresh inputs

Configure source access as specified in `configs/sources.yaml`. IMERG uses an
interactive Earthdata login without credential persistence. DSM acquisition uses
the configured Copernicus Data Space authentication. Source downloads can be large.
Run each command to successful completion before starting the next.

```powershell
uv run --no-sync floodgnn fetch --city all --source crowdflood_v1
uv run --no-sync floodgnn fetch --city all --source geoboundaries_chn_adm2
uv run --no-sync floodgnn prepare --city all --through labels
uv run --no-sync floodgnn prepare --city all --through worldcover
uv run --no-sync floodgnn prepare --city all --through osm
uv run --no-sync floodgnn prepare --city all --through worldpop
uv run --no-sync floodgnn prepare --city all --through imerg
uv run --no-sync floodgnn prepare --city all --through dsm
uv run --no-sync floodgnn prepare --city all --through design
uv run --no-sync floodgnn audit-e0
uv run --no-sync floodgnn lock-protocol
```

Every `--city all` stage prepares Zhengzhou and Beijing. The `osm`, `worldpop`,
`imerg`, and `dsm` targets each run their own source-specific stage after labels and
WorldCover. Execute all four separately. `design` consumes the existing DSM and
final scorable grids and builds terrain, features, evaluation design, and graphs.
For partial post-DSM preparation, use `--through terrain` or `--through features`.

`lock-protocol` audits the prepared artifacts and exclusively creates
`artifacts/protocol_lock.json`. The archived lock in `provenance/` documents the
published experiment. A fresh reproduction must generate its own lock from its
prepared inputs. Preserve existing locks and use a separate fresh checkout when a
new preparation requires a new protocol identity.

## Fresh local CUDA gates and primary run

```powershell
uv sync --frozen --group train
uv run --no-sync python -m floodgnn_ceus.reviewer_revision plan --output reproduced/primary_plan.json
$plan = Get-Content reproduced/primary_plan.json -Raw | ConvertFrom-Json
$runRoot = "artifacts/runs/$($plan.run_namespace)"
uv run --no-sync python -m floodgnn_ceus.reviewer_revision freeze-local
uv run --no-sync python -m floodgnn_ceus.reviewer_revision probe-local
uv run --no-sync python -m floodgnn_ceus.reviewer_revision preflight
uv run --no-sync python -m floodgnn_ceus.reviewer_revision run --phase all --start 0 --limit 50
uv run --no-sync python -m floodgnn_ceus.reviewer_revision progress
```

Check each command's exit code before continuing. Preflight must report `ready`.
The namespace is computed from current configuration and source identities, so use
`$runRoot` from the generated plan for downstream analysis. Local predictions and training receipts are written under `artifacts/runs/`.

`freeze-local` requires Python 3.12, torch 2.8.0, PyG 2.7.0, CUDA runtime 12.8,
and an available local GPU. It creates an immutable namespace-local environment
lock binding the actual environment, protocol lock, package metadata, configuration,
source code, and nine consumed files per configured city. Source provenance is
already bound by the protocol lock. The primary model runner reads the final grid,
18-feature table, five evaluation-design tables, and two graph edge tables.

The first model execution is `probe-local`, the fixed one-epoch resource probe.
It discards predictions and computes no model metrics. Its receipt must match the
current environment lock, source identity, and input identity before any primary
fit can run. Archived environment locks and probe receipts cannot authorize a fresh
run. Changes to bound inputs or environment block execution.

The configured matrix contains 50 outer units across both cities. `--start` and
`--limit` refer to outer units. Successful task receipts permit resuming the same
command after interruption. Each variant selects SMALL/MEDIUM using inner folds
only. Outer-test outcomes must never determine model or configuration selection.
Preprocessing fit scopes, reveal rules, evaluation banks, and paired seeds remain
part of the frozen protocol.

## Selected final-v8 analyses

After the primary run completes, use new output directories for each analysis.
These commands consume the dynamically computed run namespace and write under
`reproduced/`. Run from the same PowerShell session as the plan commands, or reload
`reproduced/primary_plan.json` to reconstruct `$runRoot`.

```powershell
uv run --no-sync python scripts/summarize_reviewer_training.py --root . --run-root $runRoot --output reproduced/primary_summary
uv run --no-sync python scripts/audit_reviewer_structure.py --root . --output reproduced/structure_audit
uv run --no-sync python scripts/audit_final_review_design.py --root . --run-root $runRoot --output reproduced/design_audit
uv run --no-sync python scripts/analyze_reviewer_training_structure.py --root . --run-root $runRoot --output reproduced/training_structure
uv run --no-sync python scripts/analyze_final_review_evaluation.py --root . --run-root $runRoot --output reproduced/final_evaluation
uv run --no-sync python scripts/analyze_major_revision_diagnostics.py --root . --run-root $runRoot --output reproduced/graph_diagnostics
```

Analysis scripts verify run provenance, task completeness, pairing, and inner-only
selection before reporting. Archived-reference comparisons are optional script
arguments. Fresh analyses can derive structure from the prepared city inputs.
Keep generated cell-level records and predictions outside Git. Published aggregate
results remain in `results/` for comparison with the final-v8 manuscript.
