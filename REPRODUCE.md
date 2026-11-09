# Run the experiments

Run commands from the repository root with the Python 3.12 virtual environment
activated. Dependencies are pinned in `requirements.txt` and
`requirements-train.txt`. Local inputs and generated results go to `data/`,
`artifacts/` and `reproduced/`. [Environment setup](README.md#install).

## Included results

```sh
python -m pip install -r requirements.txt
python scripts/reproduce_tables.py
python scripts/check_seed_results.py
python -m floodgnn_ceus.synthetic_diagnostic_validation --output reproduced/synthetic
```

| Result | Input data |
| --- | --- |
| Table 2, absolute performance | `results/performance/main_units.csv` |
| Table 3, attribute ablation | `results/performance/ablation_absolute_folds.csv` |
| Table 4, seed sensitivity | `results/seed/fit_group_aggregates.csv` and paired-effect tables |
| Structural contributions | `results/structural_stability/contributions_by_fold.csv` |
| Synthetic experiments | `configs/synthetic_diagnostic_validation_v1.yaml`, reference values in `results/synthetic/` |
| Configurations, epochs and seeds | `results/settings/` |

AP and recall use the 0–1 scale. Fields ending in `_pp` use percentage points.
City means average the five spatial folds equally. The seed check recomputes paired
effects and 10,000 randomized references from fit/group aggregates.

## Final manuscript diagnostics

The final revision adds paired local increments against both own attributes and
adjacency, full-to-held-out structural-group migration, and controlled ranking
validation. Start with the supplied evidence:

```sh
python -I -S reproduction/final_audit/scripts/verify_aggregates.py
python reproduction/controlled_validation/controlled_ranking/test_validation.py
```

The first command checks hashes, paired-reference identities, AP additivity,
fold averaging and migration counts without third-party dependencies. The second
tests the controlled ranking metrics using the pinned numerical environment.

Detailed commands and required inputs are in the
[final-audit guide](reproduction/final_audit/README.md) and
[controlled-validation guide](reproduction/controlled_validation/README.md).
The [operation specification](reproduction/final_audit/Operational_specification.md)
connects source data to terrain attributes, eligible-city graphs and partition
graphs. The [18-attribute ledger](reproduction/final_audit/feature_operations.csv)
records units, coverage, temporal and missing-data rules. Exact historical training
provenance is retained separately from publication-code hashes.

Regenerating full city analyses requires the local input artifacts named in each
guide. Checking aggregate evidence does not reconstruct private predictions.
Use the relative paths shown in the commands; no machine-specific project location
is required.

### Completed topology-control evidence

The degree–distance matched G7 experiment now supplies 900 outer fits, 300 independent
selections and 960 validated partition graphs. All outcomes and interpretation limits
are documented in [the topology results](results/topology_controls/README.md).
Verify the published aggregates with
`python -I -S reproduction/topology_controls/verify_aggregates.py`.

## Prepare the data

Obtain provider access using the data guide, then run each stage.
[Data access and citations](README.md#data).

```sh
floodgnn fetch --city all --source all
floodgnn prepare --city all --through worldcover
floodgnn prepare --city all --through osm
floodgnn prepare --city all --through worldpop
floodgnn prepare --city all --through imerg
floodgnn prepare --city all --through dsm
floodgnn prepare --city all --through design
floodgnn audit-e0
floodgnn lock-protocol
```

Both cities are prepared together. `design` builds terrain variables, features,
spatial partitions and graphs from the completed DSM stage. `lock-protocol`
records the prepared inputs. Use a fresh checkout for a separate preparation.

When DSM and final-grid products already exist, use only the cumulative post-DSM
targets in a separate working copy, preserving the existing protocol lock:

```sh
floodgnn prepare --city all --through terrain
floodgnn prepare --city all --through features
floodgnn prepare --city all --through design
```

## Train locally

Training uses Windows or Linux x86-64 with an NVIDIA GPU compatible with
PyTorch 2.8.0/CUDA 12.8 and PyTorch Geometric 2.7.0. The training requirements
select the CUDA 12.8 wheel from the official PyTorch download site.

```sh
python -m pip install -r requirements-train.txt
python -m pip check
python -m floodgnn_ceus.train plan --output reproduced/plan.json
python -m floodgnn_ceus.train freeze-local
python -m floodgnn_ceus.train probe-local
python -m floodgnn_ceus.train preflight
python -m floodgnn_ceus.train run --phase all --start 0 --limit 50
```

The fixed one-epoch probe records resource use before the main fits. Preflight
must report `ready`. The run covers two cities, five outer folds and five reveal
trajectories. Selection uses three inner folds. Successful task receipts allow
the same command to resume interrupted work.

## Evaluate predictions

Use the directory named by `run_namespace` in `reproduced/plan.json`.
Set `RUN` in your shell.

```sh
# Bash
RUN=$(python -c "import json; print('artifacts/runs/' + json.load(open('reproduced/plan.json'))['run_namespace'])")
```

```powershell
# PowerShell
$RUN = "artifacts/runs/$((Get-Content reproduced/plan.json -Raw | ConvertFrom-Json).run_namespace)"
```

The following commands work in both shells.

```sh
python scripts/summarize_training.py --root . --run-root "$RUN" --output reproduced/training_summary
python scripts/analyze_structure.py --root . --run-root "$RUN" --output reproduced/structure
python scripts/evaluate_predictions.py --root . --run-root "$RUN" --output reproduced/evaluation
python scripts/decompose_performance.py --root . --run-root "$RUN" --output reproduced/decomposition
python scripts/analyze_seed_sensitivity.py --root . --run-root "$RUN" --output reproduced/seed
```

These analyses use complete held-out rankings and paired model predictions.
Cell-level scores remain in the local run directory.

## Ablation

### Revised manuscript: independent selection

The revised paper uses the October replacement series, in which own-attribute
and D8 procedures each select their configuration and duration independently.
Its 520 variant selections and 1,560 refit records are provided under
[`reproduction/selection_alignment`](reproduction/selection_alignment/README.md).
The [runner guide](reproduction/selection_alignment/Runner.md) covers fresh
execution. Verify the supplied fold-level evidence with:

```sh
python -I -S reproduction/selection_alignment/verify_tables.py
```

### Legacy September attribute experiment

The following commands and missing-record notes refer to the earlier
D8-selected-reference series, which is retained for historical reproducibility.

After preparation and the local environment/probe steps, rerun the budget-100
attribute experiment with inner selection and matched reference fits.

```sh
python -m floodgnn_ceus.rq2_scarcity --root . --output-root reproduced/ablation
```

To replay the recorded budget-100 selections, use the supplied settings.

```sh
python scripts/reproduce_ablation.py run --budget 100 --output reproduced/ablation-replay
```

Own-attribute and adjacency fits inherit the D8-selected configuration and epochs.
Recovered FULL-budget records include 450 fit/bank AP summaries and the selected
configurations. They reproduce all 60 published fold/model/attribute means exactly:

```sh
python -I -S reproduction/recovery_audit/verify_records.py
```

Fifteen neural selection units still lack saved best-inner/refit epochs, and verified
integer-seed provenance is absent for 135 neural fits. Exact FULL training replay
remains unavailable; `NEURAL-1/2/3` slot labels do not fill the missing seed records.
The budget-100 interaction analysis uses the recorded FULL constants.

The original ablation used Python 3.13.5/CUDA 12.9. The main city run instead already
used Python 3.12.11, PyTorch 2.8.0+cu128 and PyG 2.7.0, with all 1,250 inner and 1,550
outer tasks completed. Frozen-prediction metric reanalysis and graph-migration replay
have numerical checks; fresh fitting with the reorganized public source has not yet
been numerically matched to the historical city fits. These are distinct verification
levels. [Search scope, recovered metadata and remaining gaps](reproduction/recovery_audit/README.md).
