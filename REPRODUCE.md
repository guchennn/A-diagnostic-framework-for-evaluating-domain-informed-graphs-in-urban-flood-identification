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

## Prepare the data

Obtain provider access using the data guide, then run each stage.
[Data access and citations](DATA.md).

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
Historical FULL-budget results are supplied as aggregates. Their refit epochs and
optimizer seeds are incomplete, so exact FULL training replay is unavailable.
The budget-100 interaction analysis uses the recorded FULL constants. The original
ablation used Python 3.13.5/CUDA 12.9, while this workflow uses Python 3.12/CUDA 12.8.
Fresh numerical replay of the city fits remains unverified.
