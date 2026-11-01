# A diagnostic framework for evaluating domain-informed graphs in urban flood identification

Experiment code and numerical data for the Zhengzhou (2021) and Beijing (2023)
case studies, covering matched graph comparisons, attribute ablation, structural
decomposition, seed sensitivity and synthetic validation.

## Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
git clone https://github.com/guchennn/A-diagnostic-framework-for-evaluating-domain-informed-graphs-in-urban-flood-identification.git flood-graph-diagnostics
cd flood-graph-diagnostics
uv sync --frozen
```

## Run with the included data

These CPU commands recompute Tables 2 and 3, check the seed statistics underlying
Table 4, and run the ten-replicate synthetic experiment.

```sh
uv run python scripts/reproduce_tables.py
uv run python scripts/check_seed_results.py
uv run python -m floodgnn_ceus.synthetic_diagnostic_validation --output reproduced/synthetic
```

Generated files go to `reproduced/`. Published numerical results are in `results/`.

## Run the city experiments

Acquire the source datasets, prepare both cities, then run local training and analysis.
Training uses PyTorch 2.8.0, PyTorch Geometric 2.7.0 and an NVIDIA GPU with CUDA 12.8.
Provider links, dataset citations and the command sequence are in the two guides.
[Data access](DATA.md) · [Experiment instructions](REPRODUCE.md).

## Contents

| Directory | Contents |
| --- | --- |
| `src/floodgnn_ceus/` | Data preparation, models and diagnostic methods |
| `configs/` | Dataset, feature, graph and experiment settings |
| `scripts/` | Evaluation and numerical reproduction |
| `results/` | Aggregate results, recorded selections and seeds |
| `tests/` | Numerical and workflow tests |

```sh
uv run pytest -q --basetemp .pytest_tmp
```

The original FULL-budget ablation is available as aggregates. Exact training replay
requires its missing refit-epoch and optimizer-seed records. The running guide
describes the supported budget-100 experiments. [Ablation scope](REPRODUCE.md#ablation).

## Citation and licence

Cite the software using `CITATION.cff` and the datasets using `data_references.bib`.
The code uses the MIT licence. Source datasets retain their provider terms.
[Software citation](CITATION.cff) · [Dataset citations](data_references.bib) · [MIT licence](LICENSE).
