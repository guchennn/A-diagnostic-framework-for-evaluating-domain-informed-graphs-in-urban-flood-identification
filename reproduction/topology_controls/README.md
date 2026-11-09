# Degree–distance matched topology controls

This portable implementation constructs G7 using the same builder as the historical
November 2026 experiment. It preserves each node's incoming/outgoing degree, edge
count, orthogonal/diagonal totals, partition membership and eight-neighbour support.
It performs constrained partial rewiring, not uniform random-graph sampling.
See `Protocol.md` for the fixed design and scientific interpretation.

## Inputs and execution

Use the repository's documented Python 3.12 training environment and preparation
pipeline first. The input contract is the existing `data/interim/<city>/` grid,
18 features, frozen evaluation design and graphs, plus `artifacts/protocol_lock.json`.
No raw data, cell-level predictions or generated graphs are included here.
From the repository root:

```text
python scripts/run_topology_controls.py cache
python scripts/run_topology_controls.py freeze
python scripts/run_topology_controls.py probe
python scripts/run_topology_controls.py train
python scripts/run_topology_controls.py status
```

`cache` is CPU-only. All 960 partition graphs must pass structural checks before
`freeze` can bind their hashes. `probe` runs the existing fixed one-epoch,
performance-blind resource test. Training refuses missing or stale bindings.
Outputs use `artifacts/runs/topology-controls-v1-20261108/`. Each graph realization
and predictor independently selects SMALL/MEDIUM using three inner folds, then
runs three outer refits. The complete matrix has 300 selections, 1,800 inner-fold
fits and 900 outer fits. Resume validates payload hashes; do not change frozen
configuration, sources or data inside a partially completed execution namespace.

The public adapter uses `floodgnn_ceus.train` local environment gates. The historical
private run additionally checked a returned AutoDL lock. A public rerun creates its
own source/input identity; it does not impersonate the historical execution.

## Paired analysis

First complete a baseline matched-model run using the repository's existing training
entry point. Supply its actual run directory explicitly:

```text
python scripts/analyze_topology_controls.py --reference-run artifacts/runs/<completed-baseline-namespace>
```

The reference must contain the fixed-D8 and two-layer-D8 GraphSAGE selections and
300 corresponding outer refits. The analyzer checks reference source/input hashes,
resource-probe bindings, all 900 G7 fits, selection durations, seed metadata, exact
evaluation banks and 960 graph identities before calculating metrics. It refuses
partial results. Both runs must use unchanged shared inputs and training sources.

Sampled AP, complete-domain AP, 1:10-weighted AP and top-10% recall retain complete
ties and exact fractional quotas. Metrics, not prediction scores, are averaged
across graph realizations within matched fold/draw/refit units. Five spatial folds
remain the summary units; 10,000 paired fold resamples use seed 20261011. Output
CSVs contain aggregates and training metadata, never cell-level predictions.
Historical G6 aggregate tables are copied separately only when their documented
source tables are available; G6 is neither refitted nor pooled with G7.

## Validation scope

CPU tests cover graph invariants, cache identities, immutable metadata, relative
local-lock paths, complete fit-grid checks, within-unit aggregation, tied-score
metrics and refusal to analyze incomplete records. During adaptation, the public
metric helpers agreed exactly with the historical implementation on 40 deterministic
synthetic tied-score datasets (four metrics each; maximum absolute difference zero).
No GPU training or city-level numerical replay was performed as part of the public
adapter validation. Historical experiment results and public-adapter test evidence
are separate artifacts.

## Completed historical evidence

The [published result tables](../../results/topology_controls/README.md) contain the
completed 900-control-fit experiment, all graph-realization outcomes and provenance.
Run `python -I -S reproduction/topology_controls/verify_aggregates.py` to check the
aggregate identities without scientific Python dependencies. Complete inner-selection
verification and a bounded four-fit exact reference replay are included alongside
this document; they are distinct from public-adapter numerical matching.
