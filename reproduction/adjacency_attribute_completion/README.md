# Adjacency completion across attribute and supervision conditions

Completed on 24 November 2026. This experiment adds the 20 missing ordinary
eight-neighbour adjacency conditions to the existing Own/D8 comparison. The four
18-attribute/100-positive main conditions are reused. The resulting 24-condition
matrix crosses two cities, fixed propagation/two-layer GraphSAGE, 18/16/14
attributes, and 100/all available training positives.

The new run completed 130 blocks, 260 independent configuration selections,
1,560 inner-fold fits and 780 outer refits. Each representation selects its own
configuration and duration using the shared inner-validation protocol.

## Results

[table6.csv](table6.csv) reports absolute AP for Own, Adjacency and D8, both
D8-minus-reference differences, descriptive 95% paired fold-resampling intervals,
and positive-fold counts. AP is in percent; differences and intervals are in pp.

The clearest additional D8 increment over adjacency occurs in Beijing with fixed
propagation, all 18 attributes and all available training positives: +1.884 pp
[0.259, 4.072], positive in four of five folds. All six Beijing fixed-propagation
means are positive; the other five intervals span zero. Zhengzhou's six GraphSAGE
means are negative, including all five folds at each of the three attribute counts
under the 100-positive budget. The complete table includes every condition.

## Verify the numerical evidence

With the repository's NumPy and pandas dependencies installed, run from its root:

```sh
python reproduction/adjacency_attribute_completion/verify_aggregates.py
```

The script checks file hashes and recomputes fold means, five-fold summaries,
10,000 paired fold resamples (seed 20261011), intervals and the complete Table 6
from the supplied fit-level AP aggregates. AP was originally computed separately
for three matched background banks and then averaged per fit; prediction scores
were never pooled across refits. The original Own/D8 aggregates were reproduced
from archived predictions to within 1.11e-16 and retained unchanged.

| File | Contents |
| --- | --- |
| `unit_metrics.csv` | 1,080 matched fold/draw/refit units, each with three-bank mean AP |
| `fold_values.csv` | 120 condition/fold rows |
| `summary.csv` | 120 condition/quantity summaries; `range_low/high` are the reported resampling intervals |
| `table6.csv` | Complete 24-condition comparison |
| `independent_selections.csv` | 260 new inner-selected configurations and refit durations |
| `refit_metadata.csv` | 780 successful outer-refit identities and prediction-file hashes |
| `config.yaml` | Frozen new-experiment matrix |
| `verification.json` | Completion counts, software environment, source/output hashes and local verification receipt digest |
| `execution_sources.zip` | Exact new runner, analyzer, configuration, two test files and execution plan |

The execution archive preserves the files used in the original frozen training
workspace. Its runner and analyzer reuse that workspace's historical helper
modules, prepared data and archived fits; the archive is a provenance record,
not a replacement for the public repository's training adapter. Aggregate
verification above runs directly with the published files. The execution source
tests passed 23 checks in the original environment. No new public-adapter GPU
replay is claimed.

Only new-experiment material is added here. Cell-level records, predictions,
graph edges, model weights and manuscript files remain outside this release.
