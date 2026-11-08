# Independently selected attribute comparisons

The revised manuscript uses the October `minor-revision-controls-v1` attribute
series. Fixed propagation and two-layer GraphSAGE each select own-attribute and
D8 procedures independently from SMALL/MEDIUM on three inner folds. The median
best epoch determines three outer refits. These are replacement comparisons, not
recovered records for the distinct September D8-selected-reference experiment.

The supplied evidence covers 520 variant selections and 1,560 outer fits:
1,200 fits for 16/14 attributes with 100 positives and 360 fits for 18/16/14
attributes with all training positives. The 18-attribute/100-positive condition
uses the original main-comparison fits. Together, archived prediction reanalysis
reproduced all 240 fold-level attribute contrasts, with a maximum absolute
discrepancy of 1.11e-14 percentage points.

Run the aggregate and metadata checks without third-party packages:

```text
python -I -S reproduction/selection_alignment/verify_tables.py
```

`independent_selections.csv` contains selected configurations and inner best
epochs. `refit_metadata.csv` records successful fits, scheduled durations, seed
slots and integer seeds reconstructed from the verified historical source rule.
Scheduled duration is not separate per-epoch runtime telemetry. Source/input
hashes and the exact validation scope accompany the CSVs. No cell-level records
or predictions are included.

The [independent-series runner](Runner.md) documents fresh execution. Archived
prediction reanalysis and public-code refit matching are distinct: the latter is
not claimed here. The September series remains available under
[`recovery_audit`](../recovery_audit/README.md), including its unresolved metadata
gaps; those gaps do not describe the replacement series used by the revised paper.
