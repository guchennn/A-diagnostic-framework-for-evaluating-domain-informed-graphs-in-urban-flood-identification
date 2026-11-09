# Local graph controls: completed city experiments

The G7 experiment completed 300 independent selections, 1,800 inner-fold fits and
900 outer fits across Beijing and Zhengzhou. Three graph realizations are averaged
within each paired fold/draw/refit unit; five spatial folds remain the summary units.
The 300 archived D8 fits provide the reference. G6 results are separately identified
historical comparisons, not new fits or repetitions of G7.

G6 preserves each receiving node's in-degree and uses same-partition eight-neighbour
candidates. G7 additionally preserves every sending node's out-degree and the total
orthogonal/diagonal edge counts, using constrained partial rewiring. Across outer
training/test partition graphs, G7 changed 8.63–13.96% of Beijing edges and
11.46–16.07% of Zhengzhou edges. It is not a uniform random graph or a complete
removal of terrain-derived connections. See the [protocol](../../reproduction/topology_controls/Protocol.md)
and its recorded pre-fit feasibility amendment.

## Results

All differences are D8 minus control in percentage points. Brackets are descriptive
95% paired fold-resampling intervals, conditional on these five folds and fitted models.

| City | Predictor | G6 sampled AP | G7 sampled AP | G7 full-domain AP |
| --- | --- | --- | --- | --- |
| Beijing | Fixed | +3.307 [0.973, 7.026] | +0.628 [−0.212, 1.870] | +0.145 [0.010, 0.287] |
| Beijing | GraphSAGE | +0.197 [−1.280, 1.525] | +0.123 [−0.254, 0.446] | +0.179 [0.007, 0.438] |
| Zhengzhou | Fixed | +0.377 [0.141, 0.602] | +0.043 [−0.308, 0.545] | +0.015 [−0.162, 0.273] |
| Zhengzhou | GraphSAGE | −0.386 [−1.584, 0.699] | −0.371 [−1.345, 0.556] | −0.240 [−0.784, 0.246] |

All four G7 sampled-AP intervals include zero, as do all twelve realization-specific
sampled-AP intervals. This does not establish equivalence. Beijing's averaged
full-domain AP contrasts are positive, but one realization-specific interval for
each predictor includes zero (Fixed realization 1; GraphSAGE realization 3).
All four realization-averaged G7 weighted-AP intervals and all four realization-averaged
top-10% recall intervals also include zero. The complete
results are retained in `paired_summary.csv` and `graph_realization_contrasts.csv`.

G6 and G7 alter different proportions of edges as well as imposing different
constraints. Their contrast sizes therefore do not isolate the effect of adding a
particular constraint. The findings support comparison-specific conclusions rather
than a general D8 advantage or identification of a physical water-transfer mechanism.
One Beijing GraphSAGE recall fold is an arithmetic zero stored as approximately
8.67e−18; its raw positive-fold count should not be interpreted as a substantive gain.
Frozen numerical outputs are preserved without rounding corrections.

## Published evidence and verification

The CSVs contain fold aggregates, graph counts and fit metadata; no cell-level
predictions, graph edges or raw labels are included. `manifest.json` preserves the
historical analysis hashes and relative artifact paths. Those private execution
artifacts are provenance references, not files claimed to be present in this checkout.

```text
python -I -S reproduction/topology_controls/verify_aggregates.py
```

The standard-library checker verifies all twelve table hashes, 900 control and
300 reference fit identities, 960 graph identities, fold means, paired differences,
three-realization averages, seeds and selected durations. The accompanying
[selection verification](../../reproduction/topology_controls/inner_selection_verification.json)
records the independent check of all 600 candidate tasks and 300 choices.
[Reference replay](../../reproduction/topology_controls/reference_replay_verification.json)
matched all prediction scores exactly for one fixed fold/draw/refit per city and
predictor (four fits). This is a bounded numerical check, not an all-fit replay.
The [public runner](../../reproduction/topology_controls/README.md) has separate
portability tests; public-code city-level numerical matching remains unverified.
