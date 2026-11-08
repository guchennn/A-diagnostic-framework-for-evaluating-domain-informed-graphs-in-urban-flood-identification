# Paired local increments under two references

The archived fixed-propagation predictions were reevaluated without training or selection changes. All models use 18 attributes and 100 revealed training positives. Every contrast pairs city, outer fold, revelation draw and seed slot; the full test candidate set and recorded-positive set are checked identical across models. Five draws and three refits give 15 fits per fold. Metrics are calculated per fit, averaged within fold, then averaged equally across five folds.

Groups 0/1/2 are no upstream / one-hop-only / multi-hop support in the held-out D8 graph. The same memberships are retained for own and adjacency contrasts. AP contributions sum positive threshold-precision changes divided by the complete fold positive count; this is not AP recomputed within groups. Rank is the positive score percentile against the complete unlabelled background, with half credit for ties.

Intervals are descriptive 95% paired fold-resampling percentiles (10,000 draws, seed 20261011). The same resampled fold indices apply to all metrics, groups and references. They are not independent-city confidence intervals. Positive-cell direction counts average the 15 within-fold contrasts before counting and do not define inferential replications.

## Results (percentage points)

| City | Group (n) | AP: D8−own [interval] | AP: D8−adjacency [interval] | Rank: D8−own [interval] | Rank: D8−adjacency [interval] |
|---|---|---|---|---|---|
| zhengzhou | 0 (420) | +0.181 [-0.020, +0.407] | -0.062 [-0.275, +0.244] | +0.491 [+0.145, +0.901] | +0.036 [-0.262, +0.328] |
| zhengzhou | 1 (138) | +0.004 [-0.030, +0.052] | -0.032 [-0.094, +0.059] | +0.471 [+0.044, +0.879] | +0.279 [-0.310, +0.986] |
| zhengzhou | 2 (233) | -0.182 [-0.355, -0.028] | -0.238 [-0.386, -0.091] | -0.932 [-1.506, -0.246] | -0.939 [-2.227, +0.249] |
| beijing | 0 (138) | +0.148 [+0.086, +0.214] | +0.082 [-0.043, +0.195] | +0.987 [+0.702, +1.275] | +0.449 [-0.349, +1.081] |
| beijing | 1 (46) | +0.017 [+0.001, +0.034] | -0.015 [-0.031, +0.000] | +1.606 [+0.485, +2.845] | -0.214 [-1.073, +0.525] |
| beijing | 2 (112) | +0.189 [+0.063, +0.325] | +0.007 [-0.169, +0.185] | +0.925 [-0.201, +2.449] | +0.126 [-0.519, +0.770] |

Zhengzhou’s multi-hop AP contribution remained negative in every fold against both references, becoming −0.238 pp against adjacency. Its background-rank mean against adjacency was −0.939 pp, but its descriptive interval crossed zero. The no-upstream contribution changed from +0.181 pp against own to −0.062 pp against adjacency; the former gain therefore does not establish a D8-specific neighbourhood benefit. In Beijing, the multi-hop contribution changed from +0.189 pp (5/5 positive folds) against own to +0.007 pp (3/5 positive folds) against adjacency. The latter interval spans zero; no population equivalence is inferred.

At the cell level, Zhengzhou had 278/420 no-upstream positives with positive mean precision change against own, but only 167/420 against adjacency; 155 were positive against own and negative against adjacency. Beijing had 107/138 and 77/138, respectively, with 40 changing from positive against own to negative against adjacency. These are descriptive counts of held-out recorded cells, not independent statistical tests.

## Reproduction

See the [package README](../README.md) for publication commands, required local artifacts and the distinction between native-run reanalysis and historical replay. Historical source and verification evidence are retained in provenance.

## Full-city versus held-out group labels (same held-out predictions)

The secondary analysis changes group labels only: each recorded test cell is assigned its full-city D8 support class while retaining exactly the same held-out predictions, candidate rankings, paired references and fold weights. It is not evaluation of an unpruned model or an estimate of the causal effect of edge removal. Total AP is invariant to this relabelling.

| City | Contrast | Group | Full-city group AP contribution | Held-out group AP contribution | Positive folds: full / held-out |
|---|---|---|---:|---:|---|
| beijing | D8 - adjacency | 0 | +0.087 | +0.082 | 4/5 / 4/5 |
| beijing | D8 - adjacency | 1 | -0.022 | -0.015 | 2/5 / 2/5 |
| beijing | D8 - adjacency | 2 | +0.008 | +0.007 | 3/5 / 3/5 |
| beijing | D8 - own | 0 | +0.139 | +0.148 | 5/5 / 5/5 |
| beijing | D8 - own | 1 | +0.018 | +0.017 | 4/5 / 4/5 |
| beijing | D8 - own | 2 | +0.196 | +0.189 | 5/5 / 5/5 |
| zhengzhou | D8 - adjacency | 0 | -0.072 | -0.062 | 1/5 / 1/5 |
| zhengzhou | D8 - adjacency | 1 | -0.029 | -0.032 | 1/5 / 1/5 |
| zhengzhou | D8 - adjacency | 2 | -0.230 | -0.238 | 1/5 / 0/5 |
| zhengzhou | D8 - own | 0 | +0.164 | +0.181 | 3/5 / 3/5 |
| zhengzhou | D8 - own | 1 | +0.006 | +0.004 | 2/5 / 2/5 |
| zhengzhou | D8 - own | 2 | -0.167 | -0.182 | 1/5 / 0/5 |

The mean signs are unchanged across all three groups and both references. Zhengzhou multi-hop contributions remain negative on average under full-city labels (−0.167 pp against own; −0.230 pp against adjacency), but are negative in four rather than five folds. The all-five-fold direction statement therefore specifically concerns held-out graph labels. Beijing multi-hop own contrasts remain positive in all five folds (+0.196 pp with full-city labels versus +0.189 pp held-out); adjacency contrasts remain positive in three folds (+0.008 versus +0.007 pp). The main reference-dependent interpretation persists while fold-level sign consistency is sensitive to the group definition.

Detailed full-city-label intervals and rank changes are in full_graph_group_summary.csv; fold-level values are in full_graph_group_folds.csv; all paired label-definition comparisons are in group_definition_sensitivity.csv.
