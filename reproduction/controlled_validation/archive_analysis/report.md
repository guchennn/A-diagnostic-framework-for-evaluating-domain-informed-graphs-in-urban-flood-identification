# Archived controlled diagnostic validation

This is a new, read-only reanalysis of completed synthetic positive–unlabelled fits. It adds exact background reweighting and paired diagnostic summaries without retraining. Synthetic results must be identified as controlled simulation results, separate from the two-city evidence.

## Design and aggregation

The frozen `synthetic-pu-regime-v1-20261029` configuration contains 192 × 96 cells, three independent normal features, four-cell horizontal chains, and a latent flood probability `sigmoid(-2 + 0.5 X1 + beta upstream_X1)`. Independently, 25% of flooded cells are recorded. The upstream coefficients are 0, 0.5 and 1.0. Row bands 0–63, 72–119 and 128–191 form training, validation and test partitions. Each test grid contains 6,144 cells. Graph edges are isolated within partitions.

The useful graph follows the generating chains; the wrong graph uses degree-preserving rewiring. The truncated graph removes chain links at fixed eight-column boundaries. Own-attribute input duplicates the local features to retain the fixed-propagation input width. The cached models use a 32-unit network with dropout 0.2, learning rate 0.001, weight decay 0.0001, maximum 250 epochs and patience 30. Each revelation draw uses 100 revealed positives and 2,000 unlabelled training cells. These settings are documentary provenance of the archived fits, not a newly conducted training run.

There are ten studies per coefficient and five independently generated grids per study. The same grid is paired across model references and evaluation weights. Five revelation draws and the first three paired fit seeds are averaged within each grid; the five grids are then averaged within a study. The additional six own-model seeds are not included in the own-versus-graph mean. Results below are means of ten study means; ranges in CSVs are observed study ranges, not confidence intervals. Coefficient conditions share generation seed identifiers and are not independent samples of cities or events.

The current helper's default coefficient list contains only 0 and 0.5, whereas the saved experiment configuration includes 1.0. This analysis follows the saved configuration and checks the common grid/replication settings. It invokes the existing cache loader, not any fitting function. All 150 base cache files and 50 zero-signal wrong-graph cache files were used. Zero-signal truncation fits are absent and are explicitly missing in the output.

## Exact background reweighting

For a positive threshold, let `p = M × cached_AP_term = TP/(TP+FP)`, where `M` is the number of recorded test positives. Giving every unlabelled test cell weight `w` changes precision to

`p_w = TP/(TP+w FP) = p/[p+w(1-p)]`.

Thus weighted AP is `mean(p_w)`. Full-domain AP uses `w=1`; 1:10 background reweighting uses `w=10M/B`, where `B=6144-M`. The transformation is applied separately to each cached positive and fitted model before averaging. This is deterministic reweighting of the same full-domain rankings, not AP on a newly sampled subset. Threshold ties retain the original inclusive-threshold convention. Background midpoint ranks do not change under uniform background reweighting.

## Reference and evaluation dependence

All changes are percentage points. “Positive studies” counts positive full-domain differences among ten study means.

| beta | Contrast | Full-domain AP | 1:10 reweighted AP | Positive studies |
|---:|---|---:|---:|---:|
| 0 | Useful − own | −0.176 | −0.398 | 0/10 |
| 0 | Useful − wrong | −0.013 | −0.032 | 4/10 |
| 0 | Wrong − own | −0.163 | −0.366 | 0/10 |
| 0.5 | Useful − own | +0.495 | +1.155 | 10/10 |
| 0.5 | Useful − wrong | +0.675 | +1.528 | 10/10 |
| 0.5 | Wrong − own | −0.180 | −0.373 | 0/10 |
| 1 | Useful − own | +1.983 | +3.962 | 10/10 |
| 1 | Useful − wrong | +2.150 | +4.283 | 10/10 |
| 1 | Wrong − own | −0.167 | −0.321 | 0/10 |

The no-upstream-signal condition preserves a negative graph-versus-own result: both useful and wrong graphs perform worse than own attributes. Their difference is near zero on average but not established as equivalent. Under positive upstream signal, useful-versus-wrong is positive in all ten studies, while wrong-versus-own remains negative. The two references therefore distinguish overall predictive increment from the advantage of the correct relationships over the rewired relationships.

Switching from own to wrong as reference increases the full-domain useful-graph increment by 0.163, 0.180 and 0.167 pp at beta 0, 0.5 and 1, respectively. Reweighting changes the useful-versus-own increment by −0.222, +0.660 and +1.979 pp, and the useful-versus-wrong increment by −0.019, +0.853 and +2.133 pp. Signs agree across weights in every study for the nonzero-signal contrasts; only the zero-signal useful-versus-wrong comparison changes sign in one study. These results show magnitude dependence on the evaluation population without presenting a broad claim of ranking reversals.

## Structural accounting and known support removal

Structural groups are fixed by the original useful graph: no incoming support, one-hop-only support and multi-hop support. They partition positive contributions to one full ranking; these are not within-group AP scores. For useful minus own, their full-domain contributions are:

| beta | No upstream | One-hop-only | Multi-hop | Total AP |
|---:|---:|---:|---:|---:|
| 0 | +0.042 | −0.066 | −0.153 | −0.176 |
| 0.5 | +0.192 | +0.105 | +0.197 | +0.495 |
| 1 | +0.372 | +0.537 | +1.074 | +1.983 |

An apparent gain in the no-upstream group is therefore possible even when upstream signal is absent. The decomposition allocates global ranking changes; it does not identify a local physical mechanism or the causal benefit of an incoming edge. At beta 0.5 and 1, AP-contribution and rank-change signs agree in all ten studies for the supported groups and in nine for the no-upstream group.

Removing support lowers total full-domain AP by 0.165 pp at beta 0.5 and 0.592 pp at beta 1; all ten studies have negative differences. The known cut-versus-retained comparison adds information beyond this total loss:

| beta | Diagnostic, truncated − useful | Cut cells | Retained cells | Cut − retained | Studies with cut more adverse |
|---:|---|---:|---:|---:|---:|
| 0.5 | Background-rank change | −2.920 | −0.532 | −2.388 | 10/10 |
| 0.5 | Threshold-precision change | +0.069 | −0.179 | +0.248 | 2/10 |
| 1 | Background-rank change | −9.371 | −1.109 | −8.261 | 10/10 |
| 1 | Threshold-precision change | −0.789 | −0.573 | −0.216 | 9/10 |

The rank contrast localizes larger losses to known support removal at both signal levels. Threshold precision gives the opposite cut-versus-retained direction in eight of ten weak-signal studies. Under 1:10 reweighting, its cut-minus-retained contrast is +0.345 pp at beta 0.5 and −0.610 pp at beta 1, with two and ten studies, respectively, showing more adverse cut-cell changes. This disagreement is substantive evidence for inspecting both quantities. Truncation models were refitted in the archived experiment, so these contrasts combine removal of support with the resulting global refit and ranking changes; they are not isolated cellwise causal effects.

## Null-reference limitation retained

The archived `location_summary.csv` reports zero-signal seed-reference exceedance rates of 60%, 90% and 100% for useful-minus-own no-upstream, one-hop and multi-hop contributions. Useful-minus-wrong rates are 10%, 20% and 10%; its total-AP rate is 20%. These numbers were read from the archived summary, not newly resampled here. An own-versus-own seed envelope is not a calibrated test of no graph information, and its exceedance rate must not be called a false-positive rate, statistical significance or a p-value. The near-zero useful-versus-wrong mean does not repair this calibration issue.

## Verification and reproducibility

From the repository root after restoring trusted original artifacts (see ../README.md):

```sh
python reproduction/controlled_validation/archive_analysis/reanalyze_archive.py --project-root . --output-dir reproduced/archive_analysis
```

The reanalysis takes approximately 3–9 seconds locally. Three direct toy-score cases, including mixed-label ties and all scores tied, were checked under three weights each; maximum AP error was 1.11 × 10⁻¹⁶. All 450 archived full-domain grid contrasts were recovered with maximum error 9.99 × 10⁻¹⁵ pp. Maximum group-additivity residual was 1.78 × 10⁻¹⁵ pp; reference-contrast identity residual was 8.88 × 10⁻¹⁶ pp. Rank change was exactly unchanged across weights. These arithmetic checks do not independently validate the original model training.

`historical_input_manifest.json` records the original SHA-256 hashes of the analysis script, two source helpers, two saved configurations, two source summary files and 200 cache files. `verification.json` records the arithmetic checks. `grid_contrasts.csv` and `study_contrasts.csv` preserve aggregation units; `ap_summary.csv`, `group_summary.csv`, `rank_summary.csv`, reference/weight interaction files, direction-agreement files and support files provide the reported summaries. Blank beta-zero support effects mean “not cached”, not zero. No raw-score, group-only AP, quota/recall curve or calibration reconstruction is attempted because the cache does not contain the required information.

The supported contribution is a controlled demonstration that reference choice, evaluation weight, additive group accounting and support-sensitive rank diagnosis answer complementary questions. It does not establish a new statistical estimator, calibrated hypothesis test, general hydrological mechanism or universal superiority of a four-dimensional framework.
