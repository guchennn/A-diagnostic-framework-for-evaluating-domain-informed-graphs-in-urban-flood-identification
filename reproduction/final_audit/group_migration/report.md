# Full-domain to held-out D8 structural-group migration

This audit uses the frozen **eligible-city G2 graph before outer-fold pruning** as the full graph. It is not the extended terrain-processing rectangle, an unrestricted river network, or a reconstruction of edges from elevation. Nodes already satisfy the final study-grid eligibility rules. The audit reads `graphs/base_edges.parquet`, `graphs/outer_edges.parquet`, `evaluation_design/outer_partitions.parquet`, `final_scorable_grid.parquet` and the STRICT-LOC label ledger. It changes no terrain, routing, eligibility, split, fit or score.

Groups use the existing manuscript definition: no upstream means no incoming edge; one-hop-only means an incoming edge but no directed path of length two ending at the cell; multi-hop means such a two-edge path exists. This is not a classification by number of incoming neighbours, nor the propagation depth used by the fixed predictor. Full and held-out memberships are calculated on the same focal cells. Each eligible cell contributes exactly once, through its outer-test fold. Pooled counts are cell-weighted descriptive counts, not equal-fold performance estimates.

## Recorded-positive transition matrix

Rows are full eligible-domain groups; columns are groups in the cell's actual held-out graph. Counts include every STRICT-LOC positive exactly once. The row total is the denominator for row percentages in `transition_matrix.csv`.

| City | Full-domain group | Held-out no upstream | Held-out one-hop-only | Held-out multi-hop | Row total |
|---|---|---:|---:|---:|---:|
| Zhengzhou | No upstream | 411 | 0 | 0 | 411 |
| Zhengzhou | One-hop-only | 2 | 123 | 0 | 125 |
| Zhengzhou | Multi-hop | 7 | 15 | 233 | 255 |
| Beijing | No upstream | 130 | 0 | 0 | 130 |
| Beijing | One-hop-only | 4 | 41 | 0 | 45 |
| Beijing | Multi-hop | 4 | 5 | 112 | 121 |

Zhengzhou has 24/791 (3.03%) positive group changes and Beijing 13/296 (4.39%). For all eligible cells, the corresponding counts are 742/25,155 (2.95%) and 1,919/61,443 (3.12%). Thus 96.97% and 95.61% of positives retain their group. All changes move toward less upstream support; none moves upward.

The full-to-held-out multi-hop totals are 255→233 positives in Zhengzhou and 121→112 in Beijing. Every held-out multi-hop positive was already multi-hop in the full eligible-domain graph. Nevertheless, pruning removes 22/255 (8.63%) and 9/121 (7.44%) of the original multi-hop membership by moving these positives to other groups. Stable group identity does not imply unchanged neighbour sets: 54/791 and 24/296 positives lose at least one unique upstream node within two hops, compared with only 24 and 13 group changes.

Held-out no-upstream positives total 420 and 138. Of these, 9/420 (2.14%) and 8/138 (5.80%) became unsupported after outer-fold pruning; the remaining 411 and 130 were already unsupported in the full eligible-domain graph. Therefore neither the no-upstream category nor the multi-hop category can be interpreted simply as a class created by fold boundaries.

## Edge retention and denominator

`edge_retention.csv` reports every fold. `train_test_retention_fraction` divides retained train-plus-test edges by all full-city edges, and therefore includes removal of buffer and cross-partition edges. `test_target_retention_fraction` instead divides retained test–test edges by full-city edges whose **target** is in that fold's test set. These denominators must not be mixed. Summing test-target numerators and denominators across folds counts each full edge once because its target has exactly one test fold. No buffered cells are assigned artificial zero support; they are excluded from that fold's prediction graph and evaluated only when they are test cells in their own fold.

The ten per-fold test-target retention fractions span 93.36–95.64% in Zhengzhou and 94.10–95.07% in Beijing. The whole-fold train-plus-test retention fractions are lower, 78.71–81.27% and 77.82–80.42%, because their denominator includes all full-city edges.

## Interpretation

The partition-local labels describe the support actually available to held-out prediction. The matrix distinguishes this operational classification from the pre-partition eligible-city graph. Most positive memberships persist, and all 233 Zhengzhou held-out multi-hop positives also have multi-hop support before pruning. This rules out an explanation in which those cells were newly classified as multi-hop by the cut. It does **not** establish that their ranking losses are independent of boundary pruning: neighbour sets can change within a stable group and models are fitted under the prescribed partitions. A causal boundary effect requires another comparison; this descriptive audit supplies no such claim. A no-upstream label is absence of retained model-graph support, not absence of physical drainage or information in the cell's own attributes.

## Manuscript-ready Methods (~110 words)

We audited structural-group migration between the eligible-domain D8 graph and the frozen graph used for each outer test fold. Full-domain and held-out memberships were computed for the same cells using the no-upstream, one-hop-only and multi-hop definitions. Each cell entered the pooled transition matrix once, in its test fold; counts were reported separately for all eligible cells and recorded positives. Rows identify full-domain groups and columns held-out groups. We verified that the stored test edges exactly equal the full graph restricted to test endpoints, that pruning cannot increase group depth, and that held-out memberships reproduce the original analysis. Edge-retention rates use full-graph edges targeting the test cells as their denominator.

## Manuscript-ready Results (~150 words)

Outer-fold isolation changed the structural group of 24 of 791 recorded positives in Zhengzhou (3.03%) and 13 of 296 in Beijing (4.39%). Corresponding rates among all eligible cells were 2.95% and 3.12%. In Zhengzhou, 255 full-domain multi-hop positives became 233 multi-hop, 15 one-hop-only and seven unsupported positives in the held-out graphs; Beijing's 121 became 112, five and four, respectively. All held-out multi-hop positives therefore belonged to the same group before pruning. Among held-out no-upstream positives, nine of 420 in Zhengzhou and eight of 138 in Beijing lost their incoming support during pruning. The remaining unsupported positives already lacked incoming edges in the eligible-domain graph. Group stability coexisted with changes in neighbour sets: 54 and 24 positives lost upstream nodes within two hops. The transition audit consequently separates structural membership from the amount of support retained under spatial isolation.

## Reproduction

See the [package README](../README.md) for publication commands, required local artifacts and the distinction between native-run reanalysis and historical replay. Historical source and verification evidence are retained in provenance.
