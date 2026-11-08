# Controlled ranking validation — frozen before test execution

This is a score-distribution experiment, separate from the archived fitted-graph experiments. It tests common-reference interpretation, not physical flood mechanisms or universal superiority of four dimensions. Created 2026-11-08 before generating test datasets. No threshold or scenario selection will use test output.

## Population and replication

Each independent dataset has 600 positives and 11,400 backgrounds. Both classes have the same three-stratum proportions, either (0.2, 0.2, 0.6) or (0.4, 0.4, 0.2), allocated exactly. Draw a shared latent score Z = 1.5Y + epsilon with epsilon ~ N(0,1). Three paired scorers have S_m = Z + t_m,G + eta_m, where eta_m ~ N(0,0.25^2) independently across scorers. All models therefore have common marginal variance v = 1.0625. Strata are synthetic labels A/B/C, not urban upstream-support classes.

Three scenarios at offsets a = 0, 0.25, 0.5, 1.0:

1. Shared displacement: own t=(0,0,0); comparison and candidate t=(-a,0,0).
2. Candidate-specific displacement: own and comparison t=(0,0,0); candidate t=(-a,0,0).
3. Redistribution: own and comparison t=(a,0,0); candidate t=(0,a,0).

In scenario 3, the equal shares of A/B give identical candidate/reference population score mixtures and exactly equal global AP, while group allocations can change. Offsets apply to both classes, so within-stratum population discrimination stays constant in every scenario. These properties follow from the specified distributions, not from observed test outcomes.

Use 300 independent null calibration datasets per composition, then 200 independent test datasets for every composition/scenario/offset. Numpy SeedSequence entropy 2026110803 with separate identifiers for calibration and each test cell; record the complete seed scheme. Pair scorers within a dataset; datasets, not cells or strata, are independent replications. Dataset draws are independent across experimental cells.

## Readouts and independent population targets

Compute all readouts from exactly the same paired predictions: global AP; AP recalculated separately inside each stratum; full-ranking AP contributions (sum of positive precision terms in a stratum divided by the total positive count); and mean positive rank against the complete background. Evaluate candidate minus own and candidate minus comparison. Evaluate full background and fixed 1:10 reweighting with w=10*600/11400.

Independent analytic background-rank target:

R_m,g = sum_h q_h Phi((1.5+t_m,g-t_m,h)/sqrt(2v)).

Population AP allocation is obtained by independent normal-mixture quadrature: integrate precision TP(s)/(TP(s)+w FP(s)) over the positive score density of each stratum and multiply by q_g. TP and FP are analytic normal survival mixtures with prevalence 0.05. Use 128-point Gauss-Hermite quadrature, verify against 256 points before accepting values. Tolerance 1e-7 in probability units; if not met, report and increase quadrature order without changing scenarios. Analytic zero within-stratum AP contrasts and identical mixture global AP provide additional independent checks. Do not equate AP-contribution signs with rank-change signs, or the shifted group with the sole affected group.

## Comparators, calibration and reporting

Reports: (i) global AP alone; (ii) global plus within-stratum AP; (iii) global plus full-background stratum rank; (iv) full-ranking AP allocation plus rank and matched reference contrasts. The rank readout in (iii) is mathematically the same as in (iv); explicitly report this equivalence. Global AP is not assigned a subgroup-localization error score. Within-stratum AP is assessed against its own zero target, not declared incorrect for estimating a different quantity.

For each composition/readout/weight/reference, independently calibrate the 95th percentile (higher empirical quantile) of maximum absolute null differences across three strata; global AP uses absolute global difference. On held-out null datasets report family-wise false-alarm rates and Wilson intervals. Under nonzero scenarios report correct-sign detection for population effects exceeding 1e-6; zero targets are reported separately. Also report bias and RMSE against matching population targets, all 200 replicates, and Monte Carlo SEs. No cell-level inferential replication or threshold tuning.

Retain all scenario strengths, nulls, two compositions and both reference comparisons in source tables. A compact manuscript figure may show one composition with the second explicitly summarized in text and retained in source data; no condition may be discarded because of its outcome. All results and scripts accompany the manuscript delivery, without invoking a supplementary manuscript section.

## Required checks and claim boundary

Test AP and ties against an independent direct threshold-count implementation, background rank against pairwise comparisons, weighted AP against direct weighted precision, full-ranking additivity, within-stratum shift invariance, analytic permutation invariance, seed disjointness, and quadrature convergence. Preserve source/config hashes and all output tables.

The experiment may establish recovery of known distribution-level changes, aggregate cancellation and reference dependence. It cannot establish physical causal attribution, empirical city transfer, or unique necessity of four dimensions. Strong conventional diagnostics may be equivalent; preserve that finding.

Pre-run interpretation clarification: thresholds calibrated at a=0 are calibration-reference screening thresholds. They do not guarantee 5% false alarms for every zero population target at a>0, because its sampling variance can differ. Report zero-target alarms separately by scenario and strength, including shared-displacement candidate–comparison and redistribution global AP; do not pool them into a universal nominal-5% claim. This clarification does not change any thresholds, scenarios or seeds.

Numerical clarification before inspecting test output: analytically known zero contrasts are set to zero; effect eligibility uses 1e-6, above the 1e-7 quadrature tolerance. This changes no test statistic or calibration threshold.
