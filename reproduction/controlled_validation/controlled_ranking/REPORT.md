# Controlled-ranking validation report

All 600 calibration and 4,800 test datasets completed. This is an independent score-distribution experiment, not fitted graph-model evidence. All probability differences below are converted to percentage points when labelled pp.

The strong conventional full-background group-rank diagnostic and the proposed rank readout are identical. No superiority over that baseline is claimed. Within-group AP estimates a different, invariant discrimination target, so its unchanged result is not an error.

Thresholds are frozen 95th percentiles of maximum absolute statistics from independent a=0 calibration datasets. They are reference-calibrated screening thresholds, not universal 5% tests. Zero population targets at nonzero offsets can have different sampling variance; all such alarm rates are reported separately.

## Execution clarification

Before any numerical test outputs were read, the parent clarified numerical eligibility from 1e-8 to 1e-6 and requested analytically known zeros be set exactly to zero. The first process was stopped while its progress log showed q0/specific/a0; no summary or replication output had been written. The complete run restarted with the same frozen seeds and thresholds. No outcome-dependent scenario, sample-size, or threshold adjustment occurred.

## Selected full-background results

### Composition 0: (0.2, 0.2, 0.6)
- Redistribution a=1, global_ap: oracle pp=[0.0]; mean pp=[-0.0088]; detection=[0.145]; correct-sign detection=[nan].
- Redistribution a=1, within_ap: oracle pp=[0.0, 0.0, 0.0]; mean pp=[0.2439, -0.1315, 0.0719]; detection=[0.05, 0.015, 0.0]; correct-sign detection=[nan, nan, nan].
- Redistribution a=1, allocation: oracle pp=[-3.2748, 3.2748, 0.0]; mean pp=[-3.3606, 3.3307, 0.0211]; detection=[1.0, 1.0, 0.01]; correct-sign detection=[1.0, 1.0, nan].
- Redistribution a=1, rank: oracle pp=[-12.9666, 12.9666, 0.0]; mean pp=[-12.8448, 12.9107, 0.0125]; detection=[1.0, 1.0, 0.0]; correct-sign detection=[1.0, 1.0, nan].
- shared a=1, rank candidate-minus-own: any-group screening detection=1.000, Wilson interval [0.981, 1.000], all targets zero=False.
- shared a=1, rank candidate-minus-comparison: any-group screening detection=0.105, Wilson interval [0.070, 0.155], all targets zero=True.
- specific a=1, rank candidate-minus-own: any-group screening detection=1.000, Wilson interval [0.981, 1.000], all targets zero=False.
- specific a=1, rank candidate-minus-comparison: any-group screening detection=1.000, Wilson interval [0.981, 1.000], all targets zero=False.
- Held-out a=0 family-wise alarms by scenario/readout:

scenario,metric,any_detection,wilson_low,wilson_high
shared,global_ap,0.03,0.0138203143401113,0.0638942924545192
shared,within_ap,0.05,0.0273826456007639,0.0895781481387759
shared,allocation,0.045,0.0238525430014929,0.0832967040018196
shared,rank,0.03,0.0138203143401113,0.0638942924545192
specific,global_ap,0.07,0.0421521718283428,0.1140548088561064
specific,within_ap,0.055,0.0309853417034588,0.0957869987723084
specific,allocation,0.065,0.0383763546491529,0.1080190792990689
specific,rank,0.025,0.0107247196517612,0.0571783404066419
redistribution,global_ap,0.075,0.045974917184382,0.1200436102362945
redistribution,within_ap,0.055,0.0309853417034588,0.0957869987723084
redistribution,allocation,0.06,0.0346521942543318,0.1019316929576627
redistribution,rank,0.03,0.0138203143401113,0.0638942924545192

### Composition 1: (0.4, 0.4, 0.2)
- Redistribution a=1, global_ap: oracle pp=[0.0]; mean pp=[-0.0223]; detection=[0.24]; correct-sign detection=[nan].
- Redistribution a=1, within_ap: oracle pp=[0.0, 0.0, 0.0]; mean pp=[-0.0691, 0.0879, 0.062]; detection=[0.005, 0.005, 0.045]; correct-sign detection=[nan, nan, nan].
- Redistribution a=1, allocation: oracle pp=[-6.1875, 6.1875, 0.0]; mean pp=[-6.2564, 6.2385, -0.0044]; detection=[1.0, 1.0, 0.0]; correct-sign detection=[1.0, 1.0, nan].
- Redistribution a=1, rank: oracle pp=[-15.0764, 15.0764, 0.0]; mean pp=[-15.1121, 15.1023, 0.0526]; detection=[1.0, 1.0, 0.05]; correct-sign detection=[1.0, 1.0, nan].
- shared a=1, rank candidate-minus-own: any-group screening detection=1.000, Wilson interval [0.981, 1.000], all targets zero=False.
- shared a=1, rank candidate-minus-comparison: any-group screening detection=0.005, Wilson interval [0.001, 0.028], all targets zero=True.
- specific a=1, rank candidate-minus-own: any-group screening detection=1.000, Wilson interval [0.981, 1.000], all targets zero=False.
- specific a=1, rank candidate-minus-comparison: any-group screening detection=1.000, Wilson interval [0.981, 1.000], all targets zero=False.
- Held-out a=0 family-wise alarms by scenario/readout:

scenario,metric,any_detection,wilson_low,wilson_high
shared,global_ap,0.055,0.0309853417034588,0.0957869987723084
shared,within_ap,0.06,0.0346521942543318,0.1019316929576627
shared,allocation,0.085,0.0537457501774756,0.1318958707156556
shared,rank,0.03,0.0138203143401113,0.0638942924545192
specific,global_ap,0.065,0.0383763546491529,0.1080190792990689
specific,within_ap,0.06,0.0346521942543318,0.1019316929576627
specific,allocation,0.07,0.0421521718283428,0.1140548088561064
specific,rank,0.03,0.0138203143401113,0.0638942924545192
redistribution,global_ap,0.03,0.0138203143401113,0.0638942924545192
redistribution,within_ap,0.06,0.0346521942543318,0.1019316929576627
redistribution,allocation,0.01,0.0027466581335444,0.0357217617161768
redistribution,rank,0.055,0.0309853417034588,0.0957869987723084

## Estimator checks

Maximum absolute bias across all group/scenario/weight/reference summaries: 0.36226 pp. Maximum RMSE: 1.98446 pp. Bias and RMSE are against matching estimands; no allocation-to-rank cross-grading.

Quadrature agreement and input hashes are in manifest.json. Five unit tests cover direct weighted precision/ties, independent pairwise rank, exact additivity, within-group shifts, oracle permutation invariance, 128/256-node convergence and seed disjointness. Dataset-level Monte Carlo SEs are in summary.csv.

## File dictionary

- run_validation.py: complete deterministic reproduction; test_validation.py: five metric/invariant tests; manifest.json: historical execution provenance.
- replicates.csv: all dataset-level differences and targets; calibration.csv: maximum-absolute null statistics; thresholds.csv: frozen thresholds.
- oracles.csv: population targets; summary.csv: bias/RMSE/MCSE and marginal detection; family_detection.csv: any-group detection and Wilson intervals by scenario.
- Numeric quantities use probability units; multiply by 100 for percentage points. weight_id 0=full background; 1=fixed 1:10 weighting; reference_id 0=own; 1=comparison; group 0/1/2=A/B/C.

This validates common-reference accounting and identifies aggregate cancellation/reference dependence. It does not establish four-dimensional methodological originality, physical causal attribution, or urban transfer.

## Post-run code hygiene

After execution, scripts were formatted with Ruff, the unused sys import was removed, and function annotations were added. Report formatting was cleaned. Numerical code behavior was unchanged. Subsequently the full frozen experiment was repeated in the repository's Python 3.12 environment; those outputs are authoritative. The earlier Python 3.13 outputs are preserved in audit_python313, and environment_comparison.json records every numeric-output comparison. All five tests, Ruff checks, formatting checks and Python compilation passed after these edits. The manifest preserves the executed script hash and records the delivered script hash, numeric-output hashes and software versions.