# Controlled diagnostic validation

These materials support Methods 3.4, Results 4.5 and Figure 8. They contain synthetic dataset/group summaries and complete CPU score-distribution code, not urban cell records. No fitting of city models is involved.

## Independent score-distribution experiment

From the repository root, with Python 3.12 and requirements.txt installed:

```sh
python reproduction/controlled_validation/controlled_ranking/test_validation.py
python reproduction/controlled_validation/controlled_ranking/run_validation.py --output-dir reproduced/controlled_ranking
python reproduction/controlled_validation/controlled_ranking/summarize.py --output-dir reproduced/controlled_ranking
```

The complete run creates 600 calibration and 4,800 independent test datasets (12,000 simulated cells each). It uses NumPy, pandas and SciPy. The supplied `controlled_ranking/*.csv` are the previously executed results: `replicates.csv` contains 192,000 dataset/group/metric rows (16.9 MB), not individual observations. Supplied historical execution/software hashes are in `manifest.json`; publication file hashes are recorded separately in `publication_manifest.json`. Screening thresholds are calibrated to the specified null distribution, not universal significance tests. The conventional complete-background rank comparator and framework rank readout are identical.

## Archived positive-unlabelled graph fits

`archive_analysis/` supplies aggregate evidence for 150 completed synthetic grids and the read-only reanalysis. To replay, restore the trusted original local archives beneath an input project directory: `results/synthetic_pu_regime_20261029/`, `results/synthetic_pu_regime_20261029_controls/`, and their corresponding `_cache` directories. These contain configuration/aggregate reference files and per-grid pickle fit caches, identified by hashes in `historical_input_manifest.json`. Original cached predictions are not distributed here. Only load trusted original pickles.

```sh
python reproduction/controlled_validation/archive_analysis/reanalyze_archive.py --project-root . --output-dir reproduced/archive_analysis
```

The loader reads cached per-positive AP terms and ranks and never calls a fitting routine. Zero-signal truncation is explicitly unavailable. No reconstruction of cached fits from aggregate tables is claimed. Original provenance retains scientific SHA-256 hashes with project-relative paths; current publication sources are hashed separately.

## Figure 8

The figure can be rendered entirely from the supplied aggregate tables. Rendering additionally uses Matplotlib (verified with 3.11.1); it no longer requires an installed author skill or private QA helper.

```sh
python -m pip install -r reproduction/controlled_validation/requirements-plot.txt
python reproduction/controlled_validation/figure8/plot_figure8.py --output-dir reproduced/figure8
```

The original PDF/SVG/PNG, caption and plotted source table are supplied. The plotting script reads its sibling aggregate directories and records input hashes beside regenerated outputs. Arial is the original font; on systems without it, Matplotlib uses its configured fallback. This does not alter data.

## Verification

Five tests cover tied threshold AP and weighting, pairwise rank, additive accounting, within-group shift invariance, population permutation/quadrature, and disjoint random streams. Publication tests also verify portable import without private data, cache-based AP reweighting, and supplied aggregate accounting. Local replay has been compared numerically with the published archive tables; no original fits were rerun.

Run `python -I -S reproduction/controlled_validation/verify_published.py` for dependency-free hash checks, the 600/4,800 dataset counts and additive AP accounting in the supplied simulated results.
