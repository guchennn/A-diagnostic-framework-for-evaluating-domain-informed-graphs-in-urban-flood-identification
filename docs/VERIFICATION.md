# Release verification

Verification date is 31 October 2026. A fresh Python 3.12.11 environment was
created from `uv.lock` with the `figures` dependency group.

| Check | Result |
| --- | --- |
| CPU test suite | 436 passed, 5 GPU-specific tests skipped |
| Ruff lint and format checks | Passed |
| Tables 2 and 3 | Recomputed fit-to-fold and fold-to-city means agree with released aggregates |
| Table 4 | Verified 1,200 fit/group aggregates, 1,200 ordered pair rows, eight reference summaries and 40 archived graph-fold effects |
| Figures 3 and 4 | Regenerated from all 50 matched-contrast fold points and 60 structural group fold points |
| Synthetic validation | All ten replicates rerun and all released CSV values matched within 1e-10 |
| Small ranking example | Executed successfully |
| Budget-100 ablation plan | Validated the released selections for 75 units and 750 fits |
| Local command interfaces | Preparation, planning and analysis entry points load successfully |

The tests cover exact-score ties, weighted metrics, additive AP accounting,
spatial partitions, fitting scopes, matched graph routing, seed identity,
immutable input/environment gates, receipt validation and corrupted-input rejection.
The published synthetic study was rerun separately from the unit tests.

## Execution coverage

City-scale data acquisition and model training require provider accounts,
prepared local inputs and a compatible GPU. These were not rerun for this release.
Budget-100 ablation uses recorded selections with the current fitting helpers.
Its full numerical replay has not been verified. The historical all-positive-label
ablation lacks the original refit-epoch and optimizer-seed records, as detailed
in the ablation guide. [Ablation scope](ABLATION.md).

The GitHub workflow installs the locked environment and repeats CPU tests,
aggregate checks and result-figure generation on Windows and Linux. Its run
status supplies the platform-specific verification record.

## Local commands

```sh
uv sync --frozen --group figures
uv run ruff check src scripts analysis plots tests
uv run ruff format --check src scripts analysis plots tests
uv run --group figures pytest -q --basetemp .pytest_tmp
uv run python scripts/reproduce_tables.py
uv run python analysis/seed/verify_aggregate_arithmetic.py
uv run --group figures python plots/make_result_figures.py --output reproduced/figures
uv run python -m floodgnn_ceus.synthetic_diagnostic_validation --output reproduced/synthetic
```

`provenance/source_files.json` records the released source hashes and the
corresponding manuscript-snapshot hashes. The original protocol identity is
archived in `provenance/protocol_lock.json`. Each fresh preparation creates its
own active protocol lock through the documented workflow.
