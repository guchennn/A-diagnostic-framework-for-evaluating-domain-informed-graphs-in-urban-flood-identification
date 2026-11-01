# FINAL-v8 attribute-ablation replay

## Supported execution

`scripts/reproduce_ablation.py` defaults to the published
`provenance/selected_ablation.csv`, which must exist. It validates all 180 original
unit/family records and replays budget 100 using `e1._outer_fold_data` and
`e1._predict_outer`. Scope is Zhengzhou, STRICT-LOC, uniform reveal, five folds,
five trajectories and the frozen 18/16/14-feature sets. There are 75 units and
750 actual fits. G0 and G1 inherit D8/G2's recorded config and median inner-fold
epoch count. XGBoost retains its separate selection and replicates its single fit
across three paired output slots.

```powershell
uv run --no-sync python scripts/reproduce_ablation.py plan --budget 100
uv run --no-sync python scripts/reproduce_ablation.py plan --budget FULL
uv run --no-sync python scripts/reproduce_ablation.py run --budget 100 --output reproduced/ablation-selection-replay
```

Training requires authorized prepared inputs, the locked train environment and the
existing reviewer-revision environment freeze and performance-blind probe workflow.
The script calls `prepare_plan` and `require_revision_environment` before reading
city data or fitting. The reviewer gate APIs retain their existing behavior.
Output must be a new directory under the prepared root's `reproduced/`. Each unit
writes local cell predictions and a fit receipt. The manifest records input-code
hashes, gate identity and completed units. Restart interrupted runs into a new
directory. Keep cell-level outputs outside public Git.

## FULL recovery result

FULL remains blocked. The reselected run's manifest records trajectory `NA`, five
folds and three seed slots. Its selection table records configurations but omits
all 15 neural refit-epoch records. The original generator and its exact optimizer
seed construction were not recovered. The available evidence does not support
filling those gaps from another run.

The released aggregate results cover both budgets. FULL values can be checked
with `scripts/reproduce_tables.py`. Replaying the FULL fits requires the original
reselected-fit epoch and seed records.

## Seed provenance and limits

Budget-100 seed identity follows the recovered RQ2/E1 source. Tasks retain
`FULL-COMMON`, experiment `E1`, and seed slots `NEURAL-1` through `NEURAL-3`.
The namespace is
`{protocol_id}/{experiment_id}/{city}/{outer_fold}/{budget}/{trajectory_id}/{model}/{config}/{seed_id}`.
SHA-256's first 16 bytes form an unsigned big-endian seed. Torch reduces it modulo
2^63 and XGBoost modulo 2^31. Graph and feature-set identities are absent from this
namespace. Fit receipts record seed names, numeric seeds, configs and epochs.
Prepared reveals, U samples, banks and train-scope preprocessing use existing E1
helpers. Selection is replayed without independent control-model tuning.

Numerical replication remains unverified. The historical RQ2 manifest records
Python 3.13.5 and torch 2.8.0+cu129, while the release gate requires Python 3.12
and torch 2.8.0/CUDA 12.8. The current XGBoost helper uses `min_child_weight=0`,
consistent with the documented local ablation lineage. This entrypoint produces
fit outputs. Downstream aggregate and bootstrap verification is separate.

## Verification

```powershell
uv run --no-sync python -m pytest tests/test_release_ablation.py -q --basetemp reproduced/pytest-ablation
uv run --no-sync python -m ruff check scripts/reproduce_ablation.py tests/test_release_ablation.py
uv run --no-sync python -m ruff format --check scripts/reproduce_ablation.py tests/test_release_ablation.py
```

Synthetic/mock tests cover feature membership, inherited schedules, paired seeds,
tree-slot replication, corruption rejection, FULL blocking, gate ordering,
complete mock-run receipts, overwrite refusal and the published default selection
file from another working directory. No city training was performed.

## Fresh budget-100 selection and fitting

The retained RQ2 implementation performs the original budget-100 inner selection,
refitting and paired fold analysis. It reads the released FULL aggregate constants
from `results/feature_ablation/inference.csv` and verifies their recorded SHA-256.
After preparing both cities and completing the local environment and resource probe,
run from the repository root.

```sh
uv run --no-sync python -m floodgnn_ceus.rq2_scarcity --root . --output-root reproduced/ablation-fresh
```

Use `--resume` to continue completed units in the same output directory. This run
estimates budget-100 effects and the specified interaction against the recorded
FULL constants. The seed and environment differences described above remain relevant
when comparing fresh values with the published results.
