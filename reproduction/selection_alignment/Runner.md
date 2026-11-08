# Independently selected attribute controls

`run_independent_attribute_controls.py` implements the independent-selection series
used in the revised manuscript. It is separate from the legacy inherited-selection
`rq2_scarcity` / `reproduce_ablation` workflow; their estimands and outputs must not
be mixed. Existing published aggregate evidence is historical. This new portable
runner has CPU contract tests but has not numerically replayed either city.

## Frozen design

Both cities use five outer spatial folds. With 100 revealed positives, the runner
uses five revelation draws and the 16- and 14-attribute sets. With all training
positives (`FULL`), it uses the single frozen draw `NA` and the 18-, 16- and
14-attribute sets. The existing main run supplies the 18-attribute/100-positive
reference; this runner does not recreate or replace it.

Each condition independently selects four procedures: fixed one-step propagation
with own attributes or D8, and two-layer GraphSAGE with own attributes or D8.
Each procedure compares SMALL/MEDIUM on three inner folds. Its own selected
configuration and median best epoch determine three fresh outer refits.
The complete matrix comprises 130 condition blocks, 520 independent selections,
3,120 inner-fold fits and 1,560 outer fits. Feature subsets preserve the original
column order. No own-attribute setting is inherited from D8.

## Local execution

Use the repository's Python 3.12 training environment and frozen preparation inputs.
All commands run from the repository root:

```text
python scripts/run_independent_attribute_controls.py plan
python scripts/run_independent_attribute_controls.py check-inputs
python scripts/run_independent_attribute_controls.py freeze
python scripts/run_independent_attribute_controls.py probe
python scripts/run_independent_attribute_controls.py train
```

`plan` reads configuration only. `check-inputs` materializes each distinct inner and
outer training condition through the existing E1 preparation helpers, without fitting
models. `freeze` repeats that check and binds all input and source hashes to a new
local execution namespace. Missing FULL-specific inner partitions, reveals,
unlabelled samples or evaluation banks cause failure; they are never substituted
with the 100-positive design or synthesized from its records. These inputs must be
present in the frozen `data/interim/<city>/evaluation_design/` tables.

The performance-blind one-epoch probe uses the existing public `train.py` gate.
Training requires both that gate and a probe bound to the attribute-run lock.
Outputs reside under `artifacts/runs/independent-attributes-v1/`. Sources and inputs
are immutable after freezing. `--city`, `--start` and `--limit` select disjoint blocks
for execution, with 65 ordered blocks per city. Resume verifies inner-score and
outer-OOF payload hashes before skipping completed work.

Selection JSON files retain the three best inner epochs, selected candidate and
scheduled refit duration. Every inner candidate also records its seed namespace,
128-bit seed and Torch seed. Outer fit metadata record the same seed information
and duration. Seed generation uses the existing E1 namespace, including model,
configuration, budget and draw. Because selection is independent, paired procedures
can select different configurations and thus different numerical seeds.

## Verification and interpretation

CPU tests check the 520/1,560 matrix, FULL/NA task identity, frozen column order,
independent selection across all four variants, median refit durations, seed
metadata, missing-input refusal and resume hash integrity. Tests mock fitting and
perform no GPU work. CLI planning, compilation and Ruff checks also pass.

A successful plan or CPU test is not a completed city experiment. Fresh numerical
agreement with the historical independent series remains unverified. Compare any
new results only after the complete paired run finishes, using the same outer
partitions, evaluation banks and full-ranking metric definitions.
