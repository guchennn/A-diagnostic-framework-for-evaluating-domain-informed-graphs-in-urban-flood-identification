# Topology-matched graph experiment (8 November 2026)

This is an author-requested, post-review experiment. The construction was chosen using
graph feasibility only, before generating or inspecting any new G7 model scores.
It is not a prospective preregistration. Existing G4/G6 results are separately identified.

## Question and comparisons

Does the tested D8 representation retain an advantage over a graph with the same node-wise
in/outdegrees, edge count, local spatial support and rook/diagonal edge counts, under the
same aggregation operator and independent model-selection rule?

Existing G6 is a completed receiver-indegree-matched local random-neighbour comparison.
It changes source outdegrees and distance composition. G7 adds those constraints.
Neither graph comparison identifies a physical water-transfer mechanism or tests every
possible graph representation.

## Graph construction fixed before fitting

Each training, inner-validation and outer-test partition is processed separately. Fixed
32-by-32-cell tiles use integer row/column coordinates and a common zero origin. Edges
crossing tile boundaries remain unchanged. Within a tile, eligible replacements join
the same-partition eight-neighbours. Binary assignment keeps every active source's one
outgoing edge, every receiver's incoming-edge count, and the tile's total diagonal-edge
count. Rook counts follow from the edge count. This preserves those quantities across
the whole partition, including zero-in/outdegree cells and the fixed-mean denominator.

The optimization prefers replacing original edges. A seeded secondary uniform cost
distinguishes otherwise tied assignments; no flood labels, features, scores or model
performance enter the objective. The solver is bounded at five seconds per tile with
relative objective gap 0.01. A feasible integer incumbent is accepted only after exact
constraint checks. A tile with no validated incumbent retains its original edges.
Every complete partition graph must change at least 7.5% of the original edges before
it is eligible for fitting. Actual changed fractions, overlap, solver gaps/fallbacks
and hashes are recorded; no uniform-random or globally optimal sampling claim is made.

Before any new model fitting, the initial 8% feasibility screen rejected three graph
realizations of Beijing outer fold 2, draw 5, inner fold 1's test partition. Each exact
tile solution replaced 941 of 11,779 edges (7.9888%); 8% would require 943. Re-solving
with zero relative gap gave the same optimum. The screen was therefore amended to
7.5% uniformly across the complete experiment. The single-origin graph algorithm and
all matching constraints remained unchanged. Initial files, failure logs and structural
probe summaries are retained. This is a graph-feasibility amendment before fitting,
not a threshold chosen from predictive results. Already accepted graph bytes were reused
after their original receipts and invariants were verified; their prior receipts are
archived and the configuration binding updated explicitly.

The tiled construction is needed because a whole-partition mixed-integer solve on the
largest Beijing training graph did not return an incumbent within its feasibility-probe
time limit. On that graph, the tiled feasibility probe changed about 12.3% of edges while
retaining the specified constraints. Stronger per-source distance-class matching was
also examined structurally; it left almost all edges fixed. These are construction
feasibility observations, not predictive model-selection results.

Three graph realizations use fixed seed namespaces. Outer graphs are constant across
the five label-revelation draws within a fold and realization. Inner graphs are bound to
their frozen fold/draw partitions. All three realizations enter the analysis; none is
selected by predictive performance.

## Fitting matrix

- Cities: Zhengzhou and Beijing; original five outer folds and five revelation draws.
- Attributes: original 18; revealed-positive budget: 100; unlabelled training sample: 2,000.
- Predictors: one-step fixed propagation and two-layer GraphSAGE.
- Each predictor/control-realization uses its own SMALL/MEDIUM selection on three inner
  folds; its median selected best epoch sets the outer refit duration.
- Three optimizer-seed slots per selected model. Numerical seed generation retains the
  original model/configuration namespace; integer seeds are recorded explicitly.
- Planned new work: 300 independent selections, 1,800 inner-fold fits and 900 outer fits.
- The frozen D8 fits supply the paired reference. The unchanged own/adjacency/G6 results
  retain their original provenance and are not represented as new fits.

Original protocol and data products remain immutable. A new execution namespace binds
the new code/configuration, original inputs, constructed graph caches and local environment.
The existing validated AutoDL environment lock is checked against the protocol; local
execution additionally requires a newly bound one-epoch, performance-blind resource probe.
No model metrics are calculated by that probe.

## Analysis fixed before fitting

Primary contrast: D8-minus-G7 sampled AP using the same three evaluation banks. Secondary
contrasts: full-domain AP and top-10% recall. Pair fold/draw/refit slot, average the three
graph realizations within each unit, then average draws/refits within folds and the five
folds equally. Report each graph-realization contrast as a sensitivity check; never
ensemble scores before computing metrics.

Use 10,000 paired fold resamples (seed 20261011) for descriptive 95% intervals, retaining
the original caution about overlapping training sets and five folds. Do not treat graph
realizations, seed slots or revelation draws as independent city replications. Publish
all outcomes, including null/negative contrasts, and the actual graph-change coverage.

The paper should integrate the new graph-control comparison into the existing Results
argument and replace the earlier statement that no usable city control existed. Keep
the matched constraints and interpretation in the main text, with full execution
metadata in the reproduction materials rather than expanding the paper into an audit log.
