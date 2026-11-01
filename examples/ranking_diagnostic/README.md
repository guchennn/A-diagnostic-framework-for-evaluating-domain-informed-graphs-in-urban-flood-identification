# Portable ranking diagnostic

This example is entirely synthetic. It requires Python and NumPy and does not
read CEUS data, fit models, or contact a service.

From the repository root:

```powershell
.venv/Scripts/python.exe scripts/diagnose_graph_ranking.py examples/ranking_diagnostic/synthetic.csv --output artifacts/runtime/ranking-example.json
```

For a standalone copy, place `diagnose_graph_ranking.py` beside `synthetic.csv`
and run `python diagnose_graph_ranking.py synthetic.csv --output result.json`.
The output file must not already exist.

Each row represents a member of one common bank, with aligned scores from two
models. `label=1` means recorded positive; `label=0` means unlabelled background,
not a known true negative. Use one nonempty group for each positive.
`bank_id,item_id` must be unique. Scores must be finite. Evaluation weights are
unit weights; non-unit weights are rejected rather than silently ignored.

Outputs give per-bank AP, paired differences, group contributions and midpoint
ECDF rank changes. Precision is always evaluated against the complete bank;
each positive receives its threshold precision, including the complete tie
block. Group contributions use the total positive count as denominator, so
they sum exactly to overall AP difference. Midpoint ranks award half credit
to exact ties. No item identifiers are included in the output.

In `perfect_vs_constant`, model A has AP 1, model B has AP 0.5, and the mean
positive midpoint-rank increment is +0.25. The two groups contribute +0.25 AP
each. This catches the upper-ECDF degeneracy: a constant score is uninformative,
not perfect. `group_accounting` shows that group contributions require the
whole bank, not independently calculated group AP.

The final two banks use the same four positives, four background members and
reference ordering. Both candidate rankings have AP 41/56 versus 11/21 for
the reference, an increment of 5/24. In `opposite_signs`, group A contributes
+11/168 to the AP difference while its mean midpoint rank changes by -3/16.
In `same_ap_different_allocation`, group A instead contributes +5/24 and its
rank changes by +1/8. Thus identical overall AP changes can have different
group allocations, and a positive AP contribution can coexist with a negative
mean rank change. These constructed examples establish those algebraic
possibilities; they do not establish an empirical effect or its cause.

These are descriptive per-bank diagnostics. Multiple banks, seeds and reveal
sequences from the same fitted spatial design are not independent geographic
replicates. CEUS aggregation first averages within each fold, then weights the
five folds equally. This small module demonstrates accounting and tie handling;
it does not validate causal mechanisms or reproduce the private case studies.

Reuse requires an exhaustive group definition chosen independently of the
evaluated differences. Attribute input, allowed computational nodes and graph
edges belong to the upstream prediction pipeline and must be documented
separately. Public availability of this example does not release restricted
reports or grant access to the case-study data.
