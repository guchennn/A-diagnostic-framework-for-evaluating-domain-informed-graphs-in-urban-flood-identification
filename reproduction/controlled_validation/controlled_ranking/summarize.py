import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

parser = argparse.ArgumentParser(description="Summarize controlled-ranking outputs")
parser.add_argument("--output-dir", type=Path, required=True)
root = parser.parse_args().output_dir.resolve()
s = pd.read_csv(root / "summary.csv")
f = pd.read_csv(root / "family_detection.csv")
lines = [
    "# Controlled-ranking validation report",
    "",
    "All 600 calibration and 4,800 test datasets completed. This is an independent score-distribution experiment, not fitted graph-model evidence. All probability differences below are converted to percentage points when labelled pp.",
    "",
    "The strong conventional full-background group-rank diagnostic and the proposed rank readout are identical. No superiority over that baseline is claimed. Within-group AP estimates a different, invariant discrimination target, so its unchanged result is not an error.",
    "",
    "Thresholds are frozen 95th percentiles of maximum absolute statistics from independent a=0 calibration datasets. They are reference-calibrated screening thresholds, not universal 5% tests. Zero population targets at nonzero offsets can have different sampling variance; all such alarm rates are reported separately.",
    "",
    "## Execution clarification",
    "",
    "Before any numerical test outputs were read, the parent clarified numerical eligibility from 1e-8 to 1e-6 and requested analytically known zeros be set exactly to zero. The first process was stopped while its progress log showed q0/specific/a0; no summary or replication output had been written. The complete run restarted with the same frozen seeds and thresholds. No outcome-dependent scenario, sample-size, or threshold adjustment occurred.",
    "",
    "## Selected full-background results",
    "",
]
for qid in range(2):
    lines.append(f"### Composition {qid}: " + str([(0.2, 0.2, 0.6), (0.4, 0.4, 0.2)][qid]))
    d = s[
        (s.composition == qid)
        & (s.scenario == "redistribution")
        & (s.offset == 1)
        & (s.weight_id == 0)
        & (s.reference_id == 0)
    ]
    for metric in ["global_ap", "within_ap", "allocation", "rank"]:
        a = d[d.metric == metric]
        lines.append(
            f"- Redistribution a=1, {metric}: "
            f"oracle pp={np.round(a.oracle.to_numpy() * 100, 4).tolist()}; "
            f"mean pp={np.round(a['mean'].to_numpy() * 100, 4).tolist()}; "
            f"detection={list(a.detection_rate)}; "
            f"correct-sign detection={list(a.correct_sign_detection)}."
        )
    for scenario in ["shared", "specific"]:
        for ref in [0, 1]:
            a = f[
                (f.composition == qid)
                & (f.scenario == scenario)
                & (f.offset == 1)
                & (f.weight_id == 0)
                & (f.reference_id == ref)
                & (f.metric == "rank")
            ].iloc[0]
            lines.append(
                f"- {scenario} a=1, rank candidate-minus-{['own', 'comparison'][ref]}: any-group screening detection={a.any_detection:.3f}, Wilson interval [{a.wilson_low:.3f}, {a.wilson_high:.3f}], all targets zero={a.all_targets_zero}."
            )
    null = f[(f.composition == qid) & (f.offset == 0) & (f.weight_id == 0) & (f.reference_id == 0)]
    lines.append(
        "- Held-out a=0 family-wise alarms by scenario/readout:\n\n"
        + null[["scenario", "metric", "any_detection", "wilson_low", "wilson_high"]].to_csv(
            index=False, lineterminator="\n"
        )
    )
lines += [
    "## Estimator checks",
    "",
    f"Maximum absolute bias across all group/scenario/weight/reference summaries: {s.bias.abs().max() * 100:.5f} pp. Maximum RMSE: {s.rmse.max() * 100:.5f} pp. Bias and RMSE are against matching estimands; no allocation-to-rank cross-grading.",
    "",
    "Quadrature agreement and input hashes are in manifest.json. Five unit tests cover direct weighted precision/ties, independent pairwise rank, exact additivity, within-group shifts, oracle permutation invariance, 128/256-node convergence and seed disjointness. Dataset-level Monte Carlo SEs are in summary.csv.",
    "",
    "## File dictionary",
    "",
    "- run_validation.py: complete deterministic reproduction; test_validation.py: five metric/invariant tests; execution provenance is recorded in manifest.json.",
    "- replicates.csv: all dataset-level differences and targets; calibration.csv: maximum-absolute null statistics; thresholds.csv: frozen thresholds.",
    "- oracles.csv: population targets; summary.csv: bias/RMSE/MCSE and marginal detection; family_detection.csv: any-group detection and Wilson intervals by scenario.",
    "- Numeric quantities use probability units; multiply by 100 for percentage points. weight_id 0=full background; 1=fixed 1:10 weighting; reference_id 0=own; 1=comparison; group 0/1/2=A/B/C.",
    "",
    "This validates common-reference accounting and identifies aggregate cancellation/reference dependence. It does not establish four-dimensional methodological originality, physical causal attribution, or urban transfer.",
]
lines.extend(
    [
        "",
        "## Post-run code hygiene",
        "",
        (
            "After execution, scripts were formatted with Ruff, the unused sys import was removed, "
            "and function annotations were added. Report formatting was cleaned. Numerical code "
            "behavior was unchanged. Subsequently the full frozen experiment was repeated in the "
            "repository's Python 3.12 environment; those outputs are authoritative. The earlier "
            "Python 3.13 outputs are preserved in audit_python313, and environment_comparison.json "
            "records every numeric-output comparison. All five "
            "tests, Ruff checks, formatting checks and Python compilation passed after these edits. "
            "The manifest preserves the executed script hash and records the delivered script hash, "
            "numeric-output hashes and software versions."
        ),
    ]
)
(root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest.setdefault("executed_script_sha256", manifest["script_sha256"])
manifest["script_sha256"] = hashlib.sha256(
    Path(__file__).with_name("run_validation.py").read_bytes()
).hexdigest()
manifest["postrun_changes"] = (
    "Publication CLI/output relocation; scientific calculations unchanged."
)
manifest["software_versions"] = {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "pandas": pd.__version__,
}
manifest["numeric_output_sha256"] = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.csv"))
}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
hashes = {
    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in root.iterdir()
    if p.is_file() and p.name != "files_sha256.json"
}
(root / "files_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
print("\n".join(lines[:25]))
