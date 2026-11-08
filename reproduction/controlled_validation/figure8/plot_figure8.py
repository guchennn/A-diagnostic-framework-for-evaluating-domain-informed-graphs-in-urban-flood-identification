"""Controlled diagnostic validation; all points come from executed experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
ROOT = parser.parse_args().output_dir.resolve()
ROOT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.55,
        "axes.edgecolor": "#a1a7ac",
        "text.color": "#303b43",
        "axes.labelcolor": "#303b43",
        "xtick.color": "#47525a",
        "ytick.color": "#47525a",
        "legend.frameon": False,
    }
)
COLORS = ["#5fc0a6", "#8fa2cb", "#ff8b63", "#8b949c"]
inputs = [
    BASE / "archive_analysis/study_contrasts.csv",
    BASE / "archive_analysis/support_study.csv",
    BASE / "controlled_ranking/summary.csv",
]
archive, support, controlled = [pd.read_csv(p) for p in inputs]
for p in inputs:
    (ROOT / p.name).write_bytes(p.read_bytes())
plotted = []
fig, axes = plt.subplots(2, 2, figsize=(7.0866141732, 5.3543307087))
fig.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.92, wspace=0.31, hspace=1.08)
titles = [
    "Graph reference and evaluation",
    "Known support removal",
    "Cancellation within total AP",
    "Common versus within-group reference",
]
for i, (ax, title) in enumerate(zip(axes.flat, titles)):
    ax.set_title(title, loc="left", pad=9)
    ax.annotate(
        "abcd"[i],
        (0, 1),
        xycoords="axes fraction",
        xytext=(-26, 9),
        textcoords="offset points",
        weight="bold",
        fontsize=10,
    )
    ax.grid(color="#e6e8eb", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, pad=3)
    ax.axhline(0, color="#a9afb5", linewidth=0.7, linestyle="--", zorder=1)


def series(ax, x, mean, se, color, label, style="-", marker="o", oracle=None, panel=""):
    x, mean, se = [np.asarray(v, float) for v in (x, mean, se)]
    assert np.all(np.diff(x) > 0) and np.all(np.isfinite(mean))
    ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.14, linewidth=0)
    if oracle is None:
        ax.plot(
            x,
            mean,
            linestyle=style,
            marker=marker,
            markersize=3.2,
            linewidth=1.2,
            color=color,
            label=label,
        )
    else:
        ax.plot(x, oracle, linestyle=style, linewidth=1.2, color=color, label=label)
        ax.plot(x, mean, linestyle="none", marker=marker, markersize=3.2, color=color)
    for j in range(len(x)):
        plotted.append(
            {
                "panel": panel,
                "series": label,
                "x": x[j],
                "mean_pp": mean[j],
                "mcse_pp": se[j],
                "oracle_pp": None if oracle is None else oracle[j],
            }
        )


ax = axes[0, 0]
for k, contrast in enumerate(["useful - own", "useful - wrong"]):
    for evaluation, style in [("full", "-"), ("1to10", "--")]:
        data = archive[
            (archive.contrast == contrast)
            & (archive.evaluation == evaluation)
            & (archive.group == "total")
        ]
        stat = data.groupby("beta").ap_contribution_pp.agg(["mean", "std", "count"])
        assert (stat["count"] == 10).all()
        label = ("Useful − own" if k == 0 else "Useful − rewired") + (
            " · full" if evaluation == "full" else " · 1:10"
        )
        series(
            ax,
            stat.index,
            stat["mean"],
            stat["std"] / np.sqrt(10),
            COLORS[k],
            label,
            style,
            panel="a",
        )
ax.set(xlabel="Upstream signal coefficient", ylabel="AP difference (pp)", xticks=[0, 0.5, 1])
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, columnspacing=1, handlelength=2.2
)

ax = axes[0, 1]
for k, metric in enumerate(["background_rank", "threshold_precision"]):
    data = support[(support.evaluation == "full") & (support.metric == metric)]
    stat = data.groupby("beta").cut_minus_retained_pp.agg(["mean", "std", "count"])
    assert (stat["count"] == 10).all()
    series(
        ax,
        stat.index,
        stat["mean"],
        stat["std"] / np.sqrt(10),
        COLORS[k],
        ["Background rank", "Threshold precision"][k],
        marker=["o", "s"][k],
        panel="b",
    )
ax.set(xlabel="Upstream signal coefficient", ylabel="Cut − retained change (pp)", xticks=[0.5, 1])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=1)

data = controlled[
    (controlled.scenario == "redistribution")
    & (controlled.weight_id == 0)
    & (controlled.reference_id == 0)
]
ax = axes[1, 0]
for group in range(4):
    sel = data[
        (data.composition == 0)
        & (data.metric == ("global_ap" if group == 3 else "allocation"))
        & (data.group == (0 if group == 3 else group))
    ].sort_values("offset")
    series(
        ax,
        sel.offset,
        100 * sel["mean"],
        100 * sel.mcse,
        COLORS[group],
        ["Group A", "Group B", "Group C", "Total AP"][group],
        style="--" if group == 3 else "-",
        marker=["o", "s", "^", "D"][group],
        panel="c",
    )
ax.set(xlabel="Score offset", ylabel="AP contribution (pp)", xticks=[0, 0.25, 0.5, 1])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2)

ax = axes[1, 1]
for composition in range(2):
    for metric, style in [("rank", "-"), ("within_ap", "--")]:
        sel = data[
            (data.composition == composition) & (data.metric == metric) & (data.group == 0)
        ].sort_values("offset")
        label = ("A: 20%" if composition == 0 else "A: 40%") + (
            " · rank" if metric == "rank" else " · within AP"
        )
        series(
            ax,
            sel.offset,
            100 * sel["mean"],
            100 * sel.mcse,
            COLORS[composition],
            label,
            style,
            marker="o" if metric == "rank" else "s",
            oracle=100 * sel.oracle.to_numpy(),
            panel="d",
        )
ax.set(xlabel="Score offset", ylabel="Group A change (pp)", xticks=[0, 0.25, 0.5, 1])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=2, columnspacing=1)

fig.canvas.draw()
stem = ROOT / "Figure8_controlled_validation"
fig.savefig(ROOT / "Figure8_controlled_validation.pdf", facecolor="white")
fig.savefig(ROOT / "Figure8_controlled_validation.svg", facecolor="white")
fig.savefig(ROOT / "Figure8_controlled_validation.png", dpi=600, facecolor="white")
pd.DataFrame(plotted).to_csv(ROOT / "plotted_source_data.csv", index=False)
(ROOT / "input_hashes.json").write_text(
    json.dumps(
        {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs + [Path(__file__)]},
        indent=2,
    ),
    encoding="utf-8",
)
print(stem)
