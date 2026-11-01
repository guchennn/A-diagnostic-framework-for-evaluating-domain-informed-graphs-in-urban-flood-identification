"""Render matched utility and structural diagnostics from reviewed aggregate CSVs.

Adapted from the manuscript figure_code/make_result_figures.py.
All retained outer folds are plotted without sampling. Use --output to choose
the destination for PDF, SVG, PNG and numeric verification outputs.
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DATA = Path(__file__).resolve().parent / "data"
CITIES = ["zhengzhou", "beijing"]
CITY_COLORS = {"zhengzhou": "#C8773C", "beijing": "#377E84"}
GROUP_COLORS = {"0": "#D9D9D9", "1": "#9ECAE1", "2": "#2F6DB5"}
GROUP_LABELS = {"0": "No upstream", "1": "One hop", "2": "Multi-hop"}
INK, BLUE = ("#263640", "#306DAD")
WIDTH_IN = 180 / 25.4
CONTRASTS = [
    ("FF-G2-1 - FF-G0-1", "Fixed propagation (1 step)\nD8 − own attributes"),
    ("FF-G2-1 - FF-G1-1", "Fixed propagation (1 step)\nD8 − adjacency"),
    ("SAGE-G2-2 - SAGE-G0-2", "GraphSAGE (2 layers)\nD8 − own attributes"),
    ("SAGE-G2-2 - SAGE-G1-2", "GraphSAGE (2 layers)\nD8 − adjacency"),
    ("FF-G2-1 - SAGE-G2-1", "Fixed propagation − GraphSAGE\nD8, matched 1 hop"),
]
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }
)


def select_one(frame, **criteria):
    out = frame
    for key, val in criteria.items():
        out = out[out[key].eq(val)]
    if len(out) != 1:
        raise ValueError((criteria, len(out)))
    return out.iloc[0]


def folded(frame, **criteria):
    out = frame
    for key, val in criteria.items():
        out = out[out[key].eq(val)]
    out = out.sort_values("outer_fold")
    if list(out.outer_fold) != [0, 1, 2, 3, 4]:
        raise ValueError(criteria)
    return out


def check_data(data, qa):
    f = pd.read_csv(data / "fig3_fold_contrasts.csv")
    s = pd.read_csv(data / "fig3_matched_contrasts.csv")
    gf = pd.read_csv(data / "fig4_structure_folds.csv", dtype={"group": str})
    gs = pd.read_csv(data / "fig4_structure_summary.csv", dtype={"group": str})
    for frame, expected in [(f, 50), (s, 10), (gf, 80), (gs, 16)]:
        if len(frame) != expected:
            raise ValueError(f"Expected {expected} figure data rows, got {len(frame)}")
    for summary in [s, gs]:
        for original, plotted in [
            ("mean", "mean_pp"),
            ("resampling_q025", "range_low_pp"),
            ("resampling_q975", "range_high_pp"),
        ]:
            if not np.allclose(summary[plotted], summary[original] * 100, atol=1e-12, rtol=0):
                raise ValueError(f"Incorrect percentage-point conversion in {plotted}")
    records = []
    for city in CITIES:
        for contrast, _ in CONTRASTS:
            folds = folded(f, city=city, contrast=contrast)
            sm = select_one(s, city=city, contrast=contrast)
            if not np.allclose(folds.effect_pp, folds.effect * 100, atol=1e-12, rtol=0):
                raise ValueError("Figure aggregate validation failed")
            if not abs(folds.effect_pp.mean() - sm.mean_pp) < 1e-10:
                raise ValueError("Figure aggregate validation failed")
            for row in folds.itertuples():
                records.append(
                    {
                        "figure": 3,
                        "city": city,
                        "contrast": contrast,
                        "fold": int(row.outer_fold),
                        "plotted_value_pp": row.effect_pp,
                    }
                )
        for metric in ["ap_contribution", "background_mean_gain"]:
            for group in ["0", "1", "2", "all"]:
                folds = folded(gf, city=city, metric=metric, group=group)
                sm = select_one(gs, city=city, metric=metric, group=group)
                if not np.allclose(folds.value_pp, folds.value * 100, atol=1e-12, rtol=0):
                    raise ValueError("Figure aggregate validation failed")
                if not abs(folds.value_pp.mean() - sm.mean_pp) < 1e-10:
                    raise ValueError("Figure aggregate validation failed")
                for row in folds.itertuples():
                    records.append(
                        {
                            "figure": 4,
                            "city": city,
                            "metric": metric,
                            "group": group,
                            "fold": int(row.outer_fold),
                            "plotted_value_pp": row.value_pp,
                            "shown_as_fold_point": group != "all",
                        }
                    )
        for fold in range(5):
            part = gf[
                (gf.city == city) & (gf.outer_fold == fold) & (gf.metric == "ap_contribution")
            ]
            if (
                not abs(
                    part[part.analysis == "L"].value.sum()
                    - part[part.analysis == "overall"].value.iloc[0]
                )
                < 1e-12
            ):
                raise ValueError("Figure aggregate validation failed")
    (qa / "results_point_verification.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return (f, s, gf, gs)


def save(fig, name, qa):
    fig.canvas.draw()
    fig.savefig(qa / f"{name}.pdf")
    fig.savefig(qa / f"{name}.svg")
    fig.savefig(qa / f"{name}.png", dpi=600)
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=8)
    ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-11, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=10,
        weight="bold",
    )


def utility(f, s, qa):
    fig = plt.figure(figsize=(WIDTH_IN, 100 / 25.4))
    grid = fig.add_gridspec(1, 2, left=0.35, right=0.975, bottom=0.21, top=0.88, wspace=0.22)
    y = np.array([4.6, 3.6, 2.6, 1.6, 0.0])
    offsets = np.linspace(0.08, 0.3, 5)
    for i, city in enumerate(CITIES):
        ax = fig.add_subplot(grid[0, i])
        for yy, (code, _) in zip(y, CONTRASTS):
            values = folded(f, city=city, contrast=code).effect_pp.to_numpy()
            sm = select_one(s, city=city, contrast=code)
            ax.scatter(
                values, yy + offsets, s=11, color="#626C73", alpha=0.85, linewidths=0, zorder=3
            )
            ax.errorbar(
                sm.mean_pp,
                yy - 0.13,
                xerr=[[sm.mean_pp - sm.range_low_pp], [sm.range_high_pp - sm.mean_pp]],
                fmt="D",
                ms=3.7,
                mfc=BLUE,
                mec="white",
                mew=0.45,
                color=BLUE,
                elinewidth=1.15,
                capsize=2,
                zorder=4,
            )
        ax.axvline(0, color=INK, lw=0.8)
        ax.axhline(0.82, color="#DCE1E4", lw=0.7)
        ax.set_xlim(-6.7, 6.7)
        ax.set_xticks([-6, -3, 0, 3, 6])
        ax.set_ylim(-0.6, 5.25)
        ax.set_yticks(y)
        ax.set_yticklabels([c[1] for c in CONTRASTS] if i == 0 else [])
        ax.tick_params(axis="y", length=0, pad=8)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color="#E9ECEE", lw=0.45)
        ax.set_axisbelow(True)
        panel(ax, "ab"[i], city.title())
        if not f[f.city.eq(city)].effect_pp.between(*ax.get_xlim()).all():
            raise ValueError("Figure aggregate validation failed")
    fig.text(0.66, 0.11, "Paired difference in sampled AP (pp)", ha="center", fontsize=8)
    fig.legend(
        handles=[
            Line2D([], [], color="#626C73", marker="o", ls="", ms=3, label="Outer fold (n = 5)"),
            Line2D(
                [], [], color=BLUE, marker="D", ms=3.5, lw=1.1, label="Mean and descriptive range"
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.62, 0.015),
        ncol=2,
        handlelength=1.3,
        columnspacing=1.8,
    )
    save(fig, "fig3_matched_increment", qa)


def structure(gf, gs, qa):
    fig = plt.figure(figsize=(WIDTH_IN, 139 / 25.4))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.975,
        bottom=0.18,
        top=0.91,
        height_ratios=[0.72, 1.0],
        hspace=0.72,
        wspace=0.35,
    )
    a = fig.add_subplot(grid[0, :])
    for y, city in zip([1, 0], CITIES):
        neg = pos = 0.0
        for group in ["0", "1", "2"]:
            sm = select_one(gs, city=city, metric="ap_contribution", group=group)
            value = sm.mean_pp
            a.barh(
                y,
                value,
                left=pos if value >= 0 else neg,
                height=0.42,
                color=GROUP_COLORS[group],
                edgecolor="white",
                linewidth=0.6,
                zorder=2,
            )
            if value >= 0:
                pos += value
            else:
                neg += value
        net = select_one(gs, city=city, metric="ap_contribution", group="all").mean_pp
        a.plot([net, net], [y - 0.27, y + 0.27], color=INK, lw=1.6, zorder=4)
    a.set_yticks([1, 0], [c.title() for c in CITIES])
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    a.set_ylim(-0.5, 1.5)
    a.set_xlim(-0.25, 0.5)
    a.set_xticks([-0.2, 0, 0.2, 0.4])
    a.axvline(0, color=INK, lw=0.65)
    a.set_xlabel("Contribution to the full-domain AP difference (pp)")
    a.grid(axis="x", color="#E9ECEE", lw=0.45)
    a.set_axisbelow(True)
    panel(a, "a", "Signed contributions and their net effect")
    group_handles = [
        Patch(facecolor=GROUP_COLORS[g], label=GROUP_LABELS[g]) for g in ["0", "1", "2"]
    ]
    group_handles.append(
        Line2D([], [], color=INK, marker="|", ms=9, ls="", mew=1.6, label="Net difference")
    )
    a.legend(
        handles=group_handles,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.36),
        handlelength=1.2,
        columnspacing=1.5,
    )
    for j, metric in enumerate(["ap_contribution", "background_mean_gain"]):
        ax = fig.add_subplot(grid[1, j])
        vals = gf[(gf.metric == metric) & (gf.analysis == "L")].value_pp
        low, high = (min(vals.min(), 0), max(vals.max(), 0))
        for ci, city in enumerate(CITIES):
            color = CITY_COLORS[city]
            for gi, group in enumerate(["0", "1", "2"]):
                values = folded(gf, city=city, metric=metric, group=group).value_pp.to_numpy()
                sm = select_one(gs, city=city, metric=metric, group=group)
                x = gi + (-0.18 if ci == 0 else 0.18)
                ax.scatter(
                    x + np.linspace(-0.055, 0.055, 5),
                    values,
                    s=10,
                    c=color,
                    alpha=0.65,
                    linewidths=0,
                    zorder=3,
                )
                ax.errorbar(
                    x,
                    sm.mean_pp,
                    yerr=[[sm.mean_pp - sm.range_low_pp], [sm.range_high_pp - sm.mean_pp]],
                    fmt="D",
                    ms=4,
                    mfc=color,
                    mec="white",
                    mew=0.45,
                    color=color,
                    elinewidth=0.9,
                    capsize=2,
                    zorder=4,
                )
                low, high = (min(low, sm.range_low_pp), max(high, sm.range_high_pp))
        pad = (high - low) * 0.14
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlim(-0.55, 2.55)
        ax.set_xticks([0, 1, 2], [GROUP_LABELS[g] for g in ["0", "1", "2"]])
        ax.axhline(0, color=INK, lw=0.7)
        ax.grid(axis="y", color="#E9ECEE", lw=0.45)
        ax.set_axisbelow(True)
        ax.set_ylabel("AP contribution (pp)" if j == 0 else "Background-rank change (pp)")
        panel(
            ax,
            "bc"[j],
            "Contributions across folds" if j == 0 else "Rank changes in the same groups",
        )
        if not vals.between(*ax.get_ylim()).all():
            raise ValueError("Figure aggregate validation failed")
    fig.text(0.54, 0.105, "Upstream support in the held-out D8 graph", ha="center", fontsize=8)
    city_handles = [
        Line2D([], [], color=CITY_COLORS[c], marker="D", ms=4, lw=0.8, label=c.title())
        for c in CITIES
    ]
    fig.legend(handles=city_handles, loc="lower center", bbox_to_anchor=(0.54, 0.055), ncol=2)
    fig.text(
        0.54,
        0.025,
        "Small circles: outer folds (n = 5)     Diamonds and whiskers: mean and descriptive range",
        ha="center",
        fontsize=7.5,
        color="#56636B",
    )
    save(fig, "fig4_structural_gains_losses", qa)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    f, s, gf, gs = check_data(args.data_dir, args.output)
    utility(f, s, args.output)
    structure(gf, gs, args.output)
    print(
        json.dumps(
            {
                "fig3_fold_points": len(f),
                "fig4_group_fold_points": int((gf.analysis == "L").sum()),
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
