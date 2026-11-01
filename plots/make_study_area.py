"""Regenerate Figure 1 from privately supplied prepared study-area inputs.

Adapted from manuscript figure_code/make_study_area.py. Only the aggregate
support-share CSV is bundled. Extraction and rendering write beneath --output.
The original six-panel design and structural calculations are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

mpl.use("Agg")
import matplotlib.pyplot as plt

SUPPORT_TABLE = Path(__file__).resolve().parent / "data/study_area_depth_shares.csv"
CITIES = ("zhengzhou", "beijing")
CITY_LABEL = {"zhengzhou": "Zhengzhou", "beijing": "Beijing"}
CRS = {"zhengzhou": "EPSG:32649", "beijing": "EPSG:32650"}
CELL_COUNTS = {"zhengzhou": 25155, "beijing": 61443}
POSITIVE_COUNTS = {"zhengzhou": 791, "beijing": 296}
CELL_M = 200
DETAIL = (96, 31, 16)
INK = "#263844"
GREY = "#77858B"
ADMIN_FILL = "#E7EAEC"
FOLD_COLORS = ["#8DA0CB", "#FC8D62", "#66C2A5", "#E78AC3", "#A6D854"]
DEPTH_COLORS = ["#D9D9D9", "#9ECAE1", "#2F6DB5"]
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.facecolor": "white",
    }
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raster(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    r0, c0 = (int(frame["row"].min()), int(frame["column"].min()))
    shape = (int(frame["row"].max()) - r0 + 1, int(frame["column"].max()) - c0 + 1)
    out = np.full(shape, np.nan, dtype=np.float64)
    out[frame["row"].to_numpy() - r0, frame["column"].to_numpy() - c0] = frame[column].to_numpy(
        float
    )
    x0 = float(frame["center_x"].min()) - CELL_M / 2
    y1 = float(frame["center_y"].max()) + CELL_M / 2
    expected_x = x0 + (frame["column"].to_numpy() - c0 + 0.5) * CELL_M
    expected_y = y1 - (frame["row"].to_numpy() - r0 + 0.5) * CELL_M
    if not np.allclose(frame["center_x"], expected_x, rtol=0, atol=1e-06):
        raise ValueError(
            "Study-area input validation failed: np.allclose(frame['center_x'], expected_x, rtol=0, atol=1e-06)"
        )
    if not np.allclose(frame["center_y"], expected_y, rtol=0, atol=1e-06):
        raise ValueError(
            "Study-area input validation failed: np.allclose(frame['center_y'], expected_y, rtol=0, atol=1e-06)"
        )
    return (out, np.array([x0, x0 + shape[1] * CELL_M, y1 - shape[0] * CELL_M, y1]))


def extract_assets(repo: Path, support_table: Path, data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    support = pd.read_csv(support_table)
    if not len(support) == 4:
        raise ValueError("Study-area input validation failed: len(support) == 4")
    if support.duplicated(["city", "scope"]).any():
        raise ValueError(
            "Study-area input validation failed: not support.duplicated(['city', 'scope']).any()"
        )
    shares = support[["share_L0", "share_L1", "share_L2"]].to_numpy(float)
    if not np.all((shares >= 0) & (shares <= 1)):
        raise ValueError(
            "Study-area input validation failed: np.all((shares >= 0) & (shares <= 1))"
        )
    if not np.allclose(shares.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError(
            "Study-area input validation failed: np.allclose(shares.sum(axis=1), 1, rtol=0, atol=1e-12)"
        )
    manifest = {
        "cell_size_m": CELL_M,
        "depth_graph": "whole-domain D8 before outer-partition pruning",
        "support_table_sha256": sha256(support_table),
        "cities": {},
    }
    for city in CITIES:
        folder = repo / "data" / "interim" / city
        files = {
            "grid": folder / "final_scorable_grid.parquet",
            "elevation": folder / "terrain_features.parquet",
            "partitions": folder / "evaluation_design" / "outer_partitions.parquet",
            "edges": folder / "d8_edges.parquet",
            "admin": folder / "admin_aoi.parquet",
        }
        frame = pd.read_parquet(
            files["grid"], columns=["grid_id", "row", "column", "center_x", "center_y"]
        )
        terrain = pd.read_parquet(files["elevation"], columns=["grid_id", "elevation_mean"])
        if not len(frame) == CELL_COUNTS[city]:
            raise ValueError("Study-area input validation failed: len(frame) == CELL_COUNTS[city]")
        if not (frame["grid_id"].is_unique and (not frame.duplicated(["row", "column"]).any())):
            raise ValueError(
                "Study-area input validation failed: frame['grid_id'].is_unique and (not frame.duplicated(['row', 'column']).any())"
            )
        frame = frame.merge(terrain, on="grid_id", validate="1:1")
        if not (len(frame) == CELL_COUNTS[city] and np.isfinite(frame["elevation_mean"]).all()):
            raise ValueError(
                "Study-area input validation failed: len(frame) == CELL_COUNTS[city] and np.isfinite(frame['elevation_mean']).all()"
            )
        partitions = pd.read_parquet(
            files["partitions"], columns=["grid_id", "outer_fold", "partition"]
        )
        test = partitions.loc[partitions["partition"].eq("test"), ["grid_id", "outer_fold"]]
        if not (test["grid_id"].is_unique and len(test) == len(frame)):
            raise ValueError(
                "Study-area input validation failed: test['grid_id'].is_unique and len(test) == len(frame)"
            )
        if not set(test["outer_fold"]) == set(range(5)):
            raise ValueError(
                "Study-area input validation failed: set(test['outer_fold']) == set(range(5))"
            )
        frame = frame.merge(test, on="grid_id", validate="1:1")
        if not len(frame) == CELL_COUNTS[city]:
            raise ValueError("Study-area input validation failed: len(frame) == CELL_COUNTS[city]")
        edges = pd.read_parquet(files["edges"], columns=["source_grid_id", "target_grid_id"])
        if not edges["source_grid_id"].is_unique:
            raise ValueError(
                "Study-area input validation failed: edges['source_grid_id'].is_unique"
            )
        ids = set(frame["grid_id"])
        if not set(edges["source_grid_id"]).issubset(ids):
            raise ValueError(
                "Study-area input validation failed: set(edges['source_grid_id']).issubset(ids)"
            )
        if not set(edges["target_grid_id"]).issubset(ids):
            raise ValueError(
                "Study-area input validation failed: set(edges['target_grid_id']).issubset(ids)"
            )
        incoming = set(edges["target_grid_id"])
        two_hop = set(edges.loc[edges["source_grid_id"].isin(incoming), "target_grid_id"])
        frame["depth"] = np.where(
            frame["grid_id"].isin(two_hop), 2, np.where(frame["grid_id"].isin(incoming), 1, 0)
        )
        measured_shares = np.bincount(frame["depth"], minlength=3) / len(frame)
        all_row = support.loc[(support.city == city) & (support.scope == "all cells")].iloc[0]
        pos_row = support.loc[
            (support.city == city) & (support.scope == "report-positive cells")
        ].iloc[0]
        if not int(all_row["cells"]) == CELL_COUNTS[city]:
            raise ValueError(
                "Study-area input validation failed: int(all_row['cells']) == CELL_COUNTS[city]"
            )
        if not int(pos_row["cells"]) == POSITIVE_COUNTS[city]:
            raise ValueError(
                "Study-area input validation failed: int(pos_row['cells']) == POSITIVE_COUNTS[city]"
            )
        if not np.allclose(
            measured_shares,
            all_row[["share_L0", "share_L1", "share_L2"]].to_numpy(float),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                "Study-area input validation failed: np.allclose(measured_shares, all_row[['share_L0', 'share_L1', 'share_L2']].to_numpy(float), rtol=0, atol=1e-12)"
            )
        positive_group_counts = pos_row[["share_L0", "share_L1", "share_L2"]].to_numpy(float) * int(
            pos_row["cells"]
        )
        if not np.allclose(
            positive_group_counts, np.rint(positive_group_counts), rtol=0, atol=1e-09
        ):
            raise ValueError(
                "Study-area input validation failed: np.allclose(positive_group_counts, np.rint(positive_group_counts), rtol=0, atol=1e-09)"
            )
        elevation, extent = raster(frame, "elevation_mean")
        folds, fold_extent = raster(frame, "outer_fold")
        depth, depth_extent = raster(frame, "depth")
        if not (np.array_equal(extent, fold_extent) and np.array_equal(extent, depth_extent)):
            raise ValueError(
                "Study-area input validation failed: np.array_equal(extent, fold_extent) and np.array_equal(extent, depth_extent)"
            )
        arrays = {"elevation": elevation, "fold": folds, "depth": depth, "extent": extent}
        details = {}
        if city == "zhengzhou":
            r0, c0, size = DETAIL
            window = frame.loc[
                frame["row"].between(r0, r0 + size - 1) & frame["column"].between(c0, c0 + size - 1)
            ].copy()
            if not len(window) == size * size:
                raise ValueError("Study-area input validation failed: len(window) == size * size")
            depth_detail, detail_extent = raster(window, "depth")
            xy = window.set_index("grid_id")[["center_x", "center_y"]]
            local = edges.loc[
                edges.source_grid_id.isin(xy.index) & edges.target_grid_id.isin(xy.index)
            ]
            segments = np.array(
                [
                    [xy.loc[s].to_numpy(float), xy.loc[t].to_numpy(float)]
                    for s, t in zip(local.source_grid_id, local.target_grid_id)
                ]
            )
            arrays.update(
                detail_depth=depth_detail, detail_extent=detail_extent, detail_edges=segments
            )
            details = {
                "detail_rows_columns": list(DETAIL),
                "detail_cells": len(window),
                "detail_edges": len(local),
                "detail_extent_m": detail_extent.tolist(),
            }
        npz = data / f"study_area_{city}.npz"
        np.savez_compressed(npz, **arrays)
        admin = gpd.read_parquet(files["admin"], columns=["geometry"])
        if not admin.crs.to_string() == CRS[city]:
            raise ValueError(
                "Study-area input validation failed: admin.crs.to_string() == CRS[city]"
            )
        geojson = data / f"study_area_admin_{city}.geojson"
        admin.to_crs("EPSG:4326").to_file(geojson, driver="GeoJSON", index=False)
        vmin, vmax = np.nanpercentile(elevation, [1, 99])
        city_manifest = {
            "crs": CRS[city],
            "cells": len(frame),
            "outer_fold_counts": {
                str(k): int(v) for k, v in frame.outer_fold.value_counts().sort_index().items()
            },
            "whole_domain_depth_counts": np.bincount(frame["depth"], minlength=3).tolist(),
            "whole_domain_depth_shares": measured_shares.tolist(),
            "supplied_positive_group_counts": np.rint(positive_group_counts).astype(int).tolist(),
            "elevation_colour_limits_m": [float(vmin), float(vmax)],
            "assets": {npz.name: sha256(npz), geojson.name: sha256(geojson)},
            **details,
            "sources": {
                key: {"file": path.relative_to(repo).as_posix(), "sha256": sha256(path)}
                for key, path in files.items()
            },
        }
        manifest["cities"][city] = city_manifest
    support_out = data / "study_area_depth_shares.csv"
    support_out.write_bytes(support_table.read_bytes())
    manifest["aggregate_asset_sha256"] = sha256(support_out)
    (data / "study_area_asset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def set_map_limits(ax, extent, admin_bounds=None):
    x0, x1, y0, y1 = [float(x) for x in extent]
    if admin_bounds is not None:
        ax0, ay0, ax1, ay1 = admin_bounds
        x0, x1, y0, y1 = (min(x0, ax0), max(x1, ax1), min(y0, ay0), max(y1, ay1))
    dx, dy = (x1 - x0, y1 - y0)
    x0 -= 0.035 * dx
    x1 += 0.035 * dx
    y0 -= 0.15 * dy
    y1 += 0.055 * dy
    box = ax.get_position(original=True)
    figw, figh = ax.figure.get_size_inches()
    target_ratio = box.width * figw / (box.height * figh)
    cx, cy = ((x0 + x1) / 2, (y0 + y1) / 2)
    span_x, span_y = (x1 - x0, y1 - y0)
    if span_x / span_y > target_ratio:
        span_y = span_x / target_ratio
    else:
        span_x = span_y * target_ratio
    ax.set_xlim(cx - span_x / 2, cx + span_x / 2)
    ax.set_ylim(cy - span_y / 2, cy + span_y / 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#B2BDC2")
        spine.set_linewidth(0.45)


def map_decorations(ax, kilometres, north=True):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bar_x = x0 + 0.06 * (x1 - x0)
    bar_y = y0 + 0.035 * (y1 - y0)
    ax.plot(
        [bar_x, bar_x + kilometres * 1000],
        [bar_y, bar_y],
        color=INK,
        lw=1.7,
        solid_capstyle="butt",
        zorder=8,
    )
    ax.text(
        bar_x + kilometres * 500,
        bar_y + 0.02 * (y1 - y0),
        f"{kilometres} km",
        ha="center",
        va="bottom",
        fontsize=7,
        color=INK,
        zorder=8,
    )
    if north:
        ax.annotate(
            "",
            xy=(0.94, 0.945),
            xytext=(0.94, 0.87),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "mutation_scale": 7, "color": INK, "lw": 0.65},
            zorder=8,
        )
        ax.text(
            0.88,
            0.915,
            "N",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color=INK,
            zorder=8,
        )


def heading(ax, letter, title):
    ax.set_title(title, loc="left", pad=5, fontsize=7.5, color=INK, linespacing=1.25)
    ax.annotate(
        letter,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-8, 5),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )


def render(data: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "study_area_asset_manifest.json").read_text())
    support = pd.read_csv(data / "study_area_depth_shares.csv")
    maps = {}
    for city in CITIES:
        with np.load(data / f"study_area_{city}.npz", allow_pickle=False) as archive:
            maps[city] = dict(archive)
    admins = {
        city: gpd.read_file(data / f"study_area_admin_{city}.geojson").to_crs(CRS[city])
        for city in CITIES
    }
    for city in CITIES:
        for name, expected in manifest["cities"][city]["assets"].items():
            if not sha256(data / name) == expected:
                raise ValueError(
                    "Study-area input validation failed: sha256(data / name) == expected"
                )
        if not np.isfinite(maps[city]["elevation"]).sum() == CELL_COUNTS[city]:
            raise ValueError(
                "Study-area input validation failed: np.isfinite(maps[city]['elevation']).sum() == CELL_COUNTS[city]"
            )
        if not set(maps[city]["fold"][np.isfinite(maps[city]["fold"])]) == set(range(5)):
            raise ValueError(
                "Study-area input validation failed: set(maps[city]['fold'][np.isfinite(maps[city]['fold'])]) == set(range(5))"
            )
    if not sha256(data / "study_area_depth_shares.csv") == manifest["aggregate_asset_sha256"]:
        raise ValueError(
            "Study-area input validation failed: sha256(data / 'study_area_depth_shares.csv') == manifest['aggregate_asset_sha256']"
        )
    width_inches, height_inches = (180 / 25.4, 155 / 25.4)
    fig = plt.figure(figsize=(width_inches, height_inches))
    positions = [(x, y, 0.265, 0.35) for y in [0.56, 0.13] for x in [0.04, 0.365, 0.69]]
    axes = [fig.add_axes(pos) for pos in positions]
    for ax, letter in zip(axes, "abcdef"):
        ax.set_gid(letter)
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad((0, 0, 0, 0))
    fold_map = ListedColormap(FOLD_COLORS)
    fold_map.set_bad((0, 0, 0, 0))
    depth_map = ListedColormap(DEPTH_COLORS)
    depth_map.set_bad((0, 0, 0, 0))
    for i, city in enumerate(CITIES):
        arrays, admin = (maps[city], admins[city])
        for row in [0, 1]:
            ax = axes[i + row * 3]
            admin.plot(
                ax=ax, color=ADMIN_FILL, edgecolor=GREY, linewidth=0.45, aspect=None, zorder=0
            )
            if row == 0:
                vmin, vmax = manifest["cities"][city]["elevation_colour_limits_m"]
                image = ax.imshow(
                    arrays["elevation"],
                    extent=arrays["extent"],
                    origin="upper",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                    rasterized=True,
                    zorder=1,
                )
                cax = fig.add_axes([0.314 + i * 0.325, 0.63, 0.011, 0.205])
                cbar = fig.colorbar(image, cax=cax)
                cbar.ax.set_title("m", fontsize=7, pad=3)
                cbar.ax.tick_params(labelsize=6.5, length=1.6, width=0.45, pad=2)
                cbar.outline.set_linewidth(0.45)
                heading(ax, "ab"[i], f"{CITY_LABEL[city]}\nFilled surface elevation")
            else:
                ax.imshow(
                    arrays["fold"],
                    extent=arrays["extent"],
                    origin="upper",
                    cmap=fold_map,
                    vmin=-0.5,
                    vmax=4.5,
                    interpolation="nearest",
                    rasterized=True,
                    zorder=1,
                )
                heading(ax, "de"[i], f"{CITY_LABEL[city]}\nOuter test folds (4 km blocks)")
            admin.boundary.plot(ax=ax, color=GREY, linewidth=0.4, aspect=None, zorder=3)
            set_map_limits(ax, arrays["extent"], admin.total_bounds)
            map_decorations(ax, 10 if city == "zhengzhou" else 20)
        if city == "zhengzhou":
            x0, x1, y0, y1 = arrays["detail_extent"]
            axes[0].add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#C8423A", lw=0.9, zorder=6
                )
            )
    detail = maps["zhengzhou"]
    ax = axes[2]
    ax.imshow(
        detail["detail_depth"],
        extent=detail["detail_extent"],
        origin="upper",
        cmap=depth_map,
        vmin=-0.5,
        vmax=2.5,
        interpolation="nearest",
        rasterized=True,
    )
    for (x0, y0), (x1, y1) in detail["detail_edges"]:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=3.5,
                color=INK,
                lw=0.35,
                shrinkA=0.5,
                shrinkB=0.7,
                zorder=5,
            )
        )
    set_map_limits(ax, detail["detail_extent"])
    map_decorations(ax, 1, north=False)
    heading(ax, "c", "D8 relationships\nZhengzhou detail (200 m cells)")
    ax = axes[5]
    for y, (city, scope) in zip(
        [0, -1, -2.4, -3.4],
        [(c, s) for c in CITIES for s in ["all cells", "report-positive cells"]],
    ):
        row = support.loc[(support.city == city) & (support.scope == scope)].iloc[0]
        left = 0.0
        for j, key in enumerate(["share_L0", "share_L1", "share_L2"]):
            share = float(row[key])
            ax.barh(
                y,
                share,
                left=left,
                color=DEPTH_COLORS[j],
                height=0.42,
                edgecolor="white",
                linewidth=0.4,
            )
            ax.text(
                left + share / 2,
                y,
                f"{share:.0%}",
                fontsize=7,
                color="white" if j == 2 else INK,
                ha="center",
                va="center",
            )
            left += share
        label = f"{CITY_LABEL[city]}: " + (
            "all cells" if scope == "all cells" else "recorded flood cells"
        )
        ax.text(0, y + 0.29, label, fontsize=7, ha="left", va="bottom", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(-3.8, 0.9)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("Share of cells (%)", fontsize=7, labelpad=5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    heading(ax, "f", "Upstream support\nWhole-domain D8 graph")
    fold_handles = [Patch(facecolor=c, label=f"Fold {i + 1}") for i, c in enumerate(FOLD_COLORS)]
    fig.legend(
        handles=fold_handles,
        loc="lower center",
        bbox_to_anchor=(0.35, 0.027),
        ncol=5,
        fontsize=7,
        handlelength=0.9,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    depth_labels = ["No upstream", "One hop", "Multi-hop"]
    depth_handles = [
        Patch(facecolor=c, edgecolor=GREY, linewidth=0.3, label=name)
        for c, name in zip(DEPTH_COLORS, depth_labels)
    ]
    fig.legend(
        handles=depth_handles,
        loc="lower center",
        bbox_to_anchor=(0.825, 0.027),
        ncol=3,
        fontsize=7,
        handlelength=0.9,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.canvas.draw()
    fig.savefig(output / "fig1_study_area.pdf", dpi=600)
    fig.savefig(output / "fig1_study_area.svg", dpi=600)
    fig.savefig(output / "fig1_study_area.png", dpi=600)
    plt.close(fig)
    print("Figure 1 exported with locally staged map assets.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Prepared project root containing data/interim for both cities.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for Figure 1 and a data/ staging directory.",
    )
    parser.add_argument(
        "--support-table",
        type=Path,
        default=SUPPORT_TABLE,
        help="Aggregate whole-domain support shares.",
    )
    args = parser.parse_args()
    data = args.output / "data"
    extract_assets(args.root, args.support_table, data)
    render(data, args.output)


if __name__ == "__main__":
    main()
