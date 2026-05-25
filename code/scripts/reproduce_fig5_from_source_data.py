# -*- coding: utf-8 -*-
"""
Reproduce Main Fig. 5 from compact public source data.

Inputs:
    data/source_data/main/Fig5/category_interaction_edges.csv
    data/source_data/main/Fig5/fig5_interaction_points.csv
    data/source_data/main/Fig5/fig5_panel_metadata.csv

Outputs:
    figures/main/Fig5_reproduced_from_source_data.png
    figures/main/Fig5_reproduced_from_source_data.pdf

The plotted interaction points are compact, plot-ready sampled points prepared
from full local SHAP outputs. The full SHAP matrices are not required for this
reproduction script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycirclize import Circos

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Paths
# =============================================================================

def find_repo_root() -> Path:
    this_dir = Path(__file__).resolve().parent
    return this_dir.parent.parent


REPO_ROOT = find_repo_root()
SOURCE_DIR = REPO_ROOT / "data" / "source_data" / "main" / "Fig5"
OUTPUT_DIR = REPO_ROOT / "figures" / "main"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDGES_PATH = SOURCE_DIR / "category_interaction_edges.csv"
POINTS_PATH = SOURCE_DIR / "fig5_interaction_points.csv"
METADATA_PATH = SOURCE_DIR / "fig5_panel_metadata.csv"

OUTPUT_PNG = OUTPUT_DIR / "Fig5_reproduced_from_source_data.png"
OUTPUT_PDF = OUTPUT_DIR / "Fig5_reproduced_from_source_data.pdf"

DPI = 300


# =============================================================================
# 2. Style and constants
# =============================================================================

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["text.color"] = "black"
plt.rcParams["font.size"] = 17

ABBR_DICT = {
    "Meteorological": "Met.",
    "Temporal": "Temp.",
    "Emission Source": "Emiss.",
    "Road Topology": "Topo.",
    "POI": "POI",
    "Urban Canopy": "UCP",
    "Landuse": "Landuse",
}

UNIFIED_FIG_WIDTH = 24.0
ORIGINAL_TOP_WIDTH = 16.0
ORIGINAL_TOP_HEIGHT = 7.0
TOP_ROW_TARGET_HEIGHT = UNIFIED_FIG_WIDTH * ORIGINAL_TOP_HEIGHT / ORIGINAL_TOP_WIDTH
ROW_TARGET_HEIGHT = 5.15
TOTAL_FIG_HEIGHT = TOP_ROW_TARGET_HEIGHT + 3 * ROW_TARGET_HEIGHT + 1.10

PANEL_LABEL_SIZE = 44
TOP_TITLE_SIZE = 20
TOP_RING_TEXT_SIZE = 14
TOP_TOP5_TITLE_SIZE = 17
TOP_TOP5_TEXT_SIZE = 13
ROW_TITLE_SIZE = 19
LOCAL_TITLE_SIZE = 15
AXIS_LABEL_SIZE = 15
TICK_SIZE = 14
COLORBAR_LABEL_SIZE = 15
COLORBAR_TICK_SIZE = 13

POINT_SIZE = 5
POINT_ALPHA = 0.78

PLOT_ORDER = [
    ("center", "0-50m"),
    ("center", "200-500m"),
    ("center", "2000-3000m"),
    ("side", "0-50m"),
    ("side", "200-500m"),
    ("side", "2000-3000m"),
]


# =============================================================================
# 3. Utility functions
# =============================================================================

def adaptive_station_sorting(plot_data: pd.DataFrame, alpha: float = 0.4) -> pd.DataFrame:
    subplot_c_min = plot_data["Color_Value"].quantile(0.05)
    subplot_c_max = plot_data["Color_Value"].quantile(0.95)
    subplot_mid = (subplot_c_min + subplot_c_max) / 2.0

    def calc_station_dev(group: pd.DataFrame) -> pd.DataFrame:
        c_min = group["Color_Value"].quantile(0.05)
        c_max = group["Color_Value"].quantile(0.95)
        if c_max > c_min:
            local_mid = (c_min + c_max) / 2.0
            diff = group["Color_Value"] - local_mid
        else:
            diff = group["Color_Value"] - subplot_mid

        dev = np.where(diff < 0, np.abs(diff) * 0.9, np.abs(diff) * 1.5)
        dev_min, dev_max = dev.min(), dev.max()
        group = group.copy()
        if dev_max > dev_min:
            group["dev_norm"] = (dev - dev_min) / (dev_max - dev_min)
        else:
            group["dev_norm"] = 0.0
        return group

    plot_data = plot_data.groupby("Station", group_keys=False).apply(calc_station_dev)
    rng = np.random.default_rng(42)
    plot_data = plot_data.copy()
    plot_data["random_noise"] = rng.uniform(0, 1, size=len(plot_data))
    plot_data["sort_score"] = alpha * plot_data["dev_norm"] + (1 - alpha) * plot_data["random_noise"]
    return plot_data.sort_values(by="sort_score", ascending=True)


def load_source_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [EDGES_PATH, POINTS_PATH, METADATA_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing Fig. 5 source-data file: {p}")

    edges = pd.read_csv(EDGES_PATH, encoding="utf-8-sig")
    points = pd.read_csv(POINTS_PATH, encoding="utf-8-sig")
    meta = pd.read_csv(METADATA_PATH, encoding="utf-8-sig")

    required_edges = {"Source", "Target", "Value"}
    if not required_edges.issubset(edges.columns):
        raise ValueError(f"{EDGES_PATH.name} must contain columns: {required_edges}")

    required_points = {
        "Panel",
        "Panel_Title",
        "Base_Feature_Display",
        "Color_Feature_Display",
        "Direction",
        "Distance_Start_m",
        "Distance_End_m",
        "Distance_Label",
        "Station",
        "X_Value",
        "SHAP_Value",
        "Color_Value",
        "Color_Vmin_P5",
        "Color_Vmax_P95",
    }
    missing = sorted(required_points - set(points.columns))
    if missing:
        raise ValueError(f"{POINTS_PATH.name} missing required columns: {missing}")

    numeric_cols = [
        "Distance_Start_m",
        "Distance_End_m",
        "X_Value",
        "SHAP_Value",
        "Color_Value",
        "Color_Vmin_P5",
        "Color_Vmax_P95",
    ]
    for col in numeric_cols:
        points[col] = pd.to_numeric(points[col], errors="coerce")

    points = points.dropna(subset=["Panel", "X_Value", "SHAP_Value", "Color_Value"]).copy()
    return edges, points, meta


# =============================================================================
# 4. Chord diagrams
# =============================================================================

def build_network_data(edges_df_all: pd.DataFrame):
    edges_df_excl = edges_df_all[
        ~edges_df_all["Source"].str.contains("Meteorological", case=False, na=False)
        & ~edges_df_all["Target"].str.contains("Meteorological", case=False, na=False)
    ].copy()

    global_cats = sorted(list(set(edges_df_all["Source"]) | set(edges_df_all["Target"])))
    global_node_total = {}
    for cat in global_cats:
        global_node_total[cat] = edges_df_all[
            (edges_df_all["Source"] == cat) | (edges_df_all["Target"] == cat)
        ]["Value"].sum()

    global_cats_sorted = sorted(global_cats, key=lambda k: global_node_total[k], reverse=True)

    cmap = plt.cm.Spectral
    num_cats = len(global_cats_sorted)
    if num_cats <= 1:
        node_colors = {cat: cmap(0.5) for cat in global_cats_sorted}
    else:
        node_colors = {cat: cmap(i / (num_cats - 1)) for i, cat in enumerate(global_cats_sorted)}

    return edges_df_all, edges_df_excl, global_cats_sorted, node_colors


def draw_chord_on_ax(ax, df, node_colors, global_cats, panel_label, title_text, show_top5=True):
    local_cats = sorted(list(set(df["Source"]) | set(df["Target"])))
    node_total_val = {}
    for cat in local_cats:
        total = df[(df["Source"] == cat) | (df["Target"] == cat)]["Value"].sum()
        node_total_val[cat] = total

    sectors = {cat: node_total_val[cat] for cat in global_cats if cat in local_cats}
    circos = Circos(sectors, space=7)

    for sector in circos.sectors:
        track = sector.add_track((95, 100))
        track.axis(fc=node_colors[sector.name], ec="white", linewidth=0.5)
        display_label = sector.name.replace("Parameters", "").strip()
        track.text(
            display_label,
            r=110,
            size=TOP_RING_TEXT_SIZE,
            weight="bold",
            color="black",
            orientation="horizontal",
            ha="center",
            va="center",
        )

    max_val = df["Value"].max()
    current_pos = {cat: 0.0 for cat in local_cats}

    for _, row in df.iterrows():
        source, target, weight = row["Source"], row["Target"], row["Value"]
        start_s, end_s = current_pos[source], current_pos[source] + weight
        current_pos[source] = end_s
        start_t, end_t = current_pos[target], current_pos[target] + weight
        current_pos[target] = end_t

        base_color = node_colors[source]
        alpha = 0.3 + 0.4 * (weight / max_val if max_val else 0.0)
        circos.link(
            (source, start_s, end_s),
            (target, start_t, end_t),
            color=(*base_color[:3], alpha),
            ec="white",
            lw=0.5,
        )

    circos.plotfig(ax=ax)
    ax.set_rlim(0, 140)
    ax.axis("off")

    ax.text(
        0.00,
        1.05,
        panel_label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        0.50,
        1.01,
        title_text,
        transform=ax.transAxes,
        fontsize=TOP_TITLE_SIZE,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    if show_top5:
        top5_df = df.sort_values(by="Value", ascending=False).head(5)
        start_x, start_y = 0.91, 0.72
        ax.text(
            start_x,
            start_y,
            "Top 5\nSynergistic Pairs:",
            fontsize=TOP_TOP5_TITLE_SIZE,
            fontweight="bold",
            transform=ax.transAxes,
            va="bottom",
        )
        for i, row in enumerate(top5_df.itertuples()):
            s_name = row.Source.replace("Parameters", "").strip()
            t_name = row.Target.replace("Parameters", "").strip()
            s_abbr = ABBR_DICT.get(s_name, s_name)
            t_abbr = ABBR_DICT.get(t_name, t_name)
            s_color = node_colors[row.Source]
            t_color = node_colors[row.Target]
            y_pos = start_y - 0.085 - (i * 0.085)
            ax.plot(
                start_x,
                y_pos,
                marker="s",
                color=s_color,
                markersize=10,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.plot(
                start_x + 0.022,
                y_pos,
                marker="s",
                color=t_color,
                markersize=10,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.text(
                start_x + 0.05,
                y_pos,
                f"{s_abbr} ↔ {t_abbr}",
                fontsize=TOP_TOP5_TEXT_SIZE,
                va="center",
                ha="left",
                transform=ax.transAxes,
            )


# =============================================================================
# 5. Interaction rows
# =============================================================================

def draw_interaction_row(fig, row_spec, row_points: pd.DataFrame, row_letter: str, row_title: str):
    block = row_spec.subgridspec(2, 1, height_ratios=[0.16, 0.84], hspace=0.015)
    title_ax = fig.add_subplot(block[0])
    title_ax.axis("off")
    title_ax.text(0.00, 0.40, row_letter, fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="center")
    title_ax.text(0.05, 0.40, row_title, fontsize=ROW_TITLE_SIZE, fontweight="bold", va="center")

    inner = block[1].subgridspec(1, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.065], wspace=0.22)
    axes = [fig.add_subplot(inner[0, i]) for i in range(6)]
    cax = fig.add_subplot(inner[0, 6])

    scatter = None
    vmin = float(row_points["Color_Vmin_P5"].dropna().iloc[0])
    vmax = float(row_points["Color_Vmax_P95"].dropna().iloc[0])
    base_feature_display = str(row_points["Base_Feature_Display"].iloc[0])
    color_feature_display = str(row_points["Color_Feature_Display"].iloc[0])

    for idx, (ax, (direction, label)) in enumerate(zip(axes, PLOT_ORDER)):
        plot_data = row_points[
            (row_points["Direction"] == direction)
            & (row_points["Distance_Label"] == label)
        ].copy()

        if plot_data.empty:
            ax.axis("off")
            continue

        plot_data = adaptive_station_sorting(plot_data, alpha=0.4)

        scatter = ax.scatter(
            plot_data["X_Value"],
            plot_data["SHAP_Value"],
            c=plot_data["Color_Value"],
            cmap=plt.cm.coolwarm,
            vmin=vmin,
            vmax=vmax,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            edgecolors="none",
        )

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_yscale("symlog", linthresh=1)

        x_min, x_max = plot_data["X_Value"].min(), plot_data["X_Value"].max()
        if x_max > x_min:
            ax.set_xticks([x_min, x_min + (x_max - x_min) / 2, x_max])
            ax.set_xticklabels(["Low", "Med", "High"], fontsize=TICK_SIZE)
        else:
            ax.set_xticks([])

        prefix = "Centre" if direction == "center" else "Side"
        ax.set_title(
            f"{prefix}: {label.replace('m', ' m')}",
            fontsize=LOCAL_TITLE_SIZE,
            fontweight="bold",
            pad=4,
        )

        if idx == 0:
            ax.set_ylabel("SHAP value\n(impact on NOₓ)", fontsize=AXIS_LABEL_SIZE)
        else:
            ax.tick_params(axis="y", labelleft=False)

        ax.tick_params(axis="x", labelsize=TICK_SIZE, pad=1)
        ax.tick_params(axis="y", labelsize=TICK_SIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[2].text(
        0.90,
        -0.26,
        base_feature_display,
        transform=axes[2].transAxes,
        fontsize=AXIS_LABEL_SIZE,
        fontweight="bold",
        ha="center",
    )

    if scatter is not None:
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.outline.set_visible(False)
        cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cbar.set_ticklabels(["Low", "Med", "High"])
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
        cbar.set_label(
            color_feature_display,
            fontsize=COLORBAR_LABEL_SIZE,
            fontweight="bold",
            rotation=270,
            labelpad=12,
        )
    else:
        cax.axis("off")


# =============================================================================
# 6. Main
# =============================================================================

def main() -> None:
    edges_df, points_df, meta_df = load_source_data()
    edges_df_all, edges_df_excl, global_cats_sorted, node_colors = build_network_data(edges_df)

    fig = plt.figure(figsize=(UNIFIED_FIG_WIDTH, TOTAL_FIG_HEIGHT), dpi=DPI)
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[TOP_ROW_TARGET_HEIGHT, ROW_TARGET_HEIGHT, ROW_TARGET_HEIGHT, ROW_TARGET_HEIGHT],
        left=0.050,
        right=0.970,
        top=0.988,
        bottom=0.038,
        hspace=0.18,
    )

    top_gs = outer[0].subgridspec(1, 2, wspace=0.14)
    ax1 = fig.add_subplot(top_gs[0, 0], polar=True)
    ax2 = fig.add_subplot(top_gs[0, 1], polar=True)

    draw_chord_on_ax(
        ax1,
        edges_df_all,
        node_colors,
        global_cats_sorted,
        "a",
        "Overall Synergistic Network",
        True,
    )
    draw_chord_on_ax(
        ax2,
        edges_df_excl,
        node_colors,
        global_cats_sorted,
        "b",
        "Network Excluding Meteorological Impacts",
        True,
    )

    for i, row_letter in enumerate(["c", "d", "e"], start=1):
        row_points = points_df[points_df["Panel"] == row_letter].copy()
        if row_points.empty:
            raise ValueError(f"No source data points found for panel {row_letter}.")

        if "Panel_Title" in row_points.columns:
            row_title = str(row_points["Panel_Title"].dropna().iloc[0])
        else:
            row_title = str(meta_df.loc[meta_df["Panel"] == row_letter, "Row_Title"].iloc[0])

        draw_interaction_row(fig, outer[i], row_points, row_letter, row_title)

    fig.savefig(OUTPUT_PNG, dpi=DPI, facecolor="white")
    fig.savefig(OUTPUT_PDF, dpi=DPI, facecolor="white")
    plt.close(fig)

    print("Main Fig. 5 reproduced from compact public source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
