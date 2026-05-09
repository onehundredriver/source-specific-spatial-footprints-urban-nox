import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from matplotlib.patheffects import withStroke


# ============================================================
# Reproduce Fig. 7 from plot-ready public source data
# Input:
#   data/source_data/main/Fig7_source_data.csv
# Output:
#   figures/main/Fig7_reproduced_from_source_data.png
#   figures/main/Fig7_reproduced_from_source_data.pdf
# ============================================================


def find_repo_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))


REPO_ROOT = find_repo_root()

SOURCE_DATA_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "source_data",
    "main",
    "Fig7_source_data.csv"
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "figures",
    "main"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Fig7_reproduced_from_source_data.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Fig7_reproduced_from_source_data.pdf")

COLOR_MAP = {
    "Ship": "#28527A",
    "Motor Vehicle": "#E74C3C",
    "Industrial Source": "#27AE60",
    "Others": "#BDC3C7",
}

SOURCE_ORDER = ["Ship", "Motor Vehicle", "Industrial Source"]
LEGEND_ORDER = ["Ship", "Motor Vehicle", "Industrial Source", "Others"]

BASELINE_R2 = 0.5875
MAX_R = 8000


def to_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    required_cols = [
        "Panel",
        "Data_Type",
        "Source_Category",
        "Plot_Order",
        "Radius_m",
        "Emission_Composition_Value",
        "Emission_Composition_Percent",
        "Distance_Bin_Left_m",
        "Distance_Bin_Right_m",
        "Distance_Bin_Center_m",
        "Frequency",
        "KDE_X_m",
        "KDE_Density",
        "Spearman_R",
        "Mean_Spatial_Validation_R2",
        "Baseline_R2",
        "Peak_Radius_m",
        "Metric_Name",
        "Metric_Value",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = [
        "Plot_Order",
        "Radius_m",
        "Emission_Composition_Value",
        "Emission_Composition_Percent",
        "Distance_Bin_Left_m",
        "Distance_Bin_Right_m",
        "Distance_Bin_Center_m",
        "Frequency",
        "KDE_X_m",
        "KDE_Density",
        "Spearman_R",
        "Mean_Spatial_Validation_R2",
        "Baseline_R2",
        "Peak_Radius_m",
        "Metric_Value",
    ]

    df = to_numeric(df, numeric_cols)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Microsoft YaHei"],
        "axes.unicode_minus": False,
        "mathtext.default": "regular",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelweight": "bold",
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "savefig.dpi": 600,
        "axes.linewidth": 2.0,
    })

    fig = plt.figure(figsize=(26, 17))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.3], hspace=0.25, wspace=0.45)

    panel_label_kwargs = dict(fontsize=40, fontweight="bold", va="bottom", color="black")
    x_offset_general = -0.15
    y_offset_panel = 1.05

    # ========================================================
    # Panel a
    # ========================================================
    ax0 = fig.add_subplot(gs[0, 0])

    pos = ax0.get_position()
    ax0.set_position([pos.x0 - 0.03, pos.y0, pos.width * 1.0, pos.height * 1.0])

    panel_a = df[
        (df["Panel"].astype(str) == "a")
        & (df["Data_Type"].astype(str) == "emission_composition")
    ].copy()

    panel_a = panel_a.sort_values("Plot_Order")

    if panel_a.empty:
        raise ValueError("Panel a source data are empty.")

    labels = panel_a["Source_Category"].astype(str).tolist()
    values = panel_a["Emission_Composition_Value"].values
    colors = [COLOR_MAP.get(x, "#999999") for x in labels]

    wedges, _, autotexts = ax0.pie(
        values,
        autopct="%1.1f%%",
        startangle=140,
        radius=1.25,
        colors=colors,
        pctdistance=0.75,
        wedgeprops={"width": 0.50, "edgecolor": "w", "linewidth": 3}
    )

    ax0.axis("off")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_frame_on(False)

    plt.setp(autotexts, size=18, weight="bold", color="white")
    ax0.text(0, 0, "Shanghai\n$NO_x$ Sources", ha="center", va="center", fontsize=22, fontweight="bold")
    ax0.text(-0.20, 1.05, "a", transform=ax0.transAxes, **panel_label_kwargs)

    legend_elements = [
        plt.Line2D([0], [0], color=COLOR_MAP[cat], lw=8, label=cat)
        for cat in LEGEND_ORDER
    ]

    ax0.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        ncol=1,
        fontsize=18,
        handletextpad=0.8,
        labelspacing=1.2
    )

    # ========================================================
    # Panel b: histogram + precomputed KDE with twin y-axis
    # ========================================================
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_hist2 = ax_hist.twinx()

    hist_data = df[
        (df["Panel"].astype(str) == "b")
        & (df["Data_Type"].astype(str) == "distance_histogram")
    ].copy()

    kde_data = df[
        (df["Panel"].astype(str) == "b")
        & (df["Data_Type"].astype(str) == "distance_kde")
    ].copy()

    for source in SOURCE_ORDER:
        subset = hist_data[hist_data["Source_Category"] == source].copy()
        subset = subset.sort_values("Distance_Bin_Left_m")

        if not subset.empty:
            bin_width = subset["Distance_Bin_Right_m"] - subset["Distance_Bin_Left_m"]

            ax_hist.bar(
                subset["Distance_Bin_Left_m"],
                subset["Frequency"],
                width=bin_width,
                align="edge",
                color=COLOR_MAP[source],
                alpha=0.7,
                edgecolor="none"
            )

        kde_subset = kde_data[kde_data["Source_Category"] == source].copy()
        kde_subset = kde_subset.dropna(subset=["KDE_X_m", "KDE_Density"]).sort_values("KDE_X_m")

        if not kde_subset.empty:
            ax_hist2.plot(
                kde_subset["KDE_X_m"],
                kde_subset["KDE_Density"],
                color="white",
                linewidth=6,
                alpha=1.0,
                zorder=10
            )
            ax_hist2.plot(
                kde_subset["KDE_X_m"],
                kde_subset["KDE_Density"],
                color=COLOR_MAP[source],
                linewidth=4,
                alpha=1.0,
                zorder=11
            )

    ax_hist.set_yscale("log")
    ax_hist.set_ylim(bottom=0.5)

    ax_hist.text(x_offset_general, y_offset_panel, "b", transform=ax_hist.transAxes, **panel_label_kwargs)
    ax_hist.set_xlabel("Distance from Station (m)", fontsize=22, fontweight="bold", labelpad=10)
    ax_hist.set_ylabel("Frequency", fontsize=22, fontweight="bold", labelpad=10)
    ax_hist2.set_ylabel("Density", fontsize=22, fontweight="bold", labelpad=10)

    ax_hist.set_xlim(0, 7000)
    ax_hist.grid(axis="y", linestyle=":", alpha=0.6)

    ax_hist.tick_params(axis="both", labelsize=18)
    ax_hist2.tick_params(axis="y", labelsize=18)
    ax_hist.spines["top"].set_visible(False)
    ax_hist2.spines["top"].set_visible(False)

    # ========================================================
    # Panel c
    # ========================================================
    ax1 = fig.add_subplot(gs[0, 1])

    panel_c = df[
        (df["Panel"].astype(str) == "c")
        & (df["Data_Type"].astype(str) == "spearman_curve")
    ].copy()

    for source in SOURCE_ORDER:
        subset = panel_c[panel_c["Source_Category"] == source].copy()
        subset = subset.dropna(subset=["Radius_m", "Spearman_R"]).sort_values("Radius_m")

        if not subset.empty:
            ax1.plot(
                subset["Radius_m"],
                subset["Spearman_R"],
                color=COLOR_MAP[source],
                lw=4,
                alpha=0.9
            )

    ax1.text(x_offset_general, y_offset_panel, "c", transform=ax1.transAxes, **panel_label_kwargs)
    ax1.set_ylabel("Spearman Correlation (R)", fontsize=22, fontweight="bold", labelpad=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="both", labelsize=18)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ========================================================
    # Panel d
    # ========================================================
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    text_y_height = 1.02
    x_offset = 1.15

    panel_d_curve = df[
        (df["Panel"].astype(str) == "d")
        & (df["Data_Type"].astype(str) == "r2_curve")
    ].copy()

    peak_summary = df[
        (df["Panel"].astype(str) == "d")
        & (df["Data_Type"].astype(str) == "peak_radius_summary")
    ].copy()

    all_r2_vals = []

    for source in SOURCE_ORDER:
        subset = panel_d_curve[panel_d_curve["Source_Category"] == source].copy()
        subset = subset.dropna(subset=["Radius_m", "Mean_Spatial_Validation_R2"]).sort_values("Radius_m")

        if subset.empty:
            continue

        all_r2_vals.append(subset["Mean_Spatial_Validation_R2"])

        ax2.plot(
            subset["Radius_m"],
            subset["Mean_Spatial_Validation_R2"],
            color=COLOR_MAP[source],
            lw=4,
            alpha=0.9,
            marker="None"
        )

        peak_subset = peak_summary[peak_summary["Source_Category"] == source].copy()

        if not peak_subset.empty:
            best_r = float(peak_subset["Peak_Radius_m"].iloc[0])
        else:
            idx_max = subset["Mean_Spatial_Validation_R2"].idxmax()
            best_r = float(subset.loc[idx_max, "Radius_m"])

        for target_ax in [ax1, ax2]:
            target_ax.axvline(
                x=best_r,
                color=COLOR_MAP[source],
                linestyle="--",
                lw=2.5,
                alpha=0.5
            )

        txt = ax2.text(
            best_r * x_offset,
            text_y_height,
            f"{int(best_r)}m",
            color=COLOR_MAP[source],
            transform=ax2.get_xaxis_transform(),
            fontweight="bold",
            ha="left",
            va="center",
            fontsize=18
        )
        txt.set_path_effects([withStroke(linewidth=4, foreground="white")])

    ax2.axhline(y=BASELINE_R2, color="black", linestyle="-.", lw=2.5, alpha=0.8)
    ax2.text(
        MAX_R * 0.98,
        BASELINE_R2 + 0.0005,
        f"Baseline (Meteo-only): {BASELINE_R2:.4f}",
        ha="right",
        va="bottom",
        fontsize=18,
        color="black",
        fontweight="bold"
    )

    if all_r2_vals:
        all_r2_vals = pd.concat(all_r2_vals)
        top_max = all_r2_vals.max()
        bot_min = all_r2_vals.min()
        plot_min = min(BASELINE_R2, bot_min) - 0.003
        ax2.set_ylim(plot_min, top_max + 0.003)
    else:
        ax2.set_ylim(0.580, 0.630)

    ax2.set_xscale("symlog", linthresh=1000)

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax2.xaxis.set_major_formatter(formatter)

    ax2.set_xticks([0, 250, 500, 750, 1000, 2000, 4000, 8000])
    ax2.set_xlim(0, MAX_R)

    ax2.text(x_offset_general, y_offset_panel, "d", transform=ax2.transAxes, **panel_label_kwargs)
    ax2.set_ylabel("5-Fold Mean Spatial Val $R^2$", fontsize=22, fontweight="bold", labelpad=10)
    ax2.set_xlabel("Buffer Radius (m)", fontsize=22, fontweight="bold", labelpad=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.tick_params(axis="both", labelsize=18)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("Fig. 7 reproduced from plot-ready source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
