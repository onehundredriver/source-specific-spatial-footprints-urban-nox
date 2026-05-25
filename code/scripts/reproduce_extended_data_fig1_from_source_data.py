import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke

warnings.filterwarnings("ignore")


# ============================================================
# Reproduce Extended Data Fig. 1 from public source data
# Input:
#   data/source_data/extended_data/Extended_Data_Fig1_source_data.xlsx
# Output:
#   figures/extended_data/Extended_Data_Fig1_reproduced_from_source_data.png
#   figures/extended_data/Extended_Data_Fig1_reproduced_from_source_data.pdf
# ============================================================


def find_repo_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))


REPO_ROOT = find_repo_root()

SOURCE_DATA_XLSX = os.path.join(
    REPO_ROOT,
    "data",
    "source_data",
    "extended_data",
    "Extended_Data_Fig1_source_data.xlsx"
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "figures",
    "extended_data"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PDF = os.path.join(
    OUTPUT_DIR,
    "Extended_Data_Fig1_reproduced_from_source_data.pdf"
)

OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    "Extended_Data_Fig1_reproduced_from_source_data.png"
)


SOURCE_ORDER = ["Motor Vehicle", "Ship", "Industrial Source"]

MAX_R = 8000
NEAR_FIELD_MAX = 500
KILOMETRE_SCALE_MIN = 1000

COLOR_MAP = {
    "Motor Vehicle": "#E74C3C",
    "Ship": "#28527A",
    "Industrial Source": "#27AE60",
}

SCALE_COLOR_MAP = {
    "near-field": "#C9D8BF",
    "intermediate": "#E8DDB6",
    "kilometre-scale": "#BFD0E0",
}

SCALE_LABEL_MAP = {
    "near-field": "near-field (≤500 m)",
    "intermediate": "intermediate (500–1000 m)",
    "kilometre-scale": "kilometre-scale (≥1000 m)",
}

PANEL_LABEL_KW = dict(fontsize=43, fontweight="bold", va="bottom", color="black")
X_OFFSET_PANEL = -0.14
Y_OFFSET_PANEL = 1.04


def apply_global_style():
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
        "axes.titlesize": 21,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.linewidth": 1.8,
        "savefig.dpi": 600,
    })

    sns.set_theme(style="ticks", font="Arial")


def to_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def classify_scale(radius_m):
    if pd.isna(radius_m):
        return "NA"
    if radius_m <= NEAR_FIELD_MAX:
        return "near-field"
    if radius_m >= KILOMETRE_SCALE_MIN:
        return "kilometre-scale"
    return "intermediate"


def read_sheet(sheet_name):
    return pd.read_excel(SOURCE_DATA_XLSX, sheet_name=sheet_name)


def load_source_data():
    if not os.path.exists(SOURCE_DATA_XLSX):
        raise FileNotFoundError(SOURCE_DATA_XLSX)

    mean_curves = read_sheet("Full_network_curves")
    full_peak_summary = read_sheet("Full_peak_summary")
    foldagg_band_df = read_sheet("Fold_aggregated_boot_bands")
    boot_peak_df = read_sheet("Station_bootstrap_peaks")
    class_prop_df = read_sheet("Scale_class_proportions")
    station_type_df = read_sheet("Station_type_sensitivity")
    seasonal_df = read_sheet("Seasonal_sensitivity")

    mean_curves = to_numeric(mean_curves, ["Radius", "Mean_Val_R2"])
    full_peak_summary = to_numeric(
        full_peak_summary,
        ["Full_Curve_Peak_Radius_m", "Full_Curve_Peak_R2"]
    )
    foldagg_band_df = to_numeric(
        foldagg_band_df,
        [
            "Radius",
            "FoldAgg_R2_p2_5",
            "FoldAgg_R2_p25",
            "FoldAgg_R2_median",
            "FoldAgg_R2_p75",
            "FoldAgg_R2_p97_5",
        ]
    )
    boot_peak_df = to_numeric(boot_peak_df, ["Bootstrap", "Peak_Radius_m", "Peak_R2"])
    class_prop_df = to_numeric(class_prop_df, ["Proportion"])
    station_type_df = to_numeric(station_type_df, ["N_Stations", "Peak_Radius_m", "Peak_R2"])
    seasonal_df = to_numeric(seasonal_df, ["N_Stations", "Peak_Radius_m", "Peak_R2"])

    return (
        mean_curves,
        full_peak_summary,
        foldagg_band_df,
        boot_peak_df,
        class_prop_df,
        station_type_df,
        seasonal_df,
    )


def plot_figure(
    mean_curves,
    full_peak_summary,
    foldagg_band_df,
    boot_peak_df,
    class_prop_df,
    station_type_df,
    seasonal_df,
):
    apply_global_style()

    fig = plt.figure(figsize=(25, 16.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # --------------------------------------------------------
    # Panel a: fold-aggregated bootstrap ribbons
    # --------------------------------------------------------
    for source in SOURCE_ORDER:
        color = COLOR_MAP[source]
        band = foldagg_band_df[
            foldagg_band_df["Source"].astype(str) == source
        ].sort_values("Radius")

        if not band.empty:
            ax_a.fill_between(
                band["Radius"].to_numpy(dtype=float),
                band["FoldAgg_R2_p25"].to_numpy(dtype=float),
                band["FoldAgg_R2_p75"].to_numpy(dtype=float),
                color=color,
                alpha=0.20,
                linewidth=0,
                zorder=2,
            )

        curve = mean_curves[
            mean_curves["Source"].astype(str) == source
        ].sort_values("Radius")

        if not curve.empty:
            ax_a.plot(
                curve["Radius"],
                curve["Mean_Val_R2"],
                color=color,
                lw=3.2,
                alpha=0.98,
                label=source,
                zorder=4,
            )

        pr = full_peak_summary[
            full_peak_summary["Source"].astype(str) == source
        ]

        if len(pr) > 0:
            peak_r = float(pr["Full_Curve_Peak_Radius_m"].iloc[0])
            ax_a.axvline(
                peak_r,
                color=color,
                linestyle="--",
                lw=1.8,
                alpha=0.65,
                zorder=3,
            )

    ax_a.set_xscale("symlog", linthresh=1000)
    ax_a.set_xlim(0, MAX_R)
    ax_a.set_xticks([0, 100, 250, 500, 1000, 2000, 4000, 8000])

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax_a.xaxis.set_major_formatter(formatter)

    all_vals = []
    for source in SOURCE_ORDER:
        vals = mean_curves.loc[
            mean_curves["Source"].astype(str) == source,
            "Mean_Val_R2"
        ].dropna().tolist()
        all_vals.extend(vals)

    if all_vals:
        ymin = max(0.52, min(all_vals) - 0.035)
        ymax = min(0.70, max(all_vals) + 0.045)
        ax_a.set_ylim(ymin, ymax)

    y_label_top = ax_a.get_ylim()[1] - 0.003

    for source in SOURCE_ORDER:
        color = COLOR_MAP[source]
        pr = full_peak_summary[
            full_peak_summary["Source"].astype(str) == source
        ]

        if len(pr) > 0:
            peak_r = float(pr["Full_Curve_Peak_Radius_m"].iloc[0])
            txt = ax_a.text(
                peak_r,
                y_label_top,
                f"{int(peak_r)} m",
                color=color,
                fontsize=16,
                fontweight="bold",
                rotation=90,
                ha="center",
                va="top",
                zorder=5,
            )
            txt.set_path_effects([withStroke(linewidth=3, foreground="white")])

    ax_a.set_xlabel("Buffer radius (m)", fontsize=20, fontweight="bold", labelpad=10)
    ax_a.set_ylabel("Mean spatial validation $R^2$", fontsize=20, fontweight="bold", labelpad=10)
    ax_a.set_title(
        "Full-network curves and fold-aggregated bootstrap envelopes",
        fontsize=21,
        fontweight="bold",
        pad=10,
    )
    ax_a.grid(axis="y", linestyle=":", alpha=0.55)

    source_handles = [
        Line2D([0], [0], color=COLOR_MAP[s], lw=3.2, label=s)
        for s in SOURCE_ORDER
    ]
    ribbon_handles = [
        Patch(
            facecolor="grey",
            edgecolor="none",
            alpha=0.20,
            label="25–75% fold-aggregated envelope",
        )
    ]

    ax_a.legend(
        handles=source_handles + ribbon_handles,
        frameon=False,
        loc="lower right",
        fontsize=16,
    )

    ax_a.text(X_OFFSET_PANEL, Y_OFFSET_PANEL, "a", transform=ax_a.transAxes, **PANEL_LABEL_KW)
    ax_a.tick_params(axis="both", labelsize=16)
    ax_a.spines[["top", "right"]].set_visible(False)

    # --------------------------------------------------------
    # Panel b: station-bootstrap peak radius distributions
    # --------------------------------------------------------
    box_data = [
        boot_peak_df.loc[
            boot_peak_df["Source"].astype(str) == s,
            "Peak_Radius_m"
        ].dropna().values
        for s in SOURCE_ORDER
    ]

    bp = ax_b.boxplot(
        box_data,
        patch_artist=True,
        widths=0.42,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2.1),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
        boxprops=dict(edgecolor="black", linewidth=1.4),
    )

    for patch, source in zip(bp["boxes"], SOURCE_ORDER):
        patch.set_facecolor(COLOR_MAP[source])
        patch.set_alpha(0.34)

    ax_b.axhline(500, color="grey", linestyle="--", linewidth=1.5, alpha=0.75)
    ax_b.axhline(1000, color="grey", linestyle=":", linewidth=1.5, alpha=0.75)

    ax_b.text(0.64, 520, "500 m", fontsize=16, color="grey", va="bottom")
    ax_b.text(0.64, 1020, "1000 m", fontsize=16, color="grey", va="bottom")

    ax_b.set_xticks(np.arange(1, len(SOURCE_ORDER) + 1))
    ax_b.set_xticklabels(SOURCE_ORDER, rotation=0)
    ax_b.set_ylabel("Station-bootstrap peak radius (m)", fontsize=20, fontweight="bold", labelpad=10)
    ax_b.set_title(
        "Station-bootstrap peak radius distributions",
        fontsize=21,
        fontweight="bold",
        pad=10,
    )
    ax_b.grid(axis="y", linestyle=":", alpha=0.55)
    ax_b.text(X_OFFSET_PANEL, Y_OFFSET_PANEL, "b", transform=ax_b.transAxes, **PANEL_LABEL_KW)
    ax_b.tick_params(axis="both", labelsize=16)
    ax_b.spines[["top", "right"]].set_visible(False)

    # --------------------------------------------------------
    # Panel c: scale-class stability
    # --------------------------------------------------------
    scale_order = ["near-field", "intermediate", "kilometre-scale"]
    x = np.arange(len(SOURCE_ORDER))
    bottom = np.zeros(len(SOURCE_ORDER))

    for scale in scale_order:
        vals = []

        for source in SOURCE_ORDER:
            sub = class_prop_df[
                (class_prop_df["Source"].astype(str) == source)
                & (class_prop_df["Scale_Class"].astype(str) == scale)
            ]

            vals.append(float(sub["Proportion"].iloc[0]) if len(sub) > 0 else 0.0)

        ax_c.bar(
            x,
            vals,
            bottom=bottom,
            color=SCALE_COLOR_MAP[scale],
            edgecolor="black",
            linewidth=1.1,
            width=0.78,
            label=SCALE_LABEL_MAP[scale],
        )

        for i, v in enumerate(vals):
            if v >= 0.12:
                ax_c.text(
                    x[i],
                    bottom[i] + v / 2,
                    f"{v * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color="black",
                )

        bottom += np.array(vals)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(SOURCE_ORDER, rotation=0)
    ax_c.set_ylim(0, 1.0)
    ax_c.set_ylabel("Proportion of bootstrap replicates", fontsize=20, fontweight="bold", labelpad=10)
    ax_c.set_title("Stability of peak-scale class", fontsize=21, fontweight="bold", pad=10)

    ax_c.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        fontsize=15,
        columnspacing=1.1,
        handlelength=1.4,
    )

    ax_c.grid(axis="y", linestyle=":", alpha=0.55)
    ax_c.text(X_OFFSET_PANEL, Y_OFFSET_PANEL, "c", transform=ax_c.transAxes, **PANEL_LABEL_KW)
    ax_c.tick_params(axis="both", labelsize=16)
    ax_c.spines[["top", "right"]].set_visible(False)

    # --------------------------------------------------------
    # Panel d: station-type and seasonal sensitivity
    # --------------------------------------------------------
    st_plot = station_type_df.copy()
    st_plot["Subset"] = st_plot["Sensitivity_Setting"].astype(str)

    se_plot = seasonal_df.copy()
    se_plot["Subset"] = "Season " + se_plot["Season"].astype(str)

    sens = pd.concat(
        [
            st_plot[["Source", "Subset", "Peak_Radius_m", "N_Stations"]],
            se_plot[["Source", "Subset", "Peak_Radius_m", "N_Stations"]],
        ],
        ignore_index=True,
    )

    subset_order = [
        "All stations",
        "Non-traffic stations",
        "Traffic-oriented stations",
        "Season DJF",
        "Season MAM",
        "Season JJA",
        "Season SON",
    ]

    y_labels_display = []

    for subset in subset_order:
        tmp = sens[sens["Subset"] == subset]

        if len(tmp) > 0 and subset in [
            "All stations",
            "Non-traffic stations",
            "Traffic-oriented stations",
        ]:
            n = int(tmp["N_Stations"].dropna().iloc[0])
            y_labels_display.append(f"{subset} (n={n})")
        else:
            y_labels_display.append(subset)

    y_positions = {
        subset: len(subset_order) - 1 - i
        for i, subset in enumerate(subset_order)
    }
    x_positions = {
        source: i
        for i, source in enumerate(SOURCE_ORDER)
    }

    for y in range(len(subset_order)):
        ax_d.axhline(y, color="#EAEAEA", lw=1.0, zorder=0)

    for x0 in range(len(SOURCE_ORDER)):
        ax_d.axvline(x0, color="#F3F3F3", lw=1.0, zorder=0)

    for _, row in sens.iterrows():
        source = row["Source"]
        subset = row["Subset"]
        peak = row["Peak_Radius_m"]

        if source not in x_positions or subset not in y_positions or pd.isna(peak):
            continue

        xx = x_positions[source]
        yy = y_positions[subset]

        ax_d.scatter(
            xx,
            yy,
            s=650,
            color=COLOR_MAP[source],
            edgecolor="black",
            linewidth=1.15,
            alpha=0.92,
            zorder=3,
        )

        txt = ax_d.text(
            xx,
            yy,
            f"{int(peak)}",
            ha="center",
            va="center",
            fontsize=15,
            color="white",
            fontweight="bold",
            zorder=4,
        )
        txt.set_path_effects([withStroke(linewidth=2.4, foreground="black")])

    ax_d.set_xlim(-0.5, len(SOURCE_ORDER) - 0.5)
    ax_d.set_ylim(-0.5, len(subset_order) - 0.5)
    ax_d.set_xticks(range(len(SOURCE_ORDER)))
    ax_d.set_xticklabels(SOURCE_ORDER, rotation=0)

    y_tick_positions = [y_positions[s] for s in subset_order]
    ax_d.set_yticks(y_tick_positions)
    ax_d.set_yticklabels(y_labels_display)

    ax_d.set_title(
        "Peak radii across station-type and seasonal subsets",
        fontsize=21,
        fontweight="bold",
        pad=10,
    )
    ax_d.text(X_OFFSET_PANEL, Y_OFFSET_PANEL, "d", transform=ax_d.transAxes, **PANEL_LABEL_KW)
    ax_d.tick_params(axis="both", labelsize=16)
    ax_d.spines[["top", "right"]].set_visible(False)

    ax_d.text(
        0.99,
        -0.12,
        "Numbers denote peak radius (m).",
        transform=ax_d.transAxes,
        fontsize=16,
        ha="right",
        va="top",
        color="black",
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.94,
        bottom=0.12,
        wspace=0.30,
        hspace=0.32,
    )

    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("Extended Data Fig. 1 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


def main():
    (
        mean_curves,
        full_peak_summary,
        foldagg_band_df,
        boot_peak_df,
        class_prop_df,
        station_type_df,
        seasonal_df,
    ) = load_source_data()

    plot_figure(
        mean_curves=mean_curves,
        full_peak_summary=full_peak_summary,
        foldagg_band_df=foldagg_band_df,
        boot_peak_df=boot_peak_df,
        class_prop_df=class_prop_df,
        station_type_df=station_type_df,
        seasonal_df=seasonal_df,
    )


if __name__ == "__main__":
    main()
