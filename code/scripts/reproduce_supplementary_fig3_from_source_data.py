import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator

warnings.filterwarnings("ignore")


# ============================================================
# Reproduce Supplementary Fig. 3 from compact public source data
# Input:
#   data/source_data/supplementary/Supplementary_Fig3_source_data.csv
# Output:
#   figures/supplementary/Supplementary_Fig3_reproduced_from_source_data.png
#   figures/supplementary/Supplementary_Fig3_reproduced_from_source_data.pdf
# ============================================================


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["axes.linewidth"] = 1.5

COLOR_CENTER = "#D55E00"
COLOR_SIDE = "#0072B2"
CMAP_SHAP = LinearSegmentedColormap.from_list("shap", ["#1E88E5", "#FF0052"])

STANDARD_LABELS = [
    "0-50m",
    "50-100m",
    "100-200m",
    "200-500m",
    "500-1000m",
    "1000-2000m",
    "2000-3000m",
]


def find_repo_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))


REPO_ROOT = find_repo_root()

SOURCE_DATA_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "source_data",
    "supplementary",
    "Supplementary_Fig3_source_data.csv"
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "figures",
    "supplementary"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Supplementary_Fig3_reproduced_from_source_data.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Supplementary_Fig3_reproduced_from_source_data.pdf")


def draw_dual_funnel_panel(fig, outer_gs, df_panel, category_name, feature_display, panel_letter):
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer_gs,
        hspace=0.1,
    )

    ax_center = fig.add_subplot(inner_gs[0])
    ax_side = fig.add_subplot(inner_gs[1], sharex=ax_center)

    sc_global = None
    y_abs_max = 0.0

    def plot_location(ax, location, color_env):
        nonlocal sc_global
        nonlocal y_abs_max

        df_env = df_panel[
            (df_panel["Location"] == location)
            & (df_panel["Data_Type"] == "envelope")
        ].copy()

        df_scatter = df_panel[
            (df_panel["Location"] == location)
            & (df_panel["Data_Type"] == "scatter_sample")
        ].copy()

        if df_env.empty:
            return

        df_env["Bin_Order"] = pd.to_numeric(df_env["Bin_Order"], errors="coerce")
        df_env["Upper_Envelope"] = pd.to_numeric(df_env["Upper_Envelope"], errors="coerce")
        df_env["Lower_Envelope"] = pd.to_numeric(df_env["Lower_Envelope"], errors="coerce")
        df_env = df_env.sort_values("Bin_Order")

        x_pos = df_env["Bin_Order"].to_numpy(dtype=float)
        upper_env = df_env["Upper_Envelope"].fillna(0).to_numpy(dtype=float)
        lower_env = df_env["Lower_Envelope"].fillna(0).to_numpy(dtype=float)

        ax.axhline(0, color="black", linewidth=1.5, alpha=0.8, zorder=2)

        if len(x_pos) > 1:
            x_smooth = np.linspace(np.nanmin(x_pos), np.nanmax(x_pos), 300)
            upper_smooth = PchipInterpolator(x_pos, upper_env)(x_smooth)
            lower_smooth = PchipInterpolator(x_pos, lower_env)(x_smooth)

            upper_smooth = np.clip(upper_smooth, 0, None)
            lower_smooth = np.clip(lower_smooth, None, 0)

            ax.fill_between(
                x_smooth,
                upper_smooth,
                lower_smooth,
                color=color_env,
                alpha=0.15,
                zorder=1,
            )
            ax.plot(
                x_smooth,
                upper_smooth,
                color=color_env,
                ls="--",
                lw=2.0,
                alpha=0.9,
                zorder=1,
            )
            ax.plot(
                x_smooth,
                lower_smooth,
                color=color_env,
                ls="--",
                lw=2.0,
                alpha=0.9,
                zorder=1,
            )

            y_abs_max = max(
                y_abs_max,
                float(np.nanmax(np.abs(upper_smooth))) if len(upper_smooth) else 0,
                float(np.nanmax(np.abs(lower_smooth))) if len(lower_smooth) else 0,
            )

        if not df_scatter.empty:
            df_scatter["Point_X"] = pd.to_numeric(df_scatter["Point_X"], errors="coerce")
            df_scatter["SHAP_Value"] = pd.to_numeric(df_scatter["SHAP_Value"], errors="coerce")
            df_scatter["Feature_Value_Norm"] = pd.to_numeric(df_scatter["Feature_Value_Norm"], errors="coerce")

            df_scatter = df_scatter.dropna(subset=["Point_X", "SHAP_Value", "Feature_Value_Norm"])

            if not df_scatter.empty:
                sc = ax.scatter(
                    df_scatter["Point_X"],
                    df_scatter["SHAP_Value"],
                    c=df_scatter["Feature_Value_Norm"],
                    cmap=CMAP_SHAP,
                    vmin=0,
                    vmax=1,
                    s=20,
                    alpha=0.7,
                    edgecolor="none",
                    zorder=3,
                )
                sc_global = sc
                y_abs_max = max(y_abs_max, float(df_scatter["SHAP_Value"].abs().max()))

        title = "UPWIND (Center)" if location == "Center" else "CROSSWIND (Side)"
        ax.text(
            0.02,
            0.90,
            title,
            transform=ax.transAxes,
            color="black",
            fontsize=18,
            fontweight="bold",
        )

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plot_location(ax_center, "Center", COLOR_CENTER)
    plot_location(ax_side, "Side", COLOR_SIDE)

    y_max = y_abs_max * 1.15 if y_abs_max > 0 else 1.0
    ax_center.set_ylim(-y_max, y_max)
    ax_side.set_ylim(-y_max, y_max)

    ax_center.tick_params(labelbottom=False, bottom=False, labelsize=18, colors="black")
    ax_side.tick_params(axis="both", labelsize=18, colors="black")

    ax_side.set_xticks(np.arange(len(STANDARD_LABELS)))
    ax_side.set_xticklabels(
        STANDARD_LABELS,
        fontsize=18,
        fontweight="bold",
        rotation=25,
        ha="center",
        color="black",
    )

    ax_center.set_ylabel("SHAP Value", fontsize=22, fontweight="bold", color="black")
    ax_side.set_ylabel("SHAP Value", fontsize=22, fontweight="bold", color="black")

    ax_center.set_title(
        f"{category_name} ({feature_display})",
        fontsize=24,
        fontweight="bold",
        pad=15,
        color="black",
    )

    ax_center.text(
        -0.09,
        1.20,
        panel_letter,
        transform=ax_center.transAxes,
        fontsize=40,
        fontweight="bold",
        va="top",
        color="black",
    )

    return sc_global


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    required_cols = [
        "Panel",
        "Data_Type",
        "Category",
        "Feature_Display",
        "Location",
        "Distance_Label",
        "Bin_Order",
        "Point_X",
        "SHAP_Value",
        "Feature_Value_Norm",
        "Upper_Envelope",
        "Lower_Envelope",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in Supplementary_Fig3_source_data.csv: {col}")

    fig = plt.figure(figsize=(26, 20))
    gs = gridspec.GridSpec(
        2,
        2,
        hspace=0.35,
        wspace=0.20,
        top=0.90,
        bottom=0.10,
        left=0.06,
        right=0.92,
    )

    panel_order = ["a", "b", "c", "d"]
    sc_global = None

    for idx, panel_letter in enumerate(panel_order):
        df_panel = df[df["Panel"] == panel_letter].copy()

        if df_panel.empty:
            continue

        category_name = df_panel["Category"].dropna().iloc[0]
        feature_display = df_panel["Feature_Display"].dropna().iloc[0]

        sc = draw_dual_funnel_panel(
            fig=fig,
            outer_gs=gs[idx // 2, idx % 2],
            df_panel=df_panel,
            category_name=category_name,
            feature_display=feature_display,
            panel_letter=panel_letter,
        )

        if sc is not None:
            sc_global = sc

    fig.text(
        0.48,
        0.96,
        "Dual-Funnel Decay of Local Impact Across Spatial Gradients",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="black",
    )

    if sc_global is not None:
        cbar_ax = fig.add_axes([0.94, 0.25, 0.01, 0.50])
        cb = fig.colorbar(sc_global, cax=cbar_ax)
        cb.set_label(
            "Feature Value (Low → High)",
            size=22,
            fontweight="bold",
            labelpad=5,
            color="black",
        )
        cb.set_ticks([0, 1])
        cb.ax.tick_params(size=0, labelsize=18, colors="black")
        cb.set_ticklabels(["Low", "High"], fontsize=18, fontweight="bold", color="black")
        cb.outline.set_visible(False)

    plt.savefig(OUTPUT_PNG, dpi=400, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Supplementary Fig. 3 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
