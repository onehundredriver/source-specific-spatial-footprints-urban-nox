import os
import warnings
from math import pi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
import seaborn as sns

warnings.filterwarnings("ignore")


# ============================================================
# Reproduce Fig. 2 from public source data
# Input:
#   data/source_data/main/Fig2_source_data.csv
# Output:
#   figures/main/Fig2_reproduced_from_source_data.png
#   figures/main/Fig2_reproduced_from_source_data.pdf
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
    "Fig2_source_data.csv"
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "figures",
    "main"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Fig2_reproduced_from_source_data.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Fig2_reproduced_from_source_data.pdf")


C_MODEL = {
    "RF": "#E64B35",
    "LGBM": "#4DBBD5",
}

COLORS_NPG = [
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#84CDCA",
    "#7E6148",
    "#DC0000",
]

PALETTE_SCALE = ["#00A087", "#84CDCA", "#F39B7F", "#E64B35"]

TIME_ORDER = ["1hour", "24hour", "7day", "31day"]
TIME_LABELS = ["1h", "24h", "7d", "31d"]

FEATURE_ORDER = [
    "Buffer",
    "Aggregated",
    "Emission",
    "Temporal",
    "Topology",
    "UCP",
    "Land Use",
    "POI",
]

LABEL_CONFIG_C = {
    "Emission": (0, 22, "center", "bottom"),
    "Buffer": (18, 10, "left", "bottom"),
    "Aggregated": (18, 0, "left", "center"),
    "Temporal": (-18, 0, "right", "center"),
    "POI": (18, 0, "left", "top"),
    "Land Use": (28, 0, "left", "top"),
    "UCP": (-28, 0, "right", "top"),
    "Topology": (18, 15, "left", "bottom"),
}


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.default"] = "regular"
    sns.set_theme(style="ticks", font="Arial")

    fig = plt.figure(figsize=(26, 18))

    gs_main = gridspec.GridSpec(
        1,
        2,
        width_ratios=[1.35, 1],
        wspace=0.15
    )

    gs_left = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs_main[0],
        height_ratios=[1, 1],
        hspace=0.3
    )

    # ------------------------------------------------------------
    # Panel b: validation R2 distributions
    # ------------------------------------------------------------
    df_b = df[df["Panel"] == "b"].copy()
    df_b["Metric_Value"] = pd.to_numeric(df_b["Metric_Value"], errors="coerce")
    df_b["Time_Order"] = pd.to_numeric(df_b["Time_Order"], errors="coerce")
    df_b = df_b.sort_values(["Time_Order", "Model", "Feature"])

    gs_b = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs_left[0],
        wspace=0.05
    )

    cv_list = ["Spatial_CV", "Temporal_CV"]
    cv_titles = ["Spatial Generalization", "Temporal Generalization"]

    for idx, cv in enumerate(cv_list):
        ax_b = fig.add_subplot(gs_b[idx])
        sub_b = df_b[df_b["CV_Strategy"] == cv].copy()

        sns.boxplot(
            data=sub_b,
            x="Time_Scale_Raw",
            y="Metric_Value",
            hue="Model",
            order=TIME_ORDER,
            hue_order=["LGBM", "RF"],
            palette=C_MODEL,
            ax=ax_b,
            width=0.6,
            linewidth=1.8,
            fliersize=4,
            whis=1.5,
            medianprops={"color": "white", "linewidth": 2.5}
        )

        ax_b.set_title(
            cv_titles[idx],
            fontsize=22,
            fontweight="bold",
            pad=40
        )

        ax_b.set_xlabel("")
        ax_b.set_xticklabels(TIME_LABELS, fontsize=18, fontweight="normal")

        if idx == 0:
            ax_b.set_ylabel(r"Validation $R^2$", fontsize=22, fontweight="bold", labelpad=15)
            ax_b.tick_params(axis="y", labelsize=18)
            ax_b.legend(title="", frameon=False, loc="upper left", fontsize=18)
            ax_b.text(-0.12, 1.25, "b", transform=ax_b.transAxes, fontsize=45, fontweight="bold", va="top")
        else:
            ax_b.set_ylabel("")
            ax_b.tick_params(axis="y", left=False, labelleft=False, labelsize=18)
            if ax_b.get_legend() is not None:
                ax_b.get_legend().remove()

        sns.despine(ax=ax_b)

    # ------------------------------------------------------------
    # Panel c: contribution quadrant
    # ------------------------------------------------------------
    ax_c = fig.add_subplot(gs_left[1])

    df_c = df[df["Panel"] == "c"].copy()
    df_c["Temporal_Contribution_Percent"] = pd.to_numeric(df_c["Temporal_Contribution_Percent"], errors="coerce")
    df_c["Spatial_Contribution_Percent"] = pd.to_numeric(df_c["Spatial_Contribution_Percent"], errors="coerce")
    df_c["Feature_Order"] = pd.to_numeric(df_c["Feature_Order"], errors="coerce")
    df_c = df_c.sort_values("Feature_Order")

    for i, row in df_c.iterrows():
        name = row["Feature"]
        x = row["Temporal_Contribution_Percent"]
        y = row["Spatial_Contribution_Percent"]
        color = COLORS_NPG[int(row["Feature_Order"] - 1) % len(COLORS_NPG)]

        ax_c.scatter(
            x,
            y,
            s=750,
            color=color,
            alpha=0.85,
            edgecolors="none",
            zorder=4
        )

        off_x, off_y, ha, va = LABEL_CONFIG_C.get(name, (12, 0, "left", "center"))

        txt = ax_c.annotate(
            name,
            (x, y),
            xytext=(off_x, off_y),
            textcoords="offset points",
            fontsize=20,
            fontweight="bold",
            color="#111111",
            ha=ha,
            va=va,
            zorder=5
        )

        txt.set_path_effects([
            path_effects.withStroke(linewidth=4, foreground="white", alpha=0.9)
        ])

    ax_c.axhline(y=0, color="#444444", ls="--", lw=1.5, alpha=0.5, zorder=3)
    ax_c.axvline(x=0, color="#444444", ls="--", lw=1.5, alpha=0.5, zorder=3)

    xmin, xmax = ax_c.get_xlim()
    ymin, ymax = ax_c.get_ylim()

    rect_left_side = patches.Rectangle(
        (xmin, ymin),
        0 - xmin,
        ymax - ymin,
        linewidth=0,
        facecolor="gray",
        alpha=0.15,
        zorder=1
    )
    ax_c.add_patch(rect_left_side)

    rect_bottom_side = patches.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        0 - ymin,
        linewidth=0,
        facecolor="gray",
        alpha=0.15,
        zorder=1
    )
    ax_c.add_patch(rect_bottom_side)

    ax_c.set_xlim(xmin, xmax)
    ax_c.set_ylim(ymin, ymax)

    ax_c.annotate(
        "",
        xy=(xmax, 0),
        xytext=(xmax - (xmax - xmin) * 0.02, 0),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.5, ls="-", alpha=0.7),
        zorder=3
    )

    ax_c.annotate(
        "",
        xy=(0, ymax),
        xytext=(0, ymax - (ymax - ymin) * 0.02),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.5, ls="-", alpha=0.7),
        zorder=3
    )

    text_st = {"fontsize": 20, "style": "italic", "color": "#888888", "alpha": 0.8}

    ax_c.text(0.96, 0.98, "Universal Drivers", transform=ax_c.transAxes, ha="right", va="top", **text_st)
    ax_c.text(0.06, 0.98, "Spatial Specialized", transform=ax_c.transAxes, ha="left", va="top", **text_st)
    ax_c.text(0.96, 0.13, "Temporal Specialized", transform=ax_c.transAxes, ha="right", va="bottom", **text_st)

    ax_c.set_xlabel(r"Temporal Contribution ($\Delta R^2_{Time}$ %)", fontsize=22, fontweight="bold", labelpad=15)
    ax_c.set_ylabel(r"Spatial Contribution ($\Delta R^2_{Spatial}$ %)", fontsize=22, fontweight="bold", labelpad=15)
    ax_c.tick_params(axis="both", labelsize=18, length=8, width=2)
    sns.despine(ax=ax_c)
    ax_c.text(-0.06, 1.1, "c", transform=ax_c.transAxes, fontsize=45, fontweight="bold", va="top")

    # ------------------------------------------------------------
    # Panel d: radar plots
    # ------------------------------------------------------------
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs_main[1],
        height_ratios=[1, 1],
        hspace=0.3
    )

    df_d = df[df["Panel"] == "d"].copy()
    df_d["Radar_Value_Percent"] = pd.to_numeric(df_d["Radar_Value_Percent"], errors="coerce")
    df_d["Feature_Order"] = pd.to_numeric(df_d["Feature_Order"], errors="coerce")

    valid_features = [
        f for f in FEATURE_ORDER
        if f in df_d["Feature"].dropna().unique().tolist()
    ]

    n_features = len(valid_features)
    angles = [n / float(n_features) * 2 * pi for n in range(n_features)]
    angles += angles[:1]

    last_ax_rad = None

    for idx, cv in enumerate(cv_list):
        ax_rad = fig.add_subplot(gs_right[idx], polar=True)
        last_ax_rad = ax_rad

        ax_rad.set_theta_offset(pi / 2)
        ax_rad.set_theta_direction(-1)

        if idx == 0:
            y_limit = 15
            y_ticks = np.arange(0, 15, 3)
        else:
            y_limit = 30
            y_ticks = np.arange(0, 30, 6)

        for j, t in enumerate(TIME_ORDER):
            sub = df_d[
                (df_d["CV_Strategy"] == cv)
                & (df_d["Time_Scale_Raw"] == t)
            ].copy()

            if sub.empty:
                continue

            vals = (
                sub
                .set_index("Feature")
                .reindex(valid_features)["Radar_Value_Percent"]
                .fillna(0)
                .values
                .flatten()
                .tolist()
            )
            vals += vals[:1]

            ax_rad.plot(angles, vals, lw=3, color=PALETTE_SCALE[j], label=t, zorder=j)
            ax_rad.fill(angles, vals, color=PALETTE_SCALE[j], alpha=0.08, zorder=j)

        ax_rad.set_xticks(angles[:-1])
        ax_rad.set_xticklabels(valid_features, color="#222222", fontsize=20, fontweight="bold")

        ax_rad.set_ylim(0, y_limit)
        ax_rad.set_rticks(y_ticks)
        ax_rad.spines["polar"].set_visible(False)

        ax_rad.set_rlabel_position(30)
        plt.setp(ax_rad.get_yticklabels(), color="black", fontsize=18, fontweight="normal")

        title_text = f"{cv.replace('_CV', '')} Potentials"
        ax_rad.set_title(rf"{title_text} ($\Delta R^2$%)", fontsize=22, fontweight="bold", pad=45)

        if idx == 0:
            ax_rad.text(-0.15, 1.25, "d", transform=ax_rad.transAxes, fontsize=45, fontweight="bold", va="top")

    handles, _ = last_ax_rad.get_legend_handles_labels()

    fig.legend(
        handles,
        ["1h", "24h", "7d", "31d"],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=18,
        bbox_to_anchor=(0.75, 0.035)
    )

    plt.tight_layout(rect=[0.01, 0.06, 0.99, 0.99])

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Fig. 2 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
