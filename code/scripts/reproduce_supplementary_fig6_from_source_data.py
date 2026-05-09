import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator

warnings.filterwarnings("ignore")


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"


TIME_SCALES = ["1hour", "24hour", "7day", "31day"]
X_LABELS = ["1-Hour", "24-Hour", "7-Day", "31-Day"]

CATEGORY_ORDER = [
    "Meteorological Parameters",
    "Temporal Parameters",
    "Emission Source Parameters",
    "Road Topology Parameters",
    "POI Parameters",
    "Urban Canopy Parameters",
    "Landuse Parameters",
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
    "Supplementary_Fig6_source_data.csv"
)

OUTPUT_DIR = os.path.join(REPO_ROOT, "figures", "supplementary")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    "Supplementary_Fig6_reproduced_from_source_data.png"
)

OUTPUT_PDF = os.path.join(
    OUTPUT_DIR,
    "Supplementary_Fig6_reproduced_from_source_data.pdf"
)


def make_contribution_matrix(df, cv_type):
    sub = df[df["CV_Type"] == cv_type].copy()

    sub["Time_Order"] = pd.to_numeric(sub["Time_Order"], errors="coerce")
    sub["Category_Order"] = pd.to_numeric(sub["Category_Order"], errors="coerce")
    sub["Contribution_Percent"] = pd.to_numeric(sub["Contribution_Percent"], errors="coerce").fillna(0)

    sub = sub.sort_values(["Time_Order", "Category_Order"])

    pivot = (
        sub
        .pivot(index="Time_Scale", columns="Category", values="Contribution_Percent")
        .reindex(index=TIME_SCALES, columns=CATEGORY_ORDER)
        .fillna(0)
    )

    colors = (
        sub
        .drop_duplicates(subset=["Category"])
        .set_index("Category")["Category_Color"]
        .to_dict()
    )

    return pivot, colors


def draw_smooth_alluvial(ax, df_contrib, category_colors, title, panel_letter):
    categories = df_contrib.columns
    x_orig = np.arange(len(TIME_SCALES))

    y_cumulative = np.zeros((len(TIME_SCALES), len(categories) + 1))

    for i, cat in enumerate(categories):
        y_cumulative[:, i + 1] = y_cumulative[:, i] + df_contrib[cat].values

    x_smooth = np.linspace(x_orig.min(), x_orig.max(), 300)
    y_smooth_cumulative = np.zeros((len(x_smooth), len(categories) + 1))

    for i in range(1, len(categories) + 1):
        interpolator = PchipInterpolator(x_orig, y_cumulative[:, i])
        y_smooth_cumulative[:, i] = interpolator(x_smooth)

    for i, cat in enumerate(categories):
        y1 = y_smooth_cumulative[:, i]
        y2 = y_smooth_cumulative[:, i + 1]
        color = category_colors.get(cat, "#CCCCCC")

        ax.fill_between(
            x_smooth,
            y1,
            y2,
            color=color,
            alpha=0.85,
            label=cat,
            edgecolor="white",
            linewidth=1.0,
        )

    ax.text(
        -0.06,
        1.10,
        panel_letter,
        transform=ax.transAxes,
        fontsize=40,
        fontweight="bold",
        va="top",
        color="black",
    )

    ax.set_xlim(x_orig.min(), x_orig.max())
    ax.set_ylim(0, 100)

    ax.set_xticks(x_orig)
    ax.set_xticklabels(
        X_LABELS,
        fontsize=16,
        fontweight="bold",
        color="black",
    )
    ax.tick_params(axis="x", pad=15)

    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(
        ["0%", "25%", "50%", "75%", "100%"],
        fontsize=16,
        color="black",
    )

    ax.set_ylabel(
        "Relative Contribution to Predictability",
        fontsize=20,
        fontweight="bold",
        labelpad=12,
        color="black",
    )

    ax.set_title(
        title,
        fontsize=24,
        fontweight="bold",
        pad=20,
        color="black",
    )

    for x in x_orig:
        ax.axvline(
            x,
            color="black",
            linestyle="--",
            alpha=0.4,
            zorder=0,
            linewidth=1.5,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    df_time, colors = make_contribution_matrix(df, "Time")
    df_station, _ = make_contribution_matrix(df, "Station")

    fig = plt.figure(figsize=(24, 10.5))

    gs = gridspec.GridSpec(
        1,
        2,
        width_ratios=[1, 1],
        wspace=0.25,
        left=0.06,
        right=0.96,
        top=0.90,
        bottom=0.20,
    )

    ax_time = fig.add_subplot(gs[0])
    ax_station = fig.add_subplot(gs[1])

    draw_smooth_alluvial(
        ax_time,
        df_time,
        colors,
        "Temporal Generalization",
        "a",
    )

    draw_smooth_alluvial(
        ax_station,
        df_station,
        colors,
        "Spatial Generalization",
        "b",
    )

    handles, labels = ax_time.get_legend_handles_labels()

    legend = fig.legend(
        handles[::-1],
        labels[::-1],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=4,
        fontsize=16,
        frameon=False,
        title="Feature Categories",
    )

    plt.setp(legend.get_title(), fontsize=18, fontweight="bold")

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Supplementary Fig. 6 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
