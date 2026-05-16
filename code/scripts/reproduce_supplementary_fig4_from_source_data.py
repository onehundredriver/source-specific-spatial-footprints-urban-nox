import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")


# ============================================================
# Reproduce Supplementary Fig. 4 from public source data
# Input:
#   data/source_data/supplementary/Supplementary_Fig4_source_data.csv
# Output:
#   figures/supplementary/Supplementary_Fig4_reproduced_from_source_data.png
#   figures/supplementary/Supplementary_Fig4_reproduced_from_source_data.pdf
# ============================================================


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"

sns.set_theme(style="white", font="Arial")

C_NEG = "#3C5488"
C_ZERO = "#FFFFFF"
C_POS = "#DC0000"
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "UrbanDiverging",
    [C_NEG, C_ZERO, C_POS],
)


def find_repo_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))


REPO_ROOT = find_repo_root()

SOURCE_DATA_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "source_data",
    "supplementary",
    "Supplementary_Fig4_source_data.csv"
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "figures",
    "supplementary"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    "Supplementary_Fig4_reproduced_from_source_data.png"
)

OUTPUT_PDF = os.path.join(
    OUTPUT_DIR,
    "Supplementary_Fig4_reproduced_from_source_data.pdf"
)


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    required_cols = [
        "Model",
        "CV_Strategy",
        "Time_Scale",
        "Time_Label",
        "Feature_Display",
        "Feature_Order",
        "Delta_R2_Percent",
        "Feature_R2",
        "Annotation_Text",
        "Column_Order",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Supplementary_Fig4_source_data.csv missing columns: {missing}")

    df["Feature_Order"] = pd.to_numeric(df["Feature_Order"], errors="coerce")
    df["Column_Order"] = pd.to_numeric(df["Column_Order"], errors="coerce")
    df["Delta_R2_Percent"] = pd.to_numeric(df["Delta_R2_Percent"], errors="coerce")
    df["Feature_R2"] = pd.to_numeric(df["Feature_R2"], errors="coerce")
    df["Annotation_Text"] = df["Annotation_Text"].fillna("")

    df = df.sort_values(["Feature_Order", "Column_Order"]).copy()

    ordered_features = (
        df[["Feature_Display", "Feature_Order"]]
        .drop_duplicates()
        .sort_values("Feature_Order")["Feature_Display"]
        .tolist()
    )

    ordered_cols_df = (
        df[["Model", "CV_Strategy", "Time_Scale", "Time_Label", "Column_Order"]]
        .drop_duplicates()
        .sort_values("Column_Order")
        .reset_index(drop=True)
    )

    ordered_cols = [
        (row["Model"], row["CV_Strategy"], row["Time_Scale"])
        for _, row in ordered_cols_df.iterrows()
    ]

    df["Col_Tuple"] = list(zip(df["Model"], df["CV_Strategy"], df["Time_Scale"]))

    pivot_delta = (
        df
        .pivot(index="Feature_Display", columns="Col_Tuple", values="Delta_R2_Percent")
        .reindex(index=ordered_features, columns=ordered_cols)
    )

    annot_df = (
        df
        .pivot(index="Feature_Display", columns="Col_Tuple", values="Annotation_Text")
        .reindex(index=ordered_features, columns=ordered_cols)
    )

    fig, (ax, cbar_ax) = plt.subplots(
        1,
        2,
        figsize=(20, 11),
        gridspec_kw={"width_ratios": [40, 1], "wspace": 0.04},
    )

    sns.heatmap(
        pivot_delta,
        cmap=CMAP_DIVERGING,
        center=0,
        annot=annot_df,
        fmt="",
        linewidths=1.2,
        linecolor="white",
        ax=ax,
        cbar_ax=cbar_ax,
        cbar_kws={"label": r"Marginal Contribution ($\Delta R^2$, %)"},
        annot_kws={"size": 13, "weight": "bold"},
    )

    cbar_ax.yaxis.label.set_size(16)
    cbar_ax.tick_params(labelsize=14)

    ax.set_ylabel(
        "Aggregation Radii",
        fontsize=18,
        labelpad=15,
        fontweight="bold",
    )

    ax.set_xticks(np.arange(len(ordered_cols)) + 0.5)
    ax.set_xticklabels(
        [row["Time_Label"] for _, row in ordered_cols_df.iterrows()],
        rotation=0,
        fontsize=15,
    )

    ax.tick_params(axis="y", labelsize=15)

    # Group separators: after LGBM spatial, after LGBM temporal / RF spatial boundary, after RF spatial.
    for x_pos in [4, 8, 12]:
        ax.axvline(
            x=x_pos,
            color="black",
            linewidth=(4 if x_pos == 8 else 1.5),
            zorder=5,
        )

    ax.text(
        4,
        -1.0,
        "LightGBM",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color="black",
        clip_on=False,
    )

    ax.text(
        12,
        -1.0,
        "Random Forest",
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="bold",
        color="black",
        clip_on=False,
    )

    for x_idx, label in zip(
        [2, 6, 10, 14],
        ["Spatial CV", "Temporal CV"] * 2,
    ):
        ax.text(
            x_idx,
            -0.3,
            label,
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color="black",
            clip_on=False,
        )

    ax.set_title(
        "Sensitivity Analysis: Impact of Aggregation Radii on Model Performance",
        fontsize=20,
        fontweight="bold",
        pad=110,
    )

    footnote = "* p < 0.05, ** p < 0.01, *** p < 0.001 (One-tailed paired t-test across 58 monitoring stations)"
    fig.text(
        0.06,
        0.015,
        footnote,
        ha="left",
        va="bottom",
        fontsize=14,
        style="italic",
        color="#333333",
    )

    plt.subplots_adjust(
        top=0.80,
        bottom=0.10,
        left=0.1,
        right=0.9,
    )

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Supplementary Fig. 4 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
