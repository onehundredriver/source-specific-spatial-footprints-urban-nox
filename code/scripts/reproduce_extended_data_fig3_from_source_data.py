import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")


# ============================================================
# Reproduce Extended Data Fig. 3 from public source data
# Input:
#   data/source_data/extended_data/Extended_Data_Fig3_source_data.csv
# Output:
#   figures/extended_data/Extended_Data_Fig3_reproduced_from_source_data.png
#   figures/extended_data/Extended_Data_Fig3_reproduced_from_source_data.pdf
# ============================================================


def find_repo_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))


REPO_ROOT = find_repo_root()

SOURCE_DATA_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "source_data",
    "extended_data",
    "Extended_Data_Fig3_source_data.csv"
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "figures",
    "extended_data"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    "Extended_Data_Fig3_reproduced_from_source_data.png"
)

OUTPUT_PDF = os.path.join(
    OUTPUT_DIR,
    "Extended_Data_Fig3_reproduced_from_source_data.pdf"
)

C_NEG = "#3C5488"
C_ZERO = "#FFFFFF"
C_POS = "#DC0000"
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "UrbanDiverging",
    [C_NEG, C_ZERO, C_POS]
)

FEATURE_ORDER = [
    "Baseline",
    "Buffer",
    "Aggregated",
    "Emission",
    "Temporal",
    "Topology",
    "UCP",
    "Land Use",
    "POI",
]

MODELS = ["LGBM", "RF"]
CV_STRATS = ["Spatial_CV", "Temporal_CV"]
TIMES = ["1hour", "24hour", "7day", "31day"]

TIME_MAP = {
    "1hour": "1h",
    "24hour": "24h",
    "7day": "7d",
    "31day": "31d",
}


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    required_cols = [
        "Feature_Domain",
        "Column_Key",
        "Column_Order",
        "Feature_Order",
        "Delta_R2_Percent",
        "Absolute_R2",
        "Annotation_Text",
        "Model",
        "CV_Strategy",
        "Time_Scale_Raw",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in Extended_Data_Fig3_source_data.csv: {col}")

    df["Column_Order"] = pd.to_numeric(df["Column_Order"], errors="coerce")
    df["Feature_Order"] = pd.to_numeric(df["Feature_Order"], errors="coerce")
    df["Delta_R2_Percent"] = pd.to_numeric(df["Delta_R2_Percent"], errors="coerce")
    df["Absolute_R2"] = pd.to_numeric(df["Absolute_R2"], errors="coerce")

    ordered_cols_df = (
        df[["Column_Key", "Column_Order", "Model", "CV_Strategy", "Time_Scale_Raw"]]
        .drop_duplicates()
        .sort_values("Column_Order")
    )

    ordered_cols = ordered_cols_df["Column_Key"].tolist()

    feature_order_df = (
        df[["Feature_Domain", "Feature_Order"]]
        .drop_duplicates()
        .sort_values("Feature_Order")
    )

    final_y_order = [
        f for f in FEATURE_ORDER
        if f in set(feature_order_df["Feature_Domain"])
    ]

    extra_features = [
        f for f in feature_order_df["Feature_Domain"].tolist()
        if f not in final_y_order
    ]

    final_y_order = final_y_order + extra_features

    pivot_delta = (
        df
        .pivot_table(
            index="Feature_Domain",
            columns="Column_Key",
            values="Delta_R2_Percent",
            aggfunc="first"
        )
        .reindex(index=final_y_order, columns=ordered_cols)
    )

    annot_df = (
        df
        .pivot_table(
            index="Feature_Domain",
            columns="Column_Key",
            values="Annotation_Text",
            aggfunc="first"
        )
        .reindex(index=final_y_order, columns=ordered_cols)
    )

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.default"] = "regular"

    sns.set_theme(style="white", font="Arial")

    fig, (ax, cbar_ax) = plt.subplots(
        1,
        2,
        figsize=(18, 12),
        gridspec_kw={"width_ratios": [40, 1], "wspace": 0.04}
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
        annot_kws={"size": 13, "weight": "bold"}
    )

    cbar_ax.yaxis.label.set_size(16)
    cbar_ax.tick_params(labelsize=14)

    ax.set_xlabel("")
    ax.set_ylabel("Feature Domains", fontsize=18, labelpad=15, fontweight="bold")

    ax.set_xticks(np.arange(len(ordered_cols)) + 0.5)
    ax.set_xticklabels(
        [
            TIME_MAP.get(x.split("|")[2], x.split("|")[2])
            for x in ordered_cols
        ],
        rotation=0,
        fontsize=15
    )

    ax.tick_params(axis="y", labelsize=15)

    for x_pos in [4, 8, 12]:
        ax.axvline(
            x=x_pos,
            color="black",
            linewidth=(4 if x_pos == 8 else 1.5),
            zorder=5
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
        clip_on=False
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
        clip_on=False
    )

    for x_idx, label in zip(
        [2, 6, 10, 14],
        ["Spatial CV", "Temporal CV"] * 2
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
            clip_on=False
        )

    ax.set_title(
        "Absolute $R^2$ and Marginal Contribution of Multi-source Features",
        fontsize=18,
        fontweight="bold",
        pad=110
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
        color="#333333"
    )

    plt.subplots_adjust(
        top=0.80,
        bottom=0.10,
        left=0.10,
        right=0.90
    )

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Extended Data Fig. 3 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
