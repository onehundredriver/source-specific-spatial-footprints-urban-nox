import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")


# ============================================================
# Reproduce Extended Data Fig. 2 from public source data
# Input:
#   data/source_data/extended_data/Extended_Data_Fig2_source_data.csv
# Output:
#   figures/extended_data/Extended_Data_Fig2_reproduced_from_source_data.png
#   figures/extended_data/Extended_Data_Fig2_reproduced_from_source_data.pdf
# ============================================================


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"

sns.set_theme(style="ticks", font="Arial")


CLUSTER_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]

CLUSTER_NAMES = [
    "Cluster 1 (Urban core)",
    "Cluster 2 (Regional background)",
    "Cluster 3 (Transitional/source-influenced)",
    "Cluster 4 (Traffic–topology responsive)",
    "Cluster 5 (Traffic–land-use responsive)",
]


def find_repo_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))


REPO_ROOT = find_repo_root()

SOURCE_DATA_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "source_data",
    "extended_data",
    "Extended_Data_Fig2_source_data.csv"
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "figures",
    "extended_data"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Extended_Data_Fig2_reproduced_from_source_data.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Extended_Data_Fig2_reproduced_from_source_data.pdf")


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")
    df_points = df[df["Data_Type"] == "environmental_gradient_point"].copy()

    required_cols = [
        "Panel",
        "Station",
        "Cluster",
        "StationType",
        "Is_Traffic_Station",
        "X_Norm",
        "Y_Norm",
        "Y_Label",
    ]

    for col in required_cols:
        if col not in df_points.columns:
            raise ValueError(f"Missing required column in Extended_Data_Fig2_source_data.csv: {col}")

    df_points["Cluster"] = pd.to_numeric(df_points["Cluster"], errors="coerce").astype(int)
    df_points["X_Norm"] = pd.to_numeric(df_points["X_Norm"], errors="coerce")
    df_points["Y_Norm"] = pd.to_numeric(df_points["Y_Norm"], errors="coerce")

    panel_order = ["a", "b", "c", "d", "e"]

    fig = plt.figure(figsize=(24, 14))
    gs = gridspec.GridSpec(2, 3, wspace=0.15, hspace=0.25)
    axes = [fig.add_subplot(gs[i]) for i in range(6)]

    np.random.seed(42)

    for idx, panel_id in enumerate(panel_order):
        ax = axes[idx]

        sub_panel = df_points[df_points["Panel"] == panel_id].copy()
        if sub_panel.empty:
            continue

        x_val = sub_panel["X_Norm"].values
        y_val = sub_panel["Y_Norm"].values
        y_label = sub_panel["Y_Label"].iloc[0]

        # Citywide physical-state density KDE.
        try:
            x_noise = x_val + np.random.normal(0, 0.01, len(x_val))
            y_noise = y_val + np.random.normal(0, 0.01, len(y_val))
            kde = gaussian_kde(np.vstack([x_noise, y_noise]))

            xi, yi = np.mgrid[-0.1:1.1:100j, -0.1:1.1:100j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

            ax.contour(
                xi,
                yi,
                zi.reshape(xi.shape),
                levels=5,
                colors="gray",
                alpha=0.3,
                linewidths=1.2,
            )
            ax.contourf(
                xi,
                yi,
                zi.reshape(xi.shape),
                levels=5,
                cmap="Greys",
                alpha=0.1,
            )
        except Exception as e:
            print(f"KDE skipped for panel {panel_id}: {e}")

        # Points by cluster and station type.
        for cluster_id in range(5):
            sub_cluster = sub_panel[sub_panel["Cluster"] == cluster_id].copy()
            if sub_cluster.empty:
                continue

            non_traffic = sub_cluster[sub_cluster["Is_Traffic_Station"] != "Yes"].copy()
            traffic = sub_cluster[sub_cluster["Is_Traffic_Station"] == "Yes"].copy()

            if not non_traffic.empty:
                ax.scatter(
                    non_traffic["X_Norm"],
                    non_traffic["Y_Norm"],
                    color=CLUSTER_COLORS[cluster_id],
                    edgecolor="white",
                    s=200,
                    linewidth=1.5,
                    marker="o",
                    alpha=0.85,
                    zorder=4,
                )

            if not traffic.empty:
                ax.scatter(
                    traffic["X_Norm"],
                    traffic["Y_Norm"],
                    color=CLUSTER_COLORS[cluster_id],
                    edgecolor="white",
                    s=250,
                    linewidth=1.5,
                    marker="P",
                    alpha=0.95,
                    zorder=5,
                )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])

        ax.set_xticklabels(["Low", "Med", "High"], fontsize=16, fontweight="bold", color="gray")
        ax.set_yticklabels(
            ["Low", "Med", "High"],
            fontsize=16,
            fontweight="bold",
            color="gray",
            rotation=90,
            va="center",
        )

        ax.set_xlabel(
            "Relative Emission Intensity $\\rightarrow$",
            fontsize=18,
            fontweight="bold",
            labelpad=10,
        )

        ax.set_ylabel(
            f"Relative {y_label} $\\rightarrow$",
            fontsize=18,
            fontweight="bold",
            labelpad=10,
        )

        ax.text(
            -0.15,
            1.05,
            panel_id,
            transform=ax.transAxes,
            fontsize=40,
            fontweight="bold",
            va="bottom",
            color="black",
        )

        sns.despine(ax=ax)

    # Sixth panel: legend.
    ax_legend = axes[5]
    ax_legend.axis("off")

    leg_elements = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=CLUSTER_COLORS[i],
            markeredgecolor="white",
            markersize=16,
            markeredgewidth=1.5,
            label=CLUSTER_NAMES[i],
        )
        for i in range(5)
    ]

    leg_elements.append(
        mlines.Line2D(
            [0],
            [0],
            marker="P",
            color="w",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=16,
            markeredgewidth=1.8,
            label="Traffic-oriented station",
        )
    )

    leg_elements.append(
        mlines.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="#e0e0e0",
            markeredgecolor="gray",
            markersize=16,
            label="Citywide physical-state density (KDE)",
        )
    )

    ax_legend.legend(
        handles=leg_elements,
        loc="center",
        fontsize=20,
        frameon=False,
        title="Micro-environmental clusters",
        title_fontsize=24,
        labelspacing=1.2,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Extended Data Fig. 2 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
