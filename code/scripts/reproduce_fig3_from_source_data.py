import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")


# =======================================================
# Reproduce Figure 3 from source data
# Current scope:
#   reproduces the provided station-level slope graph panel set
# =======================================================

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'
sns.set_theme(style="ticks", font="Arial")

COLOR_GAIN = '#E64B35'
COLOR_LOSS = '#8A9CBC'
COLOR_BASE = '#CCCCCC'
COLOR_MEAN = '#111111'


def find_repo_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))


REPO_ROOT = find_repo_root()

SOURCE_DATA_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "source_data",
    "main",
    "Fig3_source_data.csv"
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "figures",
    "main"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Fig3_reproduced_from_source_data.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Fig3_reproduced_from_source_data.pdf")


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    df_station = df[df["Data_Type"] == "station_slope_record"].copy()
    df_mean = df[df["Data_Type"] == "mean_trajectory"].copy()

    feature_list = (
        df["Feature"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    cv_strats = ["Temporal_CV", "Spatial_CV"]

    fig, axes = plt.subplots(
        2,
        len(feature_list),
        figsize=(26, 13),
        sharey=True,
        sharex=True
    )

    if len(feature_list) == 1:
        # 保证 axes 仍然是二维数组
        axes = [[axes[0]], [axes[1]]]

    for row_idx, cv in enumerate(cv_strats):
        for col_idx, feat in enumerate(feature_list):
            ax = axes[row_idx][col_idx]

            sub_station = df_station[
                (df_station["CV_Strategy"] == cv) &
                (df_station["Feature"] == feat)
            ].copy()

            sub_mean = df_mean[
                (df_mean["CV_Strategy"] == cv) &
                (df_mean["Feature"] == feat)
            ].copy()

            x_base, x_feat = 0, 1

            for _, row in sub_station.iterrows():
                y_b = row["Baseline_R2"]
                y_f = row["Added_R2"]
                delta = row["Delta_R2"]

                if pd.isna(y_b) or pd.isna(y_f):
                    continue

                if delta > 0:
                    line_color, line_alpha, line_lw, dot_color = COLOR_GAIN, 0.4, 1.2, COLOR_GAIN
                else:
                    line_color, line_alpha, line_lw, dot_color = COLOR_LOSS, 0.3, 0.8, COLOR_LOSS

                ax.plot([x_base, x_feat], [y_b, y_f], color=line_color, alpha=line_alpha, lw=line_lw, zorder=1)
                ax.scatter(x_base, y_b, color=COLOR_BASE, s=25, alpha=0.8, zorder=2)
                ax.scatter(x_feat, y_f, color=dot_color, s=35, alpha=0.8, zorder=3)

            if len(sub_mean) > 0:
                mean_b = sub_mean["Mean_Baseline_R2"].iloc[0]
                mean_f = sub_mean["Mean_Added_R2"].iloc[0]

                ax.plot(
                    [x_base, x_feat],
                    [mean_b, mean_f],
                    color=COLOR_MEAN,
                    marker='D',
                    markersize=10,
                    lw=4,
                    zorder=5
                )

            ax.axhline(0, color='black', lw=1.5, ls='--', alpha=0.3)
            ax.set_xlim(-0.3, 1.3)
            ax.set_ylim(-0.4, 1.05)
            ax.grid(axis='y', linestyle=':', alpha=0.4)
            sns.despine(ax=ax)
            ax.tick_params(axis='y', labelsize=16)

            if row_idx == 0:
                ax.set_title(feat, fontsize=22, fontweight='bold', pad=20)
                ax.tick_params(axis='x', bottom=False, labelbottom=False)

            if row_idx == 1:
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Baseline', 'Added'], fontsize=18, fontweight='normal')

            if col_idx == 0:
                cv_label = "Temporal Generalization" if cv == 'Temporal_CV' else "Spatial Generalization"
                ax.set_ylabel(f"{cv_label}\nTime-Averaged $R^2$", fontsize=20, fontweight='bold', labelpad=20)

    legend_elements = [
        Line2D([0], [0], color=COLOR_GAIN, lw=3, label='Positive Gain'),
        Line2D([0], [0], color=COLOR_LOSS, lw=3, alpha=0.6, label='Negative Gain / Noise'),
        Line2D([0], [0], color=COLOR_MEAN, marker='D', markersize=10, lw=4, label='Mean Trajectory')
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=3,
        frameon=False,
        fontsize=18
    )

    plt.suptitle(
        "Station-level Predictive Gain (Temporal vs. Spatial)",
        fontsize=26,
        fontweight='bold',
        y=0.98
    )

    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.12, right=0.95, hspace=0.15, wspace=0.02)

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Fig. 3 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
