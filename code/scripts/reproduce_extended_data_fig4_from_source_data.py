import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# Reproduce Extended Data Fig. 4 from compact public source data
# Input:
#   data/source_data/extended_data/Extended_Data_Fig4_source_data.csv
# Output:
#   figures/extended_data/Extended_Data_Fig4_reproduced_from_source_data.png
#   figures/extended_data/Extended_Data_Fig4_reproduced_from_source_data.pdf
# ============================================================


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["text.color"] = "black"
plt.rcParams["font.size"] = 16


RANDOM_SEED = 42


def find_repo_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))


REPO_ROOT = find_repo_root()

SOURCE_DATA_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "source_data",
    "extended_data",
    "Extended_Data_Fig4_source_data.csv"
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "figures",
    "extended_data"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Extended_Data_Fig4_reproduced_from_source_data.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Extended_Data_Fig4_reproduced_from_source_data.pdf")


def adaptive_station_sorting(plot_data, alpha=0.4):
    subplot_c_min = plot_data["Color_Value"].quantile(0.05)
    subplot_c_max = plot_data["Color_Value"].quantile(0.95)
    subplot_mid = (subplot_c_min + subplot_c_max) / 2.0

    def calc_station_dev(group):
        c_min = group["Color_Value"].quantile(0.05)
        c_max = group["Color_Value"].quantile(0.95)

        if c_max > c_min:
            local_mid = (c_min + c_max) / 2.0
            diff = group["Color_Value"] - local_mid
        else:
            diff = group["Color_Value"] - subplot_mid

        dev = np.where(diff < 0, abs(diff) * 0.9, abs(diff) * 1.5)

        dev_min, dev_max = dev.min(), dev.max()
        if dev_max > dev_min:
            group["dev_norm"] = (dev - dev_min) / (dev_max - dev_min)
        else:
            group["dev_norm"] = 0.0

        return group

    plot_data = plot_data.groupby("Station", group_keys=False).apply(calc_station_dev)

    rng = np.random.default_rng(RANDOM_SEED)
    plot_data["random_noise"] = rng.uniform(0, 1, size=len(plot_data))
    plot_data["sort_score"] = (alpha * plot_data["dev_norm"]) + ((1 - alpha) * plot_data["random_noise"])

    return plot_data.sort_values(by="sort_score", ascending=True)


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    required_cols = [
        "Base_Feature",
        "Color_Feature",
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
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in Extended_Data_Fig4_source_data.csv: {col}")

    df["Distance_Start_m"] = pd.to_numeric(df["Distance_Start_m"], errors="coerce")
    df["Distance_End_m"] = pd.to_numeric(df["Distance_End_m"], errors="coerce")
    df["X_Value"] = pd.to_numeric(df["X_Value"], errors="coerce")
    df["SHAP_Value"] = pd.to_numeric(df["SHAP_Value"], errors="coerce")
    df["Color_Value"] = pd.to_numeric(df["Color_Value"], errors="coerce")
    df["Color_Vmin_P5"] = pd.to_numeric(df["Color_Vmin_P5"], errors="coerce")
    df["Color_Vmax_P95"] = pd.to_numeric(df["Color_Vmax_P95"], errors="coerce")

    df = df.dropna(subset=[
        "Distance_Start_m",
        "X_Value",
        "SHAP_Value",
        "Color_Value",
    ]).copy()

    base_feature = df["Base_Feature"].iloc[0]
    color_feature = df["Color_Feature"].iloc[0]

    unique_distances = (
        df[["Distance_Start_m", "Distance_End_m", "Distance_Label"]]
        .drop_duplicates()
        .sort_values("Distance_Start_m")
        .reset_index(drop=True)
    )

    n_cols = len(unique_distances)
    n_rows = 2

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 12),
        sharey=True,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.92,
        wspace=0.15,
        hspace=0.35,
    )

    vmin = float(df["Color_Vmin_P5"].dropna().iloc[0])
    vmax = float(df["Color_Vmax_P95"].dropna().iloc[0])

    cmap = plt.cm.coolwarm
    scatter = None

    for col_idx, row_dist in unique_distances.iterrows():
        d_start = int(row_dist["Distance_Start_m"])
        d_end = int(row_dist["Distance_End_m"])
        label = row_dist["Distance_Label"]

        for row_idx, direction in enumerate(["center", "side"]):
            ax = axes[row_idx, col_idx] if n_cols > 1 else axes[row_idx]

            plot_data = df[
                (df["Distance_Start_m"] == d_start)
                & (df["Distance_End_m"] == d_end)
                & (df["Direction"] == direction)
            ].copy()

            if plot_data.empty:
                ax.axis("off")
                continue

            plot_data = adaptive_station_sorting(plot_data, alpha=0.4)

            scatter = ax.scatter(
                plot_data["X_Value"],
                plot_data["SHAP_Value"],
                c=plot_data["Color_Value"],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=15,
                alpha=0.8,
                edgecolors="none",
            )

            ax.axhline(
                0,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
            )

            ax.set_yscale("symlog", linthresh=3)

            x_min = plot_data["X_Value"].min()
            x_max = plot_data["X_Value"].max()

            if x_max > x_min:
                ax.set_xticks([x_min, x_min + (x_max - x_min) / 2, x_max])
                ax.set_xticklabels(["Low", "Med", "High"])
            else:
                ax.set_xticks([])

            if row_idx == 0:
                ax.set_title(
                    f"Distance: {label}",
                    fontsize=20,
                    fontweight="bold",
                    pad=15,
                )

            if col_idx == n_cols // 2:
                ax.set_xlabel(
                    f"Base Feature: emission_density ({direction.capitalize()})",
                    fontsize=18,
                    fontweight="bold",
                    labelpad=10,
                )

            if col_idx == 0:
                ax.set_ylabel(
                    "SHAP Value\n(Impact on NOx)",
                    fontsize=18,
                )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    if scatter is not None:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(scatter, cax=cbar_ax)
        cbar.outline.set_visible(False)
        cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
        cbar.set_ticklabels(["Low", "Med", "High"])
        cbar.set_label(
            f"Interacting Feature: {color_feature}",
            fontsize=18,
            fontweight="bold",
            rotation=270,
            labelpad=25,
        )
        cbar.ax.tick_params(labelsize=18)

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Extended Data Fig. 4 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
