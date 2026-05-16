import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# Reproduce Fig. 8 from compact public source data
# Input:
#   data/source_data/main/Fig8_source_data.csv
# Output:
#   figures/main/Fig8_reproduced_from_source_data.png
#   figures/main/Fig8_reproduced_from_source_data.pdf
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
    "main",
    "Fig8_source_data.csv"
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "figures",
    "main"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Fig8_reproduced_from_source_data.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Fig8_reproduced_from_source_data.pdf")


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


def draw_matrix_block(fig, axes_block, df_panel, panel_letter):
    df_panel = df_panel.copy()

    base_feature_display = df_panel["Base_Feature_Display"].iloc[0]
    color_feature_display = df_panel["Color_Feature_Display"].iloc[0]

    unique_distances = (
        df_panel[["Distance_Start_m", "Distance_End_m", "Distance_Label"]]
        .drop_duplicates()
        .sort_values("Distance_Start_m")
        .reset_index(drop=True)
    )

    n_cols = len(unique_distances)

    vmin = float(df_panel["Color_Vmin_P5"].dropna().iloc[0])
    vmax = float(df_panel["Color_Vmax_P95"].dropna().iloc[0])

    cmap = plt.cm.coolwarm
    scatter = None

    for col_idx, row_dist in unique_distances.iterrows():
        d_start = int(row_dist["Distance_Start_m"])
        d_end = int(row_dist["Distance_End_m"])
        label = row_dist["Distance_Label"]

        for local_row_idx, direction in enumerate(["center", "side"]):
            ax = axes_block[local_row_idx][col_idx] if n_cols > 1 else axes_block[local_row_idx]

            plot_data = df_panel[
                (df_panel["Distance_Start_m"] == d_start)
                & (df_panel["Distance_End_m"] == d_end)
                & (df_panel["Direction"] == direction)
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

            ax.set_yscale("symlog", linthresh=1)

            x_min = plot_data["X_Value"].min()
            x_max = plot_data["X_Value"].max()

            if x_max > x_min:
                ax.set_xticks([x_min, x_min + (x_max - x_min) / 2, x_max])
                ax.set_xticklabels(["Low", "Med", "High"], fontsize=12)
            else:
                ax.set_xticks([])

            if local_row_idx == 0:
                ax.set_title(
                    f"Distance: {label}",
                    fontsize=12,
                    fontweight="bold",
                    pad=12,
                )

            if col_idx == n_cols // 2:
                ax.set_xlabel(
                    f"Base Feature: {base_feature_display} ({direction.capitalize()})",
                    fontsize=12,
                    fontweight="bold",
                    labelpad=4,
                )

            if col_idx == 0:
                ax.set_ylabel(
                    "SHAP Value\n(Impact on NO$_x$)",
                    fontsize=12,
                )

            ax.tick_params(axis="both", labelsize=11)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    return scatter, vmin, vmax, color_feature_display


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    required_cols = [
        "Panel",
        "Base_Feature_Display",
        "Color_Feature_Display",
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
            raise ValueError(f"Missing required column in Fig8_source_data.csv: {col}")

    df["Distance_Start_m"] = pd.to_numeric(df["Distance_Start_m"], errors="coerce")
    df["Distance_End_m"] = pd.to_numeric(df["Distance_End_m"], errors="coerce")
    df["X_Value"] = pd.to_numeric(df["X_Value"], errors="coerce")
    df["SHAP_Value"] = pd.to_numeric(df["SHAP_Value"], errors="coerce")
    df["Color_Value"] = pd.to_numeric(df["Color_Value"], errors="coerce")
    df["Color_Vmin_P5"] = pd.to_numeric(df["Color_Vmin_P5"], errors="coerce")
    df["Color_Vmax_P95"] = pd.to_numeric(df["Color_Vmax_P95"], errors="coerce")

    df = df.dropna(subset=[
        "Panel",
        "Distance_Start_m",
        "X_Value",
        "SHAP_Value",
        "Color_Value",
    ]).copy()

    panels = ["a", "b"]

    fig, axes = plt.subplots(
        4,
        7,
        figsize=(25, 22),
        sharey=False,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.91,
        top=0.96,
        bottom=0.06,
        wspace=0.15,
        hspace=0.42,
    )

    for i, panel in enumerate(panels):
        df_panel = df[df["Panel"] == panel].copy()

        row_start = i * 2
        axes_block = axes[row_start:row_start + 2, :]

        scatter, vmin, vmax, color_feature_display = draw_matrix_block(
            fig=fig,
            axes_block=axes_block,
            df_panel=df_panel,
            panel_letter=panel,
        )

        # Panel letter.
        axes[row_start, 0].text(
            -0.30,
            1.25,
            panel,
            transform=axes[row_start, 0].transAxes,
            fontsize=28,
            fontweight="bold",
            va="top",
            color="black",
        )

        # Separate colorbar for each 2-row block.
        if scatter is not None:
            if i == 0:
                cbar_ax = fig.add_axes([0.925, 0.56, 0.012, 0.34])
            else:
                cbar_ax = fig.add_axes([0.925, 0.12, 0.012, 0.34])

            cbar = fig.colorbar(scatter, cax=cbar_ax)
            cbar.outline.set_visible(False)
            cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
            cbar.set_ticklabels(["Low", "Med", "High"])
            cbar.set_label(
                f"Interacting Feature: {color_feature_display}",
                fontsize=12,
                fontweight="bold",
                rotation=270,
                labelpad=18,
            )
            cbar.ax.tick_params(labelsize=11)

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Fig. 8 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
