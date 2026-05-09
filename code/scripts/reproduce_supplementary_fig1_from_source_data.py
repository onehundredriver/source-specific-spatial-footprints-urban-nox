import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Wedge, FancyArrowPatch
import matplotlib.patheffects as path_effects


# ============================================================
# Reproduce Supplementary Fig. 1 from public schematic source data
# Input:
#   data/source_data/supplementary/Supplementary_Fig1_source_data.csv
# Output:
#   figures/supplementary/Supplementary_Fig1_reproduced_from_source_data.png
#   figures/supplementary/Supplementary_Fig1_reproduced_from_source_data.pdf
# ============================================================


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"


def find_repo_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))


REPO_ROOT = find_repo_root()

SOURCE_DATA_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "source_data",
    "supplementary",
    "Supplementary_Fig1_source_data.csv"
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "figures",
    "supplementary"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    "Supplementary_Fig1_reproduced_from_source_data.png"
)

OUTPUT_PDF = os.path.join(
    OUTPUT_DIR,
    "Supplementary_Fig1_reproduced_from_source_data.pdf"
)


def get_color(df, name):
    row = df[(df["Data_Type"] == "color_parameter") & (df["Parameter"] == name)]
    if row.empty:
        raise ValueError(f"Missing color parameter: {name}")
    return str(row["Value"].iloc[0])


def draw_sector_with_buffers(ax, theta1, theta2, r_max, facecolor, edge_color, grid_color, num_buffers=7, alpha=0.8):
    wedge = Wedge(
        (0, 0),
        r_max,
        theta1,
        theta2,
        facecolor=facecolor,
        edgecolor=edge_color,
        alpha=alpha,
        lw=1.5,
    )
    ax.add_patch(wedge)

    theta_rad = np.linspace(np.radians(theta1), np.radians(theta2), 100)
    radii = np.linspace(r_max / num_buffers, r_max, num_buffers)

    for r in radii[:-1]:
        x_arc = r * np.cos(theta_rad)
        y_arc = r * np.sin(theta_rad)
        ax.plot(
            x_arc,
            y_arc,
            color=grid_color,
            lw=1.2,
            alpha=0.8,
            linestyle="--",
        )

    return radii


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    r_max = float(df[(df["Data_Type"] == "global_parameter") & (df["Parameter"] == "R_MAX")]["Value"].iloc[0])
    num_buffers = int(float(df[(df["Data_Type"] == "global_parameter") & (df["Parameter"] == "num_buffers")]["Value"].iloc[0]))

    edge_color = get_color(df, "EDGE_COLOR")
    grid_color = get_color(df, "GRID_COLOR")

    shadow_effect = [
        path_effects.withStroke(
            linewidth=3,
            foreground="#777777",
            alpha=0.6,
            offset=(1.2, -1.2),
        )
    ]

    panel_order = ["a", "b", "c"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=300)

    for i, panel in enumerate(panel_order):
        ax = axes[i]
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-1.5, 5.5)

        title = df[
            (df["Data_Type"] == "panel_title")
            & (df["Panel"] == panel)
        ]["Text"].iloc[0]

        ax.text(
            -0.05,
            1.15,
            panel,
            transform=ax.transAxes,
            fontsize=40,
            fontweight="bold",
            va="bottom",
            color="black",
        )

        ax.set_title(
            title,
            fontsize=24,
            fontweight="bold",
            pad=20,
        )

        df_sectors = df[
            (df["Data_Type"] == "sector")
            & (df["Panel"] == panel)
        ].copy()

        radii = None

        for _, row in df_sectors.iterrows():
            radii = draw_sector_with_buffers(
                ax=ax,
                theta1=float(row["Theta1_deg"]),
                theta2=float(row["Theta2_deg"]),
                r_max=r_max,
                facecolor=str(row["Facecolor"]),
                edge_color=edge_color,
                grid_color=grid_color,
                num_buffers=num_buffers,
                alpha=float(row["Alpha"]),
            )

        ax.plot(
            0,
            0,
            marker="o",
            color="black",
            markersize=12,
            zorder=5,
        )

        # Panel-specific arrows.
        if panel == "b":
            arrow1 = FancyArrowPatch(
                (3, 2),
                (0, -0.8),
                connectionstyle="arc3,rad=-0.3",
                color=edge_color,
                arrowstyle="simple,head_width=14,head_length=14",
                lw=2.5,
            )
            arrow2 = FancyArrowPatch(
                (-3, 2),
                (0, -0.8),
                connectionstyle="arc3,rad=0.3",
                color=edge_color,
                arrowstyle="simple,head_width=14,head_length=14",
                lw=2.5,
            )
            ax.add_patch(arrow1)
            ax.add_patch(arrow2)

        # Text labels.
        df_text = df[
            (df["Data_Type"] == "text_label")
            & (df["Panel"] == panel)
        ].copy()

        for _, row in df_text.iterrows():
            text = str(row["Text"])
            x = float(row["X"])
            y = float(row["Y"])
            text_color = str(row["Value"])

            if panel == "b" and "Spatial Aggregation" in text:
                ax.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="top",
                    fontsize=18,
                    fontweight="bold",
                    color=edge_color,
                    bbox=dict(
                        facecolor="white",
                        edgecolor=edge_color,
                        pad=6,
                        boxstyle="round,pad=0.5",
                    ),
                )
            else:
                kwargs = {}
                if text_color.lower() == "white":
                    kwargs["path_effects"] = shadow_effect

                ax.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=16,
                    **kwargs,
                )

        # Buffer labels in panel a.
        if panel == "a":
            df_buffer = df[
                (df["Data_Type"] == "buffer_label")
                & (df["Panel"] == "a")
            ].copy()

            for _, row in df_buffer.iterrows():
                x = float(row["X"])
                text = str(row["Text"])

                ax.plot(
                    x,
                    0,
                    marker="|",
                    color=edge_color,
                    markersize=10,
                    markeredgewidth=2,
                )

                ax.text(
                    x,
                    -0.3,
                    text,
                    ha="right",
                    va="top",
                    rotation=45,
                    fontsize=14,
                    fontweight="normal",
                    color=edge_color,
                )

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Supplementary Fig. 1 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
