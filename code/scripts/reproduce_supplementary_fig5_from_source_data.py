import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
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
    "Supplementary_Fig5_source_data.csv"
)

OUTPUT_DIR = os.path.join(REPO_ROOT, "figures", "supplementary")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Supplementary_Fig5_reproduced_from_source_data.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Supplementary_Fig5_reproduced_from_source_data.pdf")


CLUSTER_CMAP = LinearSegmentedColormap.from_list("shap", ["#1E88E5", "#FF0052"])


def draw_shap_bar_panel(ax_main, df, source_label, panel_letter):
    df_imp = df[
        (df["Data_Type"] == "feature_importance")
        & (df["Source"] == source_label)
    ].copy()

    df_points = df[
        (df["Data_Type"] == "shap_point")
        & (df["Source"] == source_label)
    ].copy()

    if df_imp.empty:
        raise ValueError(f"No feature_importance rows for {source_label}")

    df_imp["Feature_Rank"] = pd.to_numeric(df_imp["Feature_Rank"], errors="coerce")
    df_imp["Mean_Abs_SHAP"] = pd.to_numeric(df_imp["Mean_Abs_SHAP"], errors="coerce")
    df_imp["Category_Total_Importance"] = pd.to_numeric(df_imp["Category_Total_Importance"], errors="coerce")

    df_imp = df_imp.sort_values("Feature_Rank").head(20)

    features = df_imp["Feature"].tolist()
    display_names = df_imp["Feature_Display"].tolist()
    mean_imp = df_imp["Mean_Abs_SHAP"].to_numpy()
    categories = df_imp["Feature_Category"].tolist()

    cat_total = (
        df_imp
        .drop_duplicates(subset=["Feature_Category"])
        .set_index("Feature_Category")["Category_Total_Importance"]
        .to_dict()
    )

    sorted_cats = sorted(cat_total.keys(), key=lambda k: cat_total[k], reverse=True)

    cmap_base = plt.cm.Blues
    cat_colors = {
        cat: cmap_base(0.65 - 0.4 * (i / max(1, len(sorted_cats) - 1)))
        for i, cat in enumerate(sorted_cats)
    }

    y_positions = np.arange(len(features))[::-1]

    ax_top = ax_main.twiny()

    bar_colors = [cat_colors[c] for c in categories][::-1]
    ax_top.barh(
        y_positions,
        mean_imp[::-1],
        color=bar_colors,
        alpha=0.85,
        height=0.6,
        zorder=0,
    )

    ax_top.set_xlabel(
        "mean(|SHAP value|)",
        fontsize=22,
        fontweight="bold",
        color="black",
        labelpad=12,
    )
    ax_top.tick_params(axis="x", colors="black", labelsize=18)

    if np.nanmax(mean_imp) > 0:
        ax_top.set_xlim(0, np.nanmax(mean_imp) * 2.2)

    # Scatter points
    rng = np.random.default_rng(42)

    for rank_idx, feature in enumerate(features):
        sub = df_points[df_points["Feature"] == feature].copy()
        if sub.empty:
            continue

        sub["SHAP_Value"] = pd.to_numeric(sub["SHAP_Value"], errors="coerce")
        sub["Feature_Value_Normalized"] = pd.to_numeric(sub["Feature_Value_Normalized"], errors="coerce")

        sub = sub.dropna(subset=["SHAP_Value", "Feature_Value_Normalized"])

        y_base = len(features) - 1 - rank_idx
        jitter = rng.normal(0, 0.08, len(sub))

        ax_main.scatter(
            sub["SHAP_Value"],
            np.full(len(sub), y_base) + jitter,
            c=sub["Feature_Value_Normalized"],
            cmap=CLUSTER_CMAP,
            vmin=0,
            vmax=1,
            s=10,
            alpha=0.6,
            linewidths=0,
            zorder=3,
        )

    xmin, xmax = ax_main.get_xlim()
    span = xmax - xmin if xmax > xmin else 1
    ax_main.set_xlim(xmin - span * 0.85, xmax + span * 0.40)

    new_xmin, new_xmax = ax_main.get_xlim()
    span_new = new_xmax - new_xmin

    ax_main.set_yticks(np.arange(len(features)))
    ax_main.set_yticklabels([])
    ax_main.tick_params(axis="y", length=0)

    for i, feat_name in enumerate(display_names[::-1]):
        ax_main.text(
            new_xmin + span_new * 0.38,
            i,
            feat_name,
            fontsize=16,
            fontweight="bold",
            color="black",
            va="center",
            ha="right",
        )

    ax_main.set_xlabel(
        "SHAP value (Impact on model output)",
        fontsize=22,
        fontweight="bold",
        color="black",
        labelpad=12,
    )

    ax_main.tick_params(axis="x", labelsize=18, colors="black")

    ax_main.set_zorder(ax_top.get_zorder() + 1)
    ax_main.patch.set_visible(False)

    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    sm = plt.cm.ScalarMappable(
        cmap=CLUSTER_CMAP,
        norm=plt.Normalize(vmin=0, vmax=1),
    )

    cb = plt.colorbar(
        sm,
        ax=ax_main,
        aspect=35,
        pad=0.04,
        shrink=0.8,
        location="right",
    )
    cb.set_label(
        "Feature Value",
        size=18,
        fontweight="bold",
        color="black",
        labelpad=5,
    )
    cb.ax.tick_params(labelsize=14, colors="black")
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.outline.set_visible(False)

    ax_main.text(
        -0.05,
        1.13,
        panel_letter,
        transform=ax_main.transAxes,
        fontsize=40,
        fontweight="bold",
        va="top",
    )

    # Donut chart
    ax_sun = ax_main.inset_axes([0.73, 0.28, 0.25, 0.70])
    inner_sizes = [cat_total[c] for c in sorted_cats]
    inner_colors = [cat_colors[c] for c in sorted_cats]

    wedges, texts, autotexts = ax_sun.pie(
        inner_sizes,
        radius=1.0,
        colors=inner_colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5),
        autopct=lambda pct: f"{pct:.0f}%" if pct > 3 else "",
        pctdistance=0.75,
    )

    plt.setp(autotexts, size=15, weight="bold", color="white")

    ax_sun.legend(
        wedges,
        sorted_cats,
        title="Feature Categories",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=14,
        title_fontsize=16,
        frameon=False,
    )


def draw_rank_boxplot(ax_box, df, source_label, panel_letter):
    df_rank = df[
        (df["Data_Type"] == "bootstrap_rank")
        & (df["Source"] == source_label)
    ].copy()

    if df_rank.empty:
        raise ValueError(f"No bootstrap rank rows for {source_label}")

    df_rank["Feature_Rank"] = pd.to_numeric(df_rank["Feature_Rank"], errors="coerce")
    df_rank["Rank_Value"] = pd.to_numeric(df_rank["Rank_Value"], errors="coerce")

    features = (
        df_rank[["Feature", "Feature_Rank"]]
        .drop_duplicates()
        .sort_values("Feature_Rank")["Feature"]
        .tolist()
    )

    data_list = [
        df_rank[df_rank["Feature"] == feat]["Rank_Value"].dropna().values
        for feat in features
    ]

    positions = np.arange(len(data_list))[::-1]

    bp = ax_box.boxplot(
        data_list,
        positions=positions,
        vert=False,
        widths=0.5,
        patch_artist=True,
        showfliers=True,
    )

    for box in bp["boxes"]:
        box.set_facecolor("#E0E0E0")
        box.set_edgecolor("black")
        box.set_linewidth(1.8)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2.5)

    for whisker in bp["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(1.8)
        whisker.set_linestyle("--")

    for cap in bp["caps"]:
        cap.set_color("black")
        cap.set_linewidth(1.8)

    for flier in bp["fliers"]:
        flier.set_marker("o")
        flier.set_markerfacecolor("none")
        flier.set_markeredgecolor("black")
        flier.set_markersize(5)

    ax_box.set_ylim(-0.5, 19.5)
    ax_box.set_yticks([])
    ax_box.set_yticklabels([])

    ax_box.set_xlabel(
        "Rank (1 = Highest Importance)",
        fontsize=22,
        fontweight="bold",
        labelpad=12,
        color="black",
    )
    ax_box.tick_params(axis="x", labelsize=18, colors="black")

    d_max = np.nanmax([np.nanmax(vals) for vals in data_list if len(vals) > 0])
    ax_box.set_xlim(0, max(21, d_max + 1))
    ax_box.set_xticks([1, 5, 10, 15, 20])

    ax_box.grid(False)

    ax_box.spines["top"].set_visible(False)
    ax_box.spines["right"].set_visible(False)
    ax_box.spines["left"].set_visible(False)

    ax_box.text(
        -0.05,
        1.13,
        panel_letter,
        transform=ax_box.transAxes,
        fontsize=40,
        fontweight="bold",
        va="top",
    )


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    fig = plt.figure(figsize=(24, 20))

    gs_outer = gridspec.GridSpec(
        2,
        1,
        height_ratios=[1, 1],
        hspace=0.55,
        top=0.88,
        bottom=0.08,
        left=0.05,
        right=0.97,
    )

    gs_temp = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs_outer[0],
        width_ratios=[2.5, 1],
        wspace=-0.05,
    )
    ax_a = fig.add_subplot(gs_temp[0])
    ax_b = fig.add_subplot(gs_temp[1])

    gs_spat = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs_outer[1],
        width_ratios=[2.5, 1],
        wspace=-0.05,
    )
    ax_c = fig.add_subplot(gs_spat[0])
    ax_d = fig.add_subplot(gs_spat[1])

    draw_shap_bar_panel(ax_a, df, "Temporal_Generalization", "a")
    draw_rank_boxplot(ax_b, df, "Temporal_Generalization", "b")

    draw_shap_bar_panel(ax_c, df, "Spatial_Generalization", "c")
    draw_rank_boxplot(ax_d, df, "Spatial_Generalization", "d")

    fig.text(
        0.48,
        0.96,
        "Temporal Generalization",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="black",
    )

    fig.text(
        0.48,
        0.465,
        "Spatial Generalization",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="black",
    )

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Supplementary Fig. 5 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
