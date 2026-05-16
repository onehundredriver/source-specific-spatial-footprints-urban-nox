import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns
from shapely import wkt

warnings.filterwarnings("ignore")


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"

sns.set_theme(style="ticks", font="Arial")


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


def get_param(df, key):
    row = df[(df["Data_Type"] == "parameter") & (df["Feature"] == key)]
    if row.empty:
        raise ValueError(f"Missing parameter: {key}")
    return str(row["Value"].iloc[0])


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    n_clusters = int(float(get_param(df, "N_CLUSTERS")))
    cluster_colors = get_param(df, "CLUSTER_COLORS").split(",")

    eval_df = df[df["Data_Type"] == "cluster_evaluation"].copy()
    station_df = df[df["Data_Type"] == "station_cluster_fold"].copy()
    fingerprint_df = df[df["Data_Type"] == "cluster_fingerprint"].copy()
    traffic_df = df[df["Data_Type"] == "traffic_network"].copy()

    for col in ["K", "SSE", "Silhouette"]:
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")

    for col in ["Cluster", "Fold", "Longitude", "Latitude", "X_3857", "Y_3857"]:
        station_df[col] = pd.to_numeric(station_df[col], errors="coerce")

    fingerprint_df["Cluster"] = pd.to_numeric(fingerprint_df["Cluster"], errors="coerce").astype(int)
    fingerprint_df["Delta_R2"] = pd.to_numeric(fingerprint_df["Delta_R2"], errors="coerce")
    fingerprint_df["Feature"] = fingerprint_df["Feature"].astype(str)

    if not traffic_df.empty:
        traffic_df["avg_total_volume_bpr"] = pd.to_numeric(
            traffic_df["avg_total_volume_bpr"],
            errors="coerce",
        ).fillna(0)
        traffic_df["Line_Width"] = pd.to_numeric(
            traffic_df["Line_Width"],
            errors="coerce",
        ).fillna(0.5)
        traffic_df["geometry"] = traffic_df["geometry_wkt"].apply(wkt.loads)
        gdf_traffic = gpd.GeoDataFrame(traffic_df, geometry="geometry", crs="EPSG:3857")
    else:
        gdf_traffic = gpd.GeoDataFrame(traffic_df, geometry=[], crs="EPSG:3857")

    gdf_station = gpd.GeoDataFrame(
        station_df,
        geometry=gpd.points_from_xy(station_df["X_3857"], station_df["Y_3857"]),
        crs="EPSG:3857",
    )

    fig = plt.figure(figsize=(26, 15))

    main_gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.15)
    left_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[0], hspace=0.35)
    left_top_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=left_gs[0], wspace=0.35)

    ax_a = fig.add_subplot(left_top_gs[0])
    ax_b = fig.add_subplot(left_top_gs[1])
    ax_c = fig.add_subplot(left_gs[1])
    ax_d = fig.add_subplot(main_gs[1])

    # -------------------------------------------------------
    # Panel a: clustering evaluation
    # -------------------------------------------------------
    color1 = "#3C5488"
    color2 = "#E64B35"

    eval_df = eval_df.sort_values("K")
    k_values = eval_df["K"].astype(int).tolist()
    sse = eval_df["SSE"].tolist()
    silhouette_scores = eval_df["Silhouette"].tolist()

    ax_a.set_xlabel("Number of Clusters ($k$)", fontsize=22, fontweight="bold", color="black", labelpad=10)
    ax_a.set_ylabel("Sum of Squared Errors (SSE)", color="black", fontsize=22, fontweight="bold", labelpad=15)

    line1 = ax_a.plot(k_values, sse, marker="o", markersize=10, color=color1, lw=3, label="SSE")
    ax_a.tick_params(axis="both", labelcolor="black", labelsize=18)
    ax_a.set_xticks(k_values)
    ax_a.grid(False)

    ax_a_twin = ax_a.twinx()
    ax_a_twin.grid(False)
    ax_a_twin.set_ylabel("Silhouette Score", color="black", fontsize=22, fontweight="bold", labelpad=15)
    line2 = ax_a_twin.plot(
        k_values,
        silhouette_scores,
        marker="s",
        markersize=10,
        color=color2,
        lw=3,
        label="Silhouette",
    )
    ax_a_twin.tick_params(axis="y", labelcolor="black", labelsize=18)

    optimal_k = n_clusters
    ax_a.axvline(x=optimal_k, color="black", linestyle="--", lw=2.5, alpha=0.7, zorder=0)

    if optimal_k in k_values:
        idx = k_values.index(optimal_k)
        ax_a.scatter(optimal_k, sse[idx], color="white", edgecolor=color1, s=200, zorder=5, lw=2.5)
        ax_a_twin.scatter(
            optimal_k,
            silhouette_scores[idx],
            color="white",
            edgecolor=color2,
            s=200,
            zorder=5,
            lw=2.5,
        )

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax_a.legend(lines, labels, loc="upper right", frameon=False, fontsize=18, labelcolor="black")

    ax_a.spines["top"].set_visible(False)
    ax_a_twin.spines["top"].set_visible(False)
    ax_a.text(-0.15, 1.06, "a", transform=ax_a.transAxes, fontsize=40, fontweight="bold", va="bottom", color="black")

    # -------------------------------------------------------
    # Panel b: fold jitter scatter
    # -------------------------------------------------------
    folds = sorted(station_df["Fold"].dropna().unique().astype(int).tolist())
    n_folds = len(folds)
    rng = np.random.default_rng(42)

    for i in range(1, n_folds):
        ax_b.axvline(i + 0.5, color="gray", lw=1.5, ls="--", alpha=0.3, zorder=1)

    for i, fold in enumerate(folds):
        df_fold = station_df[station_df["Fold"].astype(int) == fold].copy()

        xs = (i + 1) + rng.uniform(-0.25, 0.25, len(df_fold))
        ys = rng.uniform(0.1, 0.9, len(df_fold))

        for idx, (_, row) in enumerate(df_fold.iterrows()):
            cluster_id = int(row["Cluster"])
            ax_b.scatter(
                xs[idx],
                ys[idx],
                facecolors=cluster_colors[cluster_id],
                edgecolors="white",
                marker="o",
                s=200,
                linewidth=1.5,
                alpha=0.9,
                zorder=3,
            )

    ax_b.set_xlim(0.5, n_folds + 0.5)
    ax_b.set_ylim(0, 1)
    ax_b.set_xticks(range(1, n_folds + 1))
    ax_b.set_xticklabels(
        [
            f"Fold {int(fold)}\n(n={len(station_df[station_df['Fold'].astype(int) == int(fold)])})"
            for fold in folds
        ],
        fontsize=18,
        fontweight="normal",
        color="black",
    )
    ax_b.tick_params(axis="x", which="major", pad=15)
    ax_b.set_yticks([])
    ax_b.set_ylabel("")
    sns.despine(ax=ax_b, left=True, top=True, right=True)
    ax_b.text(-0.05, 1.06, "b", transform=ax_b.transAxes, fontsize=40, fontweight="bold", va="bottom", color="black")

    # -------------------------------------------------------
    # Panel c: cluster response fingerprint
    # -------------------------------------------------------
    feature_order = fingerprint_df["Feature"].drop_duplicates().tolist()

    sns.barplot(
        data=fingerprint_df,
        x="Feature",
        y="Delta_R2",
        hue="Cluster",
        order=feature_order,
        palette=cluster_colors[:n_clusters],
        ax=ax_c,
        edgecolor="white",
        linewidth=1.8,
        capsize=0.1,
        err_kws={"color": "#444444", "linewidth": 2},
    )

    ax_c.set_xlabel("")
    ax_c.set_yscale("symlog", linthresh=0.01)
    ax_c.set_ylabel(r"Time-Averaged Spatial $\Delta R^2$", fontsize=22, fontweight="bold", labelpad=4, color="black")
    ax_c.axhline(0, color="black", lw=2, ls="-", alpha=0.8)

    num_features = len(feature_order)
    for i in range(1, num_features):
        ax_c.axvline(i - 0.5, color="gray", lw=1.5, ls="--", alpha=0.4, zorder=0)

    ax_c.tick_params(axis="y", labelsize=18)
    ax_c.set_xticklabels(ax_c.get_xticklabels(), rotation=0, fontsize=18, fontweight="normal", color="black")
    sns.despine(ax=ax_c)

    if ax_c.get_legend() is not None:
        ax_c.get_legend().remove()

    ax_c.text(-0.06, 1.06, "c", transform=ax_c.transAxes, fontsize=40, fontweight="bold", va="bottom", color="black")

    # -------------------------------------------------------
    # Panel d: map + traffic network
    # -------------------------------------------------------
    ax_d.set_anchor("N")

    if not gdf_traffic.empty:
        max_vol = gdf_traffic["avg_total_volume_bpr"].max()
        min_vol = gdf_traffic["avg_total_volume_bpr"].min() + 1

        if max_vol <= 0:
            max_vol = 1

        if min_vol <= 0:
            min_vol = 1

        cmap_traffic = "Greys"
        norm = mcolors.LogNorm(vmin=min_vol, vmax=max_vol)
        sm = plt.cm.ScalarMappable(cmap=cmap_traffic, norm=norm)
        sm._A = []

        gdf_traffic.plot(
            ax=ax_d,
            column="avg_total_volume_bpr",
            cmap=cmap_traffic,
            norm=norm,
            linewidth=gdf_traffic["Line_Width"],
            alpha=0.6,
            zorder=1,
        )

        cax = ax_d.inset_axes([0.60, 0.88, 0.32, 0.02])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label("Traffic Volume (log veh/60min)", fontsize=14, fontweight="bold", color="black")
        cbar.ax.tick_params(labelsize=12, colors="black", length=4)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

    for cluster_id in range(n_clusters):
        subset = gdf_station[gdf_station["Cluster"].astype(int) == cluster_id]
        if not subset.empty:
            subset.plot(
                ax=ax_d,
                color=cluster_colors[cluster_id],
                marker="o",
                edgecolor="white",
                markersize=200,
                linewidth=1.5,
                alpha=0.95,
                zorder=3,
            )

    if not gdf_station.empty:
        minx, miny, maxx, maxy = gdf_station.total_bounds
        pad_x = (maxx - minx) * 0.12
        pad_y = (maxy - miny) * 0.18
        ax_d.set_xlim(minx - pad_x, maxx + pad_x)
        ax_d.set_ylim(miny - pad_y, maxy + pad_y * 1.8)

    ax_d.axis("off")
    ax_d.text(-0.02, 1.035, "d", transform=ax_d.transAxes, fontsize=40, fontweight="bold", va="bottom", color="black")

    legend_elements = [
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=cluster_colors[i],
            markeredgecolor="white",
            markersize=14,
            markeredgewidth=1.5,
            label=f"Cluster {i + 1}",
        )
        for i in range(n_clusters)
    ]

    legend = ax_d.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        title="Micro-environment Clusters",
        title_fontsize=20,
        fontsize=18,
        frameon=False,
    )
    plt.setp(legend.get_title(), color="black", fontweight="bold")
    plt.setp(legend.get_texts(), color="black")

    plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.96])

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Supplementary Fig. 6 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
