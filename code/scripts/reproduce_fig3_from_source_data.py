import os
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
import contextily as ctx
import matplotlib.lines as mlines
from matplotlib.patheffects import withStroke

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
sns.set_theme(style="ticks", font="Arial")

CLUSTER_COLORS = ['#D55E00', '#0072B2', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#8C564B']
N_CLUSTERS = 5


def find_repo_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))


REPO_ROOT = find_repo_root()

SOURCE_DATA_PATH = os.path.join(REPO_ROOT, "data", "source_data", "main", "Fig3_source_data.csv")
TRAFFIC_DATA_PATH = os.path.join(REPO_ROOT, "data", "source_data", "main", "Fig3_traffic_network.csv")

OUTPUT_DIR = os.path.join(REPO_ROOT, "figures", "main")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Fig3_reproduced_from_source_data.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Fig3_reproduced_from_source_data.pdf")


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)
    if not os.path.exists(TRAFFIC_DATA_PATH):
        raise FileNotFoundError(TRAFFIC_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")
    df_traffic = pd.read_csv(TRAFFIC_DATA_PATH, encoding="utf-8-sig")

    fig = plt.figure(figsize=(26, 15))
    main_gs = gridspec.GridSpec(1, 2, width_ratios=[1.8, 1], wspace=0.08)
    left_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[0], hspace=0.35)
    left_top_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=left_gs[0], wspace=0.55)

    ax_a = fig.add_subplot(left_top_gs[0])
    ax_b = fig.add_subplot(left_top_gs[1])
    ax_c = fig.add_subplot(left_gs[1])
    ax_d = fig.add_subplot(main_gs[1])

    # panel a
    df_a = df[df["Panel"] == "a"].copy()
    for metric, color, marker in [("SSE", '#3C5488', 'o'), ("Silhouette", '#E64B35', 's')]:
        sub = df_a[df_a["Metric_Name"] == metric].copy()
        sub["K_Value"] = pd.to_numeric(sub["K_Value"], errors="coerce")
        sub["Metric_Value"] = pd.to_numeric(sub["Metric_Value"], errors="coerce")
        sub = sub.sort_values("K_Value")

        if metric == "SSE":
            line1 = ax_a.plot(sub["K_Value"], sub["Metric_Value"], marker=marker, markersize=10,
                              color=color, lw=3, label='SSE')
            ax_a.set_xlabel('Number of Clusters ($k$)', fontsize=22, fontweight='bold', color='black', labelpad=15)
            ax_a.set_ylabel('Sum of Squared Errors (SSE)', color='black', fontsize=22, fontweight='bold', labelpad=15)
            ax_a.tick_params(axis='both', labelcolor='black', labelsize=18)
            ax_a.set_xticks(sub["K_Value"].tolist())
            ax_a.grid(False)

            optimal_k = 5
            row_opt = sub[sub["K_Value"] == optimal_k]
            if len(row_opt) > 0:
                ax_a.axvline(x=optimal_k, color='black', linestyle='--', lw=2.5, alpha=0.7, zorder=0)
                ax_a.scatter(optimal_k, row_opt["Metric_Value"].iloc[0], color='white', edgecolor=color, s=200, zorder=5, lw=2.5)

            ax_a_twin = ax_a.twinx()
            ax_a_twin.grid(False)

        else:
            line2 = ax_a_twin.plot(sub["K_Value"], sub["Metric_Value"], marker=marker, markersize=10,
                                   color=color, lw=3, label='Silhouette')
            ax_a_twin.set_ylabel('Silhouette Score', color='black', fontsize=22, fontweight='bold', labelpad=12)
            ax_a_twin.tick_params(axis='y', labelcolor='black', labelsize=18)
            optimal_k = 5
            row_opt = sub[sub["K_Value"] == optimal_k]
            if len(row_opt) > 0:
                ax_a_twin.scatter(optimal_k, row_opt["Metric_Value"].iloc[0], color='white', edgecolor=color, s=200, zorder=5, lw=2.5)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_a.legend(lines, labels, loc='upper right', frameon=False, fontsize=18, labelcolor='black')
    ax_a.spines['top'].set_visible(False)
    ax_a_twin.spines['top'].set_visible(False)
    ax_a.text(-0.26, 1.06, 'a', transform=ax_a.transAxes, fontsize=40, fontweight='bold', va='bottom', color='black')

    # panel b
    import matplotlib.lines as mlines
    df_b = df[df["Panel"] == "b"].copy()
    df_b["X_Value"] = pd.to_numeric(df_b["X_Value"], errors="coerce")
    df_b["Y_Value"] = pd.to_numeric(df_b["Y_Value"], errors="coerce")
    df_b["Cluster"] = pd.to_numeric(df_b["Cluster"], errors="coerce")
    feat_colors = {}
    feat_markers = {}
    unique_features = df_b["Feature"].dropna().astype(str).unique().tolist()
    base_colors = ['#9b59b6', '#3498db', '#2ecc71', '#f39c12']
    base_markers = ['s', '^', 'D', 'P']
    for i, f in enumerate(unique_features):
        feat_colors[f] = base_colors[i % len(base_colors)]
        feat_markers[f] = base_markers[i % len(base_markers)]

    feat_leg_elements = []
    for f_y in unique_features:
        sub = df_b[df_b["Feature"] == f_y].copy()
        if sub.empty:
            continue

        inlier_mask = sub["Notes"].astype(str).str.contains("is_inlier=True", regex=False)
        sub_inliers = sub[inlier_mask].copy()
        sub_outliers = sub[~inlier_mask].copy()

        if not sub_inliers.empty:
            ax_b.scatter(sub_inliers["X_Value"], sub_inliers["Y_Value"],
                         facecolors=feat_colors[f_y], edgecolors='none',
                         marker=feat_markers[f_y], s=160, linewidth=0, alpha=0.85, zorder=3)

        if not sub_outliers.empty:
            ax_b.scatter(sub_outliers["X_Value"], sub_outliers["Y_Value"],
                         facecolors=feat_colors[f_y], edgecolors='none',
                         marker=feat_markers[f_y], s=160, linewidth=0, alpha=0.4, zorder=3)
            ax_b.scatter(sub_outliers["X_Value"], sub_outliers["Y_Value"],
                         facecolors='none', edgecolors=feat_colors[f_y],
                         marker='o', s=400, linewidth=2, alpha=1.0, zorder=4)

        r2_vals = pd.to_numeric(sub["Regression_R2"], errors="coerce").dropna().unique().tolist()
        if len(r2_vals) > 0 and not sub_inliers.empty:
            x_min = max(0, sub_inliers["X_Value"].min() * 0.8)
            x_max = 0.5
            x_fit = np.linspace(x_min, x_max, 100)

            # approximate linear fit from inlier data
            coef = np.polyfit(sub_inliers["X_Value"], sub_inliers["Y_Value"], 1)
            y_fit = coef[0] * x_fit + coef[1]
            ax_b.plot(x_fit, y_fit, color=feat_colors[f_y], linestyle='--', linewidth=2.5, alpha=0.8, zorder=2)

            text_x = x_fit[-1]
            text_y = y_fit[-1]
            r2_text = ax_b.text(text_x + 0.02, text_y, f"$R^2$ = {r2_vals[0]:.2f}",
                                color='black', fontsize=18, fontweight='normal',
                                ha='left', va='center', zorder=5)
            r2_text.set_path_effects([withStroke(linewidth=3, foreground='white')])

        feat_leg_elements.append(
            mlines.Line2D([0], [0], marker=feat_markers[f_y], color='w',
                          markerfacecolor=feat_colors[f_y], markeredgecolor='none',
                          markersize=12, label=f_y)
        )

    feat_leg_elements.append(
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                      markeredgecolor='gray', markersize=14, markeredgewidth=2,
                      linestyle=(0, (2, 1)), label='Excluded Points')
    )

    ax_b.set_xlim(-0.15, 1.15)
    ax_b.set_xticks([0, 0.5, 1.0])
    ax_b.set_ylim(-0.06, 0.46)
    ax_b.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax_b.set_xlabel(r"$\Delta R^2$ (Emission)", fontsize=22, fontweight='bold', labelpad=2)
    ax_b.set_ylabel(r"$\Delta R^2$ (Land Use / UCP)", fontsize=22, fontweight='bold', labelpad=2)
    ax_b.axhline(0, color='black', lw=1.5, ls='--', alpha=0.4, zorder=1)
    ax_b.axvline(0, color='black', lw=1.5, ls='--', alpha=0.4, zorder=1)
    ax_b.tick_params(axis='both', labelsize=18)
    sns.despine(ax=ax_b)
    ax_b.legend(handles=feat_leg_elements, loc='lower right', fontsize=18, frameon=False)
    ax_b.text(-0.20, 1.06, 'b', transform=ax_b.transAxes, fontsize=40, fontweight='bold', va='bottom', color='black')

    # panel c
    df_c = df[df["Panel"] == "c"].copy()
    df_c["Cluster"] = pd.to_numeric(df_c["Cluster"], errors="coerce")
    df_c["Metric_Value"] = pd.to_numeric(df_c["Metric_Value"], errors="coerce")

    sns.barplot(
        data=df_c, x='Feature', y='Metric_Value', hue='Cluster',
        palette=CLUSTER_COLORS[:N_CLUSTERS], ax=ax_c,
        edgecolor='white', linewidth=1.8, capsize=0.1, errwidth=2
    )
    ax_c.set_xlabel('')
    ax_c.set_yscale('symlog', linthresh=0.01)
    ax_c.set_ylabel(r"Time-Averaged Spatial $\Delta R^2$", fontsize=22, fontweight='bold', labelpad=4, color='black')
    ax_c.axhline(0, color='black', lw=2, ls='-', alpha=0.8)
    plot_feats = df_c["Feature"].dropna().astype(str).unique().tolist()
    num_features = len(plot_feats)
    for i in range(1, num_features):
        ax_c.axvline(i - 0.5, color='gray', lw=1.5, ls='--', alpha=0.4, zorder=0)
    ax_c.tick_params(axis='y', labelsize=18)
    ax_c.set_xticklabels(ax_c.get_xticklabels(), rotation=0, fontsize=18, fontweight='normal', color='black')
    sns.despine(ax=ax_c)
    if ax_c.get_legend() is not None:
        ax_c.get_legend().remove()
    ax_c.text(-0.11, 1.06, 'c', transform=ax_c.transAxes, fontsize=40, fontweight='bold', va='bottom', color='black')

    # panel d
    ax_d.set_anchor('N')

    # traffic
    df_traffic["geometry"] = df_traffic["Traffic_Line_WKT"].apply(wkt.loads)
    gdf_traffic = gpd.GeoDataFrame(df_traffic, geometry="geometry", crs="EPSG:4326").to_crs(epsg=3857)

    max_vol = pd.to_numeric(gdf_traffic["Traffic_Volume"], errors="coerce").max()
    min_vol = pd.to_numeric(gdf_traffic["Traffic_Volume"], errors="coerce").min() + 1

    cmap_traffic = 'Greys'
    norm = mcolors.LogNorm(vmin=min_vol, vmax=max_vol)
    sm = plt.cm.ScalarMappable(cmap=cmap_traffic, norm=norm)
    sm._A = []

    gdf_traffic.plot(
        ax=ax_d, column='Traffic_Volume', cmap=cmap_traffic,
        norm=norm, linewidth=gdf_traffic['Traffic_Line_Width'],
        alpha=0.6, zorder=1
    )

    cax = ax_d.inset_axes([0.58, 0.88, 0.35, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Traffic Volume (log veh/60min)', fontsize=16, fontweight='bold', color='black')
    cbar.ax.tick_params(labelsize=14, colors='black', length=4)
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    # station points
    df_d = df[df["Panel"] == "d"].copy()
    df_d["Longitude"] = pd.to_numeric(df_d["Longitude"], errors="coerce")
    df_d["Latitude"] = pd.to_numeric(df_d["Latitude"], errors="coerce")
    df_d["Cluster"] = pd.to_numeric(df_d["Cluster"], errors="coerce")
    gdf = gpd.GeoDataFrame(
        df_d,
        geometry=gpd.points_from_xy(df_d["Longitude"], df_d["Latitude"]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    cluster_counts = gdf["Cluster"].value_counts().to_dict()
    sorted_clusters = sorted(cluster_counts.keys(), key=lambda x: cluster_counts[x], reverse=True)

    base_zorder = 10
    for i, cluster_id in enumerate(sorted_clusters):
        subset = gdf[gdf["Cluster"] == cluster_id]
        if subset.empty:
            continue
        current_zorder = base_zorder + i
        mask_traffic = subset["StationType"] == "交通站"

        if (~mask_traffic).any():
            subset[~mask_traffic].plot(
                ax=ax_d, color=CLUSTER_COLORS[int(cluster_id)],
                marker='o', edgecolor='white',
                markersize=200, linewidth=1.5, alpha=0.95, zorder=current_zorder
            )
        if mask_traffic.any():
            subset[mask_traffic].plot(
                ax=ax_d, color=CLUSTER_COLORS[int(cluster_id)],
                marker='P', edgecolor='white',
                markersize=250, linewidth=1.5, alpha=0.95, zorder=current_zorder + 0.5
            )

    xmin, ymin, xmax, ymax = gdf.total_bounds
    zoom_margin = 0.15
    x_margin = (xmax - xmin) * zoom_margin
    y_margin = (ymax - ymin) * zoom_margin
    ax_d.set_xlim(xmin - x_margin, xmax + x_margin)
    ax_d.set_ylim(ymin - y_margin, ymax + y_margin)

    map_top_padding = 0.27
    current_ymin, current_ymax = ax_d.get_ylim()
    y_range = current_ymax - current_ymin
    ax_d.set_ylim(current_ymin, current_ymax + y_range * map_top_padding)

    max_retries = 2
    for attempt in range(max_retries):
        try:
            print(f"Loading basemap ({attempt + 1}/{max_retries})...")
            ctx.add_basemap(ax_d, source=ctx.providers.CartoDB.Positron, alpha=0.7, zorder=0)
            break
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print("Basemap loading skipped; plotting traffic and stations only.")

    ax_d.axis('off')
    ax_d.text(-0.02, 1.035, 'd', transform=ax_d.transAxes, fontsize=40, fontweight='bold', va='bottom', color='black')

    traffic_leg_element = [mlines.Line2D([0], [0], marker='P', color='w',
                                         markerfacecolor='none', markeredgecolor='black',
                                         markersize=14, markeredgewidth=1.8,
                                         label='Traffic Station')]
    traffic_leg = ax_d.legend(handles=traffic_leg_element, loc='lower right',
                              fontsize=16, frameon=False, facecolor='white',
                              framealpha=0.85, edgecolor='gray', borderpad=0.8)
    ax_d.add_artist(traffic_leg)

    for text in traffic_leg.get_texts():
        text.set_path_effects([withStroke(linewidth=3, foreground="white")])

    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=CLUSTER_COLORS[i], markeredgecolor='white',
                      markersize=14, markeredgewidth=1.5,
                      label=f'Cluster {i + 1}')
        for i in range(N_CLUSTERS)
    ]
    main_legend = ax_d.legend(
        handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3,
        title='Micro-environment Clusters', title_fontsize=20, fontsize=18, frameon=False
    )
    plt.setp(main_legend.get_title(), color='black', fontweight='bold')
    plt.setp(main_legend.get_texts(), color='black')

    plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.96])
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Fig. 3 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
