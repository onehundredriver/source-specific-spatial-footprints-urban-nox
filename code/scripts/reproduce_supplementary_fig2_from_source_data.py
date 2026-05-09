import os
import math
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyArrowPatch, Rectangle
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon, LineString

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
    "Supplementary_Fig2_source_data.csv"
)

OUTPUT_DIR = os.path.join(REPO_ROOT, "figures", "supplementary")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Supplementary_Fig2_reproduced_from_source_data.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Supplementary_Fig2_reproduced_from_source_data.pdf")


def get_param(df, name):
    row = df[(df["Data_Type"] == "parameter") & (df["Feature_ID"] == name)]
    if row.empty:
        raise ValueError(f"Missing parameter: {name}")
    return str(row["Value"].iloc[0])


def to_gdf(df, data_type):
    sub = df[df["Data_Type"] == data_type].copy()
    if sub.empty:
        return gpd.GeoDataFrame(sub, geometry=[], crs="EPSG:4326")

    sub["geometry"] = sub["geometry_wkt"].apply(wkt.loads)
    return gpd.GeoDataFrame(sub, geometry="geometry", crs="EPSG:4326")


def draw_polygon_fill(ax, geom, facecolor, edgecolor=None, alpha=1.0, lw=0.5, linestyle="-", zorder=1):
    if geom is None or geom.is_empty:
        return

    if isinstance(geom, Polygon):
        ax.fill(
            *geom.exterior.xy,
            color=facecolor,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=lw,
            linestyle=linestyle,
            zorder=zorder,
        )
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            ax.fill(
                *p.exterior.xy,
                color=facecolor,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=lw,
                linestyle=linestyle,
                zorder=zorder,
            )


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    df = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    start_radius = int(float(get_param(df, "START_RADIUS")))
    end_radius = int(float(get_param(df, "END_RADIUS")))
    origin_lat = float(get_param(df, "ORIGIN_LAT"))
    origin_lon = float(get_param(df, "ORIGIN_LON"))
    center_angle_east = float(get_param(df, "CENTER_ANGLE_EAST"))
    center_angle = float(get_param(df, "CENTER_ANGLE"))

    color_center = get_param(df, "COLOR_CENTER")
    color_side = get_param(df, "COLOR_SIDE")
    edge_color = get_param(df, "EDGE_COLOR")
    border_color = get_param(df, "BORDER_COLOR")

    grid_unscaled = to_gdf(df, "sector_grid_unscaled")
    grid_a = to_gdf(df, "sector_grid_panel_a_scaled")
    buildings_a = to_gdf(df, "building_panel_a_context")
    buildings_b = to_gdf(df, "building_panel_b_target")
    roads_a = to_gdf(df, "road_panel_a_context")
    roads_c = to_gdf(df, "road_panel_c_target")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 10), dpi=300)

    # ---------------------------------------------------
    # Panel a
    # ---------------------------------------------------
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title("Global 3×7 Upwind Sector-Grids", fontsize=24, fontweight="bold", pad=20)
    ax1.text(-0.05, 1.05, "a", transform=ax1.transAxes, fontsize=40, fontweight="bold", color="black")

    minx, miny, maxx, maxy = grid_a.total_bounds
    pad_a = 0.01
    w_span = maxx - minx
    h_span = maxy - miny

    if not roads_a.empty:
        max_em_a = pd.to_numeric(roads_a["emission_total"], errors="coerce").max()
        if pd.isna(max_em_a) or max_em_a <= 0:
            max_em_a = 1.0

        for _, r in roads_a.iterrows():
            em = pd.to_numeric(r["emission_total"], errors="coerce")
            if pd.isna(em):
                em = 0
            lw = 0.5 + (em / max_em_a) * 2.5
            geom = r.geometry
            if isinstance(geom, LineString):
                ax1.plot(*geom.xy, color="grey", lw=lw, solid_capstyle="round", alpha=0.7, zorder=0)
            elif hasattr(geom, "geoms"):
                for line in geom.geoms:
                    ax1.plot(*line.xy, color="grey", lw=lw, solid_capstyle="round", alpha=0.7, zorder=0)

    if not buildings_a.empty:
        buildings_a.plot(
            ax=ax1,
            facecolor="#F2F3F4",
            edgecolor="#D5D8DC",
            linewidth=0.5,
            alpha=0.8,
            zorder=1,
        )

    for _, row in grid_a.iterrows():
        c = color_center if abs(float(row["sector_center"]) - center_angle) < 1 else color_side
        ax1.fill(*row.geometry.exterior.xy, color=c, alpha=0.15, zorder=1.5)
        ax1.plot(*row.geometry.exterior.xy, color=c, lw=1.5, alpha=0.85, zorder=2)

    ax1.plot(origin_lon, origin_lat, marker="o", color="black", markersize=8, zorder=3)

    wind_rad_global = math.radians(center_angle_east)
    w_size = h_span * 0.1
    w_cx = minx + w_span * 0.05
    w_cy = miny + h_span * 0.05
    w_sx = w_cx + w_size * math.sin(wind_rad_global)
    w_sy = w_cy + w_size * math.cos(wind_rad_global)
    w_ex = w_cx - (w_size * 0.5) * math.sin(wind_rad_global)
    w_ey = w_cy - (w_size * 0.5) * math.cos(wind_rad_global)

    ax1.add_patch(
        FancyArrowPatch(
            (w_sx, w_sy),
            (w_ex, w_ey),
            arrowstyle="-|>",
            mutation_scale=30,
            edgecolor=edge_color,
            facecolor="#BDC3C7",
            lw=2,
            zorder=10,
        )
    )
    ax1.text(
        w_sx,
        w_sy + h_span * 0.015,
        "Wind Vector",
        fontsize=16,
        fontweight="bold",
        color=edge_color,
        ha="center",
        va="bottom",
        path_effects=[path_effects.withStroke(linewidth=3, foreground="white")],
    )

    n_cx = maxx + pad_a - w_span * 0.12
    n_cy = maxy + pad_a - h_span * 0.12
    n_size = h_span * 0.06

    ax1.fill([n_cx, n_cx - n_size * 0.25, n_cx], [n_cy - n_size, n_cy - n_size * 0.8, n_cy], color="black", zorder=10)
    ax1.fill([n_cx, n_cx + n_size * 0.25, n_cx], [n_cy - n_size, n_cy - n_size * 0.8, n_cy], color="#BDC3C7", zorder=10)
    ax1.plot([n_cx, n_cx], [n_cy - n_size, n_cy - n_size * 1.1], color="black", lw=2, zorder=10)
    ax1.text(n_cx, n_cy + n_size * 0.1, "N", fontsize=22, fontweight="bold", color="black", ha="center", va="bottom")

    ax1_xlim = (minx - 0.01, maxx + 0.01)
    ax1_ylim = (miny - 0.01, maxy + 0.01)
    ax1.set_xlim(*ax1_xlim)
    ax1.set_ylim(*ax1_ylim)

    aspect_ratio_A = (ax1_ylim[1] - ax1_ylim[0]) / (ax1_xlim[1] - ax1_xlim[0])

    # ---------------------------------------------------
    # Panel b
    # ---------------------------------------------------
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title(f"Canopy Projection in {start_radius}-{end_radius}m Sector", fontsize=24, fontweight="bold", pad=20)
    ax2.text(-0.05, 1.05, "b", transform=ax2.transAxes, fontsize=40, fontweight="bold", color="black")

    target_row = grid_unscaled[
        (grid_unscaled["r_min"].astype(float) == start_radius)
        & (grid_unscaled["r_max"].astype(float) == end_radius)
        & (abs(grid_unscaled["sector_center"].astype(float) - center_angle) < 1)
    ]

    if target_row.empty:
        raise ValueError("Target sector grid not found in source data.")

    target_poly = target_row.geometry.iloc[0]
    target_bounds = target_poly.bounds

    ax2.plot(*target_poly.exterior.xy, color=color_center, lw=2.5, zorder=5)

    wind_azi = math.radians(center_angle_east)
    proj_azi = math.radians(center_angle_east + 90)
    w_dx, w_dy = math.sin(wind_azi), math.cos(wind_azi)
    p_dx, p_dy = math.sin(proj_azi), math.cos(proj_azi)

    for _, b in buildings_b.iterrows():
        b_geom = b.geometry

        if target_poly.intersects(b_geom):
            intersect_geom = target_poly.intersection(b_geom)
            polys = []

            if isinstance(intersect_geom, Polygon):
                ax2.fill(*intersect_geom.exterior.xy, color="#3498DB", alpha=0.8, edgecolor="black", lw=1)
                polys.append(intersect_geom)
            elif isinstance(intersect_geom, MultiPolygon):
                for p in intersect_geom.geoms:
                    ax2.fill(*p.exterior.xy, color="#3498DB", alpha=0.8, edgecolor="black", lw=1)
                    polys.append(p)

            diff_geom = b_geom.difference(target_poly)
            draw_polygon_fill(ax2, diff_geom, facecolor="#D5D8DC", edgecolor="gray", alpha=0.4, lw=1, linestyle="--")

            for p in polys:
                pts = list(p.exterior.coords)
                proj_vals = [pt[0] * p_dx + pt[1] * p_dy for pt in pts]
                wind_vals = [pt[0] * w_dx + pt[1] * w_dy for pt in pts]

                min_p_idx, max_p_idx = np.argmin(proj_vals), np.argmax(proj_vals)
                min_p, max_p = proj_vals[min_p_idx], proj_vals[max_p_idx]
                min_w = min(wind_vals)
                ruler_w = min_w - 0.0001

                start_pt = (ruler_w * w_dx + min_p * p_dx, ruler_w * w_dy + min_p * p_dy)
                end_pt = (ruler_w * w_dx + max_p * p_dx, ruler_w * w_dy + max_p * p_dy)

                ax2.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], color="#E67E22", ls="--", lw=2, zorder=6)

                pt_min, pt_max = pts[min_p_idx], pts[max_p_idx]
                ax2.plot([pt_min[0], start_pt[0]], [pt_min[1], start_pt[1]], color="#E67E22", ls=":", lw=1.5, alpha=0.8, zorder=6)
                ax2.plot([pt_max[0], end_pt[0]], [pt_max[1], end_pt[1]], color="#E67E22", ls=":", lw=1.5, alpha=0.8, zorder=6)

                dx_deg = start_pt[0] - end_pt[0]
                dy_deg = start_pt[1] - end_pt[1]
                dist_m = math.sqrt((dx_deg * 111320 * math.cos(math.radians(origin_lat))) ** 2 + (dy_deg * 111320) ** 2)

                mid_x, mid_y = (start_pt[0] + end_pt[0]) / 2, (start_pt[1] + end_pt[1]) / 2

                ax2.text(
                    mid_x - 0.00005 * w_dx,
                    mid_y - 0.00005 * w_dy,
                    f"{dist_m:.1f}",
                    color="#E67E22",
                    fontsize=14,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    path_effects=[path_effects.withStroke(linewidth=2, foreground="white")],
                    zorder=7,
                )

    cx, cy = target_poly.centroid.xy
    wind_rad = math.radians(center_angle_east)

    arrow_start = (cx[0] + 0.0008 * math.sin(wind_rad), cy[0] + 0.0008 * math.cos(wind_rad))
    arrow_end = (cx[0] - 0.0010 * math.sin(wind_rad), cy[0] - 0.0010 * math.cos(wind_rad))

    ax2.add_patch(
        FancyArrowPatch(
            arrow_start,
            arrow_end,
            arrowstyle="-|>",
            mutation_scale=20,
            edgecolor=edge_color,
            facecolor="#BDC3C7",
            linewidth=1.5,
            zorder=10,
        )
    )

    ax2.text(
        arrow_start[0] - 0.00012,
        arrow_start[1] + 0.0001,
        "Wind Vector",
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="bottom",
        color=edge_color,
        path_effects=[path_effects.withStroke(linewidth=3, foreground="white")],
    )

    ax2.add_patch(Rectangle((0, 0), 1, 1, transform=ax2.transAxes, linewidth=1, edgecolor=border_color, facecolor="none", zorder=100))

    bx_min, by_min, bx_max, by_max = target_bounds
    pad_zoom = 0.00025
    target_x_span = (bx_max + pad_zoom) - (bx_min - pad_zoom)
    required_y_span = target_x_span * aspect_ratio_A
    center_y = (by_min + by_max) / 2
    ax2_ylim = (center_y - required_y_span / 2, center_y + required_y_span / 2)

    ax2.set_xlim(bx_min - pad_zoom, bx_max + pad_zoom)
    ax2.set_ylim(*ax2_ylim)

    ax2.text(
        0.5,
        0.05,
        "* Note: The projected canopy widths listed above are in meters.",
        transform=ax2.transAxes,
        fontsize=16,
        color=edge_color,
        ha="center",
        va="top",
        fontweight="normal",
    )

    # ---------------------------------------------------
    # Panel c
    # ---------------------------------------------------
    ax3.set_aspect("equal")
    ax3.axis("off")
    ax3.set_title(f"Network Aggregation in {start_radius}-{end_radius}m Sector", fontsize=24, fontweight="bold", pad=20)
    ax3.text(-0.05, 1.05, "c", transform=ax3.transAxes, fontsize=40, fontweight="bold", color="black")

    ax3.fill(*target_poly.exterior.xy, color=color_center, alpha=0.05)
    ax3.plot(*target_poly.exterior.xy, color=color_center, lw=2.5, zorder=5)

    max_em = pd.to_numeric(roads_c["emission_total"], errors="coerce").max() if not roads_c.empty else 1
    if pd.isna(max_em) or max_em <= 0:
        max_em = 1.0

    intersected_lines_info = []
    avg_meters_per_degree = math.sqrt(111320 * (111320 * math.cos(math.radians(origin_lat))))

    for _, r in roads_c.iterrows():
        r_geom = r.geometry
        em = pd.to_numeric(r["emission_total"], errors="coerce")
        if pd.isna(em):
            em = 0

        if isinstance(r_geom, LineString):
            ax3.plot(*r_geom.xy, color="#BDC3C7", lw=1, ls="--", alpha=0.6, zorder=1)
        elif hasattr(r_geom, "geoms"):
            for line in r_geom.geoms:
                ax3.plot(*line.xy, color="#BDC3C7", lw=1, ls="--", alpha=0.6, zorder=1)

        if target_poly.intersects(r_geom):
            intersect_r = target_poly.intersection(r_geom)
            lw = 1 + (em / max_em) * 5

            if isinstance(intersect_r, LineString) and not intersect_r.is_empty:
                ax3.plot(*intersect_r.xy, color="#C0392B", lw=lw, solid_capstyle="round", zorder=4)
                start_lat = intersect_r.coords[0][1]
                intersected_lines_info.append((intersect_r, em, start_lat, lw))

            elif hasattr(intersect_r, "geoms"):
                for line in intersect_r.geoms:
                    if not line.is_empty and isinstance(line, LineString):
                        ax3.plot(*line.xy, color="#C0392B", lw=lw, solid_capstyle="round", zorder=4)
                        start_lat = line.coords[0][1]
                        intersected_lines_info.append((line, em, start_lat, lw))

    ax3.text(
        cx[0],
        cy[0] + 0.0002,
        "Vector Intersection\n& Length Weighted",
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="center",
        color="#C0392B",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=4),
        zorder=12,
    )

    if intersected_lines_info:
        intersected_lines_info.sort(key=lambda x: x[2])
        target_idx = 2 if len(intersected_lines_info) >= 3 else -1
        best_line, em_total, _, original_lw = intersected_lines_info[target_idx]

        ax3.plot(*best_line.xy, color="#27AE60", lw=original_lw + 3.5, solid_capstyle="round", zorder=6)

        len_m = best_line.length * avg_meters_per_degree
        mid_pt = best_line.interpolate(0.5, normalized=True)

        anno_text = f"Length: {len_m:.1f} m\nEmission Factor: {em_total:.1f} g/km"

        ax3.annotate(
            anno_text,
            xy=(mid_pt.x, mid_pt.y),
            xytext=(mid_pt.x - 0.00025, mid_pt.y + 0.00015),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color=edge_color, lw=1.5),
            fontsize=16,
            fontweight="bold",
            color=edge_color,
            ha="center",
            va="center",
            path_effects=[path_effects.withStroke(linewidth=3, foreground="white")],
            zorder=10,
        )

    ax3.add_patch(Rectangle((0, 0), 1, 1, transform=ax3.transAxes, linewidth=1, edgecolor=border_color, facecolor="none", zorder=100))
    ax3.set_xlim(bx_min - pad_zoom, bx_max + pad_zoom)
    ax3.set_ylim(*ax2_ylim)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Supplementary Fig. 2 reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
