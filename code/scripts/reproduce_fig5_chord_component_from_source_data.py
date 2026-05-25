import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# Reproduce Fig. 5 chord component from public source data
# Input:
#   data/source_data/main/Fig5_chord_source_data.csv
# Output:
#   figures/main/Fig5_chord_reproduced_from_source_data.png
#   figures/main/Fig5_chord_reproduced_from_source_data.pdf
# ============================================================


try:
    from pycirclize import Circos
except ImportError as exc:
    raise ImportError(
        "pycirclize is required to reproduce Fig. 5 chord component. "
        "Install it with: pip install pycirclize"
    ) from exc


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["text.color"] = "black"
plt.rcParams["font.size"] = 18


ABBR_DICT = {
    "Meteorological": "Met.",
    "Temporal": "Temp.",
    "Emission Source": "Emiss.",
    "Road Topology": "Topo.",
    "POI": "POI",
    "Urban Canopy": "UCP",
    "Landuse": "Landuse",
}


def find_repo_root():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(this_dir, "..", ".."))


REPO_ROOT = find_repo_root()

SOURCE_DATA_PATH = os.path.join(
    REPO_ROOT,
    "data",
    "source_data",
    "main",
    "Fig5_chord_source_data.csv"
)

OUTPUT_DIR = os.path.join(
    REPO_ROOT,
    "figures",
    "main"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Fig5_chord_reproduced_from_source_data.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Fig5_chord_reproduced_from_source_data.pdf")


def draw_chord_on_ax(ax, df, node_colors, global_cats, panel_letter, title_text):
    local_cats = sorted(list(set(df["Source"]) | set(df["Target"])))

    node_total_val = {}
    for cat in local_cats:
        total = df[(df["Source"] == cat) | (df["Target"] == cat)]["Value"].sum()
        node_total_val[cat] = float(total)

    sectors = {
        cat: node_total_val[cat]
        for cat in global_cats
        if cat in local_cats and node_total_val[cat] > 0
    }

    if not sectors:
        ax.text(
            0.5,
            0.5,
            "No interaction edges",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="black",
        )
        ax.axis("off")
        return

    circos = Circos(sectors, space=7)

    for sector in circos.sectors:
        track = sector.add_track((95, 100))
        track.axis(fc=node_colors[sector.name], ec="white", linewidth=0.5)

        display_label = sector.name.replace("Parameters", "").strip()
        track.text(
            display_label,
            r=110,
            size=18,
            weight="bold",
            color="black",
            orientation="horizontal",
            ha="center",
            va="center",
        )

    max_val = df["Value"].max()
    if max_val <= 0:
        max_val = 1.0

    current_pos = {cat: 0.0 for cat in local_cats}

    for _, row in df.iterrows():
        source = row["Source"]
        target = row["Target"]
        weight = float(row["Value"])

        if source not in sectors or target not in sectors:
            continue

        start_s = current_pos[source]
        end_s = start_s + weight
        current_pos[source] = end_s

        start_t = current_pos[target]
        end_t = start_t + weight
        current_pos[target] = end_t

        base_color = node_colors[source]
        alpha = 0.3 + 0.4 * (weight / max_val)

        circos.link(
            (source, start_s, end_s),
            (target, start_t, end_t),
            color=(*base_color[:3], alpha),
            ec="white",
            lw=0.5,
        )

    circos.plotfig(ax=ax)

    ax.set_rlim(0, 140)
    ax.axis("off")

    ax.text(
        0.02,
        1.0,
        panel_letter,
        transform=ax.transAxes,
        fontsize=40,
        fontweight="bold",
        va="bottom",
        color="black",
    )

    ax.text(
        0.5,
        0.98,
        title_text,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        ha="center",
        va="bottom",
        color="black",
    )

    top5_df = df.sort_values(by="Value", ascending=False).head(5)

    start_x = 0.96
    start_y = 0.72

    ax.text(
        start_x,
        start_y,
        "Top 5\nSynergistic Pairs:",
        fontsize=22,
        fontweight="bold",
        transform=ax.transAxes,
        va="bottom",
        color="black",
    )

    for i, row in enumerate(top5_df.itertuples()):
        s_name = row.Source.replace("Parameters", "").strip()
        t_name = row.Target.replace("Parameters", "").strip()

        s_abbr = ABBR_DICT.get(s_name, s_name)
        t_abbr = ABBR_DICT.get(t_name, t_name)

        s_color = node_colors[row.Source]
        t_color = node_colors[row.Target]

        y_pos = start_y - 0.08 - (i * 0.09)

        ax.plot(
            start_x,
            y_pos,
            marker="s",
            color=s_color,
            markersize=16,
            transform=ax.transAxes,
            clip_on=False,
        )

        ax.plot(
            start_x + 0.022,
            y_pos,
            marker="s",
            color=t_color,
            markersize=16,
            transform=ax.transAxes,
            clip_on=False,
        )

        text_str = f"{s_abbr} ↔ {t_abbr}"

        ax.text(
            start_x + 0.05,
            y_pos,
            text_str,
            fontsize=18,
            va="center",
            ha="left",
            transform=ax.transAxes,
            color="black",
        )


def main():
    if not os.path.exists(SOURCE_DATA_PATH):
        raise FileNotFoundError(SOURCE_DATA_PATH)

    edges_df_all = pd.read_csv(SOURCE_DATA_PATH, encoding="utf-8-sig")

    required_cols = ["Source", "Target", "Value"]
    for col in required_cols:
        if col not in edges_df_all.columns:
            raise ValueError(f"Missing required column in Fig5_chord_source_data.csv: {col}")

    edges_df_all = edges_df_all[required_cols].copy()
    edges_df_all["Source"] = edges_df_all["Source"].astype(str)
    edges_df_all["Target"] = edges_df_all["Target"].astype(str)
    edges_df_all["Value"] = pd.to_numeric(edges_df_all["Value"], errors="coerce")
    edges_df_all = edges_df_all.dropna(subset=["Source", "Target", "Value"])
    edges_df_all = edges_df_all[edges_df_all["Value"] > 0].copy()

    edges_df_excl = edges_df_all[
        ~edges_df_all["Source"].str.contains("Meteorological", case=False, na=False)
        & ~edges_df_all["Target"].str.contains("Meteorological", case=False, na=False)
    ].copy()

    global_cats = sorted(list(set(edges_df_all["Source"]) | set(edges_df_all["Target"])))

    global_node_total = {}
    for cat in global_cats:
        global_node_total[cat] = edges_df_all[
            (edges_df_all["Source"] == cat) | (edges_df_all["Target"] == cat)
        ]["Value"].sum()

    global_cats_sorted = sorted(global_cats, key=lambda k: global_node_total[k], reverse=True)

    cmap = plt.cm.Spectral
    num_cats = len(global_cats_sorted)

    if num_cats <= 1:
        node_colors = {cat: cmap(0.5) for cat in global_cats_sorted}
    else:
        node_colors = {
            cat: cmap(i / (num_cats - 1))
            for i, cat in enumerate(global_cats_sorted)
        }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(25, 11),
        subplot_kw=dict(polar=True),
    )

    fig.subplots_adjust(
        wspace=0.30,
        left=0.03,
        right=0.88,
        top=0.90,
        bottom=0.0,
    )

    draw_chord_on_ax(
        axes[0],
        edges_df_all,
        node_colors,
        global_cats_sorted,
        panel_letter="a",
        title_text="Overall Synergistic Network",
    )

    draw_chord_on_ax(
        axes[1],
        edges_df_excl,
        node_colors,
        global_cats_sorted,
        panel_letter="b",
        title_text="Network Excluding Meteorological Impacts",
    )

    plt.savefig(OUTPUT_PNG, dpi=400, bbox_inches="tight")
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print("Fig. 5 chord component reproduced from source data.")
    print(f"PNG: {OUTPUT_PNG}")
    print(f"PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
