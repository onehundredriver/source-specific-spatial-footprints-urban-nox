# -*- coding: utf-8 -*-
"""
Check release package integrity for the revised Nature Cities figure set.

Run from repository root:
    python code/check_release_package_integrity.py

Expected revised figure structure:
- Main figures: Fig. 1–5
- Extended Data figures: Extended Data Fig. 1–6
- Supplementary figures: Supplementary Fig. 1–8

This checker reports:
1. missing required release files;
2. deprecated old-figure residues that should not remain in the revised release;
3. temporary local audit/renaming files that should normally not be committed;
4. large files close to GitHub's single-file size limit.
"""

from __future__ import annotations

from pathlib import Path
import csv
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Core required non-figure release files
# ---------------------------------------------------------------------

REQUIRED_PUBLIC_FILES = [
    "data/public/national_control_station_metadata.csv",
    "data/public/processed_national_control_NOx_hourly.csv",
    "data/public/NOx_QC_overall_summary.csv",
    "data/public/NOx_QC_summary_by_station.csv",
]

REQUIRED_SPLIT_FILES = [
    "data/splits/spatial_5fold_station_assignment.csv",
    "data/splits/temporal_5fold_window_assignment.csv",
    "data/splits/temporal_buffer_intervals.csv",
    "data/splits/temporal_split_metadata.csv",
    "data/splits/anomalous_station_exclusion_list.csv",
    "data/splits/final_cluster_station_labels.csv",
]

REQUIRED_SUPPLEMENTARY_DATA_FILES = [
    "data/supplementary_data/Supplementary_Data_1.xlsx",
    "data/supplementary_data/Supplementary_Data_2.xlsx",
]

REQUIRED_DOCS = [
    "README.md",
    "REPRODUCIBILITY_GUIDE.md",
    "release_manifest_public.xlsx",
]


# ---------------------------------------------------------------------
# Revised figure-set expectations
# ---------------------------------------------------------------------

MAIN_FIGS = [f"Fig{i}" for i in range(1, 6)]
EXTENDED_DATA_FIGS = [f"Extended_Data_Fig{i}" for i in range(1, 7)]
SUPP_FIGS = [f"Supplementary_Fig{i}" for i in range(1, 9)]


def file_status(rel_path: str, group: str, required: bool = True) -> dict:
    p = ROOT / rel_path
    exists = p.exists()
    return {
        "group": group,
        "relative_path": rel_path,
        "required": required,
        "exists": exists,
        "size_mb": round(p.stat().st_size / 1024 / 1024, 6) if exists and p.is_file() else "",
    }


def add_flexible_source(rows: list[dict], base_without_ext: str, group: str) -> None:
    """
    Add one source-data check accepting either CSV or XLSX.

    Example:
        base_without_ext = "data/source_data/supplementary/Supplementary_Fig1_source_data"
    """
    csv_path = f"{base_without_ext}.csv"
    xlsx_path = f"{base_without_ext}.xlsx"

    if (ROOT / csv_path).exists():
        rows.append(file_status(csv_path, group))
    elif (ROOT / xlsx_path).exists():
        rows.append(file_status(xlsx_path, group))
    else:
        # Default to CSV in the missing report, while note can be inferred from group.
        rows.append(file_status(csv_path, group))


def collect_required_files() -> list[dict]:
    rows: list[dict] = []

    # Main Fig. 1: source data are stored as a directory of compact map/schematic files.
    fig1_sources = [
        "data/source_data/main/Fig1/monitoring_sites.csv",
        "data/source_data/main/Fig1/road_network_traffic_flow_proxy.csv",
        "data/source_data/main/Fig1/fig1_schematic_parameters.csv",
        "data/source_data/main/Fig1/fig1_workflow_nodes.csv",
        "data/source_data/main/Fig1/fig1_workflow_edges.csv",
    ]
    for rel in fig1_sources:
        rows.append(file_status(rel, "main_figure_source_data"))

    rows.append(file_status("code/scripts/reproduce_fig1_from_source_data.py", "main_figure_script"))
    rows.append(file_status("figures/main/Fig1_Original.png", "main_figure_output"))
    rows.append(file_status("figures/main/Fig1_reproduced_from_source_data.png", "main_figure_output"))
    rows.append(file_status("figures/main/Fig1_reproduced_from_source_data.pdf", "main_figure_output"))

    # Main Figs. 2–4: one compact CSV source data file each.
    for fig in ["Fig2", "Fig3", "Fig4"]:
        rows.append(file_status(f"data/source_data/main/{fig}_source_data.csv", "main_figure_source_data"))
        rows.append(file_status(f"code/scripts/reproduce_{fig.lower()}_from_source_data.py", "main_figure_script"))
        rows.append(file_status(f"figures/main/{fig}_Original.png", "main_figure_output"))
        rows.append(file_status(f"figures/main/{fig}_reproduced_from_source_data.png", "main_figure_output"))
        rows.append(file_status(f"figures/main/{fig}_reproduced_from_source_data.pdf", "main_figure_output"))

    # Fig. 4 has an additional road-network source-data file.
    rows.append(file_status("data/source_data/main/Fig4_traffic_network.csv", "main_figure_source_data"))

    # Main Fig. 5: complete composite figure source data are stored as a directory.
    fig5_sources = [
        "data/source_data/main/Fig5/category_interaction_edges.csv",
        "data/source_data/main/Fig5/fig5_interaction_points.csv",
        "data/source_data/main/Fig5/fig5_panel_metadata.csv",
        "data/source_data/main/Fig5/fig5_sampling_metadata.csv",
    ]
    for rel in fig5_sources:
        rows.append(file_status(rel, "main_figure_source_data"))

    rows.append(file_status("code/scripts/reproduce_fig5_from_source_data.py", "main_figure_script"))
    rows.append(file_status("figures/main/Fig5_Original.png", "main_figure_output"))
    rows.append(file_status("figures/main/Fig5_reproduced_from_source_data.png", "main_figure_output"))
    rows.append(file_status("figures/main/Fig5_reproduced_from_source_data.pdf", "main_figure_output"))

    # Optional Fig. 5 chord component provenance. Mark as not required.
    optional_fig5_chord = [
        "data/source_data/main/Fig5_chord_source_data.csv",
        "code/scripts/reproduce_fig5_chord_component_from_source_data.py",
        "figures/main/Fig5_chord_Original.png",
        "figures/main/Fig5_chord_reproduced_from_source_data.png",
        "figures/main/Fig5_chord_reproduced_from_source_data.pdf",
    ]
    for rel in optional_fig5_chord:
        rows.append(file_status(rel, "optional_fig5_component", required=False))

    # Extended Data figures.
    for i in range(1, 7):
        fig = f"Extended_Data_Fig{i}"
        source_ext = "xlsx" if i == 1 else "csv"
        rows.append(file_status(f"data/source_data/extended_data/{fig}_source_data.{source_ext}", "extended_data_source_data"))
        rows.append(file_status(f"code/scripts/reproduce_extended_data_fig{i}_from_source_data.py", "extended_data_script"))
        rows.append(file_status(f"figures/extended_data/{fig}_Original.png", "extended_data_output"))
        rows.append(file_status(f"figures/extended_data/{fig}_reproduced_from_source_data.png", "extended_data_output"))
        rows.append(file_status(f"figures/extended_data/{fig}_reproduced_from_source_data.pdf", "extended_data_output"))

    # Supplementary figures 1–8.
    for i in range(1, 9):
        fig = f"Supplementary_Fig{i}"
        add_flexible_source(rows, f"data/source_data/supplementary/{fig}_source_data", "supplementary_figure_source_data")
        rows.append(file_status(f"code/scripts/reproduce_supplementary_fig{i}_from_source_data.py", "supplementary_figure_script"))
        rows.append(file_status(f"figures/supplementary/{fig}_reproduced_from_source_data.png", "supplementary_figure_output"))
        rows.append(file_status(f"figures/supplementary/{fig}_reproduced_from_source_data.pdf", "supplementary_figure_output"))

    # Public data, split files, supplementary data and documentation.
    for rel in REQUIRED_PUBLIC_FILES:
        rows.append(file_status(rel, "public_data"))

    for rel in REQUIRED_SPLIT_FILES:
        rows.append(file_status(rel, "split_files"))

    for rel in REQUIRED_SUPPLEMENTARY_DATA_FILES:
        rows.append(file_status(rel, "supplementary_data"))

    for rel in REQUIRED_DOCS:
        rows.append(file_status(rel, "documentation"))

    return rows


def collect_deprecated_residues() -> list[dict]:
    """
    Files that should not remain as current release items after the revised
    Nature Cities figure reorganisation.
    """
    residues: list[dict] = []

    # Old main Fig. 6–9 items.
    for i in range(6, 10):
        fig = f"Fig{i}"
        candidates = [
            f"data/source_data/main/{fig}_source_data.csv",
            f"code/scripts/reproduce_fig{i}_from_source_data.py",
            f"figures/main/{fig}_Original.png",
            f"figures/main/{fig}_reproduced_from_source_data.png",
            f"figures/main/{fig}_reproduced_from_source_data.pdf",
        ]
        for rel in candidates:
            residues.append(file_status(rel, "deprecated_old_main_figures", required=False))

    # Old Supplementary Fig. 9 items.
    old_supp9 = [
        "data/source_data/supplementary/Supplementary_Fig9_source_data.csv",
        "data/source_data/supplementary/Supplementary_Fig9_source_data.xlsx",
        "code/scripts/reproduce_supplementary_fig9_from_source_data.py",
        "figures/supplementary/Supplementary_Fig9_Original.png",
        "figures/supplementary/Supplementary_Fig9_reproduced_from_source_data.png",
        "figures/supplementary/Supplementary_Fig9_reproduced_from_source_data.pdf",
    ]
    for rel in old_supp9:
        residues.append(file_status(rel, "deprecated_old_supplementary_fig9", required=False))

    return residues


def collect_temporary_local_files() -> list[dict]:
    """
    Local audit/renaming helper files created during the figure-system update.
    These are useful locally but should normally not be committed to the public
    release unless intentionally retained under docs/audit.
    """
    candidates = [
        "_figure_rename_staging_052314",
        "figure_rename_mapping_052314.csv",
        "figure_rename_mapping_check_052314.csv",
        "local_file_inventory_before_figure_update.csv",
        "inventory_source_data_before_figure_update.csv",
        "inventory_figures_before_figure_update.csv",
        "inventory_figure_scripts_before_update.csv",
        "old_figure_label_residue_before_update.csv",
        "run_all_figure_reproductions_current.txt",
        "check_release_package_integrity_current.txt",
        "code/scripts/local_safe_rename_figures_052314.py",
        "code/scripts/local_update_paths_from_mapping_052314.py",
        "code/scripts/local_update_paths_targeted_052314.py",
        "code/scripts/local_fix_extended_data_dirs_052314.py",
        "code/scripts/local_fix_extended_data_fig1_dirs_052314.py",
    ]
    return [file_status(rel, "temporary_local_update_artifacts", required=False) for rel in candidates]


def scan_large_files(threshold_mb: float = 95.0) -> list[dict]:
    rows: list[dict] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if ".git" in p.parts:
            continue

        rel = p.relative_to(ROOT).as_posix()

        # Large model-input matrices are intentionally hosted externally
        # and ignored by Git; do not report local ignored copies as release files.
        if rel.startswith("data/model_inputs/") and rel.endswith(".parquet"):
            continue

        size_mb = p.stat().st_size / 1024 / 1024
        if size_mb >= threshold_mb:
            rows.append({
                "relative_path": rel,
                "size_mb": round(size_mb, 6),
            })
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(required_rows: list[dict], residue_rows: list[dict], temp_rows: list[dict], large_files: list[dict]) -> None:
    out_dir = ROOT / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)

    required_csv = out_dir / "release_integrity_check_required_files.csv"
    residues_csv = out_dir / "release_integrity_check_deprecated_residues.csv"
    temp_csv = out_dir / "release_integrity_check_temporary_files.csv"
    large_csv = out_dir / "release_integrity_check_large_files.csv"
    md_path = out_dir / "release_integrity_check.md"

    write_csv(
        required_csv,
        required_rows,
        ["group", "relative_path", "required", "exists", "size_mb"],
    )

    write_csv(
        residues_csv,
        residue_rows,
        ["group", "relative_path", "required", "exists", "size_mb"],
    )

    write_csv(
        temp_csv,
        temp_rows,
        ["group", "relative_path", "required", "exists", "size_mb"],
    )

    write_csv(
        large_csv,
        large_files,
        ["relative_path", "size_mb"],
    )

    missing_required = [r for r in required_rows if r["required"] and not r["exists"]]
    deprecated_present = [r for r in residue_rows if r["exists"]]
    temp_present = [r for r in temp_rows if r["exists"]]

    md_lines = [
        "# Release package integrity check",
        "",
        f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Revised figure-set expectation",
        "",
        "- Main figures: Fig. 1–5",
        "- Extended Data figures: Extended Data Fig. 1–6",
        "- Supplementary figures: Supplementary Fig. 1–8",
        "",
        "## Summary",
        "",
        f"- Required files checked: {len(required_rows)}",
        f"- Missing required files: {len(missing_required)}",
        f"- Deprecated residues checked: {len(residue_rows)}",
        f"- Deprecated residues still present: {len(deprecated_present)}",
        f"- Temporary local update artifacts present: {len(temp_present)}",
        f"- Large files >=95 MB: {len(large_files)}",
        "",
    ]

    if missing_required:
        md_lines.extend(["## Missing required files", ""])
        for r in missing_required:
            md_lines.append(f"- `{r['relative_path']}`")
        md_lines.append("")
    else:
        md_lines.extend(["## Missing required files", "", "No missing required files were detected.", ""])

    if deprecated_present:
        md_lines.extend(["## Deprecated old-figure residues still present", ""])
        for r in deprecated_present:
            md_lines.append(f"- `{r['relative_path']}`")
        md_lines.append("")
    else:
        md_lines.extend(["## Deprecated old-figure residues", "", "No deprecated old-figure residues were detected.", ""])

    if temp_present:
        md_lines.extend([
            "## Temporary local update artifacts",
            "",
            "These files/directories are useful for local audit or recovery but should normally not be committed to the public release unless intentionally retained under documentation.",
            "",
        ])
        for r in temp_present:
            md_lines.append(f"- `{r['relative_path']}`")
        md_lines.append("")
    else:
        md_lines.extend(["## Temporary local update artifacts", "", "No temporary local update artifacts were detected.", ""])

    if large_files:
        md_lines.extend(["## Large files close to GitHub size limit", ""])
        for r in large_files:
            md_lines.append(f"- `{r['relative_path']}` ({r['size_mb']} MB)")
        md_lines.append("")
    else:
        md_lines.extend(["## Large files close to GitHub size limit", "", "No files >=95 MB were detected.", ""])

    md_lines.extend([
        "## Output files",
        "",
        f"- `{required_csv.relative_to(ROOT).as_posix()}`",
        f"- `{residues_csv.relative_to(ROOT).as_posix()}`",
        f"- `{temp_csv.relative_to(ROOT).as_posix()}`",
        f"- `{large_csv.relative_to(ROOT).as_posix()}`",
        "",
    ])

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[DONE] Required-file CSV: {required_csv}")
    print(f"[DONE] Deprecated-residue CSV: {residues_csv}")
    print(f"[DONE] Temporary-file CSV: {temp_csv}")
    print(f"[DONE] Large-file CSV: {large_csv}")
    print(f"[DONE] Markdown report: {md_path}")

    if missing_required:
        print("\n[WARNING] Missing required files:")
        for r in missing_required:
            print(f"  - {r['relative_path']}")
    else:
        print("\n[DONE] No missing required files detected.")

    if deprecated_present:
        print("\n[WARNING] Deprecated old-figure residues still present:")
        for r in deprecated_present:
            print(f"  - {r['relative_path']}")
    else:
        print("[DONE] No deprecated old-figure residues detected.")

    if temp_present:
        print("\n[NOTE] Temporary local update artifacts are present:")
        for r in temp_present:
            print(f"  - {r['relative_path']}")
        print("These should usually be removed or moved to documentation before committing.")

    if large_files:
        print("\n[WARNING] Files >=95 MB detected:")
        for r in large_files:
            print(f"  - {r['relative_path']} ({r['size_mb']} MB)")


def main() -> None:
    required_rows = collect_required_files()
    residue_rows = collect_deprecated_residues()
    temp_rows = collect_temporary_local_files()
    large_files = scan_large_files(threshold_mb=95.0)
    write_report(required_rows, residue_rows, temp_rows, large_files)


if __name__ == "__main__":
    main()
