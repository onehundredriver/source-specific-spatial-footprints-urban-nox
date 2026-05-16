# -*- coding: utf-8 -*-

"""
Check release package integrity.

Run from repository root:
python code/check_release_package_integrity.py
"""

from pathlib import Path
import csv
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]

MAIN_FIGS = [f"Fig{i}" for i in range(1, 10)]
SUPP_FIGS = [f"Supplementary_Fig{i}" for i in range(1, 10)]

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

REQUIRED_DOCS = [
    "README.md",
    "REPRODUCIBILITY_GUIDE.md",
]


def file_status(rel_path: str) -> dict:
    p = ROOT / rel_path
    return {
        "relative_path": rel_path,
        "exists": p.exists(),
        "size_mb": round(p.stat().st_size / 1024 / 1024, 6) if p.exists() and p.is_file() else "",
    }


def collect_expected_files() -> list[dict]:
    rows = []

    # 主图 Fig.1 到 Fig.9
    for fig in MAIN_FIGS:
        rows.append(file_status(f"data/source_data/main/{fig}_source_data.csv"))
        rows.append(file_status(f"code/scripts/reproduce_{fig.lower()}_from_source_data.py"))
        rows.append(file_status(f"figures/main/{fig}_reproduced_from_source_data.png"))
        rows.append(file_status(f"figures/main/{fig}_reproduced_from_source_data.pdf"))

    # Fig.3 当前还有一个交通网络 source data 文件
    rows.append(file_status("data/source_data/main/Fig3_traffic_network.csv"))

    # 补充图 Supplementary Fig.1 到 Supplementary Fig.9
    for fig in SUPP_FIGS:
        csv_path = f"data/source_data/supplementary/{fig}_source_data.csv"
        xlsx_path = f"data/source_data/supplementary/{fig}_source_data.xlsx"

        if (ROOT / csv_path).exists():
            rows.append(file_status(csv_path))
        elif (ROOT / xlsx_path).exists():
            rows.append(file_status(xlsx_path))
        else:
            rows.append(file_status(csv_path))

        script_name = f"reproduce_{fig.lower()}_from_source_data.py"
        rows.append(file_status(f"code/scripts/{script_name}"))
        rows.append(file_status(f"figures/supplementary/{fig}_reproduced_from_source_data.png"))
        rows.append(file_status(f"figures/supplementary/{fig}_reproduced_from_source_data.pdf"))

    # public NOX 数据
    for rel in REQUIRED_PUBLIC_FILES:
        rows.append(file_status(rel))

    # 模型划分文件
    for rel in REQUIRED_SPLIT_FILES:
        rows.append(file_status(rel))

    # Supplementary Data 1
    rows.append(file_status("data/supplementary_data/Supplementary_Data_1.xlsx"))

    # 文档文件
    for rel in REQUIRED_DOCS:
        rows.append(file_status(rel))

    return rows


def write_report(rows: list[dict]) -> None:
    out_dir = ROOT / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "release_integrity_check.csv"
    md_path = out_dir / "release_integrity_check.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "exists", "size_mb"])
        writer.writeheader()
        writer.writerows(rows)

    missing = [r for r in rows if not r["exists"]]

    md_lines = [
        "# Release package integrity check",
        "",
        f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Total checked items: {len(rows)}",
        f"Missing items: {len(missing)}",
        "",
    ]

    if missing:
        md_lines.append("## Missing files")
        md_lines.append("")
        for r in missing:
            md_lines.append(f"- `{r['relative_path']}`")
    else:
        md_lines.append("No missing required files were detected.")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[DONE] CSV report: {csv_path}")
    print(f"[DONE] Markdown report: {md_path}")

    if missing:
        print("\n[WARNING] Missing files detected:")
        for r in missing:
            print(f"  - {r['relative_path']}")
    else:
        print("\n[DONE] No missing required files detected.")


def main() -> None:
    rows = collect_expected_files()
    write_report(rows)


if __name__ == "__main__":
    main()
