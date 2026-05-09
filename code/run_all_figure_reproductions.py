# -*- coding: utf-8 -*-

"""
Run all figure reproduction scripts.

Run from repository root:
python code/run_all_figure_reproductions.py
"""

from pathlib import Path
import subprocess
import sys
from datetime import datetime
import csv


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "code" / "scripts"
LOG_DIR = ROOT / "docs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

ORDERED_SCRIPTS = [
    "reproduce_fig1_from_source_data.py",
    "reproduce_fig2_from_source_data.py",
    "reproduce_fig3_from_source_data.py",
    "reproduce_fig4_from_source_data.py",
    "reproduce_fig5_from_source_data.py",
    "reproduce_fig6_from_source_data.py",
    "reproduce_fig7_from_source_data.py",
    "reproduce_fig8_from_source_data.py",
    "reproduce_fig9_from_source_data.py",
    "reproduce_fig10_from_source_data.py",
    "reproduce_fig11_from_source_data.py",
    "reproduce_supplementary_fig1_from_source_data.py",
    "reproduce_supplementary_fig2_from_source_data.py",
    "reproduce_supplementary_fig3_from_source_data.py",
    "reproduce_supplementary_fig4_from_source_data.py",
    "reproduce_supplementary_fig5_from_source_data.py",
    "reproduce_supplementary_fig6_from_source_data.py",
    "reproduce_supplementary_fig7_from_source_data.py",
]


def run_one(script_name: str) -> dict:
    script_path = SCRIPT_DIR / script_name
    result = {
        "script": script_name,
        "exists": script_path.exists(),
        "returncode": "",
        "status": "",
        "stdout_tail": "",
        "stderr_tail": "",
    }

    if not script_path.exists():
        result["status"] = "missing_script"
        return result

    print(f"\n[RUN] {script_name}")

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    result["returncode"] = completed.returncode
    result["status"] = "success" if completed.returncode == 0 else "failed"
    result["stdout_tail"] = completed.stdout[-1000:]
    result["stderr_tail"] = completed.stderr[-1000:]

    if completed.returncode == 0:
        print(f"[OK] {script_name}")
    else:
        print(f"[FAILED] {script_name}")
        print(completed.stderr[-1000:])

    return result


def main() -> None:
    start = datetime.now()
    rows = []

    for script_name in ORDERED_SCRIPTS:
        rows.append(run_one(script_name))

    end = datetime.now()

    csv_path = LOG_DIR / "figure_reproduction_run_report.csv"
    md_path = LOG_DIR / "figure_reproduction_run_report.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["script", "exists", "returncode", "status", "stdout_tail", "stderr_tail"],
        )
        writer.writeheader()
        writer.writerows(rows)

    failed = [r for r in rows if r["status"] != "success"]

    md_lines = [
        "# Figure reproduction run report",
        "",
        f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}",
        f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total scripts: {len(rows)}",
        f"Successful scripts: {sum(r['status'] == 'success' for r in rows)}",
        f"Failed or missing scripts: {len(failed)}",
        "",
    ]

    if failed:
        md_lines.append("## Failed or missing scripts")
        md_lines.append("")
        for r in failed:
            md_lines.append(f"- `{r['script']}`: {r['status']}")
    else:
        md_lines.append("All figure reproduction scripts completed successfully.")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\n[DONE] CSV report: {csv_path}")
    print(f"[DONE] Markdown report: {md_path}")

    if failed:
        raise SystemExit("[WARNING] Some scripts failed or were missing. See report.")
    else:
        print("[DONE] All figure reproduction scripts completed successfully.")


if __name__ == "__main__":
    main()
