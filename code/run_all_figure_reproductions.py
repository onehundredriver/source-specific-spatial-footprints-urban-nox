# -*- coding: utf-8 -*-
"""
Run all figure reproduction scripts.

Updated for the revised Nature Cities manuscript figure structure:
- Main figures: Fig. 1–5
- Extended Data figures: Extended Data Fig. 1–6
- Supplementary figures: Supplementary Fig. 1–8

This runner only reproduces figures from already released source data.
It does not run source-data preparation scripts such as:
- prepare_fig1_source_data_052314.py
- prepare_fig5_source_data_052314.py
"""

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "code" / "scripts"

MAIN_FIGURE_SCRIPTS = [
    f"reproduce_fig{i}_from_source_data.py" for i in range(1, 6)
]

EXTENDED_DATA_SCRIPTS = [
    f"reproduce_extended_data_fig{i}_from_source_data.py" for i in range(1, 7)
]

SUPPLEMENTARY_FIGURE_SCRIPTS = [
    f"reproduce_supplementary_fig{i}_from_source_data.py" for i in range(1, 9)
]

SCRIPTS = (
    MAIN_FIGURE_SCRIPTS
    + EXTENDED_DATA_SCRIPTS
    + SUPPLEMENTARY_FIGURE_SCRIPTS
)


def main() -> None:
    print("=" * 100)
    print("Running all figure reproduction scripts")
    print("=" * 100)
    print(f"Repository root: {REPO_ROOT}")
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Total scripts: {len(SCRIPTS)}")

    missing_scripts = [
        script_name for script_name in SCRIPTS
        if not (SCRIPT_DIR / script_name).exists()
    ]

    if missing_scripts:
        print("\n[ERROR] Missing reproduction scripts:")
        for script_name in missing_scripts:
            print(f"  - {SCRIPT_DIR / script_name}")
        raise FileNotFoundError(
            "One or more required figure reproduction scripts are missing."
        )

    for idx, script_name in enumerate(SCRIPTS, start=1):
        script_path = SCRIPT_DIR / script_name
        print("\n" + "-" * 100)
        print(f"[RUN {idx:02d}/{len(SCRIPTS):02d}] {script_name}")
        print("-" * 100)
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(REPO_ROOT),
            check=True,
        )

    print("\n[DONE] All figure reproduction scripts completed successfully.")


if __name__ == "__main__":
    main()
