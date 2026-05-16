# -*- coding: utf-8 -*-
"""
Run all figure reproduction scripts.

Updated after figure renumbering:
- Main figures: Fig. 1–9
- Supplementary figures: Supplementary Fig. 1–9
"""

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "code" / "scripts"

SCRIPTS = [
    *[f"reproduce_fig{i}_from_source_data.py" for i in range(1, 10)],
    *[f"reproduce_supplementary_fig{i}_from_source_data.py" for i in range(1, 10)],
]


def main():
    print("=" * 100)
    print("Running all figure reproduction scripts")
    print("=" * 100)

    for script_name in SCRIPTS:
        script_path = SCRIPT_DIR / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Missing reproduction script: {script_path}")

        print(f"\n[RUN] {script_name}")
        subprocess.run([sys.executable, str(script_path)], cwd=str(REPO_ROOT), check=True)

    print("\n[DONE] All figure reproduction scripts completed successfully.")


if __name__ == "__main__":
    main()
