import subprocess
import sys
from pathlib import Path

NOTEBOOKS = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_models_comparison.ipynb",
    "notebooks/03_hybrid_analysis.ipynb",
]


def main():
    root = Path(__file__).resolve().parents[1]
    failed = False
    for nb in NOTEBOOKS:
        path = root / nb
        if not path.exists():
            print("SKIP: " + nb + " not found")
            continue
        print("Executing " + nb + "...")
        r = subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(path)],
            cwd=str(root),
        )
        if r.returncode != 0:
            print("FAIL: " + nb)
            failed = True
    if failed:
        sys.exit(1)
    print("OK: all notebooks executed")


if __name__ == "__main__":
    main()
