"""
setup_check.py
==============
Quick environment and dataset validation script.
Run this before starting experiments to confirm all dependencies
and data files are correctly installed/placed.

Usage:
    python setup_check.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def check_imports() -> bool:
    REQUIRED = [
        ("numpy",        "numpy"),
        ("pandas",       "pandas"),
        ("scipy",        "scipy"),
        ("sklearn",      "scikit-learn"),
        ("lightgbm",     "lightgbm"),
        ("ngboost",      "ngboost"),
        ("pulp",         "pulp"),
        ("matplotlib",   "matplotlib"),
        ("seaborn",      "seaborn"),
        ("yaml",         "pyyaml"),
        ("tqdm",         "tqdm"),
        ("statsmodels",  "statsmodels"),
        ("pingouin",     "pingouin"),
    ]
    ok = True
    print("\n── Dependency Check ─────────────────────────────────────────")
    for mod, pkg in REQUIRED:
        try:
            importlib.import_module(mod)
            print(f"  ✓  {pkg:<20}")
        except ImportError:
            print(f"  ✗  {pkg:<20}  ← MISSING  (pip install {pkg})")
            ok = False
    return ok


def check_data() -> bool:
    DATA_FILES = [
        "data/raw/train.csv",
        "data/raw/features.csv",
        "data/raw/stores.csv",
    ]
    ok = True
    print("\n── Dataset Check ───────────────────────────────────────��────")
    for f in DATA_FILES:
        p = Path(f)
        if p.exists():
            kb = p.stat().st_size // 1024
            print(f"  ✓  {f:<35}  ({kb:,} KB)")
        else:
            print(f"  ✗  {f:<35}  ← NOT FOUND")
            ok = False
    if not ok:
        print(
            "\n  Download the dataset from:\n"
            "  https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast\n"
            "  and place the CSV files in data/raw/"
        )
    return ok


def check_solver() -> None:
    print("\n── Solver Check ─────────────────────────────────────────────")
    try:
        import pulp
        solvers = pulp.listSolvers(onlyAvailable=True)
        print(f"  Available PuLP solvers: {solvers}")
        if not solvers:
            print("  ⚠  No solvers found. CBC is bundled with PuLP; try:")
            print("     pip install pulp  (includes CBC by default)")
    except ImportError:
        print("  ✗  PuLP not installed.")


def main():
    print("=" * 60)
    print("  MEIO Project — Environment Setup Check")
    print("=" * 60)
    deps_ok = check_imports()
    data_ok = check_data()
    check_solver()
    print("\n── Summary ──────────────────────────────────────────────────")
    if deps_ok and data_ok:
        print("  ✓  All checks passed. Ready to run experiments!\n")
        print("  Quick start:")
        print("    python main.py --config config.yaml --quick --method ngboost\n")
    else:
        print("  ✗  Some checks failed. See above for details.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()