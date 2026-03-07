"""
run_pipeline.py — End-to-end execution of the multi-echelon inventory
optimization pipeline.

This file lives at the PROJECT ROOT (same level as config.py, src/, data/).

Pipeline stages:
  1. Data Preprocessing
  2. Exploratory Data Analysis
  3. Demand Forecasting (Baseline, LightGBM, XGBoost)
  4. Multi-Echelon Network Construction
  5. GSM Optimization
  6. Computational Experiments
  7. Visualization

Usage:
    python run_pipeline.py
"""

import sys
import time
import importlib.util
from pathlib import Path

# Project root = directory containing this file
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config


def load_module(name, filename):
    """Dynamically load a pipeline module from src/."""
    # Use a unique prefixed name to avoid collisions with any installed packages
    unique_name = f"meio_pipeline.{name}"
    filepath = PROJECT_ROOT / "src" / filename
    spec = importlib.util.spec_from_file_location(unique_name, str(filepath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module  # Register to avoid re-import issues
    spec.loader.exec_module(module)
    return module


def main():
    print("=" * 70)
    print("  MULTI-ECHELON INVENTORY OPTIMIZATION WITH ML-ENHANCED GSM")
    print("  Dynamic Safety Stock Placement Using Walmart Retail Data")
    print("=" * 70)

    total_start = time.time()

    # ================================================================
    # STAGE 1: DATA PREPROCESSING
    # ================================================================
    preprocessing = load_module("preprocessing", "01_data_preprocessing.py")
    df, aggregations = preprocessing.run_preprocessing()

    # ================================================================
    # STAGE 2: EXPLORATORY DATA ANALYSIS
    # ================================================================
    eda_mod = load_module("eda", "02_eda.py")
    eda_results = eda_mod.run_eda(df)

    # ================================================================
    # STAGE 3: DEMAND FORECASTING
    # ================================================================
    forecasting = load_module("forecasting", "03_demand_forecasting.py")
    forecast_results = forecasting.run_demand_forecasting(df)

    print("\n  Forecast Metrics Summary:")
    print(forecast_results["metrics"].to_string(index=False))

    # ================================================================
    # STAGE 6: COMPUTATIONAL EXPERIMENTS
    # (Stages 4-5 are executed within experiments for systematic comparison)
    # ================================================================
    experiments_mod = load_module("experiments", "06_experiments.py")
    experiment_results = experiments_mod.run_all_experiments(df, forecast_results)

    # ================================================================
    # STAGE 7: VISUALIZATION
    # ================================================================
    visualization = load_module("visualization", "07_visualization.py")
    visualization.generate_all_figures(experiment_results)

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)

    exp1 = experiment_results["experiment1"]
    ss_red = exp1["SS Reduction (%)"].iloc[0]
    cost_red = exp1["Cost Reduction (%)"].iloc[0]

    print(f"  Key Finding:")
    print(f"    ML-enhanced demand forecasting reduces safety stock by")
    print(f"    {ss_red:.1f}% and holding costs by {cost_red:.1f}% compared to")
    print(f"    classical statistical approaches, at the same service level.")
    print(f"")
    print(f"  Total runtime: {total_elapsed:.1f} seconds")
    print(f"  Results: {config.RESULTS_DIR}")
    print(f"  Figures: {config.FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()