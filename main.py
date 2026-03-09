"""
main.py
=======
Entry point for the MEIO Stochastic Optimization project.

This script orchestrates the complete pipeline:
  1. Load configuration.
  2. Validate dataset availability.
  3. Run the main experiment (all forecaster × optimizer combinations).
  4. Run the ablation study.
  5. Print a concise results summary to stdout.

Usage
-----
    # Full pipeline (all methods):
    python main.py --config config.yaml --all

    # Single method:
    python main.py --config config.yaml --method ngboost --optimizer stochastic_milp

    # Quick test (reduced data):
    python main.py --config config.yaml --quick

    # Ablation only:
    python main.py --config config.yaml --ablation-only --quick
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import load_config
from experiments.run_experiment import run_experiment
from experiments.ablation_study import run_ablation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MEIO Stochastic Optimization — Main Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--method", default="ngboost",
        choices=["ngboost", "quantile_regression", "conformal"],
        help="Demand forecasting method (single run)."
    )
    parser.add_argument(
        "--optimizer", default="stochastic_milp",
        choices=["stochastic_milp", "chance_constrained"],
        help="Optimization model (single run)."
    )
    parser.add_argument(
        "--output", default="results/",
        help="Directory for all output files and figures."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all forecaster × optimizer combinations."
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Also run the ablation study after the main experiment."
    )
    parser.add_argument(
        "--ablation-only", action="store_true",
        help="Skip main experiment; run ablation study only."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use reduced grid / data subset for fast testing."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_dataset(cfg: dict) -> bool:
    """Check that all required dataset files are present."""
    d = cfg.get("data", {})
    required = ["train_path", "features_path", "stores_path"]
    missing = []
    for key in required:
        p = Path(d.get(key, ""))
        if not p.exists():
            missing.append(str(p))
    if missing:
        logger.error(
            "Missing dataset files:\n  %s\n\n"
            "Please download the Walmart dataset from:\n"
            "  https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast\n"
            "and place CSV files in data/raw/",
            "\n  ".join(missing),
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results_df: pd.DataFrame) -> None:
    """Print a concise tabular summary of results to stdout."""
    if results_df.empty:
        print("No results to summarise.")
        return

    print("\n" + "=" * 72)
    print("  MEIO STOCHASTIC OPTIMIZATION — RESULTS SUMMARY")
    print("=" * 72)

    metric_map = {
        "RMSE":         "forecast_RMSE",
        "Coverage":     "forecast_Coverage",
        "WinklerScore": "forecast_WinklerScore",
        "Fill Rate":    "opt_fill_rate",
        "Hold Cost":    "opt_holding_cost",
        "Backord Cost": "opt_backorder_cost",
    }

    available = {
        label: col
        for label, col in metric_map.items()
        if col in results_df.columns
    }

    group_cols = ["forecaster", "optimizer"]
    existing_group = [c for c in group_cols if c in results_df.columns]
    if not existing_group:
        print(results_df.describe())
        return

    agg = results_df.groupby(existing_group)[list(available.values())].mean()
    agg.columns = list(available.keys())
    agg = agg.reset_index()
    agg.columns = [c.replace("forecaster", "Forecaster").replace("optimizer", "Optimizer")
                   for c in agg.columns]

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 120,
        "display.float_format", "{:.4f}".format,
    ):
        print(agg.to_string(index=False))

    print("=" * 72)

    # Best method by fill rate
    if "opt_fill_rate" in results_df.columns:
        best_idx = results_df.groupby(existing_group)["opt_fill_rate"].mean().idxmax()
        print(f"\n  Best method by fill rate: {best_idx}")

    # Best method by total cost (proxy: holding + backorder)
    cost_cols = [c for c in ["opt_holding_cost", "opt_backorder_cost", "opt_replenishment_cost"]
                 if c in results_df.columns]
    if cost_cols:
        results_df["_total_cost"] = results_df[cost_cols].sum(axis=1)
        best_cost = results_df.groupby(existing_group)["_total_cost"].mean().idxmin()
        print(f"  Best method by total cost: {best_cost}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the complete MEIO pipeline. Returns exit code."""
    args = _parse_args()

    # Apply quick-mode overrides to config
    cfg = load_config(args.config)
    if args.quick:
        cfg["data"]["store_subset"] = [1, 2]
        cfg["data"]["dept_subset"]  = [1, 2, 3]
        cfg["optimization"]["n_scenarios"] = 25
        cfg["demand"]["ngboost"]["n_estimators"] = 100
        cfg["demand"]["quantile_regression"]["n_estimators"] = 100

    logging.basicConfig(
        level=getattr(logging, cfg["output"].get("log_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    banner = r"""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║  MEIO — Stochastic Optimization with Data-Driven Demand Distributions ║
  ║  Approach 1 | OR Publication Pipeline                                 ║
  ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

    # Dataset validation
    if not _check_dataset(cfg):
        return 1

    results_df = pd.DataFrame()

    # --- Main experiment ---
    if not args.ablation_only:
        logger.info("Starting main experiment...")
        results_df = run_experiment(
            config_path=args.config,
            method=args.method,
            optimizer=args.optimizer,
            output_dir=args.output,
            all_combinations=args.all,
        )
        _print_summary(results_df)

    # --- Ablation study ---
    if args.ablation or args.ablation_only:
        logger.info("Starting ablation study...")
        ablation_df = run_ablation(
            config_path=args.config,
            output_dir=args.output,
            quick=args.quick,
        )
        print(f"\n  Ablation study complete: {len(ablation_df)} cells evaluated.")
        print(f"  Results saved to: {Path(args.output) / 'ablation_results.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())