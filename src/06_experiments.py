"""
06_experiments.py — Computational Experiments.

Experiment 1: Classical vs. ML-Enhanced GSM
Experiment 2: Service Level Sensitivity Analysis
Experiment 3: Echelon Contribution Analysis
Experiment 4: Forecasting Model Comparison
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Also add src/ itself so sibling modules can be imported by name
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import time

import config

# Import sibling modules directly (now on sys.path)
from importlib import import_module as _im
import importlib.util

def _load_sibling(module_name, filename):
    """Load a sibling module from src/ by filename."""
    unique_name = f"meio_pipeline.{module_name}"
    # Check if already loaded
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    spec = importlib.util.spec_from_file_location(
        unique_name, Path(__file__).resolve().parent / filename
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod  # Register before exec to handle circular refs
    spec.loader.exec_module(mod)
    return mod

# Lazy-load siblings to avoid circular issues
nc = None
gsm = None

def _ensure_loaded():
    global nc, gsm
    if nc is None:
        nc = _load_sibling("network_construction", "04_network_construction.py")
    if gsm is None:
        gsm = _load_sibling("gsm_optimization", "05_gsm_optimization.py")


def run_experiment_1(df, forecast_results):
    """
    Experiment 1: Classical vs ML-Enhanced Safety Stock Optimization.
    """
    _ensure_loaded()

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: CLASSICAL vs ML-ENHANCED GSM")
    print("=" * 70)

    best_model = forecast_results["best_model_name"]
    residual_stats = forecast_results["residual_stats"]

    baseline_stats = residual_stats["Historical Baseline"]
    ml_stats = residual_stats[best_model]

    comparison_rows = []

    for sl in [config.DEFAULT_SERVICE_LEVEL]:
        print(f"\n  --- Approach A: Classical (sigma_historical), SL={sl*100:.0f}% ---")
        net_classical = nc.build_network(baseline_stats, df, "sigma_historical")
        _, df_classical = gsm.run_gsm_optimization(
            net_classical, sl, f"classical_sl{int(sl*100)}"
        )

        print(f"\n  --- Approach B: ML-Enhanced (sigma_residual, {best_model}), SL={sl*100:.0f}% ---")
        net_ml = nc.build_network(ml_stats, df, "sigma_residual")
        _, df_ml = gsm.run_gsm_optimization(
            net_ml, sl, f"ml_enhanced_sl{int(sl*100)}"
        )

        classical_cost = df_classical["holding_cost_weekly"].sum()
        ml_cost = df_ml["holding_cost_weekly"].sum()
        classical_ss = df_classical["safety_stock"].sum()
        ml_ss = df_ml["safety_stock"].sum()

        reduction_cost = (1 - ml_cost / classical_cost) * 100
        reduction_ss = (1 - ml_ss / classical_ss) * 100

        comparison_rows.append({
            "Service Level": f"{sl*100:.0f}%",
            "Classical: Total SS": classical_ss,
            "Classical: Weekly Cost": classical_cost,
            "ML-Enhanced: Total SS": ml_ss,
            "ML-Enhanced: Weekly Cost": ml_cost,
            "SS Reduction (%)": reduction_ss,
            "Cost Reduction (%)": reduction_cost,
        })

    comparison_df = pd.DataFrame(comparison_rows)

    print(f"\n\n  {'='*68}")
    print(f"  EXPERIMENT 1 RESULTS: Cost Reduction from ML-Enhanced GSM")
    print(f"  {'='*68}")
    print(f"  Safety Stock Reduction:  {comparison_df['SS Reduction (%)'].iloc[0]:>6.1f}%")
    print(f"  Holding Cost Reduction:  {comparison_df['Cost Reduction (%)'].iloc[0]:>6.1f}%")
    print(f"  {'='*68}")

    comparison_df.to_csv(config.RESULTS_DIR / "experiment1_comparison.csv", index=False)
    return comparison_df


def run_experiment_2(df, forecast_results):
    """Experiment 2: Service Level Sensitivity Analysis."""
    _ensure_loaded()

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SERVICE LEVEL SENSITIVITY ANALYSIS")
    print("=" * 70)

    best_model = forecast_results["best_model_name"]
    baseline_stats = forecast_results["residual_stats"]["Historical Baseline"]
    ml_stats = forecast_results["residual_stats"][best_model]

    rows = []

    for sl in config.SENSITIVITY_SERVICE_LEVELS:
        print(f"\n  --- Service Level: {sl*100:.0f}% ---")

        net_c = nc.build_network(baseline_stats, df, "sigma_historical")
        _, df_c = gsm.run_gsm_optimization(net_c, sl, f"sensitivity_classical_{int(sl*100)}")

        net_m = nc.build_network(ml_stats, df, "sigma_residual")
        _, df_m = gsm.run_gsm_optimization(net_m, sl, f"sensitivity_ml_{int(sl*100)}")

        for approach, res_df in [("Classical", df_c), ("ML-Enhanced", df_m)]:
            for echelon in sorted(res_df["echelon"].unique()):
                edf = res_df[res_df["echelon"] == echelon]
                rows.append({
                    "service_level": sl, "approach": approach,
                    "echelon": echelon,
                    "total_safety_stock": edf["safety_stock"].sum(),
                    "total_holding_cost": edf["holding_cost_weekly"].sum(),
                    "n_nodes": len(edf),
                })
            rows.append({
                "service_level": sl, "approach": approach,
                "echelon": "Total",
                "total_safety_stock": res_df["safety_stock"].sum(),
                "total_holding_cost": res_df["holding_cost_weekly"].sum(),
                "n_nodes": len(res_df),
            })

    sensitivity_df = pd.DataFrame(rows)
    sensitivity_df.to_csv(config.RESULTS_DIR / "experiment2_sensitivity.csv", index=False)

    print(f"\n  Service Level Sensitivity (Network Total):")
    print(f"  {'SL':>5} | {'Classical Cost':>16} | {'ML Cost':>16} | {'Reduction':>10}")
    print(f"  {'-'*55}")
    for sl in config.SENSITIVITY_SERVICE_LEVELS:
        c_cost = sensitivity_df[
            (sensitivity_df["service_level"] == sl)
            & (sensitivity_df["approach"] == "Classical")
            & (sensitivity_df["echelon"] == "Total")
        ]["total_holding_cost"].iloc[0]
        m_cost = sensitivity_df[
            (sensitivity_df["service_level"] == sl)
            & (sensitivity_df["approach"] == "ML-Enhanced")
            & (sensitivity_df["echelon"] == "Total")
        ]["total_holding_cost"].iloc[0]
        red = (1 - m_cost / c_cost) * 100
        print(f"  {sl*100:>4.0f}% | {c_cost:>16,.2f} | {m_cost:>16,.2f} | {red:>9.1f}%")

    return sensitivity_df


def run_experiment_3(df, forecast_results):
    """Experiment 3: Echelon Contribution Analysis."""
    _ensure_loaded()

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: ECHELON CONTRIBUTION ANALYSIS")
    print("=" * 70)

    best_model = forecast_results["best_model_name"]
    baseline_stats = forecast_results["residual_stats"]["Historical Baseline"]
    ml_stats = forecast_results["residual_stats"][best_model]

    sl = config.DEFAULT_SERVICE_LEVEL
    rows = []

    for approach_name, stats, sigma_col in [
        ("Classical", baseline_stats, "sigma_historical"),
        ("ML-Enhanced", ml_stats, "sigma_residual"),
    ]:
        net = nc.build_network(stats, df, sigma_col)
        _, res_df = gsm.run_gsm_optimization(net, sl, f"echelon_{approach_name.lower()}")

        total_cost = res_df["holding_cost_weekly"].sum()
        total_ss = res_df["safety_stock"].sum()

        for echelon in sorted(res_df["echelon"].unique()):
            edf = res_df[res_df["echelon"] == echelon]
            e_cost = edf["holding_cost_weekly"].sum()
            e_ss = edf["safety_stock"].sum()

            rows.append({
                "approach": approach_name, "echelon": echelon,
                "echelon_type": edf["type"].iloc[0],
                "n_nodes": len(edf),
                "total_safety_stock": e_ss,
                "pct_safety_stock": e_ss / total_ss * 100,
                "total_holding_cost": e_cost,
                "pct_holding_cost": e_cost / total_cost * 100,
                "avg_tau": edf["tau"].mean(),
            })

    echelon_df = pd.DataFrame(rows)
    echelon_df.to_csv(config.RESULTS_DIR / "experiment3_echelon_analysis.csv", index=False)

    print(f"\n  Echelon Distribution of Safety Stock (SL={sl*100:.0f}%):")
    for approach in ["Classical", "ML-Enhanced"]:
        print(f"\n  {approach}:")
        adf = echelon_df[echelon_df["approach"] == approach]
        for _, row in adf.iterrows():
            print(f"    Echelon {row['echelon']} ({row['echelon_type']:>10s}): "
                  f"SS = {row['total_safety_stock']:>12,.1f} "
                  f"({row['pct_safety_stock']:>5.1f}%) | "
                  f"Cost = {row['total_holding_cost']:>10,.2f} "
                  f"({row['pct_holding_cost']:>5.1f}%)")

    return echelon_df


def run_experiment_4(df, forecast_results):
    """Experiment 4: Forecasting Model Comparison in GSM Context."""
    _ensure_loaded()

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: FORECASTING MODEL COMPARISON IN GSM CONTEXT")
    print("=" * 70)

    sl = config.DEFAULT_SERVICE_LEVEL
    rows = []
    forecast_metrics = forecast_results["metrics"]

    for model_name, stats_df in forecast_results["residual_stats"].items():
        sigma_col = "sigma_historical" if model_name == "Historical Baseline" else "sigma_residual"

        net = nc.build_network(stats_df, df, sigma_col)
        _, res_df = gsm.run_gsm_optimization(
            net, sl, f"model_comparison_{model_name.lower().replace(' ', '_')}"
        )

        model_metrics = forecast_metrics[forecast_metrics["Model"] == model_name]
        rmse = model_metrics["RMSE"].iloc[0] if len(model_metrics) > 0 else np.nan
        wape = model_metrics["WAPE (%)"].iloc[0] if len(model_metrics) > 0 else np.nan

        rows.append({
            "Forecast Model": model_name,
            "Forecast RMSE": rmse,
            "Forecast WAPE (%)": wape,
            "Total Safety Stock": res_df["safety_stock"].sum(),
            "Total Holding Cost/wk": res_df["holding_cost_weekly"].sum(),
            "Avg sigma_used": (stats_df[sigma_col].mean()
                               if sigma_col in stats_df.columns
                               else stats_df["sigma_historical"].mean()),
        })

    model_comp_df = pd.DataFrame(rows)
    model_comp_df.to_csv(config.RESULTS_DIR / "experiment4_model_comparison.csv", index=False)

    print(f"\n  Model Comparison Summary (SL={sl*100:.0f}%):")
    print(model_comp_df.to_string(index=False))

    return model_comp_df


def run_experiment_5(df, forecast_results):
    """
    Experiment 5: Monte Carlo Validation of GSM Service Levels.

    Validates that the safety stock levels computed by the GSM
    actually achieve the target service level under simulated demand.
    Tests both normal and log-normal demand distributions.
    """
    _ensure_loaded()

    mc_spec = importlib.util.spec_from_file_location(
        "meio_pipeline.monte_carlo",
        Path(__file__).resolve().parent / "08_monte_carlo_simulation.py"
    )
    if "meio_pipeline.monte_carlo" not in sys.modules:
        mc = importlib.util.module_from_spec(mc_spec)
        sys.modules["meio_pipeline.monte_carlo"] = mc
        mc_spec.loader.exec_module(mc)
    else:
        mc = sys.modules["meio_pipeline.monte_carlo"]

    print("\n" + "=" * 70)
    print("EXPERIMENT 5: MONTE CARLO SERVICE LEVEL VALIDATION")
    print("=" * 70)

    best_model = forecast_results["best_model_name"]
    baseline_stats = forecast_results["residual_stats"]["Historical Baseline"]
    ml_stats = forecast_results["residual_stats"][best_model]

    rows = []

    for sl in config.SENSITIVITY_SERVICE_LEVELS:
        for approach_name, stats, sigma_col in [
            ("Classical", baseline_stats, "sigma_historical"),
            ("ML-Enhanced", ml_stats, "sigma_residual"),
        ]:
            net = nc.build_network(stats, df, sigma_col)
            _, res_df = gsm.run_gsm_optimization(
                net, sl, f"mc_{approach_name.lower()}_{int(sl*100)}"
            )

            mc_result = mc.run_monte_carlo_validation(
                res_df, net, sl,
                label=f"{approach_name} SL={sl*100:.0f}%",
                n_trials=10000,
            )

            rows.append({
                "service_level": sl,
                "approach": approach_name,
                "target_fill_rate": sl,
                "normal_fill_rate": mc_result["normal"]["network_fill_rate"],
                "normal_mean_node_fr": mc_result["normal"]["mean_node_fill_rate"],
                "normal_min_node_fr": mc_result["normal"]["min_node_fill_rate"],
                "normal_p5_node_fr": mc_result["normal"]["p5_node_fill_rate"],
                "lognormal_fill_rate": mc_result["lognormal"]["network_fill_rate"],
                "lognormal_mean_node_fr": mc_result["lognormal"]["mean_node_fill_rate"],
            })

    mc_df = pd.DataFrame(rows)
    mc_df.to_csv(config.RESULTS_DIR / "experiment5_monte_carlo.csv", index=False)

    # Print summary table
    print(f"\n  Monte Carlo Validation Summary:")
    print(f"  {'SL':>5} {'Approach':>14} | {'Normal FR':>10} {'LN FR':>10} | "
          f"{'Gap(N)':>8} {'Gap(LN)':>8}")
    print(f"  {'-'*62}")
    for _, row in mc_df.iterrows():
        gap_n = (row["normal_fill_rate"] - row["target_fill_rate"]) * 100
        gap_ln = (row["lognormal_fill_rate"] - row["target_fill_rate"]) * 100
        print(f"  {row['service_level']*100:>4.0f}% {row['approach']:>14} | "
              f"{row['normal_fill_rate']*100:>9.2f}% "
              f"{row['lognormal_fill_rate']*100:>9.2f}% | "
              f"{gap_n:>+7.2f}pp {gap_ln:>+7.2f}pp")

    return mc_df


def run_experiment_6(df, forecast_results):
    """
    Experiment 6: SEIO Baseline Comparison (Value of Coordination).

    Compares multi-echelon inventory optimization (MEIO/GSM) against
    single-echelon independent optimization (SEIO) to quantify the
    value of supply chain coordination.
    """
    _ensure_loaded()

    seio_spec = importlib.util.spec_from_file_location(
        "meio_pipeline.seio_baseline",
        Path(__file__).resolve().parent / "09_seio_baseline.py"
    )
    if "meio_pipeline.seio_baseline" not in sys.modules:
        seio_mod = importlib.util.module_from_spec(seio_spec)
        sys.modules["meio_pipeline.seio_baseline"] = seio_mod
        seio_spec.loader.exec_module(seio_mod)
    else:
        seio_mod = sys.modules["meio_pipeline.seio_baseline"]

    print("\n" + "=" * 70)
    print("EXPERIMENT 6: SEIO vs MEIO (VALUE OF COORDINATION)")
    print("=" * 70)

    best_model = forecast_results["best_model_name"]
    baseline_stats = forecast_results["residual_stats"]["Historical Baseline"]
    ml_stats = forecast_results["residual_stats"][best_model]

    rows = []

    for sl in config.SENSITIVITY_SERVICE_LEVELS:
        for approach_name, stats, sigma_col in [
            ("Classical", baseline_stats, "sigma_historical"),
            ("ML-Enhanced", ml_stats, "sigma_residual"),
        ]:
            net = nc.build_network(stats, df, sigma_col)
            _, meio_df = gsm.run_gsm_optimization(
                net, sl, f"seio_{approach_name.lower()}_{int(sl*100)}"
            )

            seio_df, comparison = seio_mod.run_seio_analysis(
                net, meio_df, sl,
                label=f"{approach_name} SL={sl*100:.0f}%"
            )

            total = comparison["total"]
            rows.append({
                "service_level": sl,
                "approach": approach_name,
                "seio_total_ss": total["seio_ss"],
                "meio_total_ss": total["meio_ss"],
                "seio_total_cost": total["seio_cost"],
                "meio_total_cost": total["meio_cost"],
                "coordination_value_pct": total["cost_reduction_pct"],
                "coordination_value_abs": total["seio_cost"] - total["meio_cost"],
            })

    seio_comp_df = pd.DataFrame(rows)
    seio_comp_df.to_csv(config.RESULTS_DIR / "experiment6_seio_comparison.csv", index=False)

    # Print summary
    print(f"\n  SEIO vs MEIO Summary (Value of Coordination):")
    print(f"  {'SL':>5} {'Approach':>14} | {'SEIO Cost':>14} {'MEIO Cost':>14} "
          f"{'Savings':>14} {'Coord. Value':>12}")
    print(f"  {'-'*78}")
    for _, row in seio_comp_df.iterrows():
        print(f"  {row['service_level']*100:>4.0f}% {row['approach']:>14} | "
              f"${row['seio_total_cost']:>13,.2f} "
              f"${row['meio_total_cost']:>13,.2f} "
              f"${row['coordination_value_abs']:>13,.2f} "
              f"{row['coordination_value_pct']:>11.1f}%")

    return seio_comp_df


def run_experiment_7(df, forecast_results):
    """
    Experiment 7: Parameter Sensitivity Analysis & MILP Verification.

    Tests robustness of results to assumed parameters (lead times,
    holding costs) and verifies the DP solution using PuLP MILP.
    """
    _ensure_loaded()

    sens_spec = importlib.util.spec_from_file_location(
        "meio_pipeline.sensitivity",
        Path(__file__).resolve().parent / "10_sensitivity_analysis.py"
    )
    if "meio_pipeline.sensitivity" not in sys.modules:
        sens_mod = importlib.util.module_from_spec(sens_spec)
        sys.modules["meio_pipeline.sensitivity"] = sens_mod
        sens_spec.loader.exec_module(sens_mod)
    else:
        sens_mod = sys.modules["meio_pipeline.sensitivity"]

    best_model = forecast_results["best_model_name"]
    classical_stats = forecast_results["residual_stats"]["Historical Baseline"]
    ml_stats = forecast_results["residual_stats"][best_model]

    result = sens_mod.run_sensitivity_analysis(
        nc.build_network, gsm.run_gsm_optimization,
        classical_stats, ml_stats, df,
        service_level=config.DEFAULT_SERVICE_LEVEL,
    )

    return result


def run_all_experiments(df, forecast_results):
    """Run all computational experiments."""
    print("\n" + "#" * 70)
    print("#  COMPUTATIONAL EXPERIMENTS")
    print("#" * 70)

    start = time.time()

    exp1 = run_experiment_1(df, forecast_results)
    exp2 = run_experiment_2(df, forecast_results)
    exp3 = run_experiment_3(df, forecast_results)
    exp4 = run_experiment_4(df, forecast_results)
    exp5 = run_experiment_5(df, forecast_results)
    exp6 = run_experiment_6(df, forecast_results)
    exp7 = run_experiment_7(df, forecast_results)

    elapsed = time.time() - start
    print(f"\n  All experiments completed in {elapsed:.1f} seconds")

    return {
        "experiment1": exp1,
        "experiment2": exp2,
        "experiment3": exp3,
        "experiment4": exp4,
        "experiment5": exp5,
        "experiment6": exp6,
        "experiment7": exp7,
    }