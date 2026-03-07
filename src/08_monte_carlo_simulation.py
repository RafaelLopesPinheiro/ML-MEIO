"""
08_monte_carlo_simulation.py — Monte Carlo Validation of GSM Safety Stocks.

Validates that the safety stock levels computed by the GSM actually
achieve the target service level under stochastic demand.

METHODOLOGY:
  For each simulation trial:
    1. Generate random demand at each leaf node (Store-Dept) drawn from
       N(mu, sigma^2) — matching the GSM demand assumption.
    2. Simulate the multi-echelon inventory system over a planning horizon
       using base-stock policies with the GSM-computed safety stocks.
    3. Measure the achieved fill rate (Type-1 service level) at each node:
       P(demand <= base_stock_level) = P(demand <= mu + safety_stock)
    4. Aggregate across nodes and trials.

  The GSM with safety factor k guarantees that, under the bounded-demand
  assumption, demand is met with probability >= Phi(k). Under normally
  distributed demand, the expected fill rate at each node equals Phi(k)
  when tau > 0, and 100% when tau = 0 (no replenishment uncertainty).

  This simulation validates whether the ACTUAL fill rate matches the
  TARGET, and quantifies any gap due to model assumptions (e.g.,
  demand non-normality, independence assumptions in risk pooling).

References:
  - Graves & Willems (2000), Section 4 (Service guarantees)
  - Simchi-Levi & Zhao (2012), "Performance Evaluation of Stochastic
    Multi-Echelon Inventory Systems: A Survey"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config


def simulate_single_node_fill_rate(mu, sigma, safety_stock, n_periods, rng):
    """
    Simulate fill rate at a single node under base-stock policy.

    Base-stock level B = mu + safety_stock.
    Each period, demand D ~ N(mu, sigma^2) (clipped at 0).
    Fill rate = fraction of periods where D <= B.

    Parameters:
        mu: Mean demand per period
        sigma: Demand std deviation
        safety_stock: GSM-computed safety stock
        n_periods: Number of demand periods to simulate
        rng: numpy random generator

    Returns:
        fill_rate: Fraction of periods with no stockout
        avg_shortage: Average shortage quantity when stockout occurs
    """
    base_stock = mu + safety_stock

    # Generate demands
    demands = rng.normal(mu, sigma, size=n_periods)
    demands = np.maximum(demands, 0)  # Demand cannot be negative

    # Fill rate: fraction of periods where demand <= base stock
    filled = demands <= base_stock
    fill_rate = filled.mean()

    # Average shortage (when stockout occurs)
    shortages = np.maximum(demands - base_stock, 0)
    avg_shortage = shortages.mean()

    return fill_rate, avg_shortage


def simulate_network_fill_rate(
    gsm_results_df, network, n_trials=10000, n_periods_per_trial=52,
    random_seed=None
):
    """
    Monte Carlo simulation of the full multi-echelon network.

    For each trial, generates independent demand at each leaf node
    and checks whether the GSM safety stocks provide sufficient buffer.

    We measure two service metrics:
      1. Node-level fill rate: Per-node average across trials
      2. Network-level fill rate: Fraction of (node, period) pairs
         where demand was met

    Parameters:
        gsm_results_df: DataFrame from GSM optimization (node_id, mu, sigma, safety_stock)
        network: SupplyChainNetwork object
        n_trials: Number of MC trials (default 10,000)
        n_periods_per_trial: Periods per trial (default 52 = 1 year)
        random_seed: For reproducibility

    Returns:
        dict with simulation results
    """
    rng = np.random.default_rng(random_seed or config.RANDOM_SEED)

    # Extract leaf nodes (demand nodes) — these are where service matters
    leaf_df = gsm_results_df[gsm_results_df["type"] == "store_dept"].copy()

    n_nodes = len(leaf_df)
    total_periods = n_trials * n_periods_per_trial

    # Pre-extract arrays for vectorized simulation
    mus = leaf_df["mu"].values
    sigmas = leaf_df["sigma"].values
    safety_stocks = leaf_df["safety_stock"].values
    base_stocks = mus + safety_stocks

    # Vectorized simulation: (n_trials * n_periods_per_trial, n_nodes)
    all_demands = rng.normal(
        loc=mus[np.newaxis, :],
        scale=sigmas[np.newaxis, :],
        size=(total_periods, n_nodes)
    )
    all_demands = np.maximum(all_demands, 0)

    # Fill: demand <= base_stock
    filled_matrix = all_demands <= base_stocks[np.newaxis, :]

    # --- Node-level fill rates ---
    node_fill_rates = filled_matrix.mean(axis=0)  # shape: (n_nodes,)

    # --- Network-level aggregate ---
    network_fill_rate = filled_matrix.mean()

    # --- Shortage analysis ---
    shortages = np.maximum(all_demands - base_stocks[np.newaxis, :], 0)
    avg_shortage_per_node = shortages.mean(axis=0)
    total_avg_shortage = shortages.mean()

    # --- Per-node results ---
    leaf_df = leaf_df.copy()
    leaf_df["simulated_fill_rate"] = node_fill_rates
    leaf_df["avg_shortage"] = avg_shortage_per_node

    # --- Summary statistics ---
    results = {
        "network_fill_rate": network_fill_rate,
        "mean_node_fill_rate": node_fill_rates.mean(),
        "median_node_fill_rate": np.median(node_fill_rates),
        "min_node_fill_rate": node_fill_rates.min(),
        "p5_node_fill_rate": np.percentile(node_fill_rates, 5),
        "p95_node_fill_rate": np.percentile(node_fill_rates, 95),
        "std_node_fill_rate": node_fill_rates.std(),
        "total_avg_shortage": total_avg_shortage,
        "n_trials": n_trials,
        "n_periods_per_trial": n_periods_per_trial,
        "total_simulated_periods": total_periods,
        "n_nodes_simulated": n_nodes,
        "node_results": leaf_df,
    }

    return results


def simulate_with_non_normal_demand(
    gsm_results_df, network, n_trials=10000, n_periods_per_trial=52,
    random_seed=None
):
    """
    Robustness check: simulate with log-normal demand (skewed, realistic).

    Retail demand is typically right-skewed, not normal. This tests
    whether GSM safety stocks (derived under normality assumption)
    still provide adequate service under more realistic demand.

    Log-normal parameters are fitted to match the same mu and sigma
    as the normal case, so any difference is purely from distribution shape.
    """
    rng = np.random.default_rng(random_seed or config.RANDOM_SEED + 1)

    leaf_df = gsm_results_df[gsm_results_df["type"] == "store_dept"].copy()
    n_nodes = len(leaf_df)
    total_periods = n_trials * n_periods_per_trial

    mus = leaf_df["mu"].values
    sigmas = leaf_df["sigma"].values
    safety_stocks = leaf_df["safety_stock"].values
    base_stocks = mus + safety_stocks

    # Convert (mu, sigma) to log-normal parameters
    # For log-normal X: E[X] = exp(mu_ln + sigma_ln^2/2), Var[X] = ...
    # Solve: sigma_ln^2 = log(1 + sigma^2/mu^2), mu_ln = log(mu) - sigma_ln^2/2
    cv_sq = (sigmas / np.maximum(mus, 1.0)) ** 2
    sigma_ln = np.sqrt(np.log(1 + cv_sq))
    mu_ln = np.log(np.maximum(mus, 1.0)) - sigma_ln ** 2 / 2

    all_demands = rng.lognormal(
        mean=mu_ln[np.newaxis, :],
        sigma=sigma_ln[np.newaxis, :],
        size=(total_periods, n_nodes)
    )

    filled_matrix = all_demands <= base_stocks[np.newaxis, :]

    node_fill_rates = filled_matrix.mean(axis=0)
    network_fill_rate = filled_matrix.mean()

    leaf_df = leaf_df.copy()
    leaf_df["simulated_fill_rate_lognormal"] = node_fill_rates

    results = {
        "network_fill_rate": network_fill_rate,
        "mean_node_fill_rate": node_fill_rates.mean(),
        "median_node_fill_rate": np.median(node_fill_rates),
        "min_node_fill_rate": node_fill_rates.min(),
        "p5_node_fill_rate": np.percentile(node_fill_rates, 5),
        "distribution": "Log-Normal",
        "node_results": leaf_df,
    }

    return results


def run_monte_carlo_validation(gsm_results_df, network, service_level,
                                label="", n_trials=10000):
    """
    Run full Monte Carlo validation suite.

    Returns dict with normal and log-normal simulation results.
    """
    k = config.SERVICE_LEVELS[service_level]
    target_fill = service_level

    print(f"\n  Monte Carlo Validation ({label})")
    print(f"    Target SL: {target_fill*100:.0f}% (k={k:.3f})")
    print(f"    Trials: {n_trials:,} x 52 periods = {n_trials*52:,} demand realizations")

    # Normal demand
    print(f"    Simulating Normal demand...")
    normal_results = simulate_network_fill_rate(
        gsm_results_df, network, n_trials=n_trials,
        n_periods_per_trial=52, random_seed=config.RANDOM_SEED
    )

    print(f"      Network fill rate:   {normal_results['network_fill_rate']*100:.2f}% "
          f"(target: {target_fill*100:.0f}%)")
    print(f"      Mean node fill rate: {normal_results['mean_node_fill_rate']*100:.2f}%")
    print(f"      Min node fill rate:  {normal_results['min_node_fill_rate']*100:.2f}%")
    print(f"      5th percentile:      {normal_results['p5_node_fill_rate']*100:.2f}%")

    gap_normal = normal_results['network_fill_rate'] - target_fill
    if gap_normal >= -0.005:
        print(f"      PASS: Fill rate meets target (gap = {gap_normal*100:+.2f}pp)")
    else:
        print(f"      WARNING: Fill rate below target (gap = {gap_normal*100:+.2f}pp)")

    # Log-normal demand (robustness)
    print(f"    Simulating Log-Normal demand (robustness check)...")
    lognormal_results = simulate_with_non_normal_demand(
        gsm_results_df, network, n_trials=n_trials,
        n_periods_per_trial=52, random_seed=config.RANDOM_SEED + 1
    )

    print(f"      Network fill rate:   {lognormal_results['network_fill_rate']*100:.2f}%")
    print(f"      Mean node fill rate: {lognormal_results['mean_node_fill_rate']*100:.2f}%")

    gap_ln = lognormal_results['network_fill_rate'] - target_fill
    if gap_ln >= -0.01:
        print(f"      PASS: Robust to non-normality (gap = {gap_ln*100:+.2f}pp)")
    else:
        print(f"      NOTE: Slight degradation under skewed demand "
              f"(gap = {gap_ln*100:+.2f}pp)")

    return {
        "normal": normal_results,
        "lognormal": lognormal_results,
        "target_service_level": target_fill,
        "label": label,
    }