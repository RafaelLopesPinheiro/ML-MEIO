"""
09_seio_baseline.py — Single-Echelon Inventory Optimization (SEIO) Baseline.

Implements the SEIO approach where each node independently sets its own
safety stock, ignoring the multi-echelon structure. This serves as the
baseline to quantify the VALUE OF COORDINATION from the GSM/MEIO approach.

METHODOLOGY:
  In SEIO, each node j independently computes:
    SS_j = k * sigma_j * sqrt(L_j + L_upstream_avg)

  where L_upstream_avg is the AVERAGE total upstream lead time that node j
  would experience if it ordered independently (no service time guarantees).

  This is the standard (R, Q) or base-stock approach used in practice when
  echelons are managed independently — e.g., each store sets its own
  reorder point based on its own lead time from the warehouse, without
  knowing or coordinating with the DC's inventory decisions.

  The key difference from MEIO/GSM:
    - MEIO: tau_j = SI_j + L_j - S_j (optimized service times reduce tau)
    - SEIO: tau_j = full_lead_time_j (no service time optimization)

  SEIO ALWAYS results in MORE total safety stock than MEIO because:
    1. No service time coordination (each node buffers against full lead time)
    2. No strategic placement (can't shift stock to cheaper echelons)
    3. Double-counting of variability (no risk pooling optimization)

References:
  - Axsater (2015), "Inventory Control", Chapter 7
  - Simchi-Levi & Zhao (2012), "Performance Evaluation of Stochastic
    Multi-Echelon Inventory Systems: A Survey"
  - de Kok et al. (2018), "A typology and literature review on stochastic
    multi-echelon inventory models", EJOR 269(3), 955-983
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config


def compute_seio_safety_stocks(network, service_level=config.DEFAULT_SERVICE_LEVEL):
    """
    Compute safety stocks under single-echelon independent optimization.

    Each node independently buffers against its FULL effective lead time,
    which includes its own processing time plus all upstream lead times
    (since no upstream node guarantees a service time).

    For a node at echelon 3 (store-dept):
      Effective lead time = L_store + L_warehouse + L_dc + L_external
                          = 1 + 1 + 2 + 2 = 6 weeks

    For a node at echelon 2 (warehouse):
      Effective lead time = L_warehouse + L_dc + L_external
                          = 1 + 2 + 2 = 5 weeks

    For a node at echelon 1 (DC):
      Effective lead time = L_dc + L_external
                          = 2 + 2 = 4 weeks

    Safety stock: SS_j = k * sigma_j * sqrt(effective_LT_j)

    Parameters:
        network: SupplyChainNetwork object
        service_level: Target service level

    Returns:
        DataFrame with SEIO safety stock results per node
    """
    k = config.SERVICE_LEVELS[service_level]
    external_lt = config.EXTERNAL_LEAD_TIME

    records = []

    for nid, node in network.nodes.items():
        if node.node_type == "supply":
            continue

        # Compute effective lead time by tracing upstream
        effective_lt = _compute_effective_lead_time(nid, network, external_lt)

        # SEIO safety stock
        safety_stock = k * node.sigma_demand * np.sqrt(effective_lt)
        holding_cost = node.holding_cost * safety_stock

        records.append({
            "node_id": nid,
            "echelon": node.echelon,
            "type": node.node_type,
            "mu": node.mu_demand,
            "sigma": node.sigma_demand,
            "effective_lead_time": effective_lt,
            "safety_stock": safety_stock,
            "holding_cost_weekly": holding_cost,
            "holding_cost_annual": holding_cost * 52,
        })

    return pd.DataFrame(records)


def _compute_effective_lead_time(node_id, network, external_lt):
    """
    Compute the total effective lead time for a node in SEIO.

    Traces from the node upward to the root, summing all lead times.
    This represents the total time a replenishment order takes to
    traverse the entire upstream chain — since no echelon guarantees
    a service time, each node must buffer against the full pipeline.
    """
    total_lt = 0
    current_id = node_id

    while current_id and current_id in network.nodes:
        node = network.nodes[current_id]
        if node.node_type == "supply":
            total_lt += external_lt  # Add external supplier lead time
            break
        total_lt += node.lead_time
        current_id = node.parent_id

    return total_lt


def compare_seio_vs_meio(seio_df, meio_df, label=""):
    """
    Compare SEIO and MEIO results side by side.

    Computes the "value of coordination" — how much the multi-echelon
    optimization saves compared to independent per-node optimization.

    Parameters:
        seio_df: DataFrame from compute_seio_safety_stocks
        meio_df: DataFrame from GSM optimization (results_to_dataframe)
        label: Label for printing

    Returns:
        Dict with comparison metrics
    """
    # Aggregate by echelon
    comparison = {}

    for echelon in sorted(meio_df["echelon"].unique()):
        meio_ech = meio_df[meio_df["echelon"] == echelon]
        seio_ech = seio_df[seio_df["echelon"] == echelon]

        meio_ss = meio_ech["safety_stock"].sum()
        seio_ss = seio_ech["safety_stock"].sum()
        meio_cost = meio_ech["holding_cost_weekly"].sum()
        seio_cost = seio_ech["holding_cost_weekly"].sum()

        etype = meio_ech["type"].iloc[0] if len(meio_ech) > 0 else "unknown"

        comparison[f"echelon_{echelon}"] = {
            "type": etype,
            "n_nodes": len(meio_ech),
            "seio_ss": seio_ss,
            "meio_ss": meio_ss,
            "ss_reduction_pct": (1 - meio_ss / seio_ss) * 100 if seio_ss > 0 else 0,
            "seio_cost": seio_cost,
            "meio_cost": meio_cost,
            "cost_reduction_pct": (1 - meio_cost / seio_cost) * 100 if seio_cost > 0 else 0,
        }

    # Network totals
    seio_total_ss = seio_df["safety_stock"].sum()
    meio_total_ss = meio_df["safety_stock"].sum()
    seio_total_cost = seio_df["holding_cost_weekly"].sum()
    meio_total_cost = meio_df["holding_cost_weekly"].sum()

    comparison["total"] = {
        "type": "network",
        "n_nodes": len(meio_df),
        "seio_ss": seio_total_ss,
        "meio_ss": meio_total_ss,
        "ss_reduction_pct": (1 - meio_total_ss / seio_total_ss) * 100,
        "seio_cost": seio_total_cost,
        "meio_cost": meio_total_cost,
        "cost_reduction_pct": (1 - meio_total_cost / seio_total_cost) * 100,
    }

    return comparison


def run_seio_analysis(network, meio_results_df,
                      service_level=config.DEFAULT_SERVICE_LEVEL, label=""):
    """
    Run the full SEIO analysis and comparison.

    Parameters:
        network: SupplyChainNetwork object
        meio_results_df: DataFrame from GSM optimization
        service_level: Target service level
        label: Label for output

    Returns:
        Tuple (seio_df, comparison_dict)
    """
    k = config.SERVICE_LEVELS[service_level]

    print(f"\n  SEIO Baseline Analysis ({label})")
    print(f"    Service level: {service_level*100:.0f}% (k={k:.3f})")

    # Compute SEIO safety stocks
    seio_df = compute_seio_safety_stocks(network, service_level)

    # Compare with MEIO
    comparison = compare_seio_vs_meio(seio_df, meio_results_df, label)

    # Print results
    print(f"\n    {'Echelon':<12} {'SEIO Cost':>14} {'MEIO Cost':>14} "
          f"{'Reduction':>10} {'Avg Eff. LT':>12}")
    print(f"    {'-'*64}")

    for key, val in comparison.items():
        if key == "total":
            continue
        echelon = int(key.split("_")[1])
        ech_seio = seio_df[seio_df["echelon"] == echelon]
        avg_lt = ech_seio["effective_lead_time"].mean()
        print(f"    {val['type']:<12} {val['seio_cost']:>14,.2f} "
              f"{val['meio_cost']:>14,.2f} "
              f"{val['cost_reduction_pct']:>9.1f}% {avg_lt:>12.1f}")

    total = comparison["total"]
    print(f"    {'-'*64}")
    print(f"    {'TOTAL':<12} {total['seio_cost']:>14,.2f} "
          f"{total['meio_cost']:>14,.2f} "
          f"{total['cost_reduction_pct']:>9.1f}%")

    print(f"\n    Value of Multi-Echelon Coordination:")
    print(f"      SEIO total cost/wk:  ${total['seio_cost']:>14,.2f}")
    print(f"      MEIO total cost/wk:  ${total['meio_cost']:>14,.2f}")
    print(f"      Savings from MEIO:   ${total['seio_cost'] - total['meio_cost']:>14,.2f} "
          f"({total['cost_reduction_pct']:.1f}%)")

    return seio_df, comparison