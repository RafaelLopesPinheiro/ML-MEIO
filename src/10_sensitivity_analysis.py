"""
10_sensitivity_analysis.py — Parameter Sensitivity Analysis & PuLP MILP Solver.

Two components:

PART A: SENSITIVITY ANALYSIS
  Tests robustness of results to assumed (non-data-driven) parameters:
    - Lead times: L_dc, L_warehouse, L_store, L_external
    - Holding cost rates: h_dc, h_warehouse, h_store
  Uses a one-at-a-time (OAT) approach with multipliers {0.5, 0.75, 1.0, 1.25, 1.5}
  applied to each parameter group while others remain at baseline.
  Measures: total cost, cost reduction (Classical vs ML), and solution structure.

PART B: MILP REFORMULATION WITH PuLP
  Reformulates the GSM as a Mixed-Integer Linear Program following
  Magnanti et al. (2006) and Achkar et al. (2024).

  The nonlinearity in the GSM objective (sqrt(tau)) is handled via
  piecewise-linear approximation:
    sqrt(tau) ≈ sum_p  lambda_p * sqrt(breakpoint_p)
  where lambda_p are SOS2 convex combination weights.

  This allows solving with PuLP (CBC solver) for VERIFICATION of the
  DP solution and extensibility to non-tree topologies.

References:
  - Magnanti et al. (2006), "Inventory placement in acyclic supply chain
    networks", Operations Research Letters, 34(2), 228-238.
  - Achkar et al. (2024), "Extensions to the guaranteed service model for
    industrial applications of MEIO", EJOR 313(1), 192-206.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


# ============================================================================
# PART A: SENSITIVITY ANALYSIS
# ============================================================================

SENSITIVITY_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]


def run_sensitivity_lead_times(network_builder_func, gsm_runner_func,
                                residual_stats, df, sigma_type,
                                service_level):
    """
    Sensitivity analysis on lead time parameters.

    Varies lead times by multipliers while keeping holding costs fixed.
    Tests: L_external, L_dc, L_warehouse, L_store (each independently).
    """
    baseline_lt = {
        "external": config.EXTERNAL_LEAD_TIME,
        "dc": config.LEAD_TIMES["dc"],
        "warehouse": config.LEAD_TIMES["warehouse"],
        "store": config.LEAD_TIMES["store"],
    }

    rows = []
    for param_name, base_val in baseline_lt.items():
        for mult in SENSITIVITY_MULTIPLIERS:
            # Temporarily override config
            new_val = max(1, int(round(base_val * mult)))

            # Save originals
            orig_ext = config.EXTERNAL_LEAD_TIME
            orig_lt = dict(config.LEAD_TIMES)

            if param_name == "external":
                config.EXTERNAL_LEAD_TIME = new_val
            else:
                config.LEAD_TIMES[param_name] = new_val

            # Build and solve
            net = network_builder_func(residual_stats, df, sigma_type)
            _, res_df = gsm_runner_func(net, service_level,
                                         f"sens_lt_{param_name}_{mult}")

            total_cost = res_df["holding_cost_weekly"].sum()
            total_ss = res_df["safety_stock"].sum()

            rows.append({
                "parameter": f"L_{param_name}",
                "multiplier": mult,
                "value": new_val,
                "total_safety_stock": total_ss,
                "total_holding_cost": total_cost,
            })

            # Restore originals
            config.EXTERNAL_LEAD_TIME = orig_ext
            config.LEAD_TIMES.update(orig_lt)

    return pd.DataFrame(rows)


def run_sensitivity_holding_costs(network_builder_func, gsm_runner_func,
                                   residual_stats, df, sigma_type,
                                   service_level):
    """
    Sensitivity analysis on holding cost parameters.

    Varies holding cost rates by multipliers while keeping lead times fixed.
    Tests: h_dc, h_warehouse, h_store (each independently).
    Also tests the holding cost RATIO (h_store / h_dc) by varying h_store only.
    """
    baseline_hc = dict(config.HOLDING_COST_RATES)

    rows = []
    for param_name in ["dc", "warehouse", "store"]:
        base_val = baseline_hc[param_name]
        for mult in SENSITIVITY_MULTIPLIERS:
            new_val = base_val * mult

            # Save and override
            orig_hc = dict(config.HOLDING_COST_RATES)
            config.HOLDING_COST_RATES[param_name] = new_val

            net = network_builder_func(residual_stats, df, sigma_type)
            _, res_df = gsm_runner_func(net, service_level,
                                         f"sens_hc_{param_name}_{mult}")

            total_cost = res_df["holding_cost_weekly"].sum()
            total_ss = res_df["safety_stock"].sum()

            # Echelon breakdown
            store_cost = res_df[res_df["type"] == "store_dept"]["holding_cost_weekly"].sum()
            wh_cost = res_df[res_df["type"] == "warehouse"]["holding_cost_weekly"].sum()
            dc_cost = res_df[res_df["type"] == "dc"]["holding_cost_weekly"].sum()

            rows.append({
                "parameter": f"h_{param_name}",
                "multiplier": mult,
                "value": new_val,
                "total_safety_stock": total_ss,
                "total_holding_cost": total_cost,
                "store_cost_pct": store_cost / total_cost * 100 if total_cost > 0 else 0,
                "wh_cost_pct": wh_cost / total_cost * 100 if total_cost > 0 else 0,
                "dc_cost_pct": dc_cost / total_cost * 100 if total_cost > 0 else 0,
            })

            # Restore
            config.HOLDING_COST_RATES.update(orig_hc)

    return pd.DataFrame(rows)


def run_sensitivity_cost_reduction(network_builder_func, gsm_runner_func,
                                    classical_stats, ml_stats, df,
                                    service_level):
    """
    Test whether the ML cost reduction (%) is robust to parameter changes.

    For each lead time multiplier, compute both Classical and ML costs
    and check if the reduction percentage remains stable.
    """
    rows = []

    for lt_mult in SENSITIVITY_MULTIPLIERS:
        # Vary ALL lead times proportionally
        orig_ext = config.EXTERNAL_LEAD_TIME
        orig_lt = dict(config.LEAD_TIMES)

        config.EXTERNAL_LEAD_TIME = max(1, int(round(orig_ext * lt_mult)))
        for k in config.LEAD_TIMES:
            config.LEAD_TIMES[k] = max(1, int(round(orig_lt[k] * lt_mult)))

        # Classical
        net_c = network_builder_func(classical_stats, df, "sigma_historical")
        _, df_c = gsm_runner_func(net_c, service_level,
                                   f"sens_red_classical_{lt_mult}")
        cost_c = df_c["holding_cost_weekly"].sum()

        # ML
        net_m = network_builder_func(ml_stats, df, "sigma_residual")
        _, df_m = gsm_runner_func(net_m, service_level,
                                   f"sens_red_ml_{lt_mult}")
        cost_m = df_m["holding_cost_weekly"].sum()

        reduction = (1 - cost_m / cost_c) * 100 if cost_c > 0 else 0

        rows.append({
            "lead_time_multiplier": lt_mult,
            "classical_cost": cost_c,
            "ml_cost": cost_m,
            "cost_reduction_pct": reduction,
        })

        # Restore
        config.EXTERNAL_LEAD_TIME = orig_ext
        config.LEAD_TIMES.update(orig_lt)

    return pd.DataFrame(rows)


# ============================================================================
# PART B: MILP REFORMULATION WITH PuLP
# ============================================================================

def solve_gsm_milp(network, service_level=config.DEFAULT_SERVICE_LEVEL,
                    n_breakpoints=10, verbose=False):
    """
    Solve the GSM as a Mixed-Integer Linear Program using PuLP.

    FORMULATION:
      min  sum_j  h_j * k * sigma_j * PWL_sqrt(tau_j)

      s.t.
        tau_j = SI_j + L_j - S_j                    for all j
        tau_j >= 0                                    for all j
        0 <= S_j <= S_j^max                          for all j (integer)
        SI_j = S_parent(j)                           for all j with parent
        SI_root = external_lead_time

    PWL_sqrt approximation:
      sqrt(tau_j) ≈ sum_{p=0}^{P} lambda_{j,p} * sqrt(bp_p)
      tau_j       = sum_{p=0}^{P} lambda_{j,p} * bp_p
      sum lambda_{j,p} = 1
      lambda_{j,p} >= 0
      SOS2 constraint on lambda (only 2 adjacent can be nonzero)

    The SOS2 constraint is modeled with binary variables:
      z_{j,p} in {0,1}, sum z_{j,p} = 1
      lambda_{j,p} <= z_{j,p-1} + z_{j,p}  (adjacency)

    Parameters:
        network: SupplyChainNetwork
        service_level: Target service level
        n_breakpoints: Number of breakpoints for PWL approximation
        verbose: Print solver output

    Returns:
        (results_dict, objective_value, status)
    """
    if not PULP_AVAILABLE:
        print("    PuLP not available. Install with: pip install pulp")
        return None, None, "UNAVAILABLE"

    k = config.SERVICE_LEVELS[service_level]
    ext_lt = config.EXTERNAL_LEAD_TIME

    nodes = network.nodes
    node_ids = [nid for nid in nodes if nodes[nid].node_type != "supply"]

    # Determine max possible tau for breakpoints
    max_tau = ext_lt + sum(config.LEAD_TIMES.values()) + 2
    breakpoints = np.linspace(0, max_tau, n_breakpoints + 1)
    sqrt_bp = np.sqrt(breakpoints)

    # --- Create PuLP model ---
    model = pulp.LpProblem("GSM_MILP", pulp.LpMinimize)

    # Decision variables: S_j (integer outbound service times)
    S = {}
    for nid in node_ids:
        node = nodes[nid]
        S[nid] = pulp.LpVariable(f"S_{nid}", lowBound=0,
                                  upBound=node.max_service_time, cat="Integer")

    # Auxiliary: tau_j (continuous net replenishment time)
    tau = {}
    for nid in node_ids:
        tau[nid] = pulp.LpVariable(f"tau_{nid}", lowBound=0, cat="Continuous")

    # PWL variables: lambda_{j,p} and z_{j,p}
    lam = {}  # lambda[nid][p]
    z = {}    # z[nid][p] binary for SOS2
    for nid in node_ids:
        lam[nid] = {}
        z[nid] = {}
        for p in range(len(breakpoints)):
            lam[nid][p] = pulp.LpVariable(f"lam_{nid}_{p}", lowBound=0,
                                           upBound=1, cat="Continuous")
        for p in range(len(breakpoints) - 1):
            z[nid][p] = pulp.LpVariable(f"z_{nid}_{p}", cat="Binary")

    # Auxiliary: sqrt_tau_approx[j] = PWL approximation of sqrt(tau_j)
    sqrt_tau = {}
    for nid in node_ids:
        sqrt_tau[nid] = pulp.LpVariable(f"sqrt_tau_{nid}", lowBound=0,
                                          cat="Continuous")

    # --- OBJECTIVE ---
    model += pulp.lpSum(
        nodes[nid].holding_cost * k * nodes[nid].sigma_demand * sqrt_tau[nid]
        for nid in node_ids
    ), "Total_Holding_Cost"

    # --- CONSTRAINTS ---

    # 1. Tau definition: tau_j = SI_j + L_j - S_j
    for nid in node_ids:
        node = nodes[nid]
        if node.parent_id == "SUPPLY" or node.parent_id == "":
            # Root echelon: SI = external_lead_time
            si_j = ext_lt
        else:
            parent_id = node.parent_id
            if parent_id == "SUPPLY":
                # DCs: their SI = S_SUPPLY, but SUPPLY is virtual
                # S_SUPPLY is a decision variable too; max_S = ext_lt
                if "SUPPLY" not in S:
                    S["SUPPLY"] = pulp.LpVariable("S_SUPPLY", lowBound=0,
                                                    upBound=ext_lt, cat="Integer")
                si_j = S["SUPPLY"]
            else:
                si_j = S[parent_id]

        model += tau[nid] == si_j + node.lead_time - S[nid], f"tau_def_{nid}"

    # 2. SI for SUPPLY root: tau_SUPPLY is not optimized (virtual)
    # The SUPPLY node's S is bounded by ext_lt already

    # 3. PWL approximation constraints for each node
    for nid in node_ids:
        # tau_j = sum_p lambda_p * bp_p
        model += tau[nid] == pulp.lpSum(
            lam[nid][p] * breakpoints[p] for p in range(len(breakpoints))
        ), f"pwl_tau_{nid}"

        # sqrt_tau_j = sum_p lambda_p * sqrt(bp_p)
        model += sqrt_tau[nid] == pulp.lpSum(
            lam[nid][p] * sqrt_bp[p] for p in range(len(breakpoints))
        ), f"pwl_sqrt_{nid}"

        # sum lambda_p = 1 (convex combination)
        model += pulp.lpSum(
            lam[nid][p] for p in range(len(breakpoints))
        ) == 1, f"pwl_convex_{nid}"

        # SOS2 via binary z variables
        # sum z_p = 1 (exactly one segment active)
        model += pulp.lpSum(
            z[nid][p] for p in range(len(breakpoints) - 1)
        ) == 1, f"sos2_sum_{nid}"

        # lambda_0 <= z_0
        model += lam[nid][0] <= z[nid][0], f"sos2_lam0_{nid}"

        # lambda_P <= z_{P-1}
        P = len(breakpoints) - 1
        model += lam[nid][P] <= z[nid][P - 1], f"sos2_lamP_{nid}"

        # lambda_p <= z_{p-1} + z_p for 1 <= p <= P-1
        for p in range(1, P):
            model += lam[nid][p] <= z[nid][p - 1] + z[nid][p], \
                f"sos2_adj_{nid}_{p}"

    # --- SOLVE ---
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=120)
    status = model.solve(solver)
    status_str = pulp.LpStatus[status]

    if status_str != "Optimal":
        print(f"    PuLP solver status: {status_str}")
        return None, None, status_str

    # --- EXTRACT RESULTS ---
    # from src import config as _cfg  # avoid circular; already imported above

    obj_val = pulp.value(model.objective)

    results = {}
    for nid in node_ids:
        node = nodes[nid]
        s_val = int(round(S[nid].varValue)) if S[nid].varValue is not None else 0

        if node.parent_id == "SUPPLY":
            if "SUPPLY" in S and S["SUPPLY"].varValue is not None:
                si_val = int(round(S["SUPPLY"].varValue))
            else:
                si_val = ext_lt
        elif node.parent_id in S and S[node.parent_id].varValue is not None:
            si_val = int(round(S[node.parent_id].varValue))
        else:
            si_val = ext_lt

        tau_val = tau[nid].varValue if tau[nid].varValue is not None else 0
        ss = k * node.sigma_demand * np.sqrt(max(tau_val, 0))
        hc = node.holding_cost * ss

        #from src._05_gsm_optimization import GSMResult  # avoid; inline
        results[nid] = type('GSMResult', (), {
            'node_id': nid, 'SI': si_val, 'S': s_val,
            'tau': tau_val, 'safety_stock': ss, 'holding_cost': hc
        })()

    return results, obj_val, status_str


def solve_gsm_milp_simple(network, service_level=config.DEFAULT_SERVICE_LEVEL,
                           verbose=False):
    """
    Simplified PuLP MILP for GSM verification.

    Since service times are INTEGER and bounded by small values (0-3),
    we can enumerate all possible tau values directly instead of using
    PWL approximation. This is exact (no approximation error).

    For each node j, tau_j is integer in [0, max_tau_j].
    We create binary variables y_{j,t} = 1 if tau_j = t.
    Then: cost_j = h_j * k * sigma_j * sum_t y_{j,t} * sqrt(t)

    This linearizes the objective exactly for integer service times.
    """
    if not PULP_AVAILABLE:
        print("    PuLP not available. Install with: pip install pulp")
        return None, None, "UNAVAILABLE"

    k = config.SERVICE_LEVELS[service_level]
    ext_lt = config.EXTERNAL_LEAD_TIME

    nodes_dict = network.nodes
    node_ids = [nid for nid in nodes_dict if nodes_dict[nid].node_type != "supply"]

    # Max possible tau per node
    max_tau_global = ext_lt + sum(config.LEAD_TIMES.values()) + 2

    model = pulp.LpProblem("GSM_Exact_MILP", pulp.LpMinimize)

    # S_j: integer outbound service time
    S = {}
    for nid in node_ids:
        node = nodes_dict[nid]
        S[nid] = pulp.LpVariable(f"S_{nid}", lowBound=0,
                                  upBound=node.max_service_time, cat="Integer")

    # S for virtual SUPPLY
    S["SUPPLY"] = pulp.LpVariable("S_SUPPLY", lowBound=0,
                                    upBound=ext_lt, cat="Integer")

    # tau_j: continuous (but will be integer given integer S)
    tau = {}
    for nid in node_ids:
        tau[nid] = pulp.LpVariable(f"tau_{nid}", lowBound=0, cat="Continuous")

    # y_{j,t}: binary indicator for tau_j = t
    y = {}
    tau_values = list(range(max_tau_global + 1))
    for nid in node_ids:
        y[nid] = {}
        for t in tau_values:
            y[nid][t] = pulp.LpVariable(f"y_{nid}_{t}", cat="Binary")

    # --- OBJECTIVE: linearized via y indicators ---
    model += pulp.lpSum(
        nodes_dict[nid].holding_cost * k * nodes_dict[nid].sigma_demand
        * np.sqrt(t) * y[nid][t]
        for nid in node_ids
        for t in tau_values
    ), "Total_Holding_Cost"

    # --- CONSTRAINTS ---
    for nid in node_ids:
        node = nodes_dict[nid]

        # SI_j = S_parent(j)
        parent = node.parent_id if node.parent_id else "SUPPLY"
        if parent not in S:
            parent = "SUPPLY"

        # tau_j = SI_j + L_j - S_j
        model += tau[nid] == S[parent] + node.lead_time - S[nid], f"tau_def_{nid}"

        # tau_j = sum_t  t * y_{j,t}
        model += tau[nid] == pulp.lpSum(
            t * y[nid][t] for t in tau_values
        ), f"tau_ind_{nid}"

        # sum y_{j,t} = 1 (exactly one tau value)
        model += pulp.lpSum(
            y[nid][t] for t in tau_values
        ) == 1, f"tau_one_{nid}"

    # --- SOLVE ---
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=300)
    status = model.solve(solver)
    status_str = pulp.LpStatus[status]

    obj_val = pulp.value(model.objective) if status_str == "Optimal" else None

    # Extract results
    results_records = []
    if status_str == "Optimal":
        for nid in node_ids:
            node = nodes_dict[nid]
            s_val = int(round(S[nid].varValue)) if S[nid].varValue is not None else 0
            tau_val = tau[nid].varValue if tau[nid].varValue is not None else 0

            parent = node.parent_id if node.parent_id else "SUPPLY"
            if parent not in S:
                parent = "SUPPLY"
            si_val = int(round(S[parent].varValue)) if S[parent].varValue is not None else 0

            ss = k * node.sigma_demand * np.sqrt(max(tau_val, 0))
            hc = node.holding_cost * ss

            results_records.append({
                "node_id": nid, "echelon": node.echelon, "type": node.node_type,
                "SI": si_val, "S": s_val, "L": node.lead_time, "tau": tau_val,
                "sigma": node.sigma_demand, "mu": node.mu_demand,
                "safety_stock": ss, "holding_cost_weekly": hc,
                "holding_cost_annual": hc * 52,
            })

    results_df = pd.DataFrame(results_records) if results_records else pd.DataFrame()

    return results_df, obj_val, status_str


# ============================================================================
# COMBINED RUNNER
# ============================================================================

def run_sensitivity_analysis(network_builder_func, gsm_runner_func,
                              classical_stats, ml_stats, df,
                              service_level=config.DEFAULT_SERVICE_LEVEL):
    """
    Run the full sensitivity analysis suite.

    Returns dict with all sensitivity DataFrames.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)

    print("\n  --- 7a: Lead Time Sensitivity (ML-Enhanced) ---")
    lt_sens = run_sensitivity_lead_times(
        network_builder_func, gsm_runner_func,
        ml_stats, df, "sigma_residual", service_level
    )
    lt_sens.to_csv(config.RESULTS_DIR / "experiment7a_lt_sensitivity.csv", index=False)

    # Print lead time summary
    baseline_cost = lt_sens[lt_sens["multiplier"] == 1.0]["total_holding_cost"].mean()
    print(f"\n    Baseline cost: ${baseline_cost:,.2f}")
    print(f"    {'Parameter':<16} {'0.5x':>12} {'0.75x':>12} {'1.0x':>12} "
          f"{'1.25x':>12} {'1.5x':>12}")
    print(f"    {'-'*76}")
    for param in lt_sens["parameter"].unique():
        pdata = lt_sens[lt_sens["parameter"] == param]
        costs = [pdata[pdata["multiplier"] == m]["total_holding_cost"].iloc[0]
                 for m in SENSITIVITY_MULTIPLIERS]
        cost_strs = [f"${c:>10,.0f}" for c in costs]
        print(f"    {param:<16} {'  '.join(cost_strs)}")

    print("\n  --- 7b: Holding Cost Sensitivity (ML-Enhanced) ---")
    hc_sens = run_sensitivity_holding_costs(
        network_builder_func, gsm_runner_func,
        ml_stats, df, "sigma_residual", service_level
    )
    hc_sens.to_csv(config.RESULTS_DIR / "experiment7b_hc_sensitivity.csv", index=False)

    print(f"\n    {'Parameter':<16} {'0.5x':>12} {'0.75x':>12} {'1.0x':>12} "
          f"{'1.25x':>12} {'1.5x':>12}")
    print(f"    {'-'*76}")
    for param in hc_sens["parameter"].unique():
        pdata = hc_sens[hc_sens["parameter"] == param]
        costs = [pdata[pdata["multiplier"] == m]["total_holding_cost"].iloc[0]
                 for m in SENSITIVITY_MULTIPLIERS]
        cost_strs = [f"${c:>10,.0f}" for c in costs]
        print(f"    {param:<16} {'  '.join(cost_strs)}")

    print("\n  --- 7c: ML Cost Reduction Robustness ---")
    red_sens = run_sensitivity_cost_reduction(
        network_builder_func, gsm_runner_func,
        classical_stats, ml_stats, df, service_level
    )
    red_sens.to_csv(config.RESULTS_DIR / "experiment7c_reduction_robustness.csv", index=False)

    print(f"\n    Lead Time Multiplier | Classical Cost | ML Cost    | Reduction")
    print(f"    {'-'*65}")
    for _, row in red_sens.iterrows():
        print(f"    {row['lead_time_multiplier']:>20.2f}x | "
              f"${row['classical_cost']:>13,.2f} | "
              f"${row['ml_cost']:>9,.2f} | {row['cost_reduction_pct']:>8.1f}%")

    avg_red = red_sens["cost_reduction_pct"].mean()
    std_red = red_sens["cost_reduction_pct"].std()
    print(f"\n    ML cost reduction: {avg_red:.1f}% +/- {std_red:.1f}% "
          f"(across lead time scenarios)")
    if std_red < 2.0:
        print(f"    ROBUST: Reduction is stable across parameter variations")
    else:
        print(f"    NOTE: Some sensitivity to lead time parameters")

    # --- 7d: PuLP MILP Verification ---
    milp_result = None
    if PULP_AVAILABLE:
        print("\n  --- 7d: PuLP MILP Verification ---")
        print("    Solving GSM as exact MILP with PuLP/CBC...")

        net = network_builder_func(ml_stats, df, "sigma_residual")
        milp_df, milp_obj, milp_status = solve_gsm_milp_simple(
            net, service_level, verbose=False
        )

        print(f"    Solver status: {milp_status}")
        if milp_status == "Optimal" and milp_obj is not None:
            # Compare with DP solution
            _, dp_df = gsm_runner_func(net, service_level, "milp_verification_dp")
            dp_cost = dp_df["holding_cost_weekly"].sum()
            milp_cost = milp_df["holding_cost_weekly"].sum()
            gap = abs(milp_cost - dp_cost) / dp_cost * 100 if dp_cost > 0 else 0

            print(f"    DP   objective: ${dp_cost:>14,.2f}")
            print(f"    MILP objective: ${milp_cost:>14,.2f}")
            print(f"    Gap: {gap:.4f}%")

            if gap < 0.1:
                print(f"    VERIFIED: DP and MILP solutions match (gap < 0.1%)")
            else:
                print(f"    WARNING: Gap > 0.1% — check PWL resolution or solver tolerance")

            milp_result = {
                "dp_cost": dp_cost,
                "milp_cost": milp_cost,
                "gap_pct": gap,
                "status": milp_status,
                "milp_df": milp_df,
            }
    else:
        print("\n  --- 7d: PuLP MILP Verification (SKIPPED — pip install pulp) ---")

    return {
        "lt_sensitivity": lt_sens,
        "hc_sensitivity": hc_sens,
        "reduction_robustness": red_sens,
        "milp_verification": milp_result,
    }