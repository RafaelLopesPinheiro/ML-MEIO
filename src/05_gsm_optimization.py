"""
05_gsm_optimization.py — Guaranteed Service Model (GSM) Solver.

Implements the Graves & Willems (2000) GSM for spanning-tree networks.

FIX: The virtual SUPPLY root has L=0, max_S=0, h=0.
  Its only role is to set SI for the DCs (SI_dc = S_supply = 0).
  So DCs get SI = 0 and their tau = 0 + L_dc - S_dc.
  With external_lead_time absorbed into DC lead time directly.

  Actually, we set SUPPLY.lead_time = external_lead_time so that
  tau_SUPPLY = SI_ext + 0 - 0 = 0 but the DCs get SI = S_SUPPLY.
  Since SUPPLY max_S = 0, DCs get SI = 0.
  To model external lead time, we add it to the DC's lead time.

  ALTERNATIVE (cleaner): Set SUPPLY lead_time = external_lead_time,
  max_S = external_lead_time (it can promise up to that).
  This way the DP naturally propagates the external lead time.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass

import config
from config import SERVICE_LEVELS


@dataclass
class GSMResult:
    """Container for GSM optimization results at a single node."""
    node_id: str
    SI: float
    S: float
    tau: float
    safety_stock: float
    holding_cost: float


class GSMSolver:
    """
    GSM solver for spanning-tree networks via dynamic programming.

    The SUPPLY root node has:
      - SI = external_lead_time (from external supplier)
      - L = 0 (no processing)
      - max_S = external_lead_time (can pass service time downstream)
      - h = 0 (no holding cost)

    This correctly propagates the external lead time into the tree
    without needing special-case logic.
    """

    def __init__(self, service_level=config.DEFAULT_SERVICE_LEVEL,
                 external_lead_time=config.EXTERNAL_LEAD_TIME):
        self.service_level = service_level
        self.k = SERVICE_LEVELS[service_level]
        self.external_lead_time = external_lead_time

    def solve(self, network):
        """Solve the GSM. Returns dict node_id -> GSMResult."""
        nodes = network.nodes
        subtree_memo = {}

        # Override SUPPLY node parameters for correct DP behavior
        if "SUPPLY" in nodes:
            nodes["SUPPLY"].lead_time = 0
            nodes["SUPPLY"].holding_cost = 0
            nodes["SUPPLY"].max_service_time = self.external_lead_time
            nodes["SUPPLY"].sigma_demand = 0  # No cost at supply

        def subtree_cost(node_id, si_j):
            """
            Compute minimum cost of the subtree rooted at node_id,
            given that node_id receives inbound service time si_j.

            Returns (cost, optimal_S, child_decisions_dict)
            """
            if (node_id, si_j) in subtree_memo:
                return subtree_memo[(node_id, si_j)]

            node = nodes[node_id]

            if not node.children:
                # LEAF NODE
                best_cost = np.inf
                best_s = 0
                for s_j in range(node.max_service_time + 1):
                    tau = si_j + node.lead_time - s_j
                    if tau < 0:
                        continue
                    cost = node.holding_cost * self.k * node.sigma_demand * np.sqrt(tau)
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s_j
                result = (best_cost, best_s, {})
                subtree_memo[(node_id, si_j)] = result
                return result

            # INTERNAL NODE
            best_total = np.inf
            best_s_j = 0
            best_child_dec = {}

            for s_j in range(node.max_service_time + 1):
                tau = si_j + node.lead_time - s_j
                if tau < 0:
                    continue

                # This node's local holding cost
                local = node.holding_cost * self.k * node.sigma_demand * np.sqrt(tau)

                # Children costs (each child gets SI = s_j)
                child_total = 0
                child_dec = {}
                feasible = True
                for child_id in node.children:
                    c_cost, c_s, c_children = subtree_cost(child_id, s_j)
                    if c_cost == np.inf:
                        feasible = False
                        break
                    child_total += c_cost
                    child_dec[child_id] = (c_s, c_children)

                if not feasible:
                    continue

                total = local + child_total
                if total < best_total:
                    best_total = total
                    best_s_j = s_j
                    best_child_dec = child_dec

            result = (best_total, best_s_j, best_child_dec)
            subtree_memo[(node_id, si_j)] = result
            return result

        # Solve from the single root
        root_id = network.root_nodes[0]  # "SUPPLY"
        si_root = self.external_lead_time  # External supplier lead time
        total_cost, s_root, child_dec = subtree_cost(root_id, si_root)

        # Extract results top-down
        results = {}
        self._extract_results(root_id, si_root, s_root, child_dec, nodes, results)

        return results

    def _extract_results(self, node_id, si, s, child_decisions, nodes, results):
        """Recursively extract GSMResult from DP solution."""
        node = nodes[node_id]
        tau = si + node.lead_time - s
        safety_stock = self.k * node.sigma_demand * np.sqrt(max(tau, 0))
        holding_cost = node.holding_cost * safety_stock

        results[node_id] = GSMResult(
            node_id=node_id, SI=si, S=s, tau=tau,
            safety_stock=safety_stock, holding_cost=holding_cost,
        )

        for child_id, (child_s, child_children) in child_decisions.items():
            self._extract_results(child_id, s, child_s, child_children, nodes, results)


def results_to_dataframe(results, network):
    """Convert GSM results to a DataFrame (excluding SUPPLY root)."""
    records = []
    for nid, res in results.items():
        node = network.nodes[nid]
        if node.node_type == "supply":
            continue  # Skip virtual root in output
        records.append({
            "node_id": nid, "echelon": node.echelon, "type": node.node_type,
            "SI": res.SI, "S": res.S, "L": node.lead_time, "tau": res.tau,
            "sigma": node.sigma_demand, "mu": node.mu_demand,
            "safety_stock": res.safety_stock,
            "holding_cost_weekly": res.holding_cost,
            "holding_cost_annual": res.holding_cost * 52,
        })
    return pd.DataFrame(records)


def summarize_results(results_df, label=""):
    """Compute summary statistics from GSM results."""
    summary = {}
    prefix = f"{label}_" if label else ""

    for echelon in sorted(results_df["echelon"].unique()):
        edf = results_df[results_df["echelon"] == echelon]
        etype = edf["type"].iloc[0]
        summary[f"{prefix}echelon_{echelon}"] = {
            "type": etype, "n_nodes": len(edf),
            "total_safety_stock": edf["safety_stock"].sum(),
            "total_holding_cost_weekly": edf["holding_cost_weekly"].sum(),
            "avg_tau": edf["tau"].mean(),
            "avg_S": edf["S"].mean(), "avg_SI": edf["SI"].mean(),
        }

    summary[f"{prefix}total"] = {
        "type": "network", "n_nodes": len(results_df),
        "total_safety_stock": results_df["safety_stock"].sum(),
        "total_holding_cost_weekly": results_df["holding_cost_weekly"].sum(),
        "avg_tau": results_df["tau"].mean(),
        "avg_S": results_df["S"].mean(), "avg_SI": results_df["SI"].mean(),
    }
    return summary


def run_gsm_optimization(network, service_level=config.DEFAULT_SERVICE_LEVEL, label=""):
    """Run the GSM optimization on the given network."""
    header = f"STAGE 5: GSM OPTIMIZATION"
    if label:
        header += f" ({label})"

    print(f"\n{'=' * 70}")
    print(header)
    print(f"{'=' * 70}")
    print(f"  Service level: {service_level*100:.0f}% "
          f"(k = {SERVICE_LEVELS[service_level]:.3f})")
    print(f"  Network size: {len(network.nodes)} nodes")

    solver = GSMSolver(service_level=service_level,
                       external_lead_time=config.EXTERNAL_LEAD_TIME)
    results = solver.solve(network)

    results_df = results_to_dataframe(results, network)
    summary = summarize_results(results_df, label)

    print(f"\n  Optimization Results:")
    print(f"  {'Echelon':<12} {'Nodes':>6} {'Tot SS':>14} "
          f"{'Tot Cost/wk':>14} {'Avg tau':>8} {'Avg S':>8}")
    print(f"  {'-'*66}")

    for key, val in summary.items():
        print(f"  {val['type']:<12} {val['n_nodes']:>6} "
              f"{val['total_safety_stock']:>14,.1f} "
              f"{val['total_holding_cost_weekly']:>14,.2f} "
              f"{val['avg_tau']:>8.2f} {val['avg_S']:>8.2f}")

    fname = f"gsm_results_{label}.csv" if label else "gsm_results.csv"
    results_df.to_csv(config.RESULTS_DIR / fname, index=False)
    print(f"\n  Results saved: {fname}")

    return results, results_df