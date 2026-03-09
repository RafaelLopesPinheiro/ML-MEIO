"""
optimization/stochastic_milp.py
================================
Two-Stage Stochastic MILP for Multi-Echelon Inventory Optimization (MEIO).

Model Description
-----------------
We consider a two-echelon supply chain:
  - Echelon 1 (Upstream)  : Distribution Center (DC), node index 0.
  - Echelon 2 (Downstream): N retail stores, node indices 1..N.

Lead times are fixed: L_dc (supplier→DC), L_s (DC→store).
Review period: T (periodic review).

Decision Variables
------------------
First Stage (here-and-now, before demand realisation):
  - q_0     : replenishment order quantity at DC
  - q_i     : replenishment order quantity at store i
  - ss_0    : safety stock level at DC
  - ss_i    : safety stock level at store i
  - y_0, y_i: binary order placement indicators

Second Stage (recourse, for each scenario s):
  - inv_0^s : ending inventory at DC in scenario s
  - inv_i^s : ending inventory at store i in scenario s
  - bo_i^s  : backorders at store i in scenario s
  - e_i^s   : emergency replenishment at store i in scenario s (recourse)

Objective
---------
  min   Σ_i c_f * y_i + c_v * q_i              [replenishment cost]
      + Σ_i h_s * ss_i + h_dc * ss_0           [holding cost on SS]
      + E_s [ Σ_i (h_s * inv_i^s + b * bo_i^s + c_e * e_i^s) ]

Subject to:
  - Inventory balance equations (per echelon, per scenario)
  - Service level constraint: P(bo_i = 0) ≥ β  (enforced via scenarios)
  - Non-negativity, binary, and capacity constraints

Note: For tractability, we solve a deterministic equivalent (DE) MILP,
      which explicitly enumerates all scenarios.

References
----------
Birge, J. R., & Louveaux, F. (2011). Introduction to Stochastic
  Programming (2nd ed.). Springer.
Shapiro, A., Dentcheva, D., & Ruszczyński, A. (2021). Lectures on
  Stochastic Programming: Modeling and Theory (3rd ed.). SIAM.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    logging.warning("PuLP not installed. Run: pip install pulp")

from .scenario_generator import ScenarioSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MEIOSolution:
    """Container for the MEIO optimization solution."""
    status: str
    objective_value: float

    # First-stage decisions
    order_quantity_dc: float = 0.0
    order_quantity_stores: np.ndarray = field(default_factory=lambda: np.array([]))
    safety_stock_dc: float = 0.0
    safety_stock_stores: np.ndarray = field(default_factory=lambda: np.array([]))
    reorder_point_dc: float = 0.0
    reorder_point_stores: np.ndarray = field(default_factory=lambda: np.array([]))
    order_placed_dc: bool = False
    order_placed_stores: np.ndarray = field(default_factory=lambda: np.array([]))

    # Cost decomposition
    replenishment_cost: float = 0.0
    holding_cost: float = 0.0
    backorder_cost: float = 0.0
    emergency_cost: float = 0.0

    # Second-stage expected values
    expected_inventory_stores: np.ndarray = field(default_factory=lambda: np.array([]))
    expected_backorders_stores: np.ndarray = field(default_factory=lambda: np.array([]))
    fill_rate: float = 0.0

    solve_time_sec: float = 0.0
    solver_log: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "objective_value": self.objective_value,
            "order_quantity_dc": self.order_quantity_dc,
            "order_quantity_stores": self.order_quantity_stores.tolist(),
            "safety_stock_dc": self.safety_stock_dc,
            "safety_stock_stores": self.safety_stock_stores.tolist(),
            "reorder_point_dc": self.reorder_point_dc,
            "reorder_point_stores": self.reorder_point_stores.tolist(),
            "replenishment_cost": self.replenishment_cost,
            "holding_cost": self.holding_cost,
            "backorder_cost": self.backorder_cost,
            "emergency_cost": self.emergency_cost,
            "fill_rate": self.fill_rate,
            "solve_time_sec": self.solve_time_sec,
        }


# ---------------------------------------------------------------------------
# Two-Stage Stochastic MILP
# ---------------------------------------------------------------------------

class TwoStageStochasticMEIO:
    """
    Two-stage stochastic MILP for multi-echelon inventory optimization.

    Parameters
    ----------
    n_stores : int
        Number of downstream store nodes.
    cfg : dict
        Optimization configuration section from config.yaml.
    """

    def __init__(self, n_stores: int, cfg: dict) -> None:
        if not PULP_AVAILABLE:
            raise ImportError("PuLP is required: pip install pulp")
        self.n_stores = n_stores
        self.cfg = cfg
        self._sol: Optional[MEIOSolution] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        dc_scenarios: ScenarioSet,
        store_scenarios: ScenarioSet,
        initial_inventory_dc: float = 0.0,
        initial_inventory_stores: Optional[np.ndarray] = None,
    ) -> MEIOSolution:
        """
        Build and solve the deterministic equivalent of the two-stage
        stochastic MILP.

        Parameters
        ----------
        dc_scenarios : ScenarioSet
            Demand scenarios for the DC (1 node, n_scenarios scenarios).
        store_scenarios : ScenarioSet
            Demand scenarios for all stores (n_stores nodes, n_scenarios scenarios).
        initial_inventory_dc : float
            Current on-hand inventory at DC.
        initial_inventory_stores : np.ndarray, optional
            Current on-hand inventory at each store. Defaults to zeros.

        Returns
        -------
        MEIOSolution
        """
        if initial_inventory_stores is None:
            initial_inventory_stores = np.zeros(self.n_stores)

        n_s = store_scenarios.n_scenarios
        probs = store_scenarios.probabilities  # (n_s,)

        # Demand matrices: D_dc[s], D_store[i, s]
        D_dc = dc_scenarios.scenarios[0]  # (n_s,)
        D_store = store_scenarios.scenarios  # (n_stores, n_s)

        t0 = time.perf_counter()
        prob = pulp.LpProblem("TwoStageStochasticMEIO", pulp.LpMinimize)

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        h_s = self.cfg["holding_cost_store"]
        h_dc = self.cfg["holding_cost_dc"]
        b = self.cfg["backorder_cost"]
        c_f = self.cfg["replenishment_fixed_cost"]
        c_v = self.cfg["replenishment_variable_cost"]
        c_e = b * 1.5  # emergency replenishment = 1.5× backorder penalty
        L_dc = self.cfg["lead_time_supplier_to_dc"]  # weeks
        L_s = self.cfg["lead_time_dc_to_store"]  # weeks
        M = self.cfg.get("max_order_multiplier", 5.0)
        beta = self.cfg["service_level"]  # target fill rate

        # Upper bound on orders (big-M based on mean demand)
        mu_dc = float(D_dc.mean())
        mu_store = D_store.mean(axis=1)  # (n_stores,)
        Q_dc_max = max(M * (mu_dc * (L_dc + 1)), 1.0)
        Q_store_max = np.maximum(M * mu_store * (L_s + 1), 1.0)

        # ------------------------------------------------------------------
        # First-stage variables
        # ------------------------------------------------------------------

        # Order quantities
        q0 = pulp.LpVariable("q_dc", lowBound=0, upBound=Q_dc_max)
        q = [
            pulp.LpVariable(f"q_store_{i}", lowBound=0, upBound=float(Q_store_max[i]))
            for i in range(self.n_stores)
        ]

        # Safety stocks
        ss0 = pulp.LpVariable("ss_dc", lowBound=0, upBound=Q_dc_max)
        ss = [
            pulp.LpVariable(f"ss_store_{i}", lowBound=0, upBound=float(Q_store_max[i]))
            for i in range(self.n_stores)
        ]

        # Reorder point (= safety stock + expected demand during lead time)
        # Modelled as derived quantity in constraints below.

        # Binary: was an order placed?
        y0 = pulp.LpVariable("y_dc", cat="Binary")
        y = [pulp.LpVariable(f"y_store_{i}", cat="Binary") for i in range(self.n_stores)]

        # ------------------------------------------------------------------
        # Second-stage variables (per scenario)
        # ------------------------------------------------------------------
        inv0 = {}  # DC ending inventory per scenario
        inv = {}   # Store ending inventory per scenario
        bo = {}    # Store backorders per scenario
        e_rep = {} # Emergency replenishment per scenario

        for s in range(n_s):
            inv0[s] = pulp.LpVariable(f"inv_dc_{s}", lowBound=0)
            for i in range(self.n_stores):
                inv[i, s] = pulp.LpVariable(f"inv_store_{i}_{s}", lowBound=0)
                bo[i, s] = pulp.LpVariable(f"bo_store_{i}_{s}", lowBound=0)
                e_rep[i, s] = pulp.LpVariable(f"emerg_{i}_{s}", lowBound=0)

        # ------------------------------------------------------------------
        # Objective
        # ------------------------------------------------------------------
        # First-stage replenishment cost
        fs_cost = (
            c_f * y0 + c_v * q0
            + pulp.lpSum(c_f * y[i] + c_v * q[i] for i in range(self.n_stores))
        )
        # First-stage holding cost on safety stocks
        fs_hold = h_dc * ss0 + pulp.lpSum(h_s * ss[i] for i in range(self.n_stores))

        # Second-stage expected cost
        ss_cost = pulp.lpSum(
            probs[s] * (
                h_dc * inv0[s]
                + pulp.lpSum(
                    h_s * inv[i, s] + b * bo[i, s] + c_e * e_rep[i, s]
                    for i in range(self.n_stores)
                )
            )
            for s in range(n_s)
        )

        prob += fs_cost + fs_hold + ss_cost, "Total_Expected_Cost"

        # ------------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------------

        # (C1) Big-M linking: order only if y=1
        prob += q0 <= Q_dc_max * y0, "DC_BigM_order"
        for i in range(self.n_stores):
            prob += q[i] <= float(Q_store_max[i]) * y[i], f"Store_{i}_BigM_order"

        # (C2) Inventory balance per scenario — DC
        # Inventory at DC after replenishment and allocation to stores:
        # inv_dc^s = I0_dc + q0 - Σ_i q[i] - D_dc[s] (demand not met from stores)
        # We use a simplified single-period balance:
        #   inv_dc^s = I0_dc + q0 - Σ_i q[i]
        # The DC must ship to all stores; check feasibility.
        for s in range(n_s):
            prob += (
                inv0[s] == initial_inventory_dc + q0 - pulp.lpSum(q[i] for i in range(self.n_stores)),
                f"DC_balance_{s}",
            )
            prob += inv0[s] >= ss0, f"DC_SS_cover_{s}"

        # (C3) Inventory balance per scenario — Store i
        # inv_i^s + bo_i^s = I0_i + q_i + e_rep_i^s - D_i^s
        for i in range(self.n_stores):
            for s in range(n_s):
                prob += (
                    inv[i, s] - bo[i, s] + float(D_store[i, s])
                    == initial_inventory_stores[i] + q[i] + e_rep[i, s],
                    f"Store_{i}_balance_{s}",
                )
                # Safety stock lower bound
                prob += inv[i, s] >= ss[i], f"Store_{i}_SS_cover_{s}"

        # (C4) Service-level constraint via scenario enumeration
        # Fill rate ≥ β:  E[fulfilled / demand] ≥ β
        # Approximated as: Σ_s p_s * (D_i^s - bo_i^s) / D_i^s ≥ β for each i
        # Re-expressed linearly:
        # Σ_s p_s * bo_i^s ≤ (1-β) * Σ_s p_s * D_i^s
        for i in range(self.n_stores):
            expected_demand_i = float(np.dot(probs, D_store[i]))
            if expected_demand_i > 0:
                prob += (
                    pulp.lpSum(probs[s] * bo[i, s] for s in range(n_s))
                    <= (1 - beta) * expected_demand_i,
                    f"ServiceLevel_store_{i}",
                )

        # (C5) DC must have enough stock to supply stores + safety stock
        prob += (
            initial_inventory_dc + q0
            >= pulp.lpSum(q[i] for i in range(self.n_stores)) + ss0,
            "DC_feasibility",
        )

        # ------------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------------
        solver_name = self.cfg.get("solver", "CBC").upper()
        if solver_name == "CBC":
            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=300)
        elif solver_name == "GLPK":
            solver = pulp.GLPK_CMD(msg=0)
        else:
            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=300)

        logger.info(
            "Solving Two-Stage Stochastic MILP: %d stores × %d scenarios "
            "(solver=%s)...",
            self.n_stores,
            n_s,
            solver_name,
        )
        prob.solve(solver)
        t_elapsed = time.perf_counter() - t0

        status = pulp.LpStatus[prob.status]
        logger.info("MILP status: %s | Solve time: %.2f s", status, t_elapsed)

        # ------------------------------------------------------------------
        # Extract solution
        # ------------------------------------------------------------------
        obj_val = pulp.value(prob.objective) or 0.0

        q0_val = max(pulp.value(q0) or 0.0, 0.0)
        q_val = np.array([max(pulp.value(q[i]) or 0.0, 0.0) for i in range(self.n_stores)])
        ss0_val = max(pulp.value(ss0) or 0.0, 0.0)
        ss_val = np.array([max(pulp.value(ss[i]) or 0.0, 0.0) for i in range(self.n_stores)])
        y0_val = bool(round(pulp.value(y0) or 0.0))
        y_val = np.array([bool(round(pulp.value(y[i]) or 0.0)) for i in range(self.n_stores)])

        # Reorder point = safety stock + mean demand during lead time
        rop_dc = ss0_val + mu_dc * L_dc
        rop_stores = ss_val + mu_store * L_s

        # Expected second-stage values
        exp_inv = np.array([
            sum(probs[s] * (pulp.value(inv[i, s]) or 0.0) for s in range(n_s))
            for i in range(self.n_stores)
        ])
        exp_bo = np.array([
            sum(probs[s] * (pulp.value(bo[i, s]) or 0.0) for s in range(n_s))
            for i in range(self.n_stores)
        ])

        # Cost decomposition
        rc = (c_f * y0_val + c_v * q0_val
              + sum(c_f * y_val[i] + c_v * q_val[i] for i in range(self.n_stores)))
        hc = h_dc * ss0_val + sum(h_s * ss_val[i] for i in range(self.n_stores))
        bc = float(b * np.dot(probs, [
            sum(pulp.value(bo[i, s]) or 0.0 for i in range(self.n_stores))
            for s in range(n_s)
        ]))
        ec = float(c_e * np.dot(probs, [
            sum(pulp.value(e_rep[i, s]) or 0.0 for i in range(self.n_stores))
            for s in range(n_s)
        ]))

        # Fill rate
        total_demand = np.array([np.dot(probs, D_store[i]) for i in range(self.n_stores)])
        fill_rate = float(
            np.mean(
                np.where(total_demand > 0, 1 - exp_bo / np.maximum(total_demand, 1e-9), 1.0)
            )
        )

        self._sol = MEIOSolution(
            status=status,
            objective_value=obj_val,
            order_quantity_dc=q0_val,
            order_quantity_stores=q_val,
            safety_stock_dc=ss0_val,
            safety_stock_stores=ss_val,
            reorder_point_dc=rop_dc,
            reorder_point_stores=rop_stores,
            order_placed_dc=y0_val,
            order_placed_stores=y_val,
            replenishment_cost=rc,
            holding_cost=hc,
            backorder_cost=bc,
            emergency_cost=ec,
            expected_inventory_stores=exp_inv,
            expected_backorders_stores=exp_bo,
            fill_rate=fill_rate,
            solve_time_sec=t_elapsed,
        )
        return self._sol

    @property
    def solution(self) -> Optional[MEIOSolution]:
        return self._sol