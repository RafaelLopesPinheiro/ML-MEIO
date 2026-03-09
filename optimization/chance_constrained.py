"""
optimization/chance_constrained.py
=====================================
Chance-Constrained MEIO Optimization.

Instead of explicit scenario enumeration (as in the two-stage MILP),
this model directly encodes service-level requirements as probabilistic
(chance) constraints using the estimated demand quantiles.

Formulation
-----------
For each store i, the (1-α)-service level constraint is:
    P(Inventory_i ≥ Demand_i over lead time) ≥ 1 - α

Which is equivalent to:
    Reorder_Point_i ≥ Q_{α}(D_i^{LT})

where D_i^{LT} = Σ_{t=1}^{L} D_i^t is demand over the lead time L,
and Q_{α} is the α-quantile estimated by the distributional forecaster.

The chance-constrained LP minimizes holding + replenishment costs subject
to these quantile-based constraints. No binary variables are needed,
making this formulation more scalable than the full stochastic MILP.

Convex reformulation (when demand is sub-Gaussian or log-normal):
The chance constraint reduces to a Second-Order Cone constraint (SOCP).
We use the sample-based approximation instead for distribution-free validity.

References
----------
Charnes, A., & Cooper, W. W. (1959). Chance-constrained programming.
  Management Science, 6(1), 73–79.
Nemirovski, A., & Shapiro, A. (2006). Convex approximations of chance
  constrained programs. SIAM Journal on Optimization, 17(4), 969–996.
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

from .scenario_generator import ScenarioSet

logger = logging.getLogger(__name__)


@dataclass
class CCOSolution:
    """Solution container for the chance-constrained model."""
    status: str
    objective_value: float
    safety_stock_dc: float = 0.0
    safety_stock_stores: np.ndarray = field(default_factory=lambda: np.array([]))
    reorder_point_dc: float = 0.0
    reorder_point_stores: np.ndarray = field(default_factory=lambda: np.array([]))
    order_quantity_dc: float = 0.0
    order_quantity_stores: np.ndarray = field(default_factory=lambda: np.array([]))
    holding_cost: float = 0.0
    replenishment_cost: float = 0.0
    fill_rate: float = 0.0
    solve_time_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "objective_value": self.objective_value,
            "safety_stock_dc": self.safety_stock_dc,
            "safety_stock_stores": self.safety_stock_stores.tolist(),
            "reorder_point_dc": self.reorder_point_dc,
            "reorder_point_stores": self.reorder_point_stores.tolist(),
            "order_quantity_dc": self.order_quantity_dc,
            "order_quantity_stores": self.order_quantity_stores.tolist(),
            "holding_cost": self.holding_cost,
            "replenishment_cost": self.replenishment_cost,
            "fill_rate": self.fill_rate,
            "solve_time_sec": self.solve_time_sec,
        }


class ChanceConstrainedMEIO:
    """
    Chance-constrained LP for two-echelon inventory optimization.

    Uses empirical quantiles from demand distribution estimates to enforce
    service-level constraints without distributional assumptions.

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        dc_scenarios: ScenarioSet,
        store_scenarios: ScenarioSet,
    ) -> CCOSolution:
        """
        Solve the chance-constrained LP.

        Parameters
        ----------
        dc_scenarios : ScenarioSet
            DC-level demand scenarios (used to compute quantile bounds).
        store_scenarios : ScenarioSet
            Store-level demand scenarios (used to compute quantile bounds).

        Returns
        -------
        CCOSolution
        """
        alpha = 1 - self.cfg["service_level"]  # e.g., 0.05 for 95% SL
        L_s = self.cfg["lead_time_dc_to_store"]
        L_dc = self.cfg["lead_time_supplier_to_dc"]
        h_s = self.cfg["holding_cost_store"]
        h_dc = self.cfg["holding_cost_dc"]
        c_v = self.cfg["replenishment_variable_cost"]
        c_f = self.cfg["replenishment_fixed_cost"]
        M = self.cfg.get("max_order_multiplier", 5.0)

        # Demand quantiles over lead time
        # Under independence assumption: Q_α(D^{LT}) ≈ L * Q_α(D) [conservative]
        # Better: use the (1-α)-quantile of the empirical scenario sum over LT periods.
        # Here we use single-period scenarios × lead time as conservative proxy.

        q_alpha = 1 - alpha  # service level quantile
        store_demand_q = np.quantile(store_scenarios.scenarios, q_alpha, axis=1)  # (n_stores,)
        dc_demand_q = float(np.quantile(dc_scenarios.scenarios[0], q_alpha))

        # Lead-time demand quantiles (sum of L independent periods, use CLT-based scaling)
        store_lt_demand_q = store_demand_q * L_s
        dc_lt_demand_q = dc_demand_q * L_dc

        # Mean demands (for order quantity sizing)
        mu_store = store_scenarios.expected_demand()  # (n_stores,)
        mu_dc = float(dc_scenarios.expected_demand()[0])

        # Upper bounds
        Q_store_max = np.maximum(M * mu_store * (L_s + 1), 1.0)
        Q_dc_max = max(M * mu_dc * (L_dc + 1), 1.0)

        t0 = time.perf_counter()
        prob = pulp.LpProblem("ChanceConstrainedMEIO", pulp.LpMinimize)

        # ------------------------------------------------------------------
        # Decision variables
        # ------------------------------------------------------------------
        # Reorder points (= safety stock + mean LT demand)
        r0 = pulp.LpVariable("rop_dc", lowBound=0)
        r = [pulp.LpVariable(f"rop_store_{i}", lowBound=0) for i in range(self.n_stores)]

        # Order-up-to levels (base stock levels)
        S0 = pulp.LpVariable("S_dc", lowBound=0, upBound=Q_dc_max)
        S = [
            pulp.LpVariable(f"S_store_{i}", lowBound=0, upBound=float(Q_store_max[i]))
            for i in range(self.n_stores)
        ]

        # ------------------------------------------------------------------
        # Objective: holding + replenishment cost
        # Holding cost ≈ h * (S - μ_LT) = h * safety stock
        # Order quantity Q = S - ROP (when ROP is hit)
        # Expected replenishment cost per period ≈ c_v * μ (cycle stock)
        # ------------------------------------------------------------------
        hold_cost = h_dc * (S0 - dc_lt_demand_q) + pulp.lpSum(
            h_s * (S[i] - float(store_lt_demand_q[i])) for i in range(self.n_stores)
        )
        repl_cost = c_v * (mu_dc + pulp.lpSum(float(mu_store[i]) for i in range(self.n_stores)))

        prob += hold_cost + repl_cost, "Total_Cost"

        # ------------------------------------------------------------------
        # Chance constraints (reformulated via quantiles)
        # ------------------------------------------------------------------
        # P(inv_i ≥ 0) ≥ 1-α  ⟺  r_i ≥ Q_{1-α}(D_i^{LT})
        # (sufficient condition: set ROP ≥ quantile of lead-time demand)
        prob += r0 >= dc_lt_demand_q, "CC_DC_service"
        for i in range(self.n_stores):
            prob += r[i] >= float(store_lt_demand_q[i]), f"CC_Store_{i}_service"

        # Base-stock level ≥ ROP + mean demand per period (cycle stock)
        prob += S0 >= r0 + mu_dc, "DC_base_stock"
        for i in range(self.n_stores):
            prob += S[i] >= r[i] + float(mu_store[i]), f"Store_{i}_base_stock"

        # DC capacity: must cover all store replenishments
        prob += S0 >= pulp.lpSum(S[i] for i in range(self.n_stores)), "DC_supply_capacity"

        # Non-negativity of implied safety stocks
        for i in range(self.n_stores):
            prob += S[i] - float(store_lt_demand_q[i]) >= 0, f"Store_{i}_nonneg_SS"
        prob += S0 - dc_lt_demand_q >= 0, "DC_nonneg_SS"

        # ------------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------------
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=120)
        prob.solve(solver)
        t_elapsed = time.perf_counter() - t0

        status = pulp.LpStatus[prob.status]
        obj = pulp.value(prob.objective) or 0.0
        logger.info("CC-MEIO status: %s | Obj: %.2f | Time: %.2fs", status, obj, t_elapsed)

        # ------------------------------------------------------------------
        # Extract solution
        # ------------------------------------------------------------------
        r0_val = max(pulp.value(r0) or 0.0, 0.0)
        S0_val = max(pulp.value(S0) or 0.0, 0.0)
        r_val = np.array([max(pulp.value(r[i]) or 0.0, 0.0) for i in range(self.n_stores)])
        S_val = np.array([max(pulp.value(S[i]) or 0.0, 0.0) for i in range(self.n_stores)])

        ss_dc = max(S0_val - dc_lt_demand_q, 0.0)
        ss_stores = np.maximum(S_val - store_lt_demand_q, 0.0)
        q_dc = max(S0_val - r0_val, 0.0)
        q_stores = np.maximum(S_val - r_val, 0.0)

        hc = float(h_dc * ss_dc + h_s * ss_stores.sum())
        rc = float(c_v * (mu_dc + mu_store.sum()))

        # Estimated fill rate from scenarios
        store_fill = np.mean(
            [
                np.mean(S_val[i] >= store_scenarios.scenarios[i])
                for i in range(self.n_stores)
            ]
        )

        return CCOSolution(
            status=status,
            objective_value=obj,
            safety_stock_dc=ss_dc,
            safety_stock_stores=ss_stores,
            reorder_point_dc=r0_val,
            reorder_point_stores=r_val,
            order_quantity_dc=q_dc,
            order_quantity_stores=q_stores,
            holding_cost=hc,
            replenishment_cost=rc,
            fill_rate=float(store_fill),
            solve_time_sec=t_elapsed,
        )