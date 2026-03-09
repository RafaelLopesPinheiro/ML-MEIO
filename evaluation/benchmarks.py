"""
evaluation/benchmarks.py
==========================
Classical MEIO Baseline: EOQ + Normal-Distribution Safety Stock.

This module implements the classical baseline for comparison, which assumes:
  - Normal demand distribution N(μ, σ²) estimated from historical data.
  - Economic Order Quantity (EOQ) for replenishment sizing.
  - Analytical safety stock formula: SS = z_{α} * σ_LT
    where z_{α} = Normal quantile for service level α, and
    σ_LT = σ * √(lead_time) is the lead-time demand standard deviation.

The reorder point is then: ROP = μ_LT + SS = μ * L + z_α * σ * √L

This is the canonical textbook formula (Silver, Pyke & Thomas, 2017) that
our approach aims to outperform in terms of realized service levels and
cost efficiency when demand is non-Normal.

References
----------
Silver, E. A., Pyke, D. F., & Thomas, D. J. (2017). Inventory and
  Production Management in Supply Chains (4th ed.). CRC Press.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class ClassicalBaselineSolution:
    """Solution from the classical EOQ + Normal SS baseline."""
    eoq_dc: float = 0.0
    eoq_stores: np.ndarray = field(default_factory=lambda: np.array([]))
    safety_stock_dc: float = 0.0
    safety_stock_stores: np.ndarray = field(default_factory=lambda: np.array([]))
    reorder_point_dc: float = 0.0
    reorder_point_stores: np.ndarray = field(default_factory=lambda: np.array([]))
    holding_cost: float = 0.0
    replenishment_cost: float = 0.0

    def to_dict(self) -> dict:
        return {
            "eoq_dc": self.eoq_dc,
            "eoq_stores": self.eoq_stores.tolist(),
            "safety_stock_dc": self.safety_stock_dc,
            "safety_stock_stores": self.safety_stock_stores.tolist(),
            "reorder_point_dc": self.reorder_point_dc,
            "reorder_point_stores": self.reorder_point_stores.tolist(),
            "holding_cost": self.holding_cost,
            "replenishment_cost": self.replenishment_cost,
        }


class ClassicalMEIOBaseline:
    """
    Classical MEIO baseline using EOQ and Normal-distribution safety stocks.

    Parameters
    ----------
    cfg : dict
        Optimization configuration.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    def solve(
        self,
        demand_history_dc: np.ndarray,
        demand_history_stores: np.ndarray,
    ) -> ClassicalBaselineSolution:
        """
        Compute classical EOQ + SS policy from historical demand.

        Parameters
        ----------
        demand_history_dc : np.ndarray, shape (T,)
            Historical weekly demand at DC.
        demand_history_stores : np.ndarray, shape (T, n_stores)
            Historical weekly demand at each store.

        Returns
        -------
        ClassicalBaselineSolution
        """
        beta = self.cfg["service_level"]
        L_s = self.cfg["lead_time_dc_to_store"]
        L_dc = self.cfg["lead_time_supplier_to_dc"]
        h_s = self.cfg["holding_cost_store"]
        h_dc = self.cfg["holding_cost_dc"]
        c_f = self.cfg["replenishment_fixed_cost"]
        c_v = self.cfg["replenishment_variable_cost"]

        z = norm.ppf(beta)  # Normal quantile for service level

        # DC parameters
        mu_dc = float(np.mean(demand_history_dc))
        sigma_dc = float(np.std(demand_history_dc) + 1e-6)
        eoq_dc = self._eoq(mu_dc, c_f, h_dc, c_v)
        ss_dc = z * sigma_dc * np.sqrt(L_dc)
        rop_dc = mu_dc * L_dc + ss_dc

        # Store parameters
        if demand_history_stores.ndim == 1:
            demand_history_stores = demand_history_stores[:, None]
        n_stores = demand_history_stores.shape[1]

        mu_s = np.mean(demand_history_stores, axis=0)     # (n_stores,)
        sigma_s = np.std(demand_history_stores, axis=0) + 1e-6  # (n_stores,)

        eoq_s = np.array([self._eoq(mu_s[i], c_f, h_s, c_v) for i in range(n_stores)])
        ss_s = z * sigma_s * np.sqrt(L_s)
        rop_s = mu_s * L_s + ss_s

        hc = float(h_dc * ss_dc + h_s * ss_s.sum())
        rc = float(c_v * (mu_dc + mu_s.sum()))

        logger.info(
            "Classical MEIO baseline computed. DC EOQ=%.1f, SS=%.1f, ROP=%.1f",
            eoq_dc,
            ss_dc,
            rop_dc,
        )

        return ClassicalBaselineSolution(
            eoq_dc=eoq_dc,
            eoq_stores=eoq_s,
            safety_stock_dc=ss_dc,
            safety_stock_stores=ss_s,
            reorder_point_dc=rop_dc,
            reorder_point_stores=rop_s,
            holding_cost=hc,
            replenishment_cost=rc,
        )

    @staticmethod
    def _eoq(mu: float, S: float, h: float, c_v: float) -> float:
        """
        Economic Order Quantity (EOQ) formula.
            EOQ = √(2 * D * S / H)
        where D = annual demand ≈ 52 * weekly mean, S = fixed order cost,
        H = h + c_v * r (holding cost including capital), r = 0.1 (assumed).
        """
        D = 52 * mu  # annualised
        H = h + c_v * 0.1
        if D <= 0 or H <= 0:
            return 0.0
        return float(np.sqrt(2 * D * S / H))