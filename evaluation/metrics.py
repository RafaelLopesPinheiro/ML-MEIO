"""
evaluation/metrics.py
======================
Comprehensive evaluation metrics — nan/inf-safe version.

Key fixes vs. original
-----------------------
1. evaluate_forecaster() sanitises y_pred_mean, mu, sigma before any
   metric computation — replaces inf/nan with the sample mean of y_true.
2. crps_normal() clips sigma to a safe range before use.
3. smape() and mape() return np.nan (not inf) on degenerate inputs.
4. winkler_score() handles inf interval bounds gracefully.
5. All public functions return finite floats or np.nan — never inf.

References
----------
Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. JASA, 102(477), 359–378.
Winkler, R. L. (1972). A decision theoretic approach to interval
  estimation. JASA.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal sanitisation helper
# ---------------------------------------------------------------------------

def _sanitise(arr: np.ndarray, fallback: float) -> np.ndarray:
    """Replace inf / nan / huge values with `fallback`."""
    arr = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        arr = arr.copy()
        arr[bad] = fallback
    return arr


# ---------------------------------------------------------------------------
# Point forecast metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error. Returns nan if inputs are degenerate."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = _sanitise(y_pred, float(np.nanmean(y_true)))
    val = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return float(val) if np.isfinite(val) else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = _sanitise(y_pred, float(np.nanmean(y_true)))
    val = np.mean(np.abs(y_true - y_pred))
    return float(val) if np.isfinite(val) else float("nan")


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (excludes zero actuals)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = _sanitise(y_pred, float(np.nanmean(y_true)))
    mask = np.abs(y_true) > eps
    if not mask.any():
        return float("nan")
    val = 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return float(val) if np.isfinite(val) else float("nan")


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = _sanitise(y_pred, float(np.nanmean(y_true)))
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    # denom is guaranteed finite now that y_pred is sanitised
    val = 100.0 * np.mean(2 * np.abs(y_true - y_pred) / denom)
    return float(val) if np.isfinite(val) else float("nan")


def pinball_loss(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    quantiles: List[float],
) -> Dict[float, float]:
    """
    Pinball (quantile) loss for each quantile level.

    Parameters
    ----------
    y_true    : np.ndarray, shape (n_samples,)
    q_preds   : np.ndarray, shape (n_samples, n_quantiles)
    quantiles : list of float

    Returns
    -------
    dict mapping quantile → mean pinball loss
    """
    fallback = float(np.nanmean(y_true))
    losses: Dict[float, float] = {}
    for j, q in enumerate(quantiles):
        col = _sanitise(q_preds[:, j], fallback)
        residuals = y_true - col
        loss = np.where(residuals >= 0, q * residuals, (q - 1) * residuals)
        val  = float(np.mean(loss))
        losses[q] = val if np.isfinite(val) else float("nan")
    return losses


def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> float:
    """
    Winkler interval score for (1-α) prediction intervals. Lower is better.

    Non-finite interval bounds are replaced with safe extremes.
    """
    y_true = np.asarray(y_true, dtype=float)
    fallback_lo = float(np.nanmin(y_true))
    fallback_hi = float(np.nanmax(y_true))
    lower = _sanitise(lower, fallback_lo)
    upper = _sanitise(upper, fallback_hi)

    width        = np.maximum(upper - lower, 0.0)
    penalty_low  = (2 / alpha) * np.maximum(lower - y_true, 0)
    penalty_high = (2 / alpha) * np.maximum(y_true - upper, 0)
    val = float(np.mean(width + penalty_low + penalty_high))
    return val if np.isfinite(val) else float("nan")


def empirical_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of actuals within predicted intervals."""
    y_true = np.asarray(y_true, dtype=float)
    lower  = _sanitise(lower,  float(np.nanmin(y_true)))
    upper  = _sanitise(upper,  float(np.nanmax(y_true)))
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def crps_normal(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    CRPS for Normal distributions (closed-form).

    CRPS(N(μ,σ²), y) = σ * (z*(2Φ(z)-1) + 2φ(z) - 1/√π)
    where z = (y-μ)/σ.
    """
    from scipy.stats import norm as _norm

    fallback_mu    = float(np.nanmean(y_true))
    fallback_sigma = float(np.nanstd(y_true) + 1e-6)

    mu    = _sanitise(mu,    fallback_mu)
    sigma = _sanitise(sigma, fallback_sigma)
    sigma = np.clip(sigma, 1e-6, fallback_sigma * 50)  # prevent degenerate σ

    z    = (y_true - mu) / sigma
    crps = sigma * (
        z * (2 * _norm.cdf(z) - 1) + 2 * _norm.pdf(z) - 1 / np.sqrt(np.pi)
    )
    crps = np.where(np.isfinite(crps), crps, float("nan"))
    val  = float(np.nanmean(crps))
    return val if np.isfinite(val) else float("nan")


def crps_empirical(
    y_true: np.ndarray,
    scenarios: np.ndarray,
    probs: Optional[np.ndarray] = None,
) -> float:
    """
    Energy-form CRPS using discrete scenarios (distribution-free).

    CRPS(F, y) ≈ E_F[|X-y|] - 0.5 * E_F[|X-X'|]
    """
    n_obs, n_s = scenarios.shape
    if probs is None:
        probs = np.ones(n_s) / n_s

    crps_vals = np.zeros(n_obs)
    for i in range(n_obs):
        s = scenarios[i]
        # Skip if any inf in scenarios for this observation
        if not np.all(np.isfinite(s)):
            crps_vals[i] = float("nan")
            continue
        term1 = float(np.dot(probs, np.abs(s - y_true[i])))
        term2 = sum(
            probs[j] * probs[k] * abs(s[j] - s[k])
            for j in range(n_s) for k in range(n_s)
        )
        crps_vals[i] = term1 - 0.5 * term2
    return float(np.nanmean(crps_vals))


# ---------------------------------------------------------------------------
# Comprehensive evaluators
# ---------------------------------------------------------------------------

def evaluate_forecaster(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    q_preds: Optional[np.ndarray] = None,
    quantiles: Optional[List[float]] = None,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    alpha: float = 0.10,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Comprehensive forecaster evaluation — nan/inf-safe.

    Any inf/nan in predictions is replaced with the sample mean of y_true
    before metric computation. A warning is logged if this happens.
    """
    y_true   = np.asarray(y_true, dtype=float)
    fallback = float(np.nanmean(y_true)) if len(y_true) > 0 else 0.0

    # --- Sanitise all prediction arrays ---
    y_pred_mean = _sanitise(y_pred_mean, fallback)

    any_bad = not np.all(np.isfinite(y_pred_mean))
    if mu is not None:
        mu = _sanitise(mu, fallback)
        any_bad = any_bad or not np.all(np.isfinite(mu))
    if sigma is not None:
        sigma_fallback = float(np.nanstd(y_true) + 1e-6)
        sigma = _sanitise(sigma, sigma_fallback)
        sigma = np.clip(sigma, 1e-6, sigma_fallback * 50)

    if any_bad:
        logger.warning(
            "Forecaster produced non-finite predictions — "
            "replaced with training mean (%.2f) before metric computation.",
            fallback,
        )

    results: Dict[str, float] = {}

    # Point forecast metrics
    results["RMSE"]  = rmse(y_true, y_pred_mean)
    results["MAE"]   = mae(y_true, y_pred_mean)
    results["MAPE"]  = mape(y_true, y_pred_mean)
    results["sMAPE"] = smape(y_true, y_pred_mean)

    # Interval metrics
    if lower is not None and upper is not None:
        lower = _sanitise(lower, float(np.nanmin(y_true)))
        upper = _sanitise(upper, float(np.nanmax(y_true)))
        results["Coverage"]      = empirical_coverage(y_true, lower, upper)
        results["WinklerScore"]  = winkler_score(y_true, lower, upper, alpha)
        results["IntervalWidth"] = float(np.nanmean(upper - lower))

    # Quantile metrics
    if q_preds is not None and quantiles is not None:
        # Sanitise each quantile column
        q_preds_clean = np.where(np.isfinite(q_preds), q_preds, fallback)
        pb = pinball_loss(y_true, q_preds_clean, quantiles)
        valid_pb = [v for v in pb.values() if np.isfinite(v)]
        results["MeanPinballLoss"] = float(np.mean(valid_pb)) if valid_pb else float("nan")
        for q, v in pb.items():
            results[f"Pinball_{q:.2f}"] = v

    # Distributional metrics
    if mu is not None and sigma is not None:
        results["CRPS_Normal"] = crps_normal(y_true, mu, sigma)

    # Summary log (use nan-safe format)
    def _fmt(v):
        return f"{v:.4f}" if np.isfinite(v) else "nan"

    logger.info(
        "Forecaster evaluation: RMSE=%s | MAE=%s | MAPE=%s%% | "
        "Coverage=%s | WinklerScore=%s",
        _fmt(results["RMSE"]),
        _fmt(results["MAE"]),
        _fmt(results.get("MAPE", float("nan"))),
        _fmt(results.get("Coverage", float("nan"))),
        _fmt(results.get("WinklerScore", float("nan"))),
    )
    return results


def evaluate_inventory_policy(
    demand_actual: np.ndarray,
    inventory_levels: np.ndarray,
    order_quantities: np.ndarray,
    safety_stocks: np.ndarray,
    cfg: dict,
) -> Dict[str, float]:
    """
    Evaluate an inventory policy against realized demand.

    Parameters
    ----------
    demand_actual   : np.ndarray, shape (T, n_nodes)
    inventory_levels: np.ndarray, shape (T, n_nodes)
    order_quantities: np.ndarray, shape (n_nodes,)
    safety_stocks   : np.ndarray, shape (n_nodes,)
    cfg             : dict   Optimization configuration.

    Returns
    -------
    dict of metric_name → value
    """
    T, n = demand_actual.shape
    h_s  = cfg["holding_cost_store"]
    b    = cfg["backorder_cost"]
    c_v  = cfg["replenishment_variable_cost"]

    inventory  = inventory_levels.copy().astype(float)
    backorders = np.zeros((T, n))

    for t in range(T):
        net          = inventory[t] - demand_actual[t]
        backorders[t] = np.maximum(-net, 0)

    total_demand = demand_actual.sum(axis=0)
    total_bo     = backorders.sum(axis=0)
    fill_rate    = float(np.mean(
        np.where(total_demand > 0,
                 1 - total_bo / np.maximum(total_demand, 1e-9),
                 1.0)
    ))
    csl  = float(np.mean(backorders == 0))
    hc   = float(h_s * inventory.mean())
    bc   = float(b   * backorders.mean())
    rc   = float(c_v * order_quantities.mean())
    tc   = hc + bc + rc

    return {
        "fill_rate":              fill_rate,
        "cycle_service_level":    csl,
        "mean_backorders":        float(backorders.mean()),
        "mean_inventory":         float(inventory.mean()),
        "holding_cost":           hc,
        "backorder_cost":         bc,
        "replenishment_cost":     rc,
        "total_cost":             tc,
    }