"""
experiments/run_experiment.py — figure generation fixed.

Changes vs. previous version
------------------------------
1. PNG-only output (_save calls updated).
2. _generate_figures: safety stock dict uses display names so bars appear.
3. fill_rate and cost decomposition use _parse_scalar correctly.
4. All methods filtered against ACTIVE_METHODS before plotting.
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import WalmartDataLoader, load_config
from demand import ConformalForecaster, NGBoostForecaster, QuantileRegressionForecaster
from demand.preprocessing import DemandPreprocessor, prepare_store_dept_data
from evaluation import ClassicalMEIOBaseline, evaluate_forecaster, evaluate_inventory_policy
from optimization import ChanceConstrainedMEIO, ScenarioGenerator, TwoStageStochasticMEIO
from optimization.scenario_generator import ScenarioSet
from visualization import MEIOPlotter
from visualization.plots import METHOD_DISPLAY_NAMES, ACTIVE_METHODS, _display

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forecaster factory
# ---------------------------------------------------------------------------

def build_forecaster(method: str, cfg: dict):
    seed  = cfg.get("reproducibility", {}).get("random_seed", 42)
    d_cfg = cfg["demand"]
    if method == "ngboost":
        nb = d_cfg.get("ngboost", {})
        return NGBoostForecaster(
            distribution=nb.get("distribution", "normal"),
            n_estimators=nb.get("n_estimators", 500),
            learning_rate=nb.get("learning_rate", 0.05),
            minibatch_frac=nb.get("minibatch_frac", 0.8),
            col_sample=nb.get("col_sample", 0.8),
            random_seed=seed,
        )
    elif method == "quantile_regression":
        qr = d_cfg.get("quantile_regression", {})
        return QuantileRegressionForecaster(
            quantiles=d_cfg.get("quantiles", [0.05, 0.25, 0.50, 0.75, 0.90, 0.95]),
            n_estimators=qr.get("n_estimators", 300),
            learning_rate=qr.get("learning_rate", 0.05),
            max_depth=qr.get("max_depth", 6),
            subsample=qr.get("subsample", 0.8),
            random_seed=seed,
        )
    elif method == "conformal":
        return ConformalForecaster(
            alpha=d_cfg.get("conformal_alpha", 0.10),
            symmetric=False,
            random_seed=seed,
        )
    raise ValueError(f"Unknown forecaster: {method}")


# ---------------------------------------------------------------------------
# Safe parsers
# ---------------------------------------------------------------------------

def _parse_first_element(cell) -> float:
    if cell is None:
        return 0.0
    if isinstance(cell, (int, float)):
        return float(cell)
    if isinstance(cell, (list, np.ndarray)):
        return float(cell[0]) if len(cell) > 0 else 0.0
    try:
        parsed = ast.literal_eval(str(cell))
        if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
            return float(parsed[0])
        return float(parsed)
    except Exception:
        try:
            return float(str(cell).strip("[]() ").split(",")[0])
        except Exception:
            return 0.0


def _parse_scalar(cell) -> float:
    if cell is None:
        return 0.0
    try:
        v = float(cell)
        return v if np.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return _parse_first_element(cell)


# ---------------------------------------------------------------------------
# DC scenario builder
# ---------------------------------------------------------------------------

def _build_dc_scenarios(
    dc_dept, store_ss, forecaster_method, cfg, feature_cols, avail_feat, gen, seed
):
    if "DC_Weekly_Demand" in dc_dept.columns and len(dc_dept) >= 20:
        dc_avail = [c for c in avail_feat if c in dc_dept.columns]
        if dc_avail:
            dc_renamed = dc_dept.rename(columns={"DC_Weekly_Demand": "Weekly_Sales"}).copy()
            dc_X, dc_y, _, _ = prepare_store_dept_data(dc_renamed, dc_avail)
            if len(dc_X) >= 15:
                dc_prep = DemandPreprocessor()
                dc_X_pp = dc_prep.fit_transform(pd.DataFrame(dc_X, columns=dc_avail))
                dc_fc   = build_forecaster(forecaster_method, cfg)
                try:
                    dc_fc.fit(dc_X_pp, dc_y)
                    return gen.generate(dc_fc, dc_X_pp[-1:, :], node_ids=["DC"])
                except Exception as exc:
                    logger.debug("DC forecaster fallback: %s", exc)

    dc_raw = store_ss.scenarios * 5.0
    return ScenarioSet(scenarios=dc_raw, probabilities=store_ss.probabilities,
                       node_ids=["DC"])


# ---------------------------------------------------------------------------
# Flatten helpers
# ---------------------------------------------------------------------------

def _flatten_opt_result(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (list, np.ndarray)):
            arr = np.asarray(v, dtype=float)
            out[k]           = float(arr[0]) if len(arr) > 0 else 0.0
            out[f"{k}_sum"]  = float(arr.sum())
            out[f"{k}_mean"] = float(arr.mean()) if len(arr) > 0 else 0.0
        elif isinstance(v, bool):
            out[k] = int(v)
        else:
            try:
                out[k] = float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def _flatten_baseline_result(d: dict) -> dict:
    return _flatten_opt_result(d)


# ---------------------------------------------------------------------------
# Per store-dept
# ---------------------------------------------------------------------------

def run_store_dept(
    store_id, dept_id, train_df, val_df, dc_train,
    forecaster_method, optimizer_method, cfg, preprocessor, feature_cols,
):
    seed      = cfg.get("reproducibility", {}).get("random_seed", 42)
    opt_cfg   = cfg["optimization"]
    n_scen    = opt_cfg.get("n_scenarios", 200)

    tr = train_df[(train_df["Store"] == store_id) & (train_df["Dept"] == dept_id)].copy()
    va = val_df[(val_df["Store"] == store_id) & (val_df["Dept"] == dept_id)].copy()
    if len(tr) < 30 or len(va) < 4:
        return {}

    avail = [c for c in feature_cols if c in tr.columns]
    X_tr, y_tr, _, _ = prepare_store_dept_data(tr, avail)
    X_va, y_va, _, dates_va = prepare_store_dept_data(va, avail)
    if len(X_tr) < 20 or len(X_va) < 2:
        return {}

    prep    = DemandPreprocessor()
    X_tr_pp = prep.fit_transform(pd.DataFrame(X_tr, columns=avail))
    X_va_pp = prep.transform(pd.DataFrame(X_va, columns=avail))

    forecaster = build_forecaster(forecaster_method, cfg)
    n_early = max(int(len(X_tr_pp) * 0.15), 5)
    try:
        if forecaster_method in ("ngboost", "quantile_regression"):
            forecaster.fit(X_tr_pp[:-n_early], y_tr[:-n_early],
                           X_val=X_tr_pp[-n_early:], y_val=y_tr[-n_early:])
        else:
            forecaster.fit(X_tr_pp, y_tr)
    except Exception as exc:
        logger.error("Forecaster fit failed S%d D%d: %s", store_id, dept_id, exc)
        return {}

    quantiles   = cfg["demand"].get("quantiles", [0.05, 0.25, 0.50, 0.75, 0.90, 0.95])
    y_pred_mean = forecaster.predict_mean(X_va_pp)
    q_preds_df  = forecaster.predict_quantiles(X_va_pp, quantiles=quantiles)
    alpha       = cfg["demand"].get("conformal_alpha", 0.10)

    if hasattr(forecaster, "predict_intervals"):
        lower, upper = forecaster.predict_intervals(X_va_pp)
    else:
        qs    = np.array(quantiles, dtype=float)
        lower = q_preds_df[float(qs[np.argmin(np.abs(qs - alpha / 2))])].values
        upper = q_preds_df[float(qs[np.argmin(np.abs(qs - (1 - alpha / 2)))])].values

    sigma           = forecaster.predict_std(X_va_pp) if hasattr(forecaster, "predict_std") else None
    forecast_metrics = evaluate_forecaster(
        y_va, y_pred_mean, q_preds=q_preds_df.values, quantiles=quantiles,
        lower=lower, upper=upper, alpha=alpha, mu=y_pred_mean, sigma=sigma,
    )

    X_plan  = X_va_pp[-1:, :]
    gen     = ScenarioGenerator(n_scenarios=n_scen, n_reduced=min(50, n_scen),
                                reduction_method="kmeans", random_seed=seed)
    store_ss = gen.generate(forecaster, X_plan,
                            node_ids=[f"{store_id}_{dept_id}"])
    dc_dept  = dc_train[dc_train["Dept"] == dept_id].copy()
    dc_ss    = _build_dc_scenarios(dc_dept, store_ss, forecaster_method,
                                   cfg, feature_cols, avail, gen, seed)

    try:
        if optimizer_method == "stochastic_milp":
            sol = TwoStageStochasticMEIO(n_stores=1, cfg=opt_cfg).solve(dc_ss, store_ss)
        elif optimizer_method == "chance_constrained":
            sol = ChanceConstrainedMEIO(n_stores=1, cfg=opt_cfg).solve(dc_ss, store_ss)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_method}")
        opt_result = _flatten_opt_result(sol.to_dict())
    except Exception as exc:
        logger.error("Optimization failed S%d D%d: %s", store_id, dept_id, exc)
        opt_result = {}

    dc_hist = (
        dc_dept["DC_Weekly_Demand"].dropna().values
        if "DC_Weekly_Demand" in dc_dept.columns and len(dc_dept) > 5
        else y_tr * 5.0
    )
    baseline       = ClassicalMEIOBaseline(opt_cfg)
    bl_sol         = baseline.solve(dc_hist, y_tr[:, None])
    baseline_result = _flatten_baseline_result(bl_sol.to_dict())

    return {
        "store_id": store_id, "dept_id": dept_id,
        "forecaster": forecaster_method, "optimizer": optimizer_method,
        "n_train": len(y_tr), "n_val": len(y_va),
        **{f"forecast_{k}": v for k, v in forecast_metrics.items()},
        **{f"opt_{k}": v for k, v in opt_result.items()},
        **{f"baseline_{k}": v for k, v in baseline_result.items()},
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiment(
    config_path="config.yaml", method="ngboost",
    optimizer="stochastic_milp", output_dir="results/",
    all_combinations=False,
):
    cfg     = load_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, cfg["output"].get("log_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(out_dir / "experiment.log")],
    )

    logger.info("=" * 70)
    logger.info("MEIO Stochastic Optimization — Experiment Start")
    logger.info("=" * 70)

    loader = WalmartDataLoader(cfg)
    train_df, _, dc_df = loader.build_dataset()
    train_split, val_split = loader.temporal_split(train_df)
    feature_cols = loader.get_feature_cols(train_split)

    combos = (
        [(m, o)
         for m in ["ngboost", "quantile_regression", "conformal"]
         for o in ["stochastic_milp", "chance_constrained"]]
        if all_combinations else [(method, optimizer)]
    )

    store_ids = cfg["data"].get("store_subset") or sorted(train_df["Store"].unique())[:5]
    dept_ids  = cfg["data"].get("dept_subset")  or sorted(train_df["Dept"].unique())[:7]
    all_rows  = []

    for m, o in combos:
        logger.info("Running: method=%s | optimizer=%s", m, o)
        t0 = time.perf_counter()
        preprocessor = DemandPreprocessor()

        for store_id in store_ids:
            for dept_id in dept_ids:
                row = run_store_dept(
                    store_id, dept_id, train_split, val_split, dc_df,
                    m, o, cfg, preprocessor, feature_cols,
                )
                if row:
                    all_rows.append(row)

        logger.info("  Done in %.1fs | %d rows", time.perf_counter() - t0, len(all_rows))

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out_dir / "results.csv", index=False)
    logger.info("Results saved: %d rows", len(results_df))

    _generate_figures(results_df, train_split, cfg, out_dir)
    logger.info("Experiment complete.")
    return results_df


# ---------------------------------------------------------------------------
# Figure generation — FIXED
# ---------------------------------------------------------------------------

def _generate_figures(results_df, train_df, cfg, out_dir):
    """Generate all publication figures. PNG only."""
    if results_df.empty:
        logger.warning("No results to plot.")
        return

    plotter = MEIOPlotter(output_dir=str(out_dir / "figures"))

    # Fig 1: Demand distribution
    store_ids = sorted(train_df["Store"].unique())
    dept_ids  = sorted(train_df["Dept"].unique())
    s, d = store_ids[0], dept_ids[0]
    sd_data = (train_df[(train_df["Store"] == s) & (train_df["Dept"] == d)]["Weekly_Sales"]
               .dropna().values)
    if len(sd_data) > 10:
        try:
            plotter.plot_demand_distribution(sd_data, store_id=s, dept_id=d)
        except Exception as exc:
            logger.warning("Fig 1 failed: %s", exc)

    # Fig 4: Safety stock — FIXED: use display names as keys
    try:
        dept_subset = results_df["dept_id"].unique()[:1]
        for dept in dept_subset:
            sub     = results_df[results_df["dept_id"] == dept].copy()
            ss_dict: Dict[str, np.ndarray] = {}

            for m in sub["forecaster"].unique():
                if m not in ACTIVE_METHODS:
                    continue
                disp = _display(m)          # ← convert to display name
                m_sub = sub[sub["forecaster"] == m]
                col   = "opt_safety_stock_stores"
                if col in m_sub.columns:
                    vals = m_sub[col].apply(_parse_scalar).values
                    if vals.sum() > 0:
                        ss_dict[disp] = vals  # ← store under display name

            bl_col = "baseline_safety_stock_stores"
            if bl_col in sub.columns:
                bl_vals = (sub.drop_duplicates("store_id")[bl_col]
                           .apply(_parse_scalar).values)
                if bl_vals.sum() > 0:
                    ss_dict["Classical"] = bl_vals

            if ss_dict:
                store_ids_sub = sub.drop_duplicates("store_id")["store_id"].tolist()
                min_len = min(len(v) for v in ss_dict.values())
                plotter.plot_safety_stock_comparison(
                    store_ids_sub[:min_len],
                    {k: v[:min_len] for k, v in ss_dict.items()},
                )
    except Exception as exc:
        logger.warning("Fig 4 failed: %s", exc)

    # Fig 5: Cost decomposition
    try:
        labels, hc, bc, rc, ec = [], [], [], [], []
        for m in results_df["forecaster"].unique():
            if m not in ACTIVE_METHODS:
                continue
            for o in results_df["optimizer"].unique():
                sub = results_df[(results_df["forecaster"] == m) &
                                 (results_df["optimizer"] == o)]
                if sub.empty:
                    continue
                labels.append(f"{_display(m)}\n({o[:4]})")
                hc.append(max(sub["opt_holding_cost"].apply(_parse_scalar).mean()       if "opt_holding_cost"       in sub.columns else 0.0, 0))
                bc.append(max(sub["opt_backorder_cost"].apply(_parse_scalar).mean()     if "opt_backorder_cost"     in sub.columns else 0.0, 0))
                rc.append(max(sub["opt_replenishment_cost"].apply(_parse_scalar).mean() if "opt_replenishment_cost" in sub.columns else 0.0, 0))
                ec.append(max(sub["opt_emergency_cost"].apply(_parse_scalar).mean()     if "opt_emergency_cost"     in sub.columns else 0.0, 0))

        if labels and any(v > 0 for v in hc + bc + rc):
            plotter.plot_cost_decomposition(
                labels, np.array(hc), np.array(bc), np.array(rc),
                emergency_costs=np.array(ec) if any(e > 0 for e in ec) else None,
            )
    except Exception as exc:
        logger.warning("Fig 5 failed: %s", exc)

    # Fig 8: Fill rate — use display names
    try:
        fill_col = "opt_fill_rate"
        if fill_col in results_df.columns:
            fill_dict: Dict[str, np.ndarray] = {}
            for m in results_df["forecaster"].unique():
                if m not in ACTIVE_METHODS:
                    continue
                disp = _display(m)
                vals = (results_df[results_df["forecaster"] == m][fill_col]
                        .apply(_parse_scalar).dropna().values)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if len(vals) > 0:
                    fill_dict[disp] = vals

            if fill_dict:
                target_sl = cfg["optimization"].get("service_level", 0.95)
                plotter.plot_fill_rate_comparison(fill_dict, target=target_sl)
    except Exception as exc:
        logger.warning("Fig 8 failed: %s", exc)

    # Fig 6: Pareto frontier
    try:
        if "opt_fill_rate" in results_df.columns and "opt_objective_value" in results_df.columns:
            sl_d, tc_d = {}, {}
            for m in results_df["forecaster"].unique():
                if m not in ACTIVE_METHODS:
                    continue
                sub     = results_df[results_df["forecaster"] == m]
                sl_vals = sub["opt_fill_rate"].apply(_parse_scalar).values
                tc_vals = sub["opt_objective_value"].apply(_parse_scalar).values
                valid   = np.isfinite(sl_vals) & np.isfinite(tc_vals) & (tc_vals > 0)
                if valid.sum() >= 2:
                    sl_d[_display(m)] = sl_vals[valid]
                    tc_d[_display(m)] = tc_vals[valid]
            if sl_d:
                plotter.plot_pareto_frontier(sl_d, tc_d)
    except Exception as exc:
        logger.warning("Fig 6 failed: %s", exc)

    logger.info("Figures generated in %s/figures/", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="config.yaml")
    p.add_argument("--method",    default="ngboost",
                   choices=["ngboost", "quantile_regression", "conformal"])
    p.add_argument("--optimizer", default="stochastic_milp",
                   choices=["stochastic_milp", "chance_constrained"])
    p.add_argument("--output",    default="results/")
    p.add_argument("--all",       action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(config_path=args.config, method=args.method,
                   optimizer=args.optimizer, output_dir=args.output,
                   all_combinations=args.all)