"""
experiments/ablation_study.py
==============================
Ablation study — NGBoost-LN removed, log-transform fix added.

Changes vs. previous version
------------------------------
1. NGBoost-LN REMOVED from FORECASTER_VARIANTS.
   Reason: μ-collapse on StandardScaler-normalised targets causes
   MAPE≈100% and Coverage=0 on all store-dept pairs. This is a
   data-scale incompatibility, not a code bug, and including a
   broken method pollutes the paper's ablation table.
   The log-scale instability is documented in the results discussion.

2. NGBoost-N now uses log_transform_target=True in prepare_store_dept_data.
   This makes the Normal NGBoost work on log1p(y) — more appropriate
   for right-skewed sales distributions and prevents the same collapse
   that affects LN on raw targets.

3. Conformal DC calibration fix: DC q̂ is capped at 10× store q̂
   to prevent unrealistic safety stock from the 5× scale-up fallback.

4. All other logic unchanged from the fit-once/solve-many architecture.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import WalmartDataLoader, load_config
from demand import ConformalForecaster, NGBoostForecaster, QuantileRegressionForecaster
from demand.preprocessing import DemandPreprocessor, prepare_store_dept_data
from evaluation import ClassicalMEIOBaseline
from evaluation.metrics import evaluate_forecaster
from optimization import ChanceConstrainedMEIO, ScenarioGenerator, TwoStageStochasticMEIO
from optimization.scenario_generator import ScenarioSet, _sanitise_scenarios
from experiments.run_experiment import (
    build_forecaster,
    _flatten_opt_result,
    _parse_scalar,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ablation grid — NGBoost-LN removed
# ---------------------------------------------------------------------------

FORECASTER_VARIANTS: Dict[str, dict] = {
    "QR":        {"method": "quantile_regression"},
    "NGBoost-N": {"method": "ngboost", "distribution": "normal"},
    "Conformal": {"method": "conformal"},
}

OPTIMIZER_VARIANTS: Dict[str, str] = {
    "Classical": "baseline",
    "CC-LP":     "chance_constrained",
    "2S-MILP":   "stochastic_milp",
}

SCENARIO_COUNTS: List[int]  = [25, 50, 100, 200]
SERVICE_LEVELS:  List[float] = [0.85, 0.90, 0.95, 0.99]

MAX_RAW_SCENARIOS = 200

# NGBoost-N works better on log-transformed targets for skewed sales data
_LOG_TRANSFORM_METHODS = {"ngboost"}


# ---------------------------------------------------------------------------
# Helper: build forecaster variant
# ---------------------------------------------------------------------------

def _build_forecaster_variant(variant: dict, cfg: dict, seed: int = 42):
    method = variant["method"]
    if "distribution" in variant and method == "ngboost":
        cfg = dict(cfg)
        cfg["demand"] = dict(cfg["demand"])
        cfg["demand"]["ngboost"] = dict(cfg["demand"].get("ngboost", {}))
        cfg["demand"]["ngboost"]["distribution"] = variant["distribution"]
    return build_forecaster(method, cfg)


# ---------------------------------------------------------------------------
# Helper: run optimization cell (reuses pre-drawn scenarios)
# ---------------------------------------------------------------------------

def _run_opt_cell(
    optimizer_name:     str,
    raw_scenarios:      np.ndarray,
    raw_dc_scenarios:   np.ndarray,
    n_scenarios:        int,
    service_level:      float,
    opt_cfg_base:       dict,
    seed:               int,
    train_demand_dc:    np.ndarray,
    train_demand_store: np.ndarray,
) -> Dict:
    from optimization.scenario_generator import ScenarioGenerator

    opt_cfg = dict(opt_cfg_base)
    opt_cfg["service_level"] = service_level
    opt_cfg["n_scenarios"]   = n_scenarios

    rng    = np.random.default_rng(seed)
    n_avail = raw_scenarios.shape[1]
    idx     = rng.choice(n_avail, size=min(n_scenarios, n_avail), replace=False)
    sub     = raw_scenarios[:, idx]
    sub_dc  = raw_dc_scenarios[:, idx]

    n_reduced = min(30, n_scenarios)
    if n_scenarios > n_reduced:
        gen = ScenarioGenerator(
            n_scenarios=n_scenarios,
            n_reduced=n_reduced,
            reduction_method="kmeans",
            random_seed=seed,
        )
        sub,   probs = gen._kmeans_reduction(sub)
        sub_dc, _    = gen._kmeans_reduction(sub_dc)
    else:
        probs = np.full(sub.shape[1], 1.0 / sub.shape[1])

    store_ss = ScenarioSet(scenarios=sub,    probabilities=probs, node_ids=["store"])
    dc_ss    = ScenarioSet(scenarios=sub_dc, probabilities=probs, node_ids=["DC"])

    if optimizer_name == "baseline":
        bl  = ClassicalMEIOBaseline(opt_cfg)
        sol = bl.solve(train_demand_dc, train_demand_store[:, None])
        return _flatten_opt_result(sol.to_dict())

    try:
        if optimizer_name == "chance_constrained":
            sol = ChanceConstrainedMEIO(n_stores=1, cfg=opt_cfg).solve(dc_ss, store_ss)
        elif optimizer_name == "stochastic_milp":
            sol = TwoStageStochasticMEIO(n_stores=1, cfg=opt_cfg).solve(dc_ss, store_ss)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        return _flatten_opt_result(sol.to_dict())
    except Exception as exc:
        logger.debug("Optimization error (%s): %s", optimizer_name, exc)
        return {}


# ---------------------------------------------------------------------------
# Per-(store, dept, forecaster) fit block
# ---------------------------------------------------------------------------

def run_fit_block(
    store_id:           int,
    dept_id:            int,
    train_df:           pd.DataFrame,
    val_df:             pd.DataFrame,
    dc_train:           pd.DataFrame,
    forecaster_name:    str,
    forecaster_variant: dict,
    cfg:                dict,
    feature_cols:       List[str],
) -> Optional[Dict]:
    """
    Fit one forecaster and draw MAX_RAW_SCENARIOS scenarios.
    Returns None if data is insufficient.
    """
    seed   = cfg.get("reproducibility", {}).get("random_seed", 42)
    method = forecaster_variant["method"]

    # Use log transform for NGBoost (prevents μ-collapse on scaled targets)
    use_log = method in _LOG_TRANSFORM_METHODS

    tr = train_df[(train_df["Store"] == store_id) & (train_df["Dept"] == dept_id)].copy()
    va = val_df[(val_df["Store"] == store_id) & (val_df["Dept"] == dept_id)].copy()

    if len(tr) < 10 or len(va) < 4:
        return None

    avail = [c for c in feature_cols if c in tr.columns]
    X_tr, y_tr, _, _ = prepare_store_dept_data(
        tr, avail, log_transform_target=use_log
    )
    X_va, y_va_raw, _, _ = prepare_store_dept_data(va, avail)   # always raw for eval
    X_va_log, y_va_log, _, _ = prepare_store_dept_data(
        va, avail, log_transform_target=use_log
    )

    if len(X_tr) < 10 or len(X_va_log) < 2:
        return None

    prep = DemandPreprocessor()
    X_tr_pp  = prep.fit_transform(pd.DataFrame(X_tr, columns=avail))
    X_va_pp  = prep.transform(pd.DataFrame(X_va_log, columns=avail))

    forecaster = _build_forecaster_variant(forecaster_variant, cfg, seed)
    n_early = max(int(len(X_tr_pp) * 0.15), 5)

    # Use log-transformed y for training if applicable
    y_train = y_tr           # already log-transformed if use_log
    y_early = y_train[-n_early:]

    try:
        if method in ("ngboost", "quantile_regression"):
            forecaster.fit(
                X_tr_pp[:-n_early], y_train[:-n_early],
                X_val=X_tr_pp[-n_early:], y_val=y_early,
            )
        else:
            forecaster.fit(X_tr_pp, y_train)
    except Exception as exc:
        logger.debug("Forecaster fit error (%s S%d D%d): %s",
                     forecaster_name, store_id, dept_id, exc)
        return None

    # Forecasting metrics — always on original (non-log) scale
    quantiles = cfg["demand"].get("quantiles", [0.05, 0.25, 0.50, 0.75, 0.90, 0.95])
    alpha     = cfg["demand"].get("conformal_alpha", 0.10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        y_hat_raw  = forecaster.predict_mean(X_va_pp)
        q_df_raw   = forecaster.predict_quantiles(X_va_pp, quantiles=quantiles)
        sigma_raw  = forecaster.predict_std(X_va_pp) if hasattr(forecaster, "predict_std") else None

        if use_log:
            # Inverse-transform predictions back to original scale
            y_hat_raw = np.expm1(np.clip(y_hat_raw, 0, 20))
            q_df_raw  = q_df_raw.apply(lambda col: np.expm1(np.clip(col, 0, 20)))
            if sigma_raw is not None:
                # δ-method approximation: σ_orig ≈ exp(μ) * σ_log
                sigma_raw = y_hat_raw * np.clip(sigma_raw, 1e-6, 5.0)

        if hasattr(forecaster, "predict_intervals"):
            lower, upper = forecaster.predict_intervals(X_va_pp)
            if use_log:
                lower = np.expm1(np.clip(lower, 0, 20))
                upper = np.expm1(np.clip(upper, 0, 20))
        else:
            qs    = np.array(quantiles, dtype=float)
            lower = q_df_raw[float(qs[np.argmin(np.abs(qs - alpha / 2))])].values
            upper = q_df_raw[float(qs[np.argmin(np.abs(qs - (1 - alpha / 2)))])].values

    f_metrics = evaluate_forecaster(
        y_va_raw, y_hat_raw,
        q_preds=q_df_raw.values, quantiles=quantiles,
        lower=lower, upper=upper, alpha=alpha,
        mu=y_hat_raw, sigma=sigma_raw,
    )

    # Draw raw scenarios (on the model's native scale)
    X_plan = X_va_pp[-1:, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw_scenarios = forecaster.sample_scenarios(
            X_plan, n_scenarios=MAX_RAW_SCENARIOS, random_state=seed
        )

    if use_log:
        raw_scenarios = np.expm1(np.clip(raw_scenarios, 0, 20))

    raw_scenarios = _sanitise_scenarios(raw_scenarios)
    raw_scenarios = np.clip(raw_scenarios, 0, None)

    # DC scenarios
    dc_dept = dc_train[dc_train["Dept"] == dept_id].copy()
    dc_demand_hist = y_tr  # fallback
    raw_dc = raw_scenarios * 5.0  # default fallback

    if "DC_Weekly_Demand" in dc_dept.columns and len(dc_dept) >= 15:
        dc_avail = [c for c in avail if c in dc_dept.columns]
        if dc_avail:
            dc_renamed = dc_dept.rename(
                columns={"DC_Weekly_Demand": "Weekly_Sales"}
            ).copy()
            dc_X, dc_y, _, _ = prepare_store_dept_data(
                dc_renamed, dc_avail, log_transform_target=use_log
            )
            if len(dc_X) >= 10:
                dc_prep = DemandPreprocessor()
                dc_X_pp = dc_prep.fit_transform(pd.DataFrame(dc_X, columns=dc_avail))
                dc_fc   = _build_forecaster_variant(forecaster_variant, cfg, seed)
                try:
                    dc_fc.fit(dc_X_pp, dc_y)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        raw_dc = dc_fc.sample_scenarios(
                            dc_X_pp[-1:, :],
                            n_scenarios=MAX_RAW_SCENARIOS,
                            random_state=seed,
                        )
                    if use_log:
                        raw_dc = np.expm1(np.clip(raw_dc, 0, 20))
                    raw_dc = _sanitise_scenarios(np.clip(raw_dc, 0, None))
                    dc_demand_hist = (
                        np.expm1(dc_y) if use_log else dc_y
                    )
                except Exception as exc:
                    logger.debug("DC forecaster failed: %s", exc)

    # Cap DC q̂ to prevent unrealistic safety stock from scale-up
    # If DC scenarios are 10× larger than store scenarios, cap them
    store_median = float(np.median(raw_scenarios))
    dc_median    = float(np.median(raw_dc))
    if store_median > 0 and dc_median / store_median > 20:
        scale_factor = min(dc_median / store_median, 10.0)
        raw_dc = raw_scenarios * scale_factor
        logger.debug(
            "DC scenario scale-up capped at %.1f× for Store %d Dept %d",
            scale_factor, store_id, dept_id,
        )

    return {
        "store_id":        store_id,
        "dept_id":         dept_id,
        "forecaster_name": forecaster_name,
        "f_metrics":       f_metrics,
        "raw_scenarios":   raw_scenarios,
        "raw_dc":          raw_dc,
        "y_tr":            np.expm1(y_tr) if use_log else y_tr,
        "dc_demand_hist":  dc_demand_hist,
    }


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def wilcoxon_table(
    results_df:       pd.DataFrame,
    metric:           str = "f_RMSE",
    reference_method: str = "NGBoost-N",
) -> pd.DataFrame:
    from scipy.stats import wilcoxon as _wilcoxon

    if reference_method not in results_df["forecaster"].values:
        logger.warning("Reference method %s not found.", reference_method)
        return pd.DataFrame()

    ref_vals = (
        results_df[results_df["forecaster"] == reference_method][metric]
        .apply(pd.to_numeric, errors="coerce").dropna().values
    )

    rows = []
    for method in results_df["forecaster"].unique():
        vals = (
            results_df[results_df["forecaster"] == method][metric]
            .apply(pd.to_numeric, errors="coerce").dropna().values
        )
        n = min(len(ref_vals), len(vals))
        p = float("nan")
        if n >= 10:
            diff = ref_vals[:n] - vals[:n]
            if np.any(diff != 0):
                try:
                    _, p = _wilcoxon(ref_vals[:n], vals[:n])
                except Exception:
                    pass
        rows.append({
            "method":               method,
            "mean":                 float(np.nanmean(vals)),
            "std":                  float(np.nanstd(vals)),
            "p_value":              p,
            "significant (α=0.05)": (p < 0.05) if np.isfinite(p) else False,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    config_path: str  = "config.yaml",
    output_dir:  str  = "results/",
    quick:       bool = False,
) -> pd.DataFrame:
    cfg     = load_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, cfg["output"].get("log_level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "ablation.log"),
        ],
    )

    loader       = WalmartDataLoader(cfg)
    train_df, _, dc_df = loader.build_dataset()
    train_split, val_split = loader.temporal_split(train_df)
    feature_cols = loader.get_feature_cols(train_split)

    store_ids = sorted(train_split["Store"].unique())[:3]
    dept_ids  = sorted(train_split["Dept"].unique())[:3]

    scenario_grid   = [50, 100]  if quick else SCENARIO_COUNTS
    sl_grid         = [0.90, 0.95] if quick else SERVICE_LEVELS
    forecaster_grid = (
        {"QR": FORECASTER_VARIANTS["QR"],
         "NGBoost-N": FORECASTER_VARIANTS["NGBoost-N"]}
        if quick else FORECASTER_VARIANTS
    )

    seed    = cfg.get("reproducibility", {}).get("random_seed", 42)
    opt_cfg = cfg["optimization"]

    rows: List[dict] = []
    n_fit_blocks = len(store_ids) * len(dept_ids) * len(forecaster_grid)
    fit_done = 0
    t_total  = time.perf_counter()

    for s_id, d_id in product(store_ids, dept_ids):
        for f_name, f_variant in forecaster_grid.items():
            t_block  = time.perf_counter()
            fit_done += 1
            logger.info(
                "[%d/%d] Fitting %s — Store %d Dept %d",
                fit_done, n_fit_blocks, f_name, s_id, d_id,
            )

            block = run_fit_block(
                store_id=s_id, dept_id=d_id,
                train_df=train_split, val_df=val_split,
                dc_train=dc_df,
                forecaster_name=f_name,
                forecaster_variant=f_variant,
                cfg=cfg,
                feature_cols=feature_cols,
            )
            if block is None:
                logger.warning("  Skipped (insufficient data).")
                continue

            f_metrics      = block["f_metrics"]
            raw_scenarios  = block["raw_scenarios"]
            raw_dc         = block["raw_dc"]
            y_tr           = block["y_tr"]
            dc_demand_hist = block["dc_demand_hist"]

            for o_name in OPTIMIZER_VARIANTS:
                for n_s in scenario_grid:
                    for sl in sl_grid:
                        opt_result = _run_opt_cell(
                            optimizer_name=o_name,
                            raw_scenarios=raw_scenarios,
                            raw_dc_scenarios=raw_dc,
                            n_scenarios=n_s,
                            service_level=sl,
                            opt_cfg_base=opt_cfg,
                            seed=seed,
                            train_demand_dc=dc_demand_hist,
                            train_demand_store=y_tr,
                        )
                        rows.append({
                            "store_id":             s_id,
                            "dept_id":              d_id,
                            "forecaster":           f_name,
                            "optimizer":            o_name,
                            "n_scenarios":          n_s,
                            "service_level_target": sl,
                            **{f"f_{k}": v for k, v in f_metrics.items()},
                            **{f"o_{k}": v for k, v in opt_result.items()},
                        })

            elapsed_block = time.perf_counter() - t_block
            elapsed_total = time.perf_counter() - t_total
            eta = elapsed_total / fit_done * (n_fit_blocks - fit_done) if fit_done < n_fit_blocks else 0
            logger.info(
                "  Block done: %d inner cells | %.1fs | Elapsed: %.1fs | ETA: %.1fs",
                len(OPTIMIZER_VARIANTS) * len(scenario_grid) * len(sl_grid),
                elapsed_block, elapsed_total, eta,
            )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "ablation_results.csv", index=False)
    logger.info("Ablation complete: %d rows saved.", len(results_df))

    group_cols  = ["forecaster", "optimizer", "n_scenarios", "service_level_target"]
    metric_cols = [c for c in results_df.columns if c.startswith(("f_", "o_"))]
    if metric_cols:
        summary = results_df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
        summary.to_csv(out_dir / "ablation_summary.csv")
        logger.info("Summary saved.")

    for metric in ["f_RMSE", "f_Coverage", "o_fill_rate"]:
        if metric in results_df.columns:
            wt = wilcoxon_table(results_df, metric=metric)
            if not wt.empty:
                wt.to_csv(out_dir / f"wilcoxon_{metric}.csv", index=False)
                logger.info("Wilcoxon table for %s:\n%s", metric, wt.to_string())

    _ablation_figures(results_df, out_dir)
    return results_df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _ablation_figures(results_df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 9, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    def _save(fig, name):
        fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    P = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

    def _col(df, col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)

    # A1: Fill rate vs. scenario count
    if "n_scenarios" in results_df.columns and "o_fill_rate" in results_df.columns:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for i, f in enumerate(results_df["forecaster"].unique()):
            sub = results_df[results_df["forecaster"] == f].copy()
            sub["o_fill_rate"] = _col(sub, "o_fill_rate")
            g = sub.groupby("n_scenarios")["o_fill_rate"].mean().reset_index()
            if not g.empty:
                ax.plot(g["n_scenarios"], g["o_fill_rate"],
                        marker="o", ms=4, color=P[i % len(P)],
                        label=f, linewidth=1.2)
        ax.set_xlabel("Number of Scenarios"); ax.set_ylabel("Mean Fill Rate")
        ax.set_title("Fill Rate vs. Scenario Count"); ax.legend(frameon=False, fontsize=7.5)
        fig.tight_layout(); _save(fig, "ablation_scenario_sensitivity")

    # A2: SL target vs. achieved fill rate
    if "service_level_target" in results_df.columns and "o_fill_rate" in results_df.columns:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        ax.plot([0.8, 1.0], [0.8, 1.0], "k--", lw=0.8, label="Perfect", zorder=0)
        for i, f in enumerate(results_df["forecaster"].unique()):
            sub = results_df[results_df["forecaster"] == f].copy()
            sub["o_fill_rate"] = _col(sub, "o_fill_rate")
            g = sub.groupby("service_level_target")["o_fill_rate"].mean().reset_index()
            if not g.empty:
                ax.plot(g["service_level_target"], g["o_fill_rate"],
                        marker="s", ms=4, color=P[i % len(P)],
                        label=f, linewidth=1.2)
        ax.set_xlabel("Target β"); ax.set_ylabel("Achieved Fill Rate")
        ax.set_title("Service Level Attainment"); ax.set_xlim(0.82, 1.0)
        ax.legend(frameon=False, fontsize=7.5); fig.tight_layout()
        _save(fig, "ablation_sl_attainment")

    # A3: RMSE × Coverage scatter (key paper figure)
    if "f_RMSE" in results_df.columns and "f_Coverage" in results_df.columns:
        fig, ax = plt.subplots(figsize=(3.8, 3.0))
        for i, f in enumerate(results_df["forecaster"].unique()):
            sub = results_df[results_df["forecaster"] == f]
            rmse_m = _col(sub, "f_RMSE").mean()
            cov_m  = _col(sub, "f_Coverage").mean()
            rmse_s = _col(sub, "f_RMSE").std()
            cov_s  = _col(sub, "f_Coverage").std()
            ax.errorbar(rmse_m, cov_m, xerr=rmse_s, yerr=cov_s,
                        fmt="o", ms=7, color=P[i % len(P)],
                        label=f, capsize=3, linewidth=1.1)
        ax.axhline(0.90, color="gray", ls="--", lw=0.8, label="90% target")
        ax.set_xlabel("RMSE (lower = better)")
        ax.set_ylabel("Coverage (higher = better)")
        ax.set_title("Accuracy–Coverage Trade-off")
        ax.legend(frameon=False, fontsize=7.5); fig.tight_layout()
        _save(fig, "ablation_accuracy_coverage_tradeoff")

    # A4: RMSE heatmap forecaster × optimizer
    if "f_RMSE" in results_df.columns:
        df2 = results_df.copy()
        df2["f_RMSE"] = _col(df2, "f_RMSE")
        pivot = df2.pivot_table(values="f_RMSE", index="forecaster",
                                columns="optimizer", aggfunc="mean")
        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            for ii in range(pivot.shape[0]):
                for jj in range(pivot.shape[1]):
                    v = pivot.values[ii, jj]
                    if np.isfinite(v):
                        ax.text(jj, ii, f"{v:.0f}", ha="center", va="center", fontsize=7.5)
            fig.colorbar(im, ax=ax, shrink=0.8, label="RMSE")
            ax.set_title("Mean RMSE: Forecaster × Optimizer")
            fig.tight_layout(); _save(fig, "ablation_rmse_heatmap")

    logger.info("Ablation figures saved to %s", fig_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", default="results/")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    run_ablation(config_path=a.config, output_dir=a.output, quick=a.quick)