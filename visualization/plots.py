"""
visualization/plots.py
=======================
Publication-quality figures for the MEIO stochastic optimization paper.

Changes vs. previous version
------------------------------
1. PNG-only output (PDF removed).
2. METHOD_DISPLAY_NAMES dict maps internal names → display labels everywhere.
3. Safety stock figure: matches by display name so all methods appear.
4. Fill rate figure: uses display names on x-axis ticks.
5. Pinball loss: filters to only methods actually present in the data.
6. Accuracy-coverage: error bars capped at ±1 std (not full range).
7. All _save() calls removed PDF branch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method display name mapping
# ---------------------------------------------------------------------------

METHOD_DISPLAY_NAMES = {
    # Internal names (from build_forecaster / FORECASTER_VARIANTS keys)
    "ngboost":             "NGBoost-N",
    "quantile_regression": "QR",
    "conformal":           "Conformal",
    # Ablation keys (already display names — identity mapping)
    "NGBoost-N":           "NGBoost-N",
    "NGBoost-LN":          "NGBoost-LN",
    "QR":                  "QR",
    "Conformal":           "Conformal",
    # Classical baseline
    "Classical (Normal)":  "Classical",
    "Classical":           "Classical",
}

# Active methods (NGBoost-LN removed)
ACTIVE_METHODS = {"QR", "NGBoost-N", "Conformal", "ngboost",
                  "quantile_regression", "conformal"}

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

_COLORS = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
}
_PALETTE = list(_COLORS.values())

# Consistent color assignment per display name
_METHOD_COLOR = {
    "QR":              _COLORS["blue"],
    "NGBoost-N":       _COLORS["orange"],
    "Conformal":       _COLORS["green"],
    "Classical":       _COLORS["red"],
    "Classical (Normal)": _COLORS["red"],
}

_RC = {
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.linewidth":    0.6,
    "lines.linewidth":   1.2,
    "grid.linewidth":    0.3,
    "grid.alpha":        0.4,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
}

_W1 = 3.5
_W2 = 7.2
_H  = 2.6


def _apply_style() -> None:
    matplotlib.rcParams.update(_RC)


def _save(fig: plt.Figure, path: Union[str, Path]) -> None:
    """Save figure as PNG only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    logger.info("Figure saved: %s.png", path)


def _display(name: str) -> str:
    """Convert internal method name to display label."""
    return METHOD_DISPLAY_NAMES.get(str(name), str(name))


def _method_color(name: str) -> str:
    display = _display(name)
    return _METHOD_COLOR.get(display, _PALETTE[hash(display) % len(_PALETTE)])


# ---------------------------------------------------------------------------
# Plotter class
# ---------------------------------------------------------------------------

class MEIOPlotter:
    """
    Factory class for all publication figures.

    Parameters
    ----------
    output_dir : str or Path
    """

    def __init__(self, output_dir: str = "results/figures") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _apply_style()

    # ------------------------------------------------------------------
    # 1. Demand distribution comparison
    # ------------------------------------------------------------------

    def plot_demand_distribution(
        self,
        demand: np.ndarray,
        store_id: Union[int, str] = 1,
        dept_id: Union[int, str] = 1,
        bins: int = 40,
    ) -> plt.Figure:
        from scipy.stats import norm, shapiro, skew, kurtosis
        from scipy.stats import probplot

        fig, axes = plt.subplots(1, 2, figsize=(_W2, _H))
        ax = axes[0]
        mu, sigma = float(np.mean(demand)), float(np.std(demand) + 1e-8)
        x = np.linspace(demand.min() * 0.8, demand.max() * 1.2, 300)
        pdf = norm.pdf(x, mu, sigma)
        ax.hist(demand, bins=bins, density=True,
                color=_COLORS["blue"], alpha=0.55, edgecolor="white", linewidth=0.3,
                label="Empirical")
        ax.plot(x, pdf, color=_COLORS["red"], linewidth=1.5,
                label=r"Normal fit $\mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$")
        ax.set_xlabel("Weekly Sales (USD)")
        ax.set_ylabel("Density")
        ax.set_title(f"Store {store_id} | Dept {dept_id}")
        ax.legend(frameon=False)

        sk = skew(demand)
        ku = kurtosis(demand)
        stat, p_sw = shapiro(demand[:min(len(demand), 5000)])
        ax.text(0.97, 0.95,
                f"Skew = {sk:.2f}\nEx. Kurt = {ku:.2f}\nShapiro p = {p_sw:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5))

        ax2 = axes[1]
        (osm, osr), (slope, intercept, r) = probplot(demand, dist="norm", fit=True)
        ax2.scatter(osm, osr, s=6, alpha=0.5, color=_COLORS["blue"], linewidths=0)
        qq_line = np.array([osm.min(), osm.max()])
        ax2.plot(qq_line, slope * qq_line + intercept,
                 color=_COLORS["red"], linewidth=1.2, label=f"R² = {r**2:.3f}")
        ax2.set_xlabel("Theoretical Quantiles")
        ax2.set_ylabel("Sample Quantiles")
        ax2.set_title("Normal Q-Q Plot")
        ax2.legend(frameon=False)

        fig.suptitle("Demand Distribution Analysis — Motivation for Distribution-Free Methods",
                     y=1.02, fontsize=9, style="italic")
        fig.tight_layout()
        _save(fig, self.output_dir / "fig1_demand_distribution")
        return fig

    # ------------------------------------------------------------------
    # 2. Quantile calibration
    # ------------------------------------------------------------------

    def plot_quantile_calibration(
        self,
        y_true: np.ndarray,
        quantile_preds: Dict[str, pd.DataFrame],
        quantiles: List[float] = None,
    ) -> plt.Figure:
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        fig, ax = plt.subplots(figsize=(_W1 * 1.15, _W1 * 1.15))
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration", zorder=0)

        for method, df in quantile_preds.items():
            if method not in ACTIVE_METHODS:
                continue
            display = _display(method)
            empirical = []
            for q in quantiles:
                col = q if q in df.columns else min(df.columns, key=lambda c: abs(c - q))
                empirical.append(float(np.mean(y_true <= df[col].values)))
            ax.plot(quantiles, empirical, marker="o", markersize=4,
                    color=_method_color(method), label=display, linewidth=1.2)

        ax.set_xlabel("Nominal Quantile Level")
        ax.set_ylabel("Empirical Coverage")
        ax.set_title("Quantile Calibration (Reliability Diagram)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(frameon=False, loc="upper left")
        ax.set_aspect("equal")
        fig.tight_layout()
        _save(fig, self.output_dir / "fig2_quantile_calibration")
        return fig

    # ------------------------------------------------------------------
    # 3. Coverage by horizon
    # ------------------------------------------------------------------

    def plot_coverage_by_horizon(
        self,
        coverage_by_horizon: Dict[str, List[float]],
        target_coverage: float = 0.90,
        horizons: Optional[List[int]] = None,
    ) -> plt.Figure:
        if horizons is None:
            n_h = max(len(v) for v in coverage_by_horizon.values())
            horizons = list(range(1, n_h + 1))

        fig, ax = plt.subplots(figsize=(_W2 * 0.6, _H))
        ax.axhline(target_coverage, color="black", linestyle="--",
                   linewidth=0.9, label=f"Target {target_coverage*100:.0f}%", zorder=0)

        for method, covs in coverage_by_horizon.items():
            if method not in ACTIVE_METHODS:
                continue
            ax.plot(horizons[:len(covs)], covs, marker="s", markersize=4,
                    color=_method_color(method), label=_display(method), linewidth=1.2)

        ax.set_xlabel("Forecast Horizon (weeks)")
        ax.set_ylabel("Empirical Coverage")
        ax.set_title("Prediction Interval Coverage by Horizon")
        ax.set_ylim(0.5, 1.02)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.legend(frameon=False)
        fig.tight_layout()
        _save(fig, self.output_dir / "fig3_coverage_by_horizon")
        return fig

    # ------------------------------------------------------------------
    # 4. Safety stock comparison — FIXED
    # ------------------------------------------------------------------

    def plot_safety_stock_comparison(
        self,
        store_ids: List[Union[int, str]],
        ss_dict: Dict[str, np.ndarray],
    ) -> plt.Figure:
        """
        Bar chart of safety stock levels.

        FIX: normalise all keys through _display() before plotting so
        internal names (ngboost / quantile_regression / conformal) and
        display names (NGBoost-N / QR / Conformal) both render correctly.
        """
        # Normalise keys to display names
        ss_display: Dict[str, np.ndarray] = {}
        for raw_name, vals in ss_dict.items():
            disp = _display(raw_name)
            arr  = np.asarray(vals, dtype=float)
            if np.any(arr > 0):      # skip all-zero series
                ss_display[disp] = arr

        if not ss_display:
            logger.warning("Safety stock comparison: all methods have zero values — skipping.")
            return plt.figure()

        n_stores  = len(store_ids)
        methods   = list(ss_display.keys())
        n_methods = len(methods)
        x         = np.arange(n_stores)
        width     = 0.8 / n_methods

        fig, ax = plt.subplots(figsize=(_W2, _H))

        for idx, method in enumerate(methods):
            offset = (idx - n_methods / 2 + 0.5) * width
            vals   = ss_display[method][:n_stores]
            ax.bar(x + offset, vals, width=width * 0.92,
                   color=_method_color(method), label=method,
                   alpha=0.85, edgecolor="white", linewidth=0.3)

        ax.set_xlabel("Store ID")
        ax.set_ylabel("Safety Stock (USD)")
        ax.set_title("Safety Stock Allocation: Data-Driven vs. Classical Normal")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in store_ids], rotation=45, ha="right")
        ax.legend(frameon=False, ncol=n_methods)
        fig.tight_layout()
        _save(fig, self.output_dir / "fig4_safety_stock_comparison")
        return fig

    # ------------------------------------------------------------------
    # 5. Cost decomposition
    # ------------------------------------------------------------------

    def plot_cost_decomposition(
        self,
        methods: List[str],
        holding_costs: np.ndarray,
        backorder_costs: np.ndarray,
        replenishment_costs: np.ndarray,
        emergency_costs: Optional[np.ndarray] = None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(_W2 * 0.75, _H * 1.1))
        y      = np.arange(len(methods))
        bar_h  = 0.55
        labels = ["Holding", "Backorder", "Replenishment"]
        arrays = [holding_costs, backorder_costs, replenishment_costs]
        colors = [_COLORS["blue"], _COLORS["red"], _COLORS["green"]]
        if emergency_costs is not None:
            labels.append("Emergency"); arrays.append(emergency_costs)
            colors.append(_COLORS["orange"])

        lefts = np.zeros(len(methods))
        for label, arr, color in zip(labels, arrays, colors):
            ax.barh(y, arr, left=lefts, height=bar_h,
                    color=color, label=label, alpha=0.88,
                    edgecolor="white", linewidth=0.3)
            lefts += arr

        ax.set_yticks(y)
        ax.set_yticklabels([_display(m) for m in methods])
        ax.set_xlabel("Expected Total Cost (USD)")
        ax.set_title("Cost Decomposition by Method")
        ax.legend(frameon=False, loc="lower right", ncol=2)
        fig.tight_layout()
        _save(fig, self.output_dir / "fig5_cost_decomposition")
        return fig

    # ------------------------------------------------------------------
    # 6. Pareto frontier
    # ------------------------------------------------------------------

    def plot_pareto_frontier(
        self,
        service_levels: Dict[str, np.ndarray],
        total_costs: Dict[str, np.ndarray],
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(_W1 * 1.2, _W1 * 1.1))

        for method in service_levels:
            if method not in ACTIVE_METHODS:
                continue
            sl    = np.asarray(service_levels[method])
            tc    = np.asarray(total_costs[method])
            order = np.argsort(sl)
            ax.plot(sl[order], tc[order], color=_method_color(method),
                    marker="o", markersize=4, label=_display(method), linewidth=1.2)

        ax.set_xlabel("Achieved Fill Rate (Type-2 Service Level)")
        ax.set_ylabel("Expected Total Cost (USD)")
        ax.set_title("Cost–Service Level Pareto Frontier")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.legend(frameon=False)
        fig.tight_layout()
        _save(fig, self.output_dir / "fig6_pareto_frontier")
        return fig

    # ------------------------------------------------------------------
    # 7. Scenario fan chart
    # ------------------------------------------------------------------

    def plot_scenario_fan(
        self,
        dates: pd.DatetimeIndex,
        y_true: np.ndarray,
        y_pred_mean: np.ndarray,
        quantile_bands: Dict[Tuple[float, float], np.ndarray],
        title: str = "Demand Forecast with Uncertainty Bands",
        n_history: int = 26,
    ) -> plt.Figure:
        T_hist = n_history
        T_fore = len(y_pred_mean)
        hist_dates = dates[:T_hist]
        fore_dates = dates[T_hist: T_hist + T_fore]

        fig, ax = plt.subplots(figsize=(_W2, _H * 1.1))
        ax.plot(hist_dates, y_true[:T_hist],
                color=_COLORS["black"], linewidth=1.1, label="Actual (history)", zorder=5)
        if len(y_true) > T_hist:
            ax.plot(fore_dates, y_true[T_hist: T_hist + T_fore],
                    color=_COLORS["black"], linewidth=1.1, linestyle="--",
                    label="Actual (test)", zorder=5)

        alphas = np.linspace(0.12, 0.35, len(quantile_bands))
        sorted_bands = sorted(quantile_bands.keys(), key=lambda x: x[1] - x[0], reverse=True)
        for (lo_q, hi_q), band_alpha in zip(sorted_bands, alphas):
            lo_arr, hi_arr = quantile_bands[(lo_q, hi_q)]
            ax.fill_between(fore_dates, lo_arr, hi_arr,
                            color=_COLORS["blue"], alpha=band_alpha,
                            label=f"{int(round((hi_q - lo_q)*100))}% PI")

        ax.plot(fore_dates, y_pred_mean, color=_COLORS["blue"],
                linewidth=1.4, label="Forecast (median)", zorder=4)
        ax.axvline(hist_dates[-1], color="gray", linestyle=":", linewidth=0.9, zorder=3)
        ax.set_xlabel("Date"); ax.set_ylabel("Weekly Sales (USD)")
        ax.set_title(title)
        ax.legend(frameon=False, ncol=3, fontsize=7.5)
        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout()
        _save(fig, self.output_dir / "fig7_scenario_fan")
        return fig

    # ------------------------------------------------------------------
    # 8. Fill rate comparison — FIXED (display names + y-axis)
    # ------------------------------------------------------------------

    def plot_fill_rate_comparison(
        self,
        fill_rates: Dict[str, np.ndarray],
        target: float = 0.95,
    ) -> plt.Figure:
        """
        Box plot of fill rates per method.

        FIX 1: Keys normalised through _display() — internal names map to
                correct display labels.
        FIX 2: NGBoost-LN filtered out (all-zero coverage would distort).
        FIX 3: Y-axis floor set to max(0, target - 0.1) so minor variations
                around the target are visible without compressing the full 0-100% range.
        """
        # Filter and normalise
        fr_display: Dict[str, np.ndarray] = {}
        for raw_name, vals in fill_rates.items():
            disp = _display(raw_name)
            if disp == "NGBoost-LN":
                continue
            vals_arr = np.asarray(vals, dtype=float)
            vals_arr = vals_arr[np.isfinite(vals_arr)]
            if len(vals_arr) > 0:
                fr_display[disp] = vals_arr

        if not fr_display:
            logger.warning("Fill rate comparison: no valid data — skipping.")
            return plt.figure()

        methods = list(fr_display.keys())
        data    = [fr_display[m] for m in methods]

        fig, ax = plt.subplots(figsize=(_W1 * 1.3, _H))

        bp = ax.boxplot(
            data, patch_artist=True, widths=0.45,
            medianprops=dict(color="white", linewidth=1.5),
            whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
            flierprops=dict(marker="o", markersize=3, alpha=0.4, linewidth=0),
        )
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(_method_color(method))
            patch.set_alpha(0.80)

        ax.axhline(target, color=_COLORS["red"], linestyle="--",
                   linewidth=0.9, label=f"Target {target*100:.0f}%")
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("Fill Rate (Type-2 SL)")
        ax.set_title("Fill Rate Comparison Across Methods")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

        # FIX: set sensible y-axis range based on actual data
        all_vals = np.concatenate(data)
        y_min = max(0.0, float(np.nanmin(all_vals)) - 0.02)
        y_max = min(1.02, float(np.nanmax(all_vals)) + 0.02)
        ax.set_ylim(y_min, y_max)

        ax.legend(frameon=False)
        fig.tight_layout()
        _save(fig, self.output_dir / "fig8_fill_rate_comparison")
        return fig

    # ------------------------------------------------------------------
    # 9. CRPS comparison
    # ------------------------------------------------------------------

    def plot_crps_comparison(
        self,
        crps_scores: Dict[str, np.ndarray],
        node_labels: Optional[List[str]] = None,
    ) -> plt.Figure:
        methods = [m for m in crps_scores if _display(m) != "NGBoost-LN"]
        data    = [crps_scores[m] for m in methods]
        if not methods:
            return plt.figure()

        fig, ax = plt.subplots(figsize=(_W1 * 1.1, _H))
        parts   = ax.violinplot(data, positions=range(1, len(methods) + 1),
                                showmedians=True, showextrema=True, widths=0.5)
        for pc, method in zip(parts["bodies"], methods):
            pc.set_facecolor(_method_color(method)); pc.set_alpha(0.70)
        parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(1.5)
        for key in ["cmins", "cmaxes", "cbars"]:
            parts[key].set_linewidth(0.7); parts[key].set_color("gray")

        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([_display(m) for m in methods], rotation=20, ha="right")
        ax.set_ylabel("CRPS (lower = better)")
        ax.set_title("Distributional Forecast Quality (CRPS)")
        fig.tight_layout()
        _save(fig, self.output_dir / "fig9_crps_comparison")
        return fig

    # ------------------------------------------------------------------
    # 10. Reorder point heatmap
    # ------------------------------------------------------------------

    def plot_reorder_point_heatmap(
        self,
        rop_matrix: np.ndarray,
        store_ids: List[Union[int, str]],
        dept_ids: List[Union[int, str]],
        method: str = "NGBoost-N + Stochastic MILP",
    ) -> plt.Figure:
        fig, ax = plt.subplots(
            figsize=(_W2, max(2.5, 0.28 * len(store_ids) + 1.0))
        )
        im   = ax.imshow(rop_matrix, aspect="auto", cmap="YlOrRd")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("Reorder Point (USD)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax.set_xticks(range(len(dept_ids)))
        ax.set_xticklabels([str(d) for d in dept_ids], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(store_ids)))
        ax.set_yticklabels([str(s) for s in store_ids], fontsize=7)
        ax.set_xlabel("Department ID"); ax.set_ylabel("Store ID")
        ax.set_title(f"Optimal Reorder Points — {_display(method)}")
        fig.tight_layout()
        _save(fig, self.output_dir / "fig10_reorder_point_heatmap")
        return fig

    # ------------------------------------------------------------------
    # Summary radar chart
    # ------------------------------------------------------------------

    def plot_method_summary(
        self,
        results_df: pd.DataFrame,
        metric_cols: List[str] = None,
    ) -> plt.Figure:
        if metric_cols is None:
            metric_cols = [c for c in results_df.columns if c != "method"]

        methods = (results_df["method"].tolist()
                   if "method" in results_df.columns
                   else results_df.index.tolist())
        values  = results_df[metric_cols].values
        N       = len(metric_cols)
        angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(_W1 * 1.2, _W1 * 1.2),
                               subplot_kw=dict(polar=True))
        for idx, (method, row) in enumerate(zip(methods, values)):
            vals = row.tolist() + row[:1].tolist()
            ax.plot(angles, vals, color=_method_color(method),
                    linewidth=1.3, label=_display(method))
            ax.fill(angles, vals, color=_method_color(method), alpha=0.12)

        ax.set_thetagrids(np.degrees(angles[:-1]), metric_cols, fontsize=7.5)
        ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.50, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6.5)
        ax.set_title("Method Performance Summary", pad=18)
        ax.legend(frameon=False, loc="upper right",
                  bbox_to_anchor=(1.35, 1.15), fontsize=7.5)
        fig.tight_layout()
        _save(fig, self.output_dir / "fig_summary_radar")
        return fig