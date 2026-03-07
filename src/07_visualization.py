"""
07_visualization.py — Publication-quality results visualization.

Generates all figures for the thesis/paper.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import config

# Publication style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def fig_cost_comparison(exp1_df):
    """Bar chart comparing Classical vs ML-Enhanced total costs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Classical\n(sigma historical)", "ML-Enhanced\n(sigma residual)"]
    ss_vals = [exp1_df["Classical: Total SS"].iloc[0],
               exp1_df["ML-Enhanced: Total SS"].iloc[0]]
    colors = ["#e74c3c", "#27ae60"]

    axes[0].bar(labels, ss_vals, color=colors, edgecolor="white", width=0.5)
    axes[0].set_ylabel("Total Safety Stock (units)")
    axes[0].set_title("(a) Safety Stock Comparison")

    reduction = exp1_df["SS Reduction (%)"].iloc[0]
    axes[0].annotate(f"down {reduction:.1f}%",
                     xy=(1, ss_vals[1]), xytext=(1, ss_vals[1] * 1.08),
                     fontsize=14, fontweight="bold", color="#27ae60", ha="center")

    cost_vals = [exp1_df["Classical: Weekly Cost"].iloc[0],
                 exp1_df["ML-Enhanced: Weekly Cost"].iloc[0]]
    axes[1].bar(labels, cost_vals, color=colors, edgecolor="white", width=0.5)
    axes[1].set_ylabel("Weekly Holding Cost ($)")
    axes[1].set_title("(b) Holding Cost Comparison")

    cost_red = exp1_df["Cost Reduction (%)"].iloc[0]
    axes[1].annotate(f"down {cost_red:.1f}%",
                     xy=(1, cost_vals[1]), xytext=(1, cost_vals[1] * 1.08),
                     fontsize=14, fontweight="bold", color="#27ae60", ha="center")

    plt.suptitle("Experiment 1: Classical vs. ML-Enhanced Safety Stock Optimization",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig_cost_comparison.png",
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: fig_cost_comparison.png")


def fig_sensitivity_analysis(exp2_df):
    """Service level vs. total cost curves for both approaches."""
    totals = exp2_df[exp2_df["echelon"] == "Total"].copy()
    totals["service_level_pct"] = totals["service_level"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for approach, color, marker in [
        ("Classical", "#e74c3c", "o"),
        ("ML-Enhanced", "#27ae60", "s"),
    ]:
        adf = totals[totals["approach"] == approach]
        axes[0].plot(adf["service_level_pct"], adf["total_holding_cost"],
                     marker=marker, color=color, label=approach, lw=2, ms=8)

    axes[0].set_xlabel("Service Level (%)")
    axes[0].set_ylabel("Total Weekly Holding Cost ($)")
    axes[0].set_title("(a) Cost-Service Tradeoff")
    axes[0].legend()

    reductions = []
    for sl in totals["service_level"].unique():
        c_cost = totals[(totals["service_level"] == sl) &
                        (totals["approach"] == "Classical")]["total_holding_cost"].iloc[0]
        m_cost = totals[(totals["service_level"] == sl) &
                        (totals["approach"] == "ML-Enhanced")]["total_holding_cost"].iloc[0]
        reductions.append({
            "service_level_pct": sl * 100,
            "cost_reduction_pct": (1 - m_cost / c_cost) * 100,
        })
    red_df = pd.DataFrame(reductions)

    axes[1].bar(red_df["service_level_pct"].astype(str),
                red_df["cost_reduction_pct"],
                color="#3498db", edgecolor="white", width=0.5)
    axes[1].set_xlabel("Service Level (%)")
    axes[1].set_ylabel("Cost Reduction from ML Enhancement (%)")
    axes[1].set_title("(b) ML-Driven Cost Reduction by Service Level")

    for i, row in red_df.iterrows():
        axes[1].text(i, row["cost_reduction_pct"] + 0.3,
                     f"{row['cost_reduction_pct']:.1f}%",
                     ha="center", fontweight="bold", fontsize=11)

    plt.suptitle("Experiment 2: Service Level Sensitivity Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig_sensitivity_analysis.png",
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: fig_sensitivity_analysis.png")


def fig_echelon_breakdown(exp3_df):
    """Stacked bar chart showing safety stock distribution by echelon."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    echelon_labels = {1: "DC", 2: "Warehouse", 3: "Store-Dept"}
    echelon_colors = {1: "#3498db", 2: "#f39c12", 3: "#e74c3c"}

    for idx, (metric, title) in enumerate([
        ("pct_safety_stock", "(a) Safety Stock Distribution (%)"),
        ("pct_holding_cost", "(b) Holding Cost Distribution (%)"),
    ]):
        ax = axes[idx]
        approaches = exp3_df["approach"].unique()
        x = np.arange(len(approaches))
        bottom = np.zeros(len(approaches))

        for echelon in sorted(exp3_df["echelon"].unique()):
            vals = []
            for approach in approaches:
                v = exp3_df[
                    (exp3_df["approach"] == approach) & (exp3_df["echelon"] == echelon)
                ][metric].iloc[0]
                vals.append(v)
            vals = np.array(vals)
            ax.bar(x, vals, width=0.4, bottom=bottom,
                   color=echelon_colors[echelon],
                   label=echelon_labels[echelon], edgecolor="white")
            for j, v in enumerate(vals):
                if v > 5:
                    ax.text(j, bottom[j] + v / 2, f"{v:.1f}%",
                            ha="center", va="center", fontsize=9,
                            fontweight="bold", color="white")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(approaches)
        ax.set_ylabel("Percentage (%)")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.set_ylim(0, 110)

    plt.suptitle("Experiment 3: Echelon Contribution Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig_echelon_breakdown.png",
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: fig_echelon_breakdown.png")


def fig_model_comparison(exp4_df):
    """Scatter of forecast accuracy (RMSE) vs. inventory cost."""
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {"Historical Baseline": "#e74c3c", "LightGBM": "#27ae60", "XGBoost": "#3498db"}
    markers = {"Historical Baseline": "o", "LightGBM": "s", "XGBoost": "D"}

    for _, row in exp4_df.iterrows():
        model = row["Forecast Model"]
        ax.scatter(row["Forecast RMSE"], row["Total Holding Cost/wk"],
                   s=200, color=colors.get(model, "gray"),
                   marker=markers.get(model, "o"),
                   label=model, zorder=5, edgecolors="white", linewidth=1.5)

    ax.set_xlabel("Forecast RMSE (lower = better forecast)")
    ax.set_ylabel("Total Weekly Holding Cost ($)")
    ax.set_title("Experiment 4: Forecast Accuracy Impact on Inventory Cost")
    ax.legend(title="Forecasting Model", fontsize=11)

    ax.annotate("Better forecasting\n-> Lower inventory cost",
                xy=(exp4_df["Forecast RMSE"].min() * 1.02,
                    exp4_df["Total Holding Cost/wk"].min() * 1.02),
                fontsize=10, fontstyle="italic", color="gray", ha="left")

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig_model_comparison.png",
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: fig_model_comparison.png")


def fig_network_topology(network=None):
    """Schematic of the multi-echelon network topology."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 9)
    ax.axis("off")

    ax.text(5, 8.5, "Multi-Echelon Supply Chain Network Topology",
            fontsize=16, fontweight="bold", ha="center")

    echelon_x = {1: 1.5, 2: 5, 3: 9}
    for ech, x in echelon_x.items():
        label = {1: "Echelon 1\nDistribution\nCenters",
                 2: "Echelon 2\nRegional\nWarehouses",
                 3: "Echelon 3\nStore-Dept\n(Demand Nodes)"}[ech]
        ax.text(x, 8, label, fontsize=10, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1",
                          edgecolor="#bdc3c7"))

    dc_y = [6, 4, 2]
    dc_labels = ["DC R1\n(Type A)", "DC R2\n(Type B)", "DC R3\n(Type C)"]
    dc_colors = ["#3498db", "#f39c12", "#e74c3c"]

    for i, (y, label, color) in enumerate(zip(dc_y, dc_labels, dc_colors)):
        circle = plt.Circle((1.5, y), 0.6, color=color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(1.5, y, label, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white")

    wh_positions = [
        (5, 7, "WH R1-A"), (5, 5.5, "WH R1-B"),
        (5, 4, "WH R2-A"), (5, 2.5, "WH R2-B"),
        (5, 1, "WH R3-C"),
    ]
    for x, y, label in wh_positions:
        rect = mpatches.FancyBboxPatch((x - 0.7, y - 0.35), 1.4, 0.7,
                                        boxstyle="round,pad=0.1",
                                        facecolor="#2ecc71", alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white")

    store_y_positions = [7.5, 7, 6.5, 5.5, 4.5, 4, 3, 2.5, 1.5, 0.5]
    for i, y in enumerate(store_y_positions):
        rect = mpatches.FancyBboxPatch((8.5, y - 0.2), 1.2, 0.4,
                                        boxstyle="round,pad=0.05",
                                        facecolor="#9b59b6", alpha=0.7)
        ax.add_patch(rect)
        ax.text(9.1, y, f"S{i+1}_D{np.random.randint(1,20)}", ha="center",
                va="center", fontsize=7, color="white")

    for dc_y_val, wh_ys in [(6, [7, 5.5]), (4, [4, 2.5]), (2, [1])]:
        for wy in wh_ys:
            ax.annotate("", xy=(4.3, wy), xytext=(2.1, dc_y_val),
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        lw=1.5, alpha=0.6))

    wh_store_map = {7: [7.5, 7], 5.5: [6.5, 5.5], 4: [4.5, 4],
                    2.5: [3, 2.5], 1: [1.5, 0.5]}
    for wy, sys_list in wh_store_map.items():
        for sy in sys_list:
            ax.annotate("", xy=(8.5, sy), xytext=(5.7, wy),
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        lw=1, alpha=0.4))

    ax.annotate("External\nSupply", xy=(0.9, 4), xytext=(-0.5, 4),
                fontsize=9, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="black", lw=2))

    ax.text(0, -0.5,
            "Parameters per node: Lead time (Lj), Holding cost (hj), "
            "Service time bounds (0 <= Sj <= Sj_max), Demand (mu_j, sigma_j)",
            fontsize=9, fontstyle="italic", color="gray")

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig_network_topology.png",
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: fig_network_topology.png")


def fig_monte_carlo_validation(exp5_df):
    """Figure: MC simulated fill rates vs target service levels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Normal demand: simulated vs target fill rate
    ax = axes[0]
    for approach, color, marker in [
        ("Classical", "#e74c3c", "o"),
        ("ML-Enhanced", "#27ae60", "s"),
    ]:
        adf = exp5_df[exp5_df["approach"] == approach]
        ax.plot(adf["service_level"] * 100, adf["normal_fill_rate"] * 100,
                marker=marker, color=color, label=f"{approach} (simulated)",
                lw=2, ms=8, zorder=5)

    # Perfect target line
    sls = sorted(exp5_df["service_level"].unique())
    ax.plot([s * 100 for s in sls], [s * 100 for s in sls],
            ls="--", color="gray", lw=1.5, label="Target = Simulated", zorder=1)

    ax.set_xlabel("Target Service Level (%)")
    ax.set_ylabel("Simulated Fill Rate (%)")
    ax.set_title("(a) Normal Demand: GSM Validation")
    ax.legend(fontsize=9)
    ax.set_xlim(88, 100)
    ax.set_ylim(88, 100)

    # (b) Normal vs Log-Normal comparison
    ax = axes[1]
    width = 0.18
    sls_pct = [int(s * 100) for s in sls]
    x = np.arange(len(sls_pct))

    for i, approach in enumerate(["Classical", "ML-Enhanced"]):
        adf = exp5_df[exp5_df["approach"] == approach]
        gap_normal = (adf["normal_fill_rate"].values - adf["target_fill_rate"].values) * 100
        gap_ln = (adf["lognormal_fill_rate"].values - adf["target_fill_rate"].values) * 100

        offset = -width / 2 if i == 0 else width / 2
        color_n = "#3498db" if i == 0 else "#27ae60"
        color_ln = "#e74c3c" if i == 0 else "#f39c12"

        ax.bar(x + offset - width / 2, gap_normal, width, label=f"{approach} Normal",
               color=color_n, edgecolor="white", alpha=0.85)
        ax.bar(x + offset + width / 2, gap_ln, width, label=f"{approach} Log-Normal",
               color=color_ln, edgecolor="white", alpha=0.85)

    ax.axhline(0, color="black", lw=1, ls="-")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}%" for s in sls_pct])
    ax.set_xlabel("Target Service Level")
    ax.set_ylabel("Fill Rate Gap (pp) [Simulated - Target]")
    ax.set_title("(b) Fill Rate Gap by Distribution")
    ax.legend(fontsize=8, ncol=2)

    plt.suptitle("Experiment 5: Monte Carlo Service Level Validation",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig_monte_carlo_validation.png",
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: fig_monte_carlo_validation.png")


def fig_seio_vs_meio(exp6_df):
    """Figure: SEIO vs MEIO cost comparison showing value of coordination."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sls = sorted(exp6_df["service_level"].unique())

    # (a) Absolute costs: SEIO vs MEIO at each service level
    ax = axes[0]
    for approach, color_seio, color_meio in [
        ("Classical", "#e74c3c", "#c0392b"),
        ("ML-Enhanced", "#27ae60", "#1e8449"),
    ]:
        adf = exp6_df[exp6_df["approach"] == approach]
        sl_pct = adf["service_level"] * 100

        ax.plot(sl_pct, adf["seio_total_cost"] / 1000, marker="o",
                color=color_seio, ls="--", lw=1.5, ms=7, alpha=0.7,
                label=f"{approach} SEIO")
        ax.plot(sl_pct, adf["meio_total_cost"] / 1000, marker="s",
                color=color_meio, ls="-", lw=2, ms=7,
                label=f"{approach} MEIO")

    ax.set_xlabel("Service Level (%)")
    ax.set_ylabel("Total Weekly Holding Cost ($K)")
    ax.set_title("(a) SEIO vs MEIO Cost Comparison")
    ax.legend(fontsize=9)

    # (b) Coordination value (%) by service level and approach
    ax = axes[1]
    width = 0.3
    x = np.arange(len(sls))

    for i, approach in enumerate(["Classical", "ML-Enhanced"]):
        adf = exp6_df[exp6_df["approach"] == approach]
        offset = -width / 2 if i == 0 else width / 2
        color = "#e74c3c" if i == 0 else "#27ae60"

        bars = ax.bar(x + offset, adf["coordination_value_pct"].values,
                      width, label=approach, color=color, edgecolor="white")

        for j, v in enumerate(adf["coordination_value_pct"].values):
            ax.text(x[j] + offset, v + 0.5, f"{v:.1f}%",
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s*100)}%" for s in sls])
    ax.set_xlabel("Service Level")
    ax.set_ylabel("Value of Coordination (%)\n[SEIO cost - MEIO cost] / SEIO cost")
    ax.set_title("(b) Value of Multi-Echelon Coordination")
    ax.legend()

    plt.suptitle("Experiment 6: Single-Echelon vs Multi-Echelon Optimization",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig_seio_vs_meio.png",
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: fig_seio_vs_meio.png")


def fig_parameter_sensitivity(exp7_results):
    """Figure: Tornado chart + reduction robustness for sensitivity analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Tornado chart: cost variation per parameter
    ax = axes[0]
    lt_df = exp7_results.get("lt_sensitivity")
    hc_df = exp7_results.get("hc_sensitivity")

    if lt_df is not None and hc_df is not None:
        combined = pd.concat([
            lt_df[["parameter", "multiplier", "total_holding_cost"]],
            hc_df[["parameter", "multiplier", "total_holding_cost"]],
        ])

        baseline_cost = combined[combined["multiplier"] == 1.0]["total_holding_cost"].mean()
        params = combined["parameter"].unique()

        # For each parameter, compute range (min cost at 0.5x, max cost at 1.5x)
        tornado_data = []
        for p in params:
            pdata = combined[combined["parameter"] == p]
            low = pdata[pdata["multiplier"] == 0.5]["total_holding_cost"].iloc[0]
            high = pdata[pdata["multiplier"] == 1.5]["total_holding_cost"].iloc[0]
            tornado_data.append({
                "parameter": p,
                "low_pct": (low - baseline_cost) / baseline_cost * 100,
                "high_pct": (high - baseline_cost) / baseline_cost * 100,
                "range": abs(high - low) / baseline_cost * 100,
            })

        tdf = pd.DataFrame(tornado_data).sort_values("range", ascending=True)

        y_pos = range(len(tdf))
        colors_low = "#3498db"
        colors_high = "#e74c3c"

        for i, (_, row) in enumerate(tdf.iterrows()):
            ax.barh(i, row["low_pct"], height=0.6, color=colors_low,
                    alpha=0.8, edgecolor="white")
            ax.barh(i, row["high_pct"], height=0.6, color=colors_high,
                    alpha=0.8, edgecolor="white")

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(tdf["parameter"].values)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Cost Change from Baseline (%)")
        ax.set_title("(a) Tornado Chart: Parameter Impact on Total Cost")

        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color=colors_low, label="Parameter x 0.5"),
            Patch(color=colors_high, label="Parameter x 1.5"),
        ], loc="lower right", fontsize=9)

    # (b) ML reduction robustness across lead time scenarios
    ax = axes[1]
    red_df = exp7_results.get("reduction_robustness")

    if red_df is not None:
        ax.bar(red_df["lead_time_multiplier"].astype(str),
               red_df["cost_reduction_pct"],
               color="#27ae60", edgecolor="white", width=0.5)

        for i, (_, row) in enumerate(red_df.iterrows()):
            ax.text(i, row["cost_reduction_pct"] + 0.3,
                    f"{row['cost_reduction_pct']:.1f}%",
                    ha="center", fontweight="bold", fontsize=10)

        avg = red_df["cost_reduction_pct"].mean()
        ax.axhline(avg, color="#e74c3c", ls="--", lw=1.5,
                   label=f"Mean = {avg:.1f}%")
        ax.set_xlabel("Lead Time Multiplier")
        ax.set_ylabel("ML Cost Reduction (%)")
        ax.set_title("(b) ML Benefit Robustness to Lead Time Assumptions")
        ax.legend()

    plt.suptitle("Experiment 7: Parameter Sensitivity Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "fig_parameter_sensitivity.png",
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("  Saved: fig_parameter_sensitivity.png")


def generate_all_figures(experiment_results):
    """Generate all publication figures."""
    print("\n" + "=" * 70)
    print("STAGE 7: GENERATING PUBLICATION FIGURES")
    print("=" * 70)

    fig_network_topology()
    fig_cost_comparison(experiment_results["experiment1"])
    fig_sensitivity_analysis(experiment_results["experiment2"])
    fig_echelon_breakdown(experiment_results["experiment3"])
    fig_model_comparison(experiment_results["experiment4"])
    fig_monte_carlo_validation(experiment_results["experiment5"])
    fig_seio_vs_meio(experiment_results["experiment6"])
    fig_parameter_sensitivity(experiment_results["experiment7"])

    print(f"\n  All figures saved to {config.FIGURES_DIR}")