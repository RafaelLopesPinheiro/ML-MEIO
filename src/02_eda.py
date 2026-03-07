"""
02_eda.py — Exploratory Data Analysis.

Generates diagnostic plots and summary statistics to characterize
demand patterns, variability structure, and hierarchical relationships
in the Walmart dataset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import config


def demand_distribution_analysis(df):
    """
    Characterize demand distributions at multiple aggregation levels.
    Tests normality and computes descriptive statistics.
    """
    print("\n  --- Demand Distribution Analysis ---")
    results = {}

    for level, group_cols in [
        ("Store-Dept", ["Store", "Dept"]),
        ("Store", ["Store"]),
        ("Region", ["Region"]),
    ]:
        grouped = df.groupby(group_cols)["Weekly_Sales"].agg(
            ["mean", "std", "median", "skew", "count"]
        )
        grouped["cv"] = grouped["std"] / grouped["mean"]

        results[level] = {
            "mean_demand": grouped["mean"].mean(),
            "mean_std": grouped["std"].mean(),
            "mean_cv": grouped["cv"].mean(),
            "median_cv": grouped["cv"].median(),
            "mean_skewness": grouped["skew"].mean(),
            "n_groups": len(grouped),
        }

        print(f"    {level:>12s}: {results[level]['n_groups']:>5d} groups | "
              f"mean CV = {results[level]['mean_cv']:.3f} | "
              f"mean skew = {results[level]['mean_skewness']:.3f}")

    return results


def plot_demand_timeseries(df):
    """Plot aggregate weekly demand time series by region."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    total = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    axes[0].plot(total["Date"], total["Weekly_Sales"] / 1e6, color="#2c3e50", lw=1.5)
    axes[0].set_ylabel("Weekly Sales (M$)")
    axes[0].set_title("Aggregate Weekly Demand - All Stores")
    axes[0].grid(True, alpha=0.3)

    for region in sorted(df["Region"].unique()):
        reg_data = (
            df[df["Region"] == region]
            .groupby("Date")["Weekly_Sales"].sum().reset_index()
        )
        axes[1].plot(reg_data["Date"], reg_data["Weekly_Sales"] / 1e6,
                     label=region, lw=1.2)

    axes[1].set_ylabel("Weekly Sales (M$)")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Weekly Demand by Region (DC Echelon)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "demand_timeseries.png",
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("    Saved: demand_timeseries.png")


def plot_demand_variability(df):
    """Plot coefficient of variation (CV) across store-departments."""
    cv_data = (
        df.groupby(["Store", "Dept"])["Weekly_Sales"]
        .agg(["mean", "std"]).reset_index()
    )
    cv_data["CV"] = cv_data["std"] / cv_data["mean"]
    cv_data = cv_data.replace([np.inf, -np.inf], np.nan).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(cv_data["CV"], bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].axvline(cv_data["CV"].median(), color="#e74c3c", ls="--", lw=2,
                    label=f"Median CV = {cv_data['CV'].median():.2f}")
    axes[0].set_xlabel("Coefficient of Variation")
    axes[0].set_ylabel("Count (Store-Dept pairs)")
    axes[0].set_title("Demand Variability Distribution")
    axes[0].legend()

    axes[1].scatter(cv_data["mean"], cv_data["std"], alpha=0.3, s=10, color="#2c3e50")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Mean Weekly Sales (log)")
    axes[1].set_ylabel("Std Weekly Sales (log)")
    axes[1].set_title("Mean-Variance Relationship (Store-Dept Level)")

    log_mean = np.log(cv_data["mean"].values)
    log_std = np.log(cv_data["std"].values)
    slope, intercept, r_val, _, _ = stats.linregress(log_mean, log_std)
    x_fit = np.linspace(log_mean.min(), log_mean.max(), 100)
    axes[1].plot(np.exp(x_fit), np.exp(intercept + slope * x_fit),
                 color="#e74c3c", lw=2,
                 label=f"sigma ~ mu^{slope:.2f} (R2={r_val**2:.3f})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "demand_variability.png",
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("    Saved: demand_variability.png")


def plot_seasonal_patterns(df):
    """Decompose and visualize seasonal demand patterns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    monthly = df.groupby("Month")["Weekly_Sales"].mean()
    axes[0].bar(monthly.index, monthly.values / 1e3, color="#3498db", edgecolor="white")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Avg Weekly Sales ($K)")
    axes[0].set_title("Monthly Seasonality Pattern")
    axes[0].set_xticks(range(1, 13))

    holiday_sales = df.groupby("IsHoliday")["Weekly_Sales"].mean()
    colors = ["#95a5a6", "#e74c3c"]
    axes[1].bar(["Non-Holiday", "Holiday"], holiday_sales.values / 1e3,
                color=colors, edgecolor="white")
    axes[1].set_ylabel("Avg Weekly Sales ($K)")
    axes[1].set_title("Holiday Effect on Demand")

    pct_diff = (holiday_sales[True] - holiday_sales[False]) / holiday_sales[False] * 100
    axes[1].annotate(f"+{pct_diff:.1f}%", xy=(1, holiday_sales[True] / 1e3),
                     fontsize=14, ha="center", va="bottom", fontweight="bold",
                     color="#e74c3c")

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "seasonal_patterns.png",
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("    Saved: seasonal_patterns.png")


def plot_echelon_demand_structure(df):
    """Visualize demand aggregation across echelons (risk-pooling effect)."""
    echelon_stats = []

    for level, gcols in [
        ("Store-Dept\n(Demand Nodes)", ["Store", "Dept"]),
        ("Store\n(Retail Echelon)", ["Store"]),
        ("Region\n(DC Echelon)", ["Region"]),
    ]:
        grouped = df.groupby(gcols)["Weekly_Sales"].agg(["mean", "std"])
        grouped["CV"] = grouped["std"] / grouped["mean"]
        for _, row in grouped.iterrows():
            echelon_stats.append({"Echelon": level, "CV": row["CV"]})

    edf = pd.DataFrame(echelon_stats).dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=edf, x="Echelon", y="CV", palette="Set2", ax=ax, fliersize=2)
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Demand Variability by Supply Chain Echelon\n"
                 "(Risk pooling effect at higher aggregation)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "echelon_variability.png",
                dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("    Saved: echelon_variability.png")


def run_eda(df):
    """Execute all EDA analyses."""
    print("\n" + "=" * 70)
    print("STAGE 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    dist_results = demand_distribution_analysis(df)

    print("\n  Generating figures...")
    plot_demand_timeseries(df)
    plot_demand_variability(df)
    plot_seasonal_patterns(df)
    plot_echelon_demand_structure(df)

    summary_df = pd.DataFrame(dist_results).T
    summary_df.to_csv(config.RESULTS_DIR / "eda_demand_statistics.csv")
    print(f"\n  EDA results saved to {config.RESULTS_DIR / 'eda_demand_statistics.csv'}")

    return dist_results


if __name__ == "__main__":
    df = pd.read_csv(config.RESULTS_DIR / "processed_data.csv", parse_dates=["Date"])
    run_eda(df)