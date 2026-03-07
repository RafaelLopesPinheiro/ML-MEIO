"""
01_data_preprocessing.py — Data loading, cleaning, merging, and feature engineering.

This module transforms the raw Walmart datasets (train.csv, stores.csv, features.csv)
into a unified analytical dataset suitable for demand forecasting and inventory
optimization.
"""

import sys
from pathlib import Path
# Ensure project root is on sys.path (for standalone execution)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import config


def load_raw_data():
    """Load the three raw Walmart CSV files."""
    print("=" * 70)
    print("STAGE 1: DATA PREPROCESSING")
    print("=" * 70)

    train = pd.read_csv(config.TRAIN_FILE)
    stores = pd.read_csv(config.STORES_FILE)
    features = pd.read_csv(config.FEATURES_FILE)

    print(f"  train.csv    : {train.shape[0]:>8,} rows x {train.shape[1]} cols")
    print(f"  stores.csv   : {stores.shape[0]:>8,} rows x {stores.shape[1]} cols")
    print(f"  features.csv : {features.shape[0]:>8,} rows x {features.shape[1]} cols")

    return train, stores, features


def merge_datasets(train, stores, features):
    """
    Merge train with stores and features on shared keys.

    Join logic:
      train <-(Store)-> stores
      train <-(Store, Date, IsHoliday)-> features
    """
    df = train.merge(stores, on="Store", how="left")
    df = df.merge(features, on=["Store", "Date", "IsHoliday"], how="left")

    print(f"\n  Merged dataset: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def engineer_temporal_features(df):
    """
    Extract temporal features from the Date column.

    Features created:
      - Year, Month, WeekOfYear, DayOfYear, Quarter
      - IsMonthStart, IsMonthEnd
      - SinWeek, CosWeek, SinMonth, CosMonth (cyclical encodings)
    """
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Quarter"] = df["Date"].dt.quarter
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)

    # Cyclical encoding for week of year
    df["SinWeek"] = np.sin(2 * np.pi * df["WeekOfYear"] / 52)
    df["CosWeek"] = np.cos(2 * np.pi * df["WeekOfYear"] / 52)

    # Cyclical encoding for month
    df["SinMonth"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["CosMonth"] = np.cos(2 * np.pi * df["Month"] / 12)

    print("  Temporal features engineered (12 new features)")
    return df


def handle_missing_values(df):
    """
    Impute missing values with principled strategies.

    - MarkDown columns: Fill NaN with 0 (no markdown active)
    - CPI and Unemployment: Forward-fill within store groups
    """
    markdown_cols = [c for c in df.columns if "MarkDown" in c]
    df[markdown_cols] = df[markdown_cols].fillna(0)

    for col in ["CPI", "Unemployment"]:
        if col in df.columns:
            df[col] = df.groupby("Store")[col].transform(
                lambda x: x.ffill().bfill()
            )

    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if len(remaining_nulls) > 0:
        print(f"  Warning: Remaining nulls:\n{remaining_nulls}")
    else:
        print("  All missing values handled successfully")

    return df


def create_store_state_mapping(df):
    """
    Map stores to synthetic state-level regions based on store Type.

    Mapping logic for multi-echelon network:
      - Type A stores (large)  -> Region "R1" (high-volume DC)
      - Type B stores (medium) -> Region "R2" (medium-volume DC)
      - Type C stores (small)  -> Region "R3" (low-volume DC)
    """
    type_to_region = {"A": "R1", "B": "R2", "C": "R3"}
    df["Region"] = df["Type"].map(type_to_region)
    print(f"  Store-to-Region mapping created ({df['Region'].nunique()} regions)")
    return df


def compute_aggregated_demand(df):
    """
    Compute demand at multiple aggregation levels for the multi-echelon network.
    """
    aggregations = {}

    aggregations["store_dept"] = (
        df.groupby(["Region", "Store", "Dept", "Date"])["Weekly_Sales"]
        .sum().reset_index()
    )
    aggregations["store"] = (
        df.groupby(["Region", "Store", "Date"])["Weekly_Sales"]
        .sum().reset_index()
    )
    aggregations["region"] = (
        df.groupby(["Region", "Date"])["Weekly_Sales"].sum().reset_index()
    )
    aggregations["total"] = (
        df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    )

    for key, agg_df in aggregations.items():
        print(f"  Aggregation '{key}': {agg_df.shape[0]:,} rows")

    return aggregations


def select_top_departments(df, n):
    """Select top N departments by total sales volume."""
    dept_sales = df.groupby("Dept")["Weekly_Sales"].sum().sort_values(ascending=False)
    top_depts = dept_sales.head(n).index.tolist()
    print(f"  Top {n} departments by volume: {top_depts}")
    return top_depts


def run_preprocessing():
    """Execute the full preprocessing pipeline."""
    train, stores, features = load_raw_data()
    df = merge_datasets(train, stores, features)
    df = engineer_temporal_features(df)
    df = handle_missing_values(df)
    df = create_store_state_mapping(df)

    top_depts = select_top_departments(df, config.TOP_N_DEPARTMENTS)
    df_filtered = df[df["Dept"].isin(top_depts)].copy()
    print(f"\n  Filtered to top {config.TOP_N_DEPARTMENTS} depts: "
          f"{df_filtered.shape[0]:,} rows")

    aggregations = compute_aggregated_demand(df_filtered)

    output_path = config.RESULTS_DIR / "processed_data.csv"
    df_filtered.to_csv(output_path, index=False)
    print(f"\n  Processed data saved to {output_path}")

    print(f"\n  Dataset Summary:")
    print(f"    Stores       : {df_filtered['Store'].nunique()}")
    print(f"    Departments  : {df_filtered['Dept'].nunique()}")
    print(f"    Regions      : {df_filtered['Region'].nunique()}")
    print(f"    Date range   : {df_filtered['Date'].min()} to {df_filtered['Date'].max()}")
    print(f"    Total records: {df_filtered.shape[0]:,}")

    return df_filtered, aggregations


if __name__ == "__main__":
    df, aggs = run_preprocessing()