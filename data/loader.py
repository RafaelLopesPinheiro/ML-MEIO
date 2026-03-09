"""
data/loader.py
==============
Dataset loading, merging, and feature engineering for the
Walmart Sales Forecasting dataset.

The dataset models a two-echelon supply chain:
  - Echelon 1 (Upstream)  : Virtual Distribution Center (DC) — aggregation
                             of all store-level demand per department.
  - Echelon 2 (Downstream): Individual Walmart stores.

References
----------
Kaggle dataset: https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a DataFrame that has a 'Date' column with calendar features."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["year"] = df["Date"].dt.year
    df["week_index"] = (
        (df["year"] - df["year"].min()) * 52 + df["week_of_year"]
    )
    return df


def _add_lag_features(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    lags: List[int] = [1, 2, 4, 8, 13, 26, 52],
    group_cols: List[str] = ["Store", "Dept"],
) -> pd.DataFrame:
    """Add lag features for demand forecasting."""
    df = df.copy()
    df = df.sort_values(group_cols + ["Date"])
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
    return df


def _add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Weekly_Sales",
    windows: List[int] = [4, 13, 26, 52],
    group_cols: List[str] = ["Store", "Dept"],
) -> pd.DataFrame:
    """Add rolling mean and std features."""
    df = df.copy()
    df = df.sort_values(group_cols + ["Date"])
    for w in windows:
        grp = df.groupby(group_cols)[target_col]
        df[f"{target_col}_roll_mean_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"{target_col}_roll_std_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0)
        )
    return df


def _add_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill MarkDown columns (sparse) and create aggregate markdown feature."""
    md_cols = [c for c in df.columns if c.startswith("MarkDown")]
    if not md_cols:
        return df
    df = df.copy()
    for c in md_cols:
        df[c] = df[c].fillna(0.0).clip(lower=0)
    df["total_markdown"] = df[md_cols].sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class WalmartDataLoader:
    """
    Load, merge, and feature-engineer the Walmart Sales Forecasting dataset.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration dictionary (``data`` section).
    """

    FEATURE_COLS = [
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "IsHoliday",
    ]

    def __init__(self, config: dict) -> None:
        self.cfg = config["data"]
        self.opt_cfg = config["optimization"]
        self.random_seed = config.get("reproducibility", {}).get("random_seed", 42)
        np.random.seed(self.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_raw(self) -> Dict[str, pd.DataFrame]:
        """Load all raw CSV files and return as a dict."""
        paths = {
            "train": self.cfg["train_path"],
            "test": self.cfg["test_path"],
            "features": self.cfg["features_path"],
            "stores": self.cfg["stores_path"],
        }
        dfs = {}
        for name, path in paths.items():
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(
                    f"Dataset file not found: {p}.\n"
                    "Please download the Walmart dataset from Kaggle:\n"
                    "  https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast\n"
                    "and place the CSV files in data/raw/"
                )
            dfs[name] = pd.read_csv(p)
            logger.info("Loaded %s: %s rows, %s cols", name, *dfs[name].shape)
        return dfs

    def build_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build the full feature-engineered dataset.

        Returns
        -------
        train_df : pd.DataFrame
            Training data with all features.
        test_df : pd.DataFrame
            Test data with all features.
        dc_df : pd.DataFrame
            Aggregated DC-level (upstream echelon) demand dataframe.
        """
        raw = self.load_raw()

        train = self._merge_and_filter(raw["train"], raw["features"], raw["stores"])
        test = self._merge_and_filter(raw["test"], raw["features"], raw["stores"])

        # Engineer features
        for df in [train, test]:
            df["Weekly_Sales"] = df.get("Weekly_Sales", pd.Series(np.nan))

        train = self._engineer(train)
        test = self._engineer(test, is_test=True)

        # Echelon 1: aggregate DC demand per Dept × Date
        dc_df = self._build_dc_echelon(train)

        logger.info(
            "Dataset built. Train: %s rows | Test: %s rows | DC echelon: %s rows",
            len(train),
            len(test),
            len(dc_df),
        )
        return train, test, dc_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge_and_filter(
        self,
        sales_df: pd.DataFrame,
        features_df: pd.DataFrame,
        stores_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge sales, features, and stores; apply subset filters."""
        df = sales_df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        features_df = features_df.copy()
        features_df["Date"] = pd.to_datetime(features_df["Date"])

        # Merge features (store × date)
        df = df.merge(features_df, on=["Store", "Date", "IsHoliday"], how="left")
        # Merge store metadata
        df = df.merge(stores_df, on="Store", how="left")

        # Optional subsetting for faster experimentation
        if self.cfg.get("store_subset"):
            df = df[df["Store"].isin(self.cfg["store_subset"])]
        if self.cfg.get("dept_subset"):
            df = df[df["Dept"].isin(self.cfg["dept_subset"])]

        # Store type encoding
        df["store_type"] = df["Type"].map({"A": 0, "B": 1, "C": 2}).fillna(0)
        df["IsHoliday"] = df["IsHoliday"].astype(int)

        return df

    def _engineer(self, df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = _add_calendar_features(df)
        df = _add_markdown_features(df)
        if not is_test:
            df = _add_lag_features(df)
            df = _add_rolling_features(df)
        df = df.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
        return df

    def _build_dc_echelon(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate store-level demand to a virtual Distribution Center (DC)
        per department and date — representing upstream echelon demand.
        """
        agg = (
            train.groupby(["Dept", "Date"])["Weekly_Sales"]
            .sum()
            .reset_index()
            .rename(columns={"Weekly_Sales": "DC_Weekly_Demand"})
        )
        agg = _add_calendar_features(agg.rename(columns={"Date": "Date"}))
        agg = agg.sort_values(["Dept", "Date"]).reset_index(drop=True)

        # Lag / rolling features for DC-level
        agg = agg.rename(columns={"DC_Weekly_Demand": "Weekly_Sales"})
        agg = _add_lag_features(agg, group_cols=["Dept"])
        agg = _add_rolling_features(agg, group_cols=["Dept"])
        agg = agg.rename(columns={"Weekly_Sales": "DC_Weekly_Demand"})
        return agg

    # ------------------------------------------------------------------
    # Train/test split utilities
    # ------------------------------------------------------------------

    def temporal_split(
        self, df: pd.DataFrame, test_weeks: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Temporal train/validation split (no leakage).

        Parameters
        ----------
        df : pd.DataFrame
        test_weeks : int, optional
            Number of final weeks to hold out. Defaults to config value.

        Returns
        -------
        train_df, val_df : pd.DataFrame, pd.DataFrame
        """
        n = test_weeks or self.cfg.get("test_weeks", 26)
        cutoff = df["Date"].max() - pd.Timedelta(weeks=n)
        train_df = df[df["Date"] <= cutoff].copy()
        val_df = df[df["Date"] > cutoff].copy()
        logger.info(
            "Temporal split: train up to %s (%d rows), val from %s (%d rows)",
            cutoff.date(),
            len(train_df),
            (cutoff + pd.Timedelta(days=1)).date(),
            len(val_df),
        )
        return train_df, val_df

    def get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Return all usable feature column names (excludes target and keys)."""
        exclude = {
            "Weekly_Sales",
            "DC_Weekly_Demand",
            "Date",
            "Store",
            "Dept",
            "Type",
        }
        return [c for c in df.columns if c not in exclude and df[c].dtype != object]


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg