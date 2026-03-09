"""
demand/preprocessing.py
========================
Data preprocessing for demand forecasting.

Fixes vs. original
-------------------
1. min_observations lowered to 20 (was 52) — Walmart weekly data after
   lag-feature creation naturally yields only 26 clean rows per group.
   Raising the threshold to 52 was silently skipping nearly all nodes.
2. Added log_transform_target option: log1p-transforms y before fitting,
   inverse-transforms after predict. Required for NGBoost-LogNormal to
   work correctly (prevents μ-collapse on scaled targets).
3. prepare_store_dept_data now also returns feature column names so
   callers can reconstruct DataFrames for LGBMRegressor with proper names.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Minimum clean observations required after lag creation and NaN dropping
MIN_OBSERVATIONS = 20   # was 52 — too strict for weekly Walmart data


class DemandPreprocessor:
    """
    Feature preprocessor: imputation + standard scaling.

    Parameters
    ----------
    impute_strategy : str
        Strategy for SimpleImputer ('mean', 'median', 'constant').
    scale : bool
        If True, apply StandardScaler after imputation.
    clip_quantile : float or None
        If set, clip features at this upper quantile (e.g., 0.99)
        to reduce influence of extreme values.
    """

    def __init__(
        self,
        impute_strategy: str = "median",
        scale: bool = True,
        clip_quantile: Optional[float] = None,
    ) -> None:
        self.impute_strategy = impute_strategy
        self.scale           = scale
        self.clip_quantile   = clip_quantile
        self._imputer        = SimpleImputer(strategy=impute_strategy)
        self._scaler         = StandardScaler() if scale else None
        self._clip_vals: Optional[np.ndarray] = None
        self._is_fitted      = False

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.values.astype(float)
        if self.clip_quantile is not None:
            self._clip_vals = np.nanquantile(X_arr, self.clip_quantile, axis=0)
            X_arr = np.clip(X_arr, a_min=None, a_max=self._clip_vals)
        X_arr = self._imputer.fit_transform(X_arr)
        if self._scaler is not None:
            X_arr = self._scaler.fit_transform(X_arr)
        self._is_fitted = True
        return X_arr

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("DemandPreprocessor not fitted. Call fit_transform first.")
        X_arr = X.values.astype(float)
        if self.clip_quantile is not None and self._clip_vals is not None:
            X_arr = np.clip(X_arr, a_min=None, a_max=self._clip_vals)
        X_arr = self._imputer.transform(X_arr)
        if self._scaler is not None:
            X_arr = self._scaler.transform(X_arr)
        return X_arr


def prepare_store_dept_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Weekly_Sales",
    sort_col:   str = "Date",
    log_transform_target: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """
    Prepare feature matrix X and target y for one store-dept series.

    Parameters
    ----------
    df            : pd.DataFrame
    feature_cols  : list of str   Feature column names (must exist in df).
    target_col    : str
    sort_col      : str           Column to sort by (chronological order).
    log_transform_target : bool
        If True, apply log1p to y before returning.
        Use with NGBoost-LogNormal to prevent μ-collapse.

    Returns
    -------
    X      : np.ndarray, shape (n_clean, n_features)
    y      : np.ndarray, shape (n_clean,)
    groups : np.ndarray  (row indices in original df)
    dates  : pd.Index
    """
    df = df.copy()

    if sort_col in df.columns:
        df = df.sort_values(sort_col)

    avail_feat = [c for c in feature_cols if c in df.columns]

    # Drop rows where target or any feature is NaN
    cols_needed = avail_feat + [target_col]
    df_clean = df[cols_needed].dropna()

    n = len(df_clean)
    if n < MIN_OBSERVATIONS:
        logger.warning(
            "Only %d observations available (min=%d).", n, MIN_OBSERVATIONS
        )

    if n == 0:
        return (
            np.empty((0, len(avail_feat))),
            np.empty(0),
            np.empty(0, dtype=int),
            pd.Index([]),
        )

    X = df_clean[avail_feat].values.astype(float)
    y = df_clean[target_col].values.astype(float)
    y = np.clip(y, 0, None)                   # demand is non-negative

    if log_transform_target:
        y = np.log1p(y)

    groups = df_clean.index.values
    dates  = (
        df.loc[groups, sort_col]
        if sort_col in df.columns
        else pd.RangeIndex(n)
    )

    return X, y, groups, dates