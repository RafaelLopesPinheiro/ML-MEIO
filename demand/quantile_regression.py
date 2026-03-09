"""
demand/quantile_regression.py
==============================
Gradient-Boosting Quantile Regression Forecaster.

Fixed: predict_quantiles() now accepts an optional `quantiles` keyword
argument for API consistency with NGBoostForecaster and ConformalForecaster.
When provided, the closest available trained quantile is used via interpolation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logging.warning("LightGBM not available. QuantileRegressionForecaster will be limited.")

logger = logging.getLogger(__name__)


class QuantileRegressionForecaster:
    """
    Multi-quantile gradient-boosting forecaster using LightGBM.

    One model is trained per quantile level. Quantile crossing is
    corrected via isotonic regression at prediction time.

    Parameters
    ----------
    quantiles : list of float
        Quantile levels to estimate, e.g. [0.05, 0.25, 0.50, 0.75, 0.95].
    n_estimators : int
    learning_rate : float
    max_depth : int
    subsample : float
    random_seed : int
    """

    def __init__(
        self,
        quantiles: List[float] = None,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        random_seed: int = 42,
    ) -> None:
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.quantiles = sorted(quantiles)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_seed = random_seed
        self._models: Dict[float, "lgb.Booster"] = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "QuantileRegressionForecaster":
        """Train one LightGBM quantile regression model per quantile level."""
        if not LGB_AVAILABLE:
            raise ImportError("LightGBM is required. Run: pip install lightgbm")

        logger.info(
            "Training QuantileRegressionForecaster for %d quantiles on %d samples.",
            len(self.quantiles),
            len(y_train),
        )

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = None
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        for q in self.quantiles:
            params = {
                "objective": "quantile",
                "alpha": q,
                "metric": "quantile",
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "subsample": self.subsample,
                "subsample_freq": 1,
                "verbose": -1,
                "seed": self.random_seed,
                "n_jobs": -1,
            }

            callbacks = []
            if val_data is not None:
                callbacks.append(
                    lgb.early_stopping(stopping_rounds=30, verbose=verbose)
                )
            if verbose:
                callbacks.append(lgb.log_evaluation(period=50))

            valid_sets = [val_data] if val_data else []
            model = lgb.train(
                params,
                train_data,
                valid_sets=valid_sets,
                callbacks=callbacks if callbacks else None,
            )
            self._models[q] = model
            logger.debug(
                "  Quantile %.2f trained. Best iter: %d",
                q,
                model.best_iteration or self.n_estimators,
            )

        self._is_fitted = True
        logger.info("QuantileRegressionForecaster training complete.")
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: List[float] = None,
    ) -> pd.DataFrame:
        """
        Predict quantiles for new observations.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        quantiles : list of float, optional
            Quantile levels to return. If provided, values are interpolated
            from the trained quantile grid. If None, returns all trained quantiles.

        Returns
        -------
        pd.DataFrame, shape (n_samples, n_quantiles)
            Columns are quantile levels (float).
            Crossing corrected by isotonic regression row-wise.
        """
        self._check_fitted()

        # Always predict over all trained quantiles first
        raw = {q: self._models[q].predict(X) for q in self.quantiles}
        df_all = pd.DataFrame(raw, columns=self.quantiles)
        df_all = self._fix_crossing(df_all)
        df_all = df_all.clip(lower=0)

        # If no specific quantiles requested, return all trained ones
        if quantiles is None:
            return df_all

        # Interpolate to the requested quantile levels
        trained_levels = np.array(self.quantiles, dtype=float)
        result: Dict[float, np.ndarray] = {}
        for q in quantiles:
            q = float(q)
            if q in df_all.columns:
                result[q] = df_all[q].values
            else:
                # Linear interpolation across trained quantile columns
                cols = df_all.values  # (n_samples, n_trained_quantiles)
                interpolated = np.array([
                    float(np.interp(q, trained_levels, cols[i]))
                    for i in range(len(cols))
                ])
                result[q] = interpolated

        return pd.DataFrame(result, columns=quantiles).clip(lower=0)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Predict the median (0.50 quantile) as point forecast."""
        preds = self.predict_quantiles(X)
        if 0.50 in preds.columns:
            return preds[0.50].values
        return preds.mean(axis=1).values

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate conditional std from the interquartile range:
        σ̂ ≈ (Q75 - Q25) / 1.349  (consistent for Normal).
        """
        preds = self.predict_quantiles(X, quantiles=[0.25, 0.75])
        if 0.25 in preds.columns and 0.75 in preds.columns:
            iqr = preds[0.75].values - preds[0.25].values
            return np.maximum(iqr / 1.349, 1e-6)
        return np.ones(len(X))

    def predict_intervals(
        self,
        X: np.ndarray,
        alpha: float = 0.10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict (1-alpha) prediction interval [lower, upper].

        Parameters
        ----------
        X : np.ndarray
        alpha : float
            Miscoverage level (e.g., 0.10 → 90% interval).

        Returns
        -------
        lower, upper : np.ndarray, shape (n_samples,)
        """
        q_lo = alpha / 2
        q_hi = 1 - alpha / 2
        preds = self.predict_quantiles(X, quantiles=[q_lo, q_hi])
        return preds[q_lo].values, preds[q_hi].values

    def sample_scenarios(
        self,
        X: np.ndarray,
        n_scenarios: int = 200,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Draw demand scenarios via quantile-inverse-CDF interpolation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        n_scenarios : int
        random_state : int

        Returns
        -------
        scenarios : np.ndarray, shape (n_samples, n_scenarios)
        """
        rng = np.random.default_rng(random_state)
        q_df = self.predict_quantiles(X)
        n_obs = len(X)
        scenarios = np.zeros((n_obs, n_scenarios))
        q_levels = np.array(q_df.columns, dtype=float)

        for i in range(n_obs):
            q_vals = q_df.iloc[i].values.astype(float)
            u_samples = rng.uniform(0, 1, n_scenarios)
            scenarios[i] = np.interp(u_samples, q_levels, q_vals)

        return scenarios.clip(min=0)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fix_crossing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply row-wise isotonic regression to remove quantile crossings."""
        q_levels = np.array(df.columns, dtype=float)
        result = df.values.copy()
        iso = IsotonicRegression(increasing=True)
        for i in range(len(result)):
            result[i] = iso.fit_transform(q_levels, result[i])
        return pd.DataFrame(result, columns=df.columns)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

    @property
    def feature_importances(self) -> Optional[Dict[float, np.ndarray]]:
        if not self._is_fitted:
            return None
        return {q: m.feature_importance(importance_type="gain") for q, m in self._models.items()}