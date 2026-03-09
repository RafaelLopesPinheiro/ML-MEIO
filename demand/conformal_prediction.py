"""
demand/conformal_prediction.py
================================
Split Conformal Prediction — feature-name-warning-free version.

Fix: LGBMRegressor is always fitted and predicted with numpy arrays
(not DataFrames), eliminating the sklearn UserWarning:
  "X does not have valid feature names, but LGBMRegressor was fitted
   with feature names"

This warning was firing thousands of times in the ablation loop,
causing significant terminal I/O overhead and log spam.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)


class ConformalForecaster:
    """
    Split Conformal Prediction wrapper for demand forecasting.

    Parameters
    ----------
    base_estimator : object, optional
        Scikit-learn compatible regressor for point forecasting.
        Defaults to LightGBM if available, else GradientBoostingRegressor.
    alpha : float
        Miscoverage level (e.g., 0.10 → 90% coverage guarantee).
    calibration_frac : float
        Fraction of training data used as calibration set.
    symmetric : bool
        If True, use symmetric intervals ŷ ± q.
        If False, use one-sided upper bound ŷ + q (recommended for inventory).
    random_seed : int
    """

    def __init__(
        self,
        base_estimator=None,
        alpha: float = 0.10,
        calibration_frac: float = 0.20,
        symmetric: bool = False,
        random_seed: int = 42,
    ) -> None:
        self.alpha            = alpha
        self.calibration_frac = calibration_frac
        self.symmetric        = symmetric
        self.random_seed      = random_seed
        self._q_hat: Optional[float] = None
        self._is_fitted       = False

        if base_estimator is not None:
            self._base = base_estimator
        elif LGB_AVAILABLE:
            # Use verbose=-1 and feature_name="auto" suppressed via predict with numpy
            self._base = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=random_seed,
                verbose=-1,
                n_jobs=-1,
            )
        else:
            self._base = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=random_seed,
            )

    # ------------------------------------------------------------------
    # Internal: safe numpy conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(X) -> np.ndarray:
        """Convert any array-like to a plain numpy array (avoids feature-name warnings)."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "ConformalForecaster":
        """
        Fit the base estimator and compute conformal quantile q̂_{1-α}.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        # Always convert to numpy to avoid LGBMRegressor feature-name warnings
        X_train = self._to_numpy(X_train)
        y_train = np.asarray(y_train, dtype=float)

        n   = len(y_train)
        rng = np.random.default_rng(self.random_seed)
        idx = rng.permutation(n)
        n_cal    = max(int(n * self.calibration_frac), 20)
        n_proper = n - n_cal

        if n_proper < 20:
            logger.warning(
                "Proper training set has only %d samples. "
                "Consider reducing calibration_frac.", n_proper
            )

        proper_idx = idx[:n_proper]
        cal_idx    = idx[n_proper:]

        X_proper = X_train[proper_idx]
        y_proper = y_train[proper_idx]
        X_cal    = X_train[cal_idx]
        y_cal    = y_train[cal_idx]

        # Fit base estimator with plain numpy arrays
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._base.fit(X_proper, y_proper)

        logger.debug(
            "ConformalForecaster base estimator fitted on %d samples.", n_proper
        )

        # Non-conformity scores on calibration set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            y_cal_pred = self._base.predict(X_cal)

        if self.symmetric:
            scores = np.abs(y_cal - y_cal_pred)
        else:
            scores = y_cal - y_cal_pred  # positive = under-prediction

        # Conformal quantile
        n_cal_   = len(scores)
        level    = np.ceil((1 - self.alpha) * (n_cal_ + 1)) / n_cal_
        level    = min(level, 1.0)
        self._q_hat = float(np.quantile(scores, level))
        self._is_fitted = True

        logger.info(
            "ConformalForecaster fitted. α=%.2f → q̂_{1-α}=%.4f "
            "(calibration n=%d, proper n=%d).",
            self.alpha, self._q_hat, n_cal_, n_proper,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict_base(self, X) -> np.ndarray:
        """Run base estimator prediction suppressing feature-name warnings."""
        X = self._to_numpy(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return np.clip(self._base.predict(X), 0, None)

    def predict_mean(self, X) -> np.ndarray:
        """Return point forecast from base estimator."""
        self._check_fitted()
        return self._predict_base(X)

    def predict_std(self, X) -> np.ndarray:
        """
        Approximate conditional std from the conformal score q̂.

        Treats q̂ as a proxy for ~1.28σ (consistent with 90% Normal interval).
        """
        self._check_fitted()
        sigma = abs(self._q_hat) / 1.28
        return np.full(len(self._to_numpy(X)), max(sigma, 1e-6))

    def predict_intervals(
        self,
        X,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute conformal prediction intervals with (1-alpha) coverage guarantee.

        Returns
        -------
        lower, upper : np.ndarray, shape (n_samples,)
        """
        self._check_fitted()
        y_hat = self._predict_base(X)
        if self.symmetric:
            lower = np.clip(y_hat - self._q_hat, 0, None)
            upper = y_hat + self._q_hat
        else:
            lower = np.zeros(len(y_hat))
            upper = np.clip(y_hat + self._q_hat, 0, None)
        return lower, upper

    def predict_quantile(self, X, q: float) -> np.ndarray:
        """
        Approximate the q-th quantile via scaled conformal score.

        Parameters
        ----------
        X : array-like
        q : float in (0, 1)
        """
        self._check_fitted()
        y_hat = self._predict_base(X)
        denom = max(1 - self.alpha, 1e-9)
        adjustment = self._q_hat * (q / denom)
        return np.clip(y_hat + adjustment, 0, None)

    def predict_quantiles(
        self,
        X,
        quantiles: list = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of quantile estimates.

        Parameters
        ----------
        X : array-like
        quantiles : list of float
        """
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95]
        rows = {q: self.predict_quantile(X, q) for q in quantiles}
        return pd.DataFrame(rows, columns=quantiles).clip(lower=0)

    def sample_scenarios(
        self,
        X,
        n_scenarios: int = 200,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Sample demand scenarios uniformly between conformal lower and upper bounds.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        n_scenarios : int
        random_state : int

        Returns
        -------
        scenarios : np.ndarray, shape (n_samples, n_scenarios)
        """
        rng          = np.random.default_rng(random_state)
        lower, upper = self.predict_intervals(X)
        n_obs        = len(lower)
        # Ensure strictly positive width
        width = np.maximum(upper - lower, 1e-6)
        u     = rng.uniform(0, 1, (n_obs, n_scenarios))
        scenarios = lower[:, None] + u * width[:, None]
        return scenarios.clip(min=0)

    @property
    def q_hat(self) -> Optional[float]:
        """Calibrated conformal quantile."""
        return self._q_hat

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")