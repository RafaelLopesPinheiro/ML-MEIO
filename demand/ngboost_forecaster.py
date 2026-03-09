"""
demand/ngboost_forecaster.py
============================
NGBoost Distributional Forecaster — overflow-safe version.

Key fixes vs. original
-----------------------
1. LogNormal σ (scale parameter on log scale) is clamped to SIGMA_MAX=2.5
   before any exp() operation, preventing overflow to inf.
2. All exp() calls are wrapped in np.clip to guard against edge cases.
3. predict_distribution() validates and sanitises output params.
4. sample_scenarios() validates drawn samples and replaces inf/nan with
   the empirical mean before returning to scenario generator.
5. predict_mean/std return finite arrays even if NGBoost diverges.

References
----------
Duan, T., Avati, A., Ding, D. Y., Thai, K. K., Basu, S., Ng, A. Y.,
  & Schuler, A. (2020). NGBoost: Natural Gradient Boosting for
  Probabilistic Prediction. ICML.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm
from sklearn.tree import DecisionTreeRegressor

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal, LogNormal
    NGB_AVAILABLE = True
except ImportError:
    NGB_AVAILABLE = False
    logging.warning(
        "NGBoost not installed. Run: pip install ngboost\n"
        "Falling back to Gaussian approximation from sample statistics."
    )

logger = logging.getLogger(__name__)

# Safety clamp: log-scale σ > SIGMA_MAX is pathological for sales data
SIGMA_MAX = 2.5          # log-scale; exp(2.5) ≈ 12× mean — already wide
MU_MAX    = 20.0         # log-scale μ > 20 → mean > $485M/week — impossible
EXP_CLIP  = 700.0        # np.exp(700) is the largest safe double


class NGBoostForecaster:
    """
    Distributional forecaster using NGBoost.

    Parameters
    ----------
    distribution : str
        'normal' or 'lognormal'.
    n_estimators : int
    learning_rate : float
    minibatch_frac : float
    col_sample : float
    random_seed : int
    """

    _DIST_MAP = {
        "normal":    Normal    if NGB_AVAILABLE else None,
        "lognormal": LogNormal if NGB_AVAILABLE else None,
    }

    def __init__(
        self,
        distribution: str = "normal",
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        minibatch_frac: float = 0.8,
        col_sample: float = 0.8,
        random_seed: int = 42,
    ) -> None:
        self.distribution      = distribution.lower()
        self.n_estimators      = n_estimators
        self.learning_rate     = learning_rate
        self.minibatch_frac    = minibatch_frac
        self.col_sample        = col_sample
        self.random_seed       = random_seed
        self._model: Optional["NGBRegressor"] = None
        self._is_fitted        = False
        self._fallback_mean:   Optional[float] = None
        self._fallback_std:    Optional[float] = None
        # Stored from training data for emergency fallback
        self._train_mean:      float = 1.0
        self._train_std:       float = 1.0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "NGBoostForecaster":
        """Fit the NGBoost distributional model."""
        y_train = np.clip(y_train, a_min=1e-3, a_max=None)
        self._train_mean = float(np.mean(y_train))
        self._train_std  = float(np.std(y_train) + 1e-6)

        if not NGB_AVAILABLE:
            logger.warning("NGBoost not available. Using Gaussian fallback.")
            self._fallback_mean = self._train_mean
            self._fallback_std  = self._train_std
            self._is_fitted = True
            return self

        dist_cls = self._DIST_MAP.get(self.distribution)
        if dist_cls is None:
            raise ValueError(
                f"Unknown distribution: {self.distribution}. "
                "Use 'normal' or 'lognormal'."
            )

        base_learner = DecisionTreeRegressor(
            max_depth=4,
            random_state=self.random_seed,
        )

        self._model = NGBRegressor(
            Dist=dist_cls,
            Base=base_learner,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            col_sample=self.col_sample,
            random_state=self.random_seed,
            verbose=False,
        )

        if X_val is not None and y_val is not None:
            y_val = np.clip(y_val, a_min=1e-3, a_max=None)
            self._model.fit(
                X_train, y_train,
                X_val=X_val, Y_val=y_val,
                early_stopping_rounds=30,
            )
        else:
            self._model.fit(X_train, y_train)

        self._is_fitted = True
        logger.info(
            "NGBoostForecaster (%s) fitted on %d samples.",
            self.distribution, len(y_train),
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_distribution(self, X: np.ndarray) -> dict:
        """
        Return sanitised distribution parameters for each observation.

        Returns
        -------
        dict with keys:
            'loc'   : np.ndarray  — mean parameter (μ)
            'scale' : np.ndarray  — spread parameter (σ), clamped
            'dist'  : str         — distribution name
        """
        self._check_fitted()
        n = len(X)

        if not NGB_AVAILABLE or self._model is None:
            return {
                "loc":   np.full(n, self._fallback_mean),
                "scale": np.full(n, self._fallback_std),
                "dist":  "normal_fallback",
            }

        try:
            pred_dist = self._model.pred_dist(X)
        except Exception as exc:
            logger.warning("NGBoost pred_dist failed (%s). Using fallback.", exc)
            return {
                "loc":   np.full(n, self._train_mean),
                "scale": np.full(n, self._train_std),
                "dist":  "normal_fallback",
            }

        if self.distribution == "normal":
            loc   = np.asarray(pred_dist.params["loc"],   dtype=float)
            scale = np.asarray(pred_dist.params["scale"], dtype=float)
            # Sanitise
            loc   = np.where(np.isfinite(loc),   loc,   self._train_mean)
            scale = np.abs(scale)
            scale = np.where(np.isfinite(scale), scale, self._train_std)
            scale = np.clip(scale, 1e-6, self._train_std * 20)
        else:
            # LogNormal: NGBoost returns (s, scale) where scale=exp(μ), s=σ
            raw_loc   = pred_dist.params.get("loc",   np.zeros(n))
            raw_scale = pred_dist.params.get("scale", np.ones(n))
            loc   = np.asarray(raw_loc,   dtype=float)
            scale = np.asarray(raw_scale, dtype=float)

            # --- CRITICAL FIX: clamp σ before any exp() ---
            # For log-normal σ is the shape parameter on the log scale.
            # σ > SIGMA_MAX produces exp(σ²/2) overflow.
            scale = np.abs(scale)
            scale = np.clip(scale, 1e-6, SIGMA_MAX)   # ← KEY CLAMP

            # Also clamp μ (log-scale mean)
            # Reasonable range for weekly USD sales: log(1) to log(1e6) = 0–14
            loc = np.clip(loc, -2.0, MU_MAX)

            # Replace any remaining non-finite values
            loc   = np.where(np.isfinite(loc),   loc,   np.log(max(self._train_mean, 1.0)))
            scale = np.where(np.isfinite(scale), scale, 0.5)

        return {"loc": loc, "scale": scale, "dist": self.distribution}

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: List[float] = None,
    ) -> pd.DataFrame:
        """
        Predict specified quantiles from the fitted distribution.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        quantiles : list of float

        Returns
        -------
        pd.DataFrame, shape (n_samples, len(quantiles))
        """
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95]
        self._check_fitted()
        params = self.predict_distribution(X)

        results: Dict[float, np.ndarray] = {}
        for q in quantiles:
            if params["dist"] in ("normal", "normal_fallback"):
                vals = norm.ppf(q, loc=params["loc"], scale=params["scale"])
            else:  # lognormal
                # scipy lognorm: ppf(q, s=σ, scale=exp(μ))
                # scale parameter passed to scipy = exp(μ), already clamped
                exp_loc = np.exp(
                    np.clip(params["loc"], -EXP_CLIP, EXP_CLIP)
                )
                vals = lognorm.ppf(q, s=params["scale"], scale=exp_loc)

            # Sanitise output
            vals = np.where(np.isfinite(vals), vals, self._train_mean)
            results[q] = vals

        df = pd.DataFrame(results, columns=quantiles)
        return df.clip(lower=0)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Return the conditional mean E[Y | X = x]."""
        params = self.predict_distribution(X)
        if params["dist"] == "lognormal":
            # E[LN(μ,σ²)] = exp(μ + σ²/2)   — with clamped σ
            mu    = np.clip(params["loc"],   -EXP_CLIP, EXP_CLIP)
            sigma = np.clip(params["scale"], 1e-6,      SIGMA_MAX)
            result = np.exp(
                np.clip(mu + sigma ** 2 / 2, -EXP_CLIP, EXP_CLIP)
            )
        else:
            result = params["loc"]
        result = np.where(np.isfinite(result), result, self._train_mean)
        return np.clip(result, 0, None)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Return the conditional std Std[Y | X = x]."""
        params = self.predict_distribution(X)
        if params["dist"] == "lognormal":
            mu    = np.clip(params["loc"],   -EXP_CLIP, EXP_CLIP)
            sigma = np.clip(params["scale"], 1e-6,      SIGMA_MAX)
            # Var[LN] = (exp(σ²)-1) * exp(2μ+σ²)
            exp_s2 = np.exp(np.clip(sigma ** 2,          0, EXP_CLIP))
            exp_2m = np.exp(np.clip(2 * mu + sigma ** 2, -EXP_CLIP, EXP_CLIP))
            var    = np.clip((exp_s2 - 1) * exp_2m, 0, None)
            result = np.sqrt(var)
        else:
            result = params["scale"]
        result = np.where(np.isfinite(result), result, self._train_std)
        return np.clip(result, 1e-6, None)

    def sample_scenarios(
        self,
        X: np.ndarray,
        n_scenarios: int = 200,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Draw Monte Carlo demand scenarios from the fitted distribution.

        All non-finite draws are replaced with the training mean to keep
        the scenario matrix finite for k-means reduction.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        n_scenarios : int
        random_state : int

        Returns
        -------
        scenarios : np.ndarray, shape (n_samples, n_scenarios)
        """
        rng    = np.random.default_rng(random_state)
        params = self.predict_distribution(X)
        n_obs  = len(X)
        scenarios = np.zeros((n_obs, n_scenarios))

        for i in range(n_obs):
            mu    = float(params["loc"][i])
            sigma = float(params["scale"][i])

            if params["dist"] == "lognormal":
                sigma = min(sigma, SIGMA_MAX)
                mu    = np.clip(mu, -2.0, MU_MAX)
                draws = rng.lognormal(mean=mu, sigma=sigma, size=n_scenarios)
            else:
                draws = rng.normal(loc=mu, scale=sigma, size=n_scenarios)

            # --- CRITICAL FIX: sanitise draws before returning ---
            draws = np.where(np.isfinite(draws), draws, self._train_mean)
            scenarios[i] = np.clip(draws, 0, None)

        return scenarios

    def predictive_entropy(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictive entropy H[Y | X = x].

        For Normal:    H = 0.5 * ln(2πe σ²)
        For LogNormal: H = ln(σ √(2πe)) + μ
        """
        params = self.predict_distribution(X)
        if params["dist"] == "lognormal":
            sigma = np.clip(params["scale"], 1e-6, SIGMA_MAX)
            mu    = params["loc"]
            h = np.log(sigma * np.sqrt(2 * np.pi * np.e)) + mu
        else:
            sigma = np.clip(params["scale"], 1e-6, None)
            h = 0.5 * np.log(2 * np.pi * np.e * sigma ** 2)
        return np.where(np.isfinite(h), h, 0.0)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")