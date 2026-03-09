"""
demand/
=======
Demand distribution estimation sub-package.

Modules
-------
preprocessing        : Time-series preprocessing, feature selection, scaling.
quantile_regression  : Gradient-boosting quantile regression (QRF-style).
ngboost_forecaster   : NGBoost distributional forecasting.
conformal_prediction : Split conformal prediction intervals.
"""

from .quantile_regression import QuantileRegressionForecaster
from .ngboost_forecaster import NGBoostForecaster
from .conformal_prediction import ConformalForecaster

__all__ = [
    "QuantileRegressionForecaster",
    "NGBoostForecaster",
    "ConformalForecaster",
]