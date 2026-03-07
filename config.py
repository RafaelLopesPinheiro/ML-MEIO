"""
config.py — Global configuration and experimental parameters.
This file lives at the PROJECT ROOT (same level as src/, data/).
"""

import os
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================
TRAIN_FILE = DATA_DIR / "train.csv"
STORES_FILE = DATA_DIR / "stores.csv"
FEATURES_FILE = DATA_DIR / "features.csv"
TEST_FILE = DATA_DIR / "test.csv"

# ============================================================================
# TEMPORAL PARAMETERS
# ============================================================================
TEST_WEEKS = 12
FORECAST_HORIZON = 4

# ============================================================================
# MULTI-ECHELON NETWORK PARAMETERS
# ============================================================================
# Lead times in weeks
LEAD_TIMES = {
    "dc": 2,
    "warehouse": 1,
    "store": 1,
}

# Holding cost per unit per week (fraction of unit cost)
HOLDING_COST_RATES = {
    "dc": 0.005,
    "warehouse": 0.010,
    "store": 0.020,
}

# Service time bounds (weeks) — key GSM decision variables
# IMPORTANT: Store-level max_service_time = 0 means the store MUST
# fulfill customer demand immediately (S_store = 0), which forces
# tau_store = SI + L - 0 > 0, and thus safety stock > 0 at stores.
# This is the standard GSM assumption for customer-facing nodes
# (Graves & Willems, 2000, Section 3).
MAX_SERVICE_TIMES = {
    "dc": 3,
    "warehouse": 2,
    "store": 0,       # Customer-facing: guaranteed immediate service
}

# External supply lead time to the most upstream node
EXTERNAL_LEAD_TIME = 2

# ============================================================================
# DEMAND FORECASTING PARAMETERS
# ============================================================================
RANDOM_SEED = 42

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "random_state": RANDOM_SEED,
}

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbosity": 0,
    "random_state": RANDOM_SEED,
}

# ============================================================================
# GSM OPTIMIZATION PARAMETERS
# ============================================================================
SERVICE_LEVELS = {
    0.90: 1.282,
    0.95: 1.645,
    0.98: 2.054,
    0.99: 2.326,
}

DEFAULT_SERVICE_LEVEL = 0.95

# ============================================================================
# EXPERIMENTAL DESIGN
# ============================================================================
TOP_N_DEPARTMENTS = 10
STORE_SAMPLE_SIZE = 20
SENSITIVITY_SERVICE_LEVELS = [0.90, 0.95, 0.98, 0.99]

# Temporal cross-validation for sigma estimation
CV_FOLDS = 5             # Number of expanding-window CV folds
CV_MIN_TRAIN_WEEKS = 40  # Minimum weeks in initial training window

# ============================================================================
# VISUALIZATION
# ============================================================================
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
COLOR_PALETTE = "Set2"