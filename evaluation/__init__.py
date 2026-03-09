"""
evaluation/
===========
Performance evaluation sub-package.

Modules
-------
metrics     : Service level, cost, coverage, and probabilistic metrics.
benchmarks  : Classical MEIO baseline (EOQ + Normal distribution safety stocks).
"""

from .metrics import evaluate_forecaster, evaluate_inventory_policy
from .benchmarks import ClassicalMEIOBaseline

__all__ = [
    "evaluate_forecaster",
    "evaluate_inventory_policy",
    "ClassicalMEIOBaseline",
]