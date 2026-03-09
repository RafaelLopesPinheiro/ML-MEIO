"""
optimization/
=============
Stochastic optimization sub-package for Multi-Echelon Inventory Optimization.

Modules
-------
stochastic_milp      : Two-stage stochastic MILP formulation (PuLP).
chance_constrained   : Chance-constrained LP/MILP formulation.
scenario_generator   : Demand scenario sampling and reduction utilities.
"""

from .stochastic_milp import TwoStageStochasticMEIO
from .chance_constrained import ChanceConstrainedMEIO
from .scenario_generator import ScenarioGenerator

__all__ = [
    "TwoStageStochasticMEIO",
    "ChanceConstrainedMEIO",
    "ScenarioGenerator",
]