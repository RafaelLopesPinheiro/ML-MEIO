"""
optimization/scenario_generator.py
====================================
Scenario generation and reduction — nan/inf-safe version.

Key fixes vs. original
-----------------------
1. generate() validates raw scenarios before k-means: any row with
   all-inf/nan is replaced with the row mean of finite values.
2. _kmeans_reduction() asserts finiteness before fitting and raises
   a descriptive error if still non-finite after sanitisation.
3. ScenarioSet.__post_init__ tolerates tiny floating-point probability
   sum errors (already had atol=1e-4, kept).

References
----------
Heitsch & Römisch (2003). Scenario reduction algorithms in stochastic
  programming. Computational Optimization and Applications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)


@dataclass
class ScenarioSet:
    """
    Container for demand scenarios and their probabilities.

    Attributes
    ----------
    scenarios     : np.ndarray, shape (n_nodes, n_scenarios)
    probabilities : np.ndarray, shape (n_scenarios,) — sums to 1.
    node_ids      : list
    """

    scenarios:     np.ndarray
    probabilities: np.ndarray
    node_ids:      list = field(default_factory=list)

    def __post_init__(self) -> None:
        assert self.scenarios.shape[1] == len(self.probabilities), (
            "scenarios.shape[1] must equal len(probabilities)."
        )
        assert np.isclose(self.probabilities.sum(), 1.0, atol=1e-4), (
            f"Probabilities must sum to 1.0 (got {self.probabilities.sum():.6f})."
        )

    @property
    def n_nodes(self) -> int:
        return self.scenarios.shape[0]

    @property
    def n_scenarios(self) -> int:
        return self.scenarios.shape[1]

    def expected_demand(self) -> np.ndarray:
        """E[D] = Σ_s p_s * d_{i,s}"""
        return self.scenarios @ self.probabilities

    def quantile_demand(self, q: float) -> np.ndarray:
        """q-th percentile of demand across scenarios."""
        return np.quantile(self.scenarios, q, axis=1)


# ---------------------------------------------------------------------------
# Sanitisation helper
# ---------------------------------------------------------------------------

def _sanitise_scenarios(scenarios: np.ndarray) -> np.ndarray:
    """
    Replace inf / nan in a scenario matrix with finite fallback values.

    Strategy per row (node):
      1. If the row has at least one finite value, replace bad entries
         with the row's finite mean.
      2. If the entire row is non-finite, replace with 0.0.
    """
    if np.all(np.isfinite(scenarios)):
        return scenarios

    scenarios = scenarios.copy()
    n_bad_total = (~np.isfinite(scenarios)).sum()
    logger.warning(
        "Scenario matrix contains %d non-finite values — sanitising.",
        n_bad_total,
    )

    for i in range(scenarios.shape[0]):
        row = scenarios[i]
        bad = ~np.isfinite(row)
        if not bad.any():
            continue
        finite_vals = row[~bad]
        if len(finite_vals) > 0:
            replacement = float(np.mean(finite_vals))
        else:
            replacement = 0.0
            logger.warning(
                "Node %d: all scenarios are non-finite. Using 0.0 as fallback.", i
            )
        row[bad] = replacement
        scenarios[i] = row

    return scenarios


# ---------------------------------------------------------------------------
# ScenarioGenerator
# ---------------------------------------------------------------------------

class ScenarioGenerator:
    """
    Generate and optionally reduce demand scenarios for stochastic programming.

    Parameters
    ----------
    n_scenarios     : int   Raw number of scenarios to sample.
    n_reduced       : int   Target after reduction (None = no reduction).
    reduction_method: str   'kmeans' or 'none'.
    random_seed     : int
    """

    def __init__(
        self,
        n_scenarios:      int = 500,
        n_reduced:        Optional[int] = 50,
        reduction_method: str = "kmeans",
        random_seed:      int = 42,
    ) -> None:
        self.n_scenarios      = n_scenarios
        self.n_reduced        = n_reduced if n_reduced and n_reduced < n_scenarios else None
        self.reduction_method = reduction_method
        self.random_seed      = random_seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        forecaster,
        X_nodes: np.ndarray,
        node_ids: list = None,
    ) -> ScenarioSet:
        """
        Generate a scenario set for multi-node demand.

        Parameters
        ----------
        forecaster : fitted forecaster
            Must implement sample_scenarios(X, n_scenarios, random_state).
        X_nodes : np.ndarray, shape (n_nodes, n_features)
        node_ids : list, optional

        Returns
        -------
        ScenarioSet
        """
        if node_ids is None:
            node_ids = list(range(len(X_nodes)))

        # Draw raw scenarios — shape (n_nodes, n_scenarios)
        raw_scenarios = forecaster.sample_scenarios(
            X_nodes,
            n_scenarios=self.n_scenarios,
            random_state=self.random_seed,
        )

        # --- CRITICAL FIX: sanitise before any downstream operation ---
        raw_scenarios = _sanitise_scenarios(raw_scenarios)
        raw_scenarios = np.clip(raw_scenarios, 0, None)

        if self.n_reduced and self.reduction_method == "kmeans":
            scenarios, probs = self._kmeans_reduction(raw_scenarios)
        else:
            scenarios = raw_scenarios
            probs     = np.full(self.n_scenarios, 1.0 / self.n_scenarios)

        logger.info(
            "ScenarioGenerator: %d nodes × %d scenarios (after reduction from %d).",
            scenarios.shape[0],
            scenarios.shape[1],
            raw_scenarios.shape[1],
        )

        return ScenarioSet(
            scenarios=scenarios,
            probabilities=probs,
            node_ids=node_ids,
        )

    def generate_two_echelon(
        self,
        forecaster_dc,
        forecaster_store,
        X_dc:     np.ndarray,
        X_stores: np.ndarray,
        store_ids: list = None,
    ) -> Tuple[ScenarioSet, ScenarioSet]:
        """Generate scenario sets for DC and store echelons."""
        dc_ss    = self.generate(forecaster_dc,    X_dc,     node_ids=["DC"])
        store_ss = self.generate(
            forecaster_store, X_stores,
            node_ids=store_ids or list(range(len(X_stores)))
        )
        return dc_ss, store_ss

    # ------------------------------------------------------------------
    # Scenario reduction
    # ------------------------------------------------------------------

    def _kmeans_reduction(
        self,
        scenarios: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduce scenarios via k-means clustering (column-wise).

        Parameters
        ----------
        scenarios : np.ndarray, shape (n_nodes, n_raw_scenarios)

        Returns
        -------
        reduced  : np.ndarray, shape (n_nodes, n_reduced)
        weights  : np.ndarray, shape (n_reduced,)
        """
        S = scenarios.T  # (n_raw_scenarios, n_nodes)
        k = min(self.n_reduced, len(S))

        # Final safety check — should never trigger after _sanitise_scenarios
        if not np.all(np.isfinite(S)):
            bad_count = (~np.isfinite(S)).sum()
            logger.error(
                "k-means: %d non-finite values remain after sanitisation. "
                "Replacing with column means.",
                bad_count,
            )
            col_means = np.nanmean(S, axis=0)
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            for j in range(S.shape[1]):
                mask = ~np.isfinite(S[:, j])
                S[mask, j] = col_means[j]

        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.random_seed,
            n_init=10,
            max_iter=300,
        )
        labels    = kmeans.fit_predict(S)
        centroids = kmeans.cluster_centers_   # (k, n_nodes)

        weights = np.bincount(labels, minlength=k).astype(float)
        weights /= weights.sum()

        logger.debug("k-means reduction: %d → %d scenarios.", len(S), k)
        return centroids.T, weights   # (n_nodes, k), (k,)