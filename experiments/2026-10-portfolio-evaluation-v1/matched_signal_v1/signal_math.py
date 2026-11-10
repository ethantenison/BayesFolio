"""Pure normalized ranks and independent raw-allocation constraint checks.

Research numerical layer: no I/O, fitting, optimizer or realized-return access.
Scores and weights are dimensionless; covariance and budgets are decimal squared.
Usage: imported by run_matched_signal.py and synthetic tests.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def normalized_scores(signal: np.ndarray) -> np.ndarray:
    """Convert each complete signal cross-section to the frozen hard-rank scale.

    Args:
        signal: Finite (..., assets) values, with larger meaning more attractive.
            Momentum magnitudes are not interpreted as one-month return forecasts.

    Returns:
        Ascending average ranks divided by asset count, exactly matching the GP
        control convention. No top-seven selection or alphabetical tie-breaking.
    """
    if signal.ndim < 2 or signal.shape[-1] < 2 or not np.isfinite(signal).all():
        raise ValueError("Complete finite signal cross-sections required")
    return rankdata(signal, axis=-1, method="average") / signal.shape[-1]


def audit_weights(weights: np.ndarray, covariance: np.ndarray, budget: np.ndarray, minimum: int = 7) -> dict:
    """Check raw solved weights against the original acceptance tolerances.

    Args:
        weights: Shape (runs, months, assets), raw unmodified weight fractions.
        covariance: Shape (runs, months, assets, assets), common matched GP risk.
        budget: Shape (months,), original strictly positive variance ceilings.
        minimum: Unchanged effective asset count floor.

    Returns:
        Monthly variance, variance-to-budget ratio and effective count arrays.
        Violations raise; no clipping, normalization or constraint relaxation.
    """
    if weights.ndim != 3 or covariance.shape != (*weights.shape, weights.shape[-1]):
        raise ValueError("Matched weight and covariance axes required")
    if budget.shape != weights.shape[1:2] or not np.isfinite(budget).all() or not (budget > 0).all():
        raise ValueError("Complete positive monthly budgets required")
    if not np.isfinite(weights).all() or not np.isfinite(covariance).all():
        raise ValueError("Finite raw weights and covariance required")
    concentration = np.sum(weights**2, axis=-1)
    variance = np.einsum("sta,stab,stb->st", weights, covariance, weights)
    if (
        weights.min() < -1e-8
        or np.max(np.abs(weights.sum(axis=-1) - 1)) > 1e-7
        or (concentration > 1 / minimum + 1e-7).any()
        or (variance > budget * (1 + 1e-6)).any()
    ):
        raise ValueError("Raw weights violate the unchanged allocation constraints")
    return {"variance": variance, "variance_ratio": variance / budget, "effective_count": 1 / concentration}
