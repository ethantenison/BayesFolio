"""Single-portfolio November hard-rank allocation using frozen original numerics.

No portfolio-state fallback, clipping, covariance repair or risk relaxation.
The analytical KKT optimum is used only when feasible for the unchanged risk
ceiling; otherwise the original strict conic allocator solves the same problem.
Inputs are decimal means, squared-decimal covariance/budget and weight fractions.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache

import numpy as np
from common import FRAMEWORK, TUNING
from scipy.stats import rankdata


@lru_cache(maxsize=None)
def dependency(kind: str):
    """Load an exact frozen numerical dependency without editing its source.

    Args:
        kind: "rank" for pure KKT math or "conic" for original allocation.

    Returns:
        Loaded read-only numerical module.
    """
    path = (TUNING / "three_way_joint_v1/joint_math.py" if kind == "rank" else
            FRAMEWORK / "monthly_extension_v1/extension_math.py")
    name = "_november_frozen_" + kind
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def allocate(mean: np.ndarray, covariance: np.ndarray, budget: float) -> dict:
    """Allocate one pooled portfolio with an exact or strict-conic certificate.

    Args:
        mean: (15,) finite pooled mean forecasts, decimal returns.
        covariance: (15,15) observed risk matrix in squared-decimal units.
        budget: Positive causal expanding historical-EW variance ceiling.

    Returns:
        Raw weights, accepted solver/certificate, and recomputed feasibility.
    """
    if (mean.shape != (15,) or covariance.shape != (15, 15) or not np.isfinite(mean).all()
            or not np.isfinite(covariance).all() or not np.isfinite(budget) or budget <= 0
            or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14)):
        raise ValueError("Complete finite symmetric pooled moments and positive budget required")
    np.linalg.cholesky(covariance)
    scores = rankdata(mean) / 15
    weights = dependency("rank")._linear_rank_optimum(scores)
    variance = float(weights @ covariance @ weights)
    if variance <= budget * (1 + 1e-12):
        result = {"weights": weights.tolist(), "solver_status": "analytic_kkt",
                  "numerical_attempts": [], "rank_objective": float(scores @ weights - 1e-8 * (weights @ weights)),
                  "certificate": "KKT optimum on simplex/NE sphere; unchanged risk ceiling is feasible"}
    else:
        result = dependency("conic").hard_rank(mean, covariance, budget)
        weights = np.asarray(result["weights"])
        result["solver_status"] = "optimal"
        result["rank_objective"] = float(scores @ weights - 1e-8 * (weights @ weights))
    variance = float(weights @ covariance @ weights)
    if (weights.min() < -1e-8 or abs(weights.sum() - 1) > 1e-7
            or weights @ weights > 1 / 7 + 1e-7 or variance > budget * (1 + 1e-6)):
        raise ValueError("Raw weights violate the unchanged investment problem")
    result.update(predicted_variance=variance, variance_budget=float(budget),
                  effective_assets=float(1 / (weights @ weights)), fallback_used=False,
                  clipping_or_normalization_used=False)
    return result


def allocate_path(mean: np.ndarray, covariance: np.ndarray, budget: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Allocate complete monthly pooled-moment paths independently.

    Args:
        mean: (months,15) decimal pooled forecasts.
        covariance: (months,15,15) squared-decimal pooled observed covariance.
        budget: (months,) strictly pre-origin variance ceilings.

    Returns:
        Raw monthly weights and complete numerical acceptance records.
    """
    if covariance.shape != (*mean.shape, 15) or budget.shape != (len(mean),):
        raise ValueError("Aligned moment and budget paths required")
    records = [allocate(m, c, float(b)) for m, c, b in zip(mean, covariance, budget, strict=True)]
    return np.asarray([item["weights"] for item in records]), records
