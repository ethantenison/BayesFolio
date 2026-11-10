"""Pure append-only portfolio numerics for four additional completed months.

Research-only allocation layer. Returns are decimal discounted excess, observed
covariance and budgets decimal squared, weights fractions. No fits, I/O, fallback,
constraint relaxation or post-solve weight repair. Imported by run_extension.py.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from scipy.stats import rankdata


def blend(broad_mean: np.ndarray, compact_mean: np.ndarray,
          broad_covariance: np.ndarray, compact_covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen Sharpe-leader family blend and 50/50 covariance blend.

    Args:
        broad_mean: (..., 15) broad-model means in fixed universe order.
        compact_mean: Matching compact-model means, decimal target units.
        broad_covariance: (..., 15, 15) observed predictive covariance.
        compact_covariance: Matching compact-model observed covariance.

    Returns:
        Blended decimal means and positive-definite observed covariance.
    """
    if broad_mean.shape != compact_mean.shape or broad_mean.shape[-1] != 15:
        raise ValueError("Identical fifteen-task mean axes required")
    if broad_covariance.shape != (*broad_mean.shape, 15) or compact_covariance.shape != broad_covariance.shape:
        raise ValueError("Identical fifteen-task covariance axes required")
    if not all(np.isfinite(value).all() for value in
               [broad_mean, compact_mean, broad_covariance, compact_covariance]):
        raise ValueError("Complete finite forecasts required")
    alpha = np.r_[np.full(4, 0.5), np.zeros(4), np.full(4, 0.375), np.full(3, 0.25)]
    mean = (1 - alpha) * broad_mean + alpha * compact_mean
    covariance = 0.5 * broad_covariance + 0.5 * compact_covariance
    np.linalg.cholesky(covariance)
    return mean, covariance


def hard_rank(mean: np.ndarray, covariance: np.ndarray, budget: float) -> dict:
    """Solve one identical hard-rank problem using the sealed factored cones.

    Args:
        mean: Fifteen decimal forecasts, used only for ascending average ranks.
        covariance: Fifteen-by-fifteen observed GP risk matrix, decimal squared.
        budget: Expanding training-only historical equal-weight variance ceiling.

    Returns:
        Raw accepted weights and audit; no clipping, renormalization or fallback.
        Long-only, fully invested, effective count >=7, no position cap, rank
        score/15 minus 1e-8 concentration tie-break, unchanged numerical retries.
    """
    if mean.shape != (15,) or covariance.shape != (15, 15) or not np.isfinite(budget) or budget <= 0:
        raise ValueError("Fifteen finite forecasts and a positive variance budget required")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Finite GP inputs required")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("Symmetric covariance required; no repair")
    weights = cp.Variable(15)
    scores = cp.Parameter(15, value=rankdata(mean) / 15)
    factor = np.linalg.cholesky(covariance / budget).T
    centered = scores - cp.sum(scores) / 15
    problem = cp.Problem(cp.Maximize(centered @ weights - 1e-8 * cp.sum_squares(weights)), [
        weights >= 0, cp.sum(weights) == 1, cp.norm(factor @ weights, 2) <= 1,
        cp.norm(weights, 2) <= 1 / np.sqrt(7),
    ])
    attempts = []
    for step, regularization in [(0.9, 1e-12), (0.8, 0.0), (0.95, 1e-12), (0.99, 1e-12)]:
        settings = {"max_step_fraction": step, "static_regularization_enable": regularization > 0}
        if regularization > 0:
            settings["static_regularization_constant"] = regularization
        try:
            problem.solve(solver="CLARABEL", warm_start=False, enforce_dpp=True,
                          tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9, **settings)
        except cp.error.SolverError as error:
            attempts.append({**settings, "status": "solver_error", "error": str(error)})
            continue
        attempts.append({**settings, "status": problem.status, "iterations": problem.solver_stats.num_iters})
        if problem.status != cp.OPTIMAL_INACCURATE:
            break
    if attempts[-1]["status"] in {"optimal_inaccurate", "solver_error"}:
        problem.solve(solver="SCS", warm_start=False, enforce_dpp=True,
                      eps_abs=1e-10, eps_rel=1e-10, max_iters=100000)
        attempts.append({"solver": "SCS", "status": problem.status})
    if problem.status != cp.OPTIMAL or weights.value is None:
        raise RuntimeError(f"Strict optimal hard-rank solve required: {attempts}")
    raw = np.asarray(weights.value).copy()
    variance, squared = float(raw @ covariance @ raw), float(raw @ raw)
    if (not np.isfinite(raw).all() or raw.min() < -1e-8 or abs(raw.sum() - 1) > 1e-7
            or variance > budget * (1 + 1e-6) or squared > 1 / 7 + 1e-7):
        raise ValueError("Raw hard-rank weights violate original tolerances")
    return {"weights": raw.tolist(), "predicted_variance": variance, "variance_budget": float(budget),
            "effective_count": 1 / squared, "numerical_attempts": attempts, "fallback_used": False}


def append_paths(old: dict[str, np.ndarray], tail: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Append new observations without recomputing or altering any earlier entry.

    Args:
        old: Frozen original dates, returns, weights, forecasts and wealth.
        tail: Four new months; wealth keys contain four new terminal NAVs only.

    Returns:
        Full 28-month panel, with bitwise-identical original array prefixes.
    """
    if set(tail) != set(old) - {"assets", "seeds"}:
        raise ValueError("Complete matching tail keys required")
    result = {"assets": old["assets"].copy(), "seeds": old["seeds"].copy()}
    for key, value in tail.items():
        axis = 1 if key in {"mean", "covariance"} or key.startswith("gp_") else 0
        if value.shape[axis] != 4:
            raise ValueError(f"Exactly four appended entries required for {key}")
        result[key] = np.concatenate([old[key], value], axis=axis)
        np.testing.assert_array_equal(np.take(result[key], range(old[key].shape[axis]), axis=axis), old[key])
    return result
