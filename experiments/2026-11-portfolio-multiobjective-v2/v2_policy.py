"""Pure eligible-14 hard-rank allocation for registered v2 policy constraints.

Means/returns use decimals, covariance/budget squared decimals, weights fractions.
An exact simplex/concentration KKT solution is used only if all registered risk
and exposure constraints also hold. Otherwise a strict conic solve enforces the
identical constraints. No clipping, renormalization, fallback or relaxation.
"""

from __future__ import annotations

from functools import lru_cache

import cvxpy as cp
import numpy as np
from scipy.stats import rankdata
from v2_common import ASSETS, ELIGIBLE_INDEX


@lru_cache(maxsize=256)
def _rank_template(n: int, effective: int) -> np.ndarray:
    scores = np.arange(1, n + 1, dtype=float) / n
    if not 1 < effective <= n:
        raise ValueError("Effective count outside allowed simplex")
    if effective == n:
        return np.full(n, 1 / n)
    for active in range(effective + 1, n + 1):
        top = scores[-active:]
        centered = top - top.mean()
        alpha = np.sqrt((1 / effective - 1 / active) / (centered @ centered))
        threshold = top.mean() - 1 / (active * alpha)
        weights = alpha * (top - threshold)
        outside = scores[-active - 1] if active < n else -np.inf
        if weights.min() >= -1e-14 and outside <= threshold + 1e-14:
            result = np.zeros(n)
            result[-active:] = weights
            return result
    raise ValueError("No certified rank-simplex solution found")


def _analytic(scores: np.ndarray, effective: int) -> np.ndarray | None:
    if effective == len(scores):
        return np.full(len(scores), 1 / len(scores))
    if len(np.unique(scores)) != len(scores):
        return None
    weights = np.empty(len(scores))
    weights[np.argsort(scores)] = _rank_template(len(scores), effective)
    return weights


def raw_checks(weights: np.ndarray, covariance: np.ndarray, effective: int,
               budget: float | None, equity_cap: float | None) -> dict:
    """Recompute raw feasibility for a single eligible-axis portfolio.

    Args:
        weights: Fourteen eligible ETF fractions.
        covariance: Matching positive-definite covariance, squared decimals.
        effective: Registered minimum effective asset count.
        budget: Optional absolute variance ceiling, squared decimals.
        equity_cap: Optional static equity fraction ceiling.

    Returns:
        Raw checks and explicit feasibility with numerical acceptance tolerances.
    """
    variance = float(weights @ covariance @ weights)
    squared = float(weights @ weights)
    equity = float(weights[:7].sum())
    passed = (np.isfinite(weights).all() and weights.min() >= -1e-8 and abs(weights.sum() - 1) <= 1e-7
              and squared <= 1 / effective + 1e-7
              and (budget is None or variance <= budget * (1 + 1e-6) + 1e-14)
              and (equity_cap is None or equity <= equity_cap + 1e-7))
    return {"passed": bool(passed), "weight_sum": float(weights.sum()), "minimum_weight": float(weights.min()),
            "effective_assets": 1 / squared, "sum_squared_weights": squared, "predicted_variance": variance,
            "variance_ceiling": budget, "variance_ratio": None if budget is None else variance / budget,
            "equity_weight": equity, "equity_cap": equity_cap,
            "variance_binding": budget is not None and variance >= budget * (1 - 1e-5),
            "equity_binding": equity_cap is not None and equity >= equity_cap - 1e-6,
            "concentration_binding": abs(squared - 1 / effective) <= 1e-6}


def allocate(mean15: np.ndarray, covariance15: np.ndarray, *, min_effective_assets: int = 7,
             variance_ceiling: float | None = None, equity_cap: float | None = None) -> dict:
    """Solve the registered portfolio problem and reinsert exact MGK zero.

    Args:
        mean15: Fifteen finite pooled GP means, approved order, decimal returns.
        covariance15: Fifteen-task observed covariance, squared decimals.
        min_effective_assets: Integer floor 3..14 over eligible ETF weights.
        variance_ceiling: Optional positive squared-decimal risk ceiling.
        equity_cap: Optional static equity ceiling in [0,1].

    Returns:
        Complete raw 15-weight portfolio plus strict numerical evidence, or an
        explicit infeasible record. Numerical failure raises with all attempts.
    """
    if (mean15.shape != (15,) or covariance15.shape != (15, 15)
            or not np.isfinite(mean15).all() or not np.isfinite(covariance15).all()
            or not np.allclose(covariance15, covariance15.T, rtol=1e-10, atol=1e-14)
            or min_effective_assets not in range(3, 15)
            or (variance_ceiling is not None and (not np.isfinite(variance_ceiling) or variance_ceiling <= 0))
            or (equity_cap is not None and not 0 <= equity_cap <= 1)):
        raise ValueError("Invalid registered finite portfolio inputs")
    covariance = covariance15[np.ix_(ELIGIBLE_INDEX, ELIGIBLE_INDEX)]
    np.linalg.cholesky(covariance)
    scores = rankdata(mean15[ELIGIBLE_INDEX]) / 14
    attempts = []
    weights = _analytic(scores, min_effective_assets)
    check = None if weights is None else raw_checks(weights, covariance, min_effective_assets,
                                                    variance_ceiling, equity_cap)
    if check is not None and check["passed"]:
        solver_status = "analytic_kkt"
        certificate = "Exact eligible-rank simplex/NE KKT optimum; every registered extra constraint is feasible"
    elif min_effective_assets == 14:
        return {"passed": False, "infeasible": True, "reason": "NE14 uniquely requires equal eligible weights",
                "raw_checks": check, "numerical_attempts": [], "fallback_used": False}
    else:
        w = cp.Variable(14)
        constraints = [w >= 0, cp.sum(w) == 1, cp.sum_squares(w) <= 1 / min_effective_assets]
        if variance_ceiling is not None:
            scale = max(float(np.diag(covariance).max()), variance_ceiling)
            factor = np.linalg.cholesky(covariance / scale).T
            constraints.append(cp.norm(factor @ w, 2) <= np.sqrt(variance_ceiling / scale))
        if equity_cap is not None:
            constraints.append(cp.sum(w[:7]) <= equity_cap)
        problem = cp.Problem(cp.Maximize((scores - scores.mean()) @ w - 1e-8 * cp.sum_squares(w)), constraints)
        accepted = False
        for regularization, step in [(1e-10, .95), (1e-12, .9), (1e-14, .8)]:
            try:
                problem.solve(solver="CLARABEL", max_iter=250, tol_gap_abs=1e-9, tol_gap_rel=1e-9,
                              tol_feas=1e-9, static_regularization_constant=regularization,
                              max_step_fraction=step, warm_start=False, verbose=False)
                attempt = {"status": problem.status, "iterations": problem.solver_stats.num_iters,
                           "static_regularization_constant": regularization, "max_step_fraction": step}
                attempts.append(attempt)
                if problem.status == "infeasible":
                    return {"passed": False, "infeasible": True, "reason": "Registered constraints infeasible",
                            "numerical_attempts": attempts, "fallback_used": False}
                if problem.status != "optimal" or w.value is None:
                    continue
                weights = np.asarray(w.value, dtype=float).reshape(14)
                check = raw_checks(weights, covariance, min_effective_assets, variance_ceiling, equity_cap)
                attempt["raw_checks"] = check
                if check["passed"]:
                    accepted = True
                    break
            except cp.SolverError as error:
                attempts.append({"status": "solver_error", "error": str(error),
                                 "static_regularization_constant": regularization, "max_step_fraction": step})
        if not accepted:
            raise RuntimeError(f"Unaccepted identical constrained rank solve: {attempts}")
        solver_status, certificate = "optimal", None
    if weights is None or check is None or not check["passed"]:
        raise ValueError("No accepted raw weights")
    delivered = np.zeros(len(ASSETS))
    delivered[ELIGIBLE_INDEX] = weights
    return {"passed": True, "weights": delivered.tolist(), "solver_status": solver_status,
            "numerical_attempts": attempts, "certificate": certificate, "raw_checks": check,
            "rank_objective": float(scores @ weights - 1e-8 * (weights @ weights)),
            "eligible_ranks": (scores * 14).tolist(), "fallback_used": False,
            "raw_weights_changed": False}
