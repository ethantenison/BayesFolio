"""Pure signed-Sharpe scenario allocation with an explicit diversification policy.

Research numerical layer, with no I/O or GP fitting. Means are decimal discounted
excess returns, covariance decimal squared and weights fractions. The only policy
change in the NE7 experiment is minimum effective assets 10 -> 7; cap25% remains.
Positive optima use the repaired, dimensionless conic ratio formulation. Negative
optima use deterministic direct signed-ratio multistart, without a return floor,
weight fallback, covariance repair or objective replacement.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class Policy:
    """Immutable allocation constraints, not tuning or fitted parameters.

    Attributes:
        minimum_effective_assets: Lower bound on 1/sum(weights squared).
        maximum_weight: Position cap as a fraction of portfolio value.
    """

    minimum_effective_assets: float = 7.0
    maximum_weight: float = .25

    def __post_init__(self) -> None:
        if not (np.isfinite(self.minimum_effective_assets) and 1 < self.minimum_effective_assets < 15
                and np.isfinite(self.maximum_weight) and 1 / 15 < self.maximum_weight <= 1):
            raise ValueError("A feasible finite fifteen-asset policy is required")


def audit_weights(weights: np.ndarray, policy: Policy) -> dict[str, float]:
    """Audit original raw weights without clipping or renormalization.

    Args:
        weights: Fifteen finite decimal portfolio weights.
        policy: Explicit minimum effective assets and position cap.

    Returns:
        Original-weight sum, extrema, squared concentration and effective count.

    Raises:
        ValueError: Any original constraint fails its unchanged tolerance.
    """
    weights = np.asarray(weights)
    if weights.shape != (15,) or not np.isfinite(weights).all():
        raise ValueError("Fifteen finite weights required")
    squared = float(weights @ weights)
    if (weights.min() < -1e-8 or abs(float(weights.sum()) - 1) > 1e-7
            or weights.max() > policy.maximum_weight + 1e-8
            or squared > 1 / policy.minimum_effective_assets + 1e-8):
        raise ValueError("Raw weights violate explicit allocation policy; no repair")
    return {"sum": float(weights.sum()), "minimum_weight": float(weights.min()),
            "maximum_weight": float(weights.max()), "sum_squared": squared, "effective_count": 1 / squared}


def _moments(mean: np.ndarray, covariance: np.ndarray) -> None:
    if (mean.shape != (15,) or covariance.shape != (15, 15)
            or not np.isfinite(mean).all() or not np.isfinite(covariance).all()):
        raise ValueError("Fifteen finite aligned scenario means and covariance required")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("Symmetric covariance required; no repair")
    np.linalg.cholesky(covariance)


def _strict_solve(problem: cp.Problem) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-9, max_iter=1000)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"Strict conic optimal status required: {problem.status}")
    solution = problem._solver_cache["CLARABEL"].get_solution()
    result = {"status": problem.status, "primal_residual": float(solution.r_prim),
              "dual_residual": float(solution.r_dual), "duality_gap": float(solution.obj_val - solution.obj_val_dual),
              "dual_upper_bound": -float(solution.obj_val_dual), "iterations": int(problem.solver_stats.num_iters),
              "largest_constraint_violation": max(float(np.max(c.violation())) for c in problem.constraints),
              "warnings": [str(w.message) for w in caught]}
    if (result["primal_residual"] > 1e-8 or result["dual_residual"] > 1e-8
            or abs(result["duality_gap"]) > 1e-8 or result["largest_constraint_violation"] > 1e-8):
        raise ValueError(f"Numerical certificate failed: {result}")
    return result


def _maximum_mean(mean: np.ndarray, policy: Policy) -> tuple[np.ndarray, dict]:
    center = np.full(15, 1 / 15)
    if float(np.ptp(mean)) < 1e-14:
        return center, {"maximum_feasible_mean": float(mean @ center), "route": "equal_mean_analytic"}
    scale = float(np.max(np.abs(mean)))
    w = cp.Variable(15)
    problem = cp.Problem(cp.Maximize((mean / scale) @ w), [
        w >= 0, cp.sum(w) == 1, w <= policy.maximum_weight,
        cp.SOC(1 / np.sqrt(policy.minimum_effective_assets), w),
    ])
    certificate = _strict_solve(problem)
    weights = np.asarray(w.value).copy()
    audit_weights(weights, policy)
    return weights, {"maximum_feasible_mean": float(mean @ weights),
                     "maximum_mean_dual_upper_bound": certificate["dual_upper_bound"] * scale,
                     "route": "policy_constrained_conic_maximum_mean", "certificate": certificate}


def _positive(mean: np.ndarray, covariance: np.ndarray, policy: Policy) -> dict:
    factor = np.linalg.cholesky(covariance)
    center = np.full(15, 1 / 15)
    scale = float(np.sqrt(center @ covariance @ center))
    transform = factor.T / scale
    x = cp.Variable(15)
    t = cp.sum(x)
    problem = cp.Problem(cp.Maximize((mean / scale) @ x), [
        x >= 0, x <= policy.maximum_weight * t,
        cp.SOC(t / np.sqrt(policy.minimum_effective_assets), x), cp.SOC(1., transform @ x),
    ])
    certificate = _strict_solve(problem)
    holdings = np.asarray(x.value).copy()
    if float(holdings.sum()) <= 0:
        raise ValueError("No positive-Sharpe homogeneous solution")
    weights = holdings / float(holdings.sum())
    audit = audit_weights(weights, policy)
    score = float(mean @ weights / np.sqrt(weights @ covariance @ weights))
    risk_norm = float(np.linalg.norm(transform @ holdings))
    if (score <= 0 or abs(risk_norm - 1) > 1e-7 or abs(score - float(problem.value)) > 1e-8):
        raise ValueError("Positive-ratio backtransform certificate failed")
    return {"weights": weights, "raw_weights": weights.copy(), "audit": audit,
            "monthly_forecast_sharpe": score, "solver_route": "dimensionless_positive_sharpe_socp",
            "solver_status": "optimal", "global_optimality_certified": True,
            "certificate": {**certificate, "risk_cone_norm": risk_norm}}


def _loss_gradient(w: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> tuple[float, np.ndarray]:
    product = covariance @ w
    std = float(np.sqrt(w @ product))
    mu = float(mean @ w)
    return -mu / std, -mean / std + mu * product / std**3


def _boundary(direction: np.ndarray, policy: Policy) -> np.ndarray:
    center = np.full(15, 1 / 15)
    direction = direction - direction.mean()
    norm = float(np.linalg.norm(direction))
    if norm < 1e-14:
        return center
    direction = direction / norm
    positive, negative = direction > 0, direction < 0
    limit = min(float(np.sqrt(1 / policy.minimum_effective_assets - center @ center)),
                float(np.min((policy.maximum_weight - center[positive]) / direction[positive])),
                float(np.min(-center[negative] / direction[negative])))
    return center + .999 * limit * direction


def solve_direct(mean: np.ndarray, covariance: np.ndarray, policy: Policy, *, seed: int,
                 random_starts: int = 64) -> dict[str, Any]:
    """Optimize signed Sharpe directly, including negative feasible means.

    Args:
        mean: Fifteen decimal scenario means, with either sign allowed.
        covariance: Strictly positive-definite sample covariance, decimal squared.
        policy: Explicit diversification and cap constraints.
        seed: Fixed independent optimizer-start RNG seed.
        random_starts: Additional deterministic starts; ordinary runs use 64.

    Returns:
        Successful raw weights whose best signed-ratio objective was replicated
        by at least two successful feasible starts. This nonconvex branch is
        not represented as a globally certified optimum. No fallback portfolio.
    """
    _moments(mean, covariance)
    maximum, feasibility = _maximum_mean(mean, policy)
    center = np.full(15, 1 / 15)
    scale = float(np.sqrt(center @ covariance @ center))
    _, vectors = np.linalg.eigh(covariance)
    directions = [*np.eye(15), *vectors.T, *(-vectors.T),
                  *np.random.default_rng(seed).standard_normal((random_starts, 15))]
    starts = [center, maximum, *[_boundary(d, policy) for d in directions]]
    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(15)},
        {"type": "ineq", "fun": lambda w: 1 - policy.minimum_effective_assets * (w @ w),
         "jac": lambda w: -2 * policy.minimum_effective_assets * w},
    ]
    candidates, records = [], []
    for index, start in enumerate(starts):
        result = minimize(_loss_gradient, start, args=(mean / scale, covariance / scale**2),
                          jac=True, method="SLSQP", bounds=[(0, policy.maximum_weight)] * 15,
                          constraints=constraints, options={"ftol": 1e-12, "maxiter": 1000})
        record = {"start": index, "success": bool(result.success), "status": int(result.status),
                  "iterations": int(result.nit), "message": str(result.message)}
        if result.success:
            try:
                audit_weights(result.x, policy)
            except ValueError as error:
                record["rejected"] = str(error)
            else:
                score = -_loss_gradient(result.x, mean, covariance)[0]
                candidates.append((score, result.x.copy(), index))
                record["score"] = score
        records.append(record)
    if not candidates:
        raise RuntimeError("No successful feasible signed-Sharpe solution; no fallback")
    score, weights, best = max(candidates, key=lambda row: row[0])
    matches = sum(abs(value - score) <= 1e-7 for value, _, _ in candidates)
    if matches < 2:
        raise RuntimeError("Best signed-Sharpe objective was not independently reproduced")
    return {"weights": weights, "raw_weights": weights.copy(), "audit": audit_weights(weights, policy),
            "monthly_forecast_sharpe": score, "solver_route": "direct_signed_sharpe_multistart",
            "global_optimality_certified": False, "best_start": best, "best_objective_reproductions": matches,
            "successful_starts": len(candidates), "starts": records, "feasibility": feasibility}


def allocate_scenarios(scenarios: np.ndarray, assets: list[str], *, seed: int,
                       policy: Policy = Policy()) -> dict[str, Any]:
    """Allocate exact frozen joint draws under one explicit policy.

    Args:
        scenarios: (5000,15) finite decimal discounted-excess returns. Negative
            observations, ETF means and optimal portfolio means are valid.
        assets: Fifteen unique ETF column names in original order.
        seed: Deterministic multistart seed, not a scenario resampling key.
        policy: Diversification and cap; experiment defaults are NE7/cap25%.

    Returns:
        Audited unmodified weights and sign-aware numerical diagnostics. Every
        return remains unchanged. No CVaR, variance ceiling or replacement weights.
    """
    if (scenarios.shape != (5000, 15) or len(assets) != 15 or len(set(assets)) != 15
            or not np.isfinite(scenarios).all()):
        raise ValueError("Exactly 5000 finite scenarios and fifteen unique aligned ETFs required")
    mean, covariance = scenarios.mean(0), np.cov(scenarios, rowvar=False)
    _moments(mean, covariance)
    _, feasibility = _maximum_mean(mean, policy)
    if feasibility["maximum_feasible_mean"] > 1e-9:
        result = _positive(mean, covariance, policy)
    else:
        result = solve_direct(mean, covariance, policy, seed=seed)
    result.update(feasibility=feasibility, negative_scenario_observations=int((scenarios < 0).sum()),
                  negative_asset_means=int((mean < 0).sum()), minimum_portfolio_return_constraint=False)
    return result
