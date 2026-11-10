"""Sign-safe maximum-Sharpe numerics for frozen GP-scenario research.

No fitting, data I/O or model selection. Scenario returns are decimal discounted
excess returns, covariance decimal squared, weights fractions. Preserve the MV
objective and all constraints. Positive feasible means use actual Riskfolio;
nonpositive means use deterministic multistart optimization of the original
signed mean/standard-deviation ratio, not a positive-return normalization.
The nonpositive branch is nonconvex; multistart agreement is recorded and is
not represented as a proof of global optimality.
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import pandas as pd
import riskfolio as rp
from scenario_allocator import audit_weights
from scipy.optimize import minimize


def _maximum_mean(mean: np.ndarray) -> tuple[np.ndarray, float]:
    # KKT water-filling solution of a linear objective over the simplex intersected
    # with sum(w**2)<=0.1. For 11..16 assets its largest possible weight is <=0.25,
    # so the cap is redundant. This avoids a numerically fragile solver sign test.
    dimension = len(mean)
    if not 10 < dimension <= 16:
        raise ValueError("Closed-form feasibility certificate supports 11 through 16 assets")
    order = np.argsort(mean, kind="stable")[::-1]
    for support in range(dimension, 9, -1):
        selected = mean[order[:support]]
        centered = selected - selected.mean()
        squared = float(centered @ centered)
        if squared < 1e-26:
            local = np.ones(support) / support
        else:
            radius_squared = 0.1 - 1 / support
            if radius_squared <= 0:
                continue
            slope = np.sqrt(radius_squared / squared)
            local = 1 / support + slope * centered
            threshold = selected.mean() - 1 / (support * slope)
            if local.min() < -1e-12:
                continue
            if support < dimension and mean[order[support]] > threshold + 1e-12:
                continue
        weights = np.zeros(dimension)
        weights[order[:support]] = local
        audit_weights(weights)
        return weights, float(mean @ weights)
    raise ValueError("Closed-form maximum-feasible-mean KKT certificate failed")


def _loss_gradient(weights: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> tuple[float, np.ndarray]:
    product = covariance @ weights
    deviation = float(np.sqrt(weights @ product))
    expected = float(mean @ weights)
    return -expected / deviation, -mean / deviation + expected * product / deviation**3


def _feasible_boundary(direction: np.ndarray) -> np.ndarray:
    center = np.ones(len(direction)) / len(direction)
    direction = direction - direction.mean()
    norm = float(np.linalg.norm(direction))
    if norm < 1e-14:
        return center
    direction = direction / norm
    positive, negative = direction > 0, direction < 0
    radius = np.sqrt(0.1 - center @ center)
    limit = min(
        float(radius),
        float(np.min((0.25 - center[positive]) / direction[positive])),
        float(np.min(-center[negative] / direction[negative])),
    )
    return center + 0.999 * limit * direction


def solve_signed(mean: np.ndarray, covariance: np.ndarray, *, seed: int, random_starts: int = 64) -> dict[str, Any]:
    """Maximize the signed Sharpe ratio under unchanged allocation constraints.

    Args:
        mean: (assets,) decimal scenario means, eleven through sixteen assets.
        covariance: Matching strictly positive-definite scenario covariance.
        seed: Fixed optimizer-start RNG seed; independent of realized outcomes.
        random_starts: Additional deterministic random directions to initialize.

    Returns:
        Raw, unaltered feasible weights and every multistart solver diagnostic.
        Objective is mean @ w / sqrt(w @ covariance @ w), regardless of sign.
        Starts include equal weights, mean maximizer and signed eigen/directional
        probes. Starting portfolios are never substituted for a failed solve.
        All candidates must converge successfully and pass raw constraints.
    """
    if mean.ndim != 1 or not 10 < len(mean) <= 16 or covariance.shape != (len(mean), len(mean)):
        raise ValueError("Aligned means/covariance and eleven through sixteen ETFs required")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Complete finite forecast moments required")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("Symmetric covariance required; no repair")
    np.linalg.cholesky(covariance)
    maximum_weights, maximum_mean = _maximum_mean(mean)
    center = np.ones(len(mean)) / len(mean)
    scale = float(np.sqrt(center @ covariance @ center))
    scaled_mean, scaled_covariance = mean / scale, covariance / scale**2
    _, eigenvectors = np.linalg.eigh(covariance)
    directions = [*np.eye(len(mean)), *eigenvectors.T, *(-eigenvectors.T)]
    directions.extend(np.random.default_rng(seed).standard_normal((random_starts, len(mean))))
    starts = [center, maximum_weights, *[_feasible_boundary(direction) for direction in directions]]
    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        {"type": "ineq", "fun": lambda w: 1 - 10 * (w @ w), "jac": lambda w: -20 * w},
    ]
    records, candidates = [], []
    for index, start in enumerate(starts):
        result = minimize(
            _loss_gradient,
            start,
            args=(scaled_mean, scaled_covariance),
            jac=True,
            method="SLSQP",
            bounds=[(0, 0.25)] * len(mean),
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        record = {
            "start": index,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
        }
        if result.success:
            try:
                audit = audit_weights(result.x)
            except ValueError as error:
                record["rejected"] = str(error)
            else:
                score = -_loss_gradient(result.x, mean, covariance)[0]
                record.update(score=score, audit=audit)
                candidates.append((score, result.x.copy(), index))
        records.append(record)
    if not candidates:
        raise RuntimeError("No successful feasible signed-Sharpe solve; no fallback")
    score, weights, best_index = max(candidates, key=lambda row: row[0])
    best_matches = sum(abs(value - score) <= 1e-7 for value, _, _ in candidates)
    if best_matches < 2:
        raise RuntimeError("Best signed-Sharpe objective was not independently reproduced across starts")
    return {
        "weights": weights,
        "raw_weights": weights.copy(),
        "monthly_forecast_sharpe": score,
        "solver_route": "direct_signed_sharpe_multistart",
        "maximum_feasible_mean": maximum_mean,
        "best_start": best_index,
        "best_objective_reproductions": best_matches,
        "successful_starts": len(candidates),
        "starts": records,
        "global_optimality_certified": False,
    }


def allocate_scenarios(scenarios: np.ndarray, assets: list[str], *, seed: int) -> dict[str, Any]:
    """Allocate GP scenarios using Riskfolio MV and a sign-safe formulation.

    Args:
        scenarios: (5000, assets) frozen joint GP decimal target-return draws.
        assets: Exact, unique ETF column order; all fifteen are retained.
        seed: Local deterministic RNG seed for signed-Sharpe solver starts.

    Returns:
        Raw/returned weights, feasibility audit and exact solver formulation.
        Both branches maximize conventional forecast Sharpe with rf zero, subject
        to long-only, full investment, effective assets >=10 and 25-percent cap.
        No CVaR, objective replacement, constraint relaxation or reused weights.
    """
    if scenarios.ndim != 2 or scenarios.shape[1] != len(assets) or len(set(assets)) != len(assets):
        raise ValueError("Scenario/ETF axes differ")
    if not np.isfinite(scenarios).all() or len(assets) <= 10:
        raise ValueError("Finite scenarios and more than ten unique ETFs required")
    mean, covariance = scenarios.mean(axis=0), np.cov(scenarios, rowvar=False)
    _, maximum_mean = _maximum_mean(mean)
    if maximum_mean <= 1e-9:
        result = solve_signed(mean, covariance, seed=seed)
        result["audit"] = audit_weights(result["weights"])
        return result
    portfolio = rp.Portfolio(returns=pd.DataFrame(scenarios, columns=assets), nea=10)
    portfolio.upperlng, portfolio.lowerlng, portfolio.card = 0.25, 0.0, None
    portfolio.solvers = ["CLARABEL"]
    portfolio.sol_params = {
        "CLARABEL": {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10, "max_iter": 1000}
    }
    portfolio.assets_stats(method_mu="hist", method_cov="hist")
    captured = []
    original_solve = cp.Problem.solve

    def _tracked_solve(problem: cp.Problem, *args: Any, **kwargs: Any) -> Any:
        captured.append(problem)
        return original_solve(problem, *args, **kwargs)

    with warnings.catch_warnings(record=True) as caught, patch.object(cp.Problem, "solve", _tracked_solve):
        frame = portfolio.optimization(model="Classic", rm="MV", obj="Sharpe", rf=0.0, hist=True)
    if len(captured) != 1 or captured[0].status != cp.OPTIMAL or frame is None:
        raise RuntimeError(f"Strict Riskfolio optimal status required: {[problem.status for problem in captured]}")
    if frame.index.tolist() != assets:
        raise ValueError("Riskfolio changed ETF ordering")
    problem = captured[0]
    vectors = [variable for variable in problem.variables() if variable.shape == (len(assets), 1)]
    scalars = [variable for variable in problem.variables() if variable.shape == (1, 1)]
    if len(vectors) != 1 or not scalars:
        raise ValueError("Cannot identify Riskfolio's raw homogeneous weights")
    scale = min(scalars, key=lambda variable: variable.id)
    if scale.value is None or vectors[0].value is None or float(scale.value.item()) <= 0:
        raise ValueError("Invalid homogeneous scale")
    raw = np.asarray(vectors[0].value).ravel() / float(scale.value.item())
    weights = frame.iloc[:, 0].to_numpy(dtype=float)
    audit_weights(raw)
    audit = audit_weights(weights)
    alteration = float(np.max(np.abs(weights - raw)))
    if alteration > 1e-8:
        raise ValueError("Riskfolio internally altered weights beyond numerical tolerance")
    return {
        "weights": weights,
        "raw_weights": raw,
        "solver_route": "riskfolio_MV_positive_mean",
        "maximum_feasible_mean": maximum_mean,
        "monthly_forecast_sharpe": -_loss_gradient(weights, mean, covariance)[0],
        "global_optimality_certified": True,
        "solver_status": problem.status,
        "audit": audit,
        "package_output_max_alteration": alteration,
        "warnings": [str(item.message) for item in caught],
    }
