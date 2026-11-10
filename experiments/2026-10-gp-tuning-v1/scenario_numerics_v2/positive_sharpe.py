"""Research-only, dimensionless conic maximum-Sharpe numerical repair.

Inputs are exact frozen scenario sample means (decimal returns) and covariance
(decimal squared). Outputs are unmodified long-only fully-invested weights.
This is the same positive-mean Sharpe problem as Riskfolio MV/Sharpe/rf=0,
with effective assets >=10 and a 25-percent position cap. No GP fitting, data
refresh, covariance repair, weight repair or alternative objective occurs here.

For any feasible portfolio w with positive mean, x=w/||A w|| and t=sum(x)
obey ||A x||<=1, where A.T@A=covariance/scale**2. Maximizing mean@x/scale
therefore maximizes the original mean/volatility ratio. Conversely x/t is a
feasible portfolio. This avoids Riskfolio's mean@x=1 normalization, which
creates huge variables when the greatest feasible mean is close to zero.
"""

from __future__ import annotations

import warnings
from typing import Any

import cvxpy as cp
import numpy as np


def solve_positive(mean: np.ndarray, covariance: np.ndarray) -> dict[str, Any]:
    """Solve the unchanged positive-Sharpe allocation using a scaled SOCP.

    Args:
        mean: Fifteen finite scenario sample means in decimal returns. At least
            one portfolio under the constraints must have positive mean.
        covariance: Symmetric strictly positive-definite (15,15) sample
            covariance in decimal-return squared units, never repaired.

    Returns:
        Raw homogeneous-backtransformed weights and numerical certificates.
        A strict optimal status, primal/dual residuals, duality gap, risk-cone
        saturation and original weight constraints must all pass. The solver's
        dual objective is an independently recorded upper bound on Sharpe.

    Raises:
        ValueError: Invalid inputs or failed raw-weight/certificate audit.
        RuntimeError: No strictly optimal positive-Sharpe solution. Inaccurate
            statuses are rejected, not accepted or replaced by fallback weights.
    """
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if mean.shape != (15,) or covariance.shape != (15, 15):
        raise ValueError("Exactly fifteen aligned ETF means and covariance required")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Finite scenario moments required")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("Symmetric covariance required; no repair")
    factor = np.linalg.cholesky(covariance)
    equal = np.full(15, 1 / 15)
    scale = float(np.sqrt(equal @ covariance @ equal))
    transform = factor.T / scale
    x = cp.Variable(15, name="scaled_holdings")
    t = cp.sum(x)
    constraints = [x >= 0, x <= .25 * t, cp.SOC(t / np.sqrt(10), x), cp.SOC(1., transform @ x)]
    problem = cp.Problem(cp.Maximize((mean / scale) @ x), constraints)
    settings = {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-9, "max_iter": 1000}
    with warnings.catch_warnings(record=True) as caught:
        problem.solve(solver="CLARABEL", **settings)
    if problem.status != cp.OPTIMAL or x.value is None:
        raise RuntimeError(f"Strict conic optimal status required: {problem.status}")
    holdings = np.asarray(x.value).copy()
    homogeneous_scale = float(holdings.sum())
    if homogeneous_scale <= 0:
        raise ValueError("No positive feasible Sharpe solution")
    weights = holdings / homogeneous_scale
    squared = float(weights @ weights)
    audit = {"sum": float(weights.sum()), "min": float(weights.min()), "max": float(weights.max()),
             "sum_squared": squared, "effective_count": 1 / squared}
    if (not np.isfinite(weights).all() or audit["min"] < -1e-8 or audit["max"] > .25 + 1e-8
            or abs(audit["sum"] - 1) > 1e-7 or squared > .1 + 1e-8):
        raise ValueError(f"Raw weights violate original constraints: {audit}")
    score = float(mean @ weights / np.sqrt(weights @ covariance @ weights))
    if score <= 0:
        raise ValueError("Positive-Sharpe branch cannot certify a nonpositive optimum")
    # CVXPY exposes the actual Clarabel primal/dual solution, not rounded logs.
    solution = problem._solver_cache["CLARABEL"].get_solution()
    dual_upper = -float(solution.obj_val_dual)  # Canonical minimization negates our objective.
    gap = dual_upper - float(problem.value)
    risk_norm = float(np.linalg.norm(transform @ holdings))
    certificate = {"primal_residual": float(solution.r_prim), "dual_residual": float(solution.r_dual),
                   "primal_sharpe_objective": float(problem.value), "dual_sharpe_upper_bound": dual_upper,
                   "duality_gap": gap, "risk_cone_norm": risk_norm,
                   "backtransform_objective_difference": abs(score - float(problem.value)),
                   "largest_constraint_violation": max(float(np.max(c.violation())) for c in constraints)}
    if (certificate["primal_residual"] > 1e-8 or certificate["dual_residual"] > 1e-8
            or abs(gap) > 1e-8 or abs(risk_norm - 1) > 1e-7
            or certificate["backtransform_objective_difference"] > 1e-8
            or certificate["largest_constraint_violation"] > 1e-8):
        raise ValueError(f"Conic numerical certificate failed: {certificate}")
    return {"weights": weights, "raw_weights": weights.copy(), "audit": audit,
            "monthly_forecast_sharpe": score, "solver_route": "scaled_positive_sharpe_socp_v2",
            "solver_status": problem.status, "global_optimality_certified": True,
            "certificate": certificate, "homogeneous_scale": homogeneous_scale,
            "equal_weight_volatility_scale": scale, "solver_settings": settings,
            "iterations": int(problem.solver_stats.num_iters), "warnings": [str(w.message) for w in caught]}
