"""Training-only feasibility checks for the approved portfolio policy.

Research numerical layer. Covariances and budgets are decimal return squared.
No realized returns, rank scores, forecast fitting or I/O enter the check.
An equal-weight witness or a valid quadratic dual lower bound separates feasible
policies from conflicting constraints. Numerical uncertainty is not infeasibility.
Usage: imported by review.py; allocate_checked is the future evaluation entrypoint.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def preflight(covariance: np.ndarray, budget: np.ndarray, minimum: int | None) -> list[dict]:
    """Certify monthly feasibility before choosing allocations from forecasts.

    Args:
        covariance: Positive-definite return covariances, shape (months, assets, assets).
        budget: Positive monthly variance ceilings, shape (months,), decimal squared.
        minimum: Optional effective-count floor, between one and the asset count.

    Returns:
        One record per month. Infeasibility requires a dual lower bound above the
        budget; an unsuccessful numerical solve remains explicitly undetermined.
    """
    if covariance.ndim != 3 or covariance.shape[1] != covariance.shape[2]:
        raise ValueError("Expected square covariance matrices")
    months, assets, _ = covariance.shape
    if budget.shape != (months,) or not np.isfinite(budget).all() or not (budget > 0).all():
        raise ValueError("Positive complete variance budgets required")
    if not np.isfinite(covariance).all() or not np.allclose(
        covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-14
    ):
        raise ValueError("Finite symmetric covariance required; no repair")
    np.linalg.cholesky(covariance)
    if minimum is not None and (not isinstance(minimum, int) or not 1 <= minimum <= assets):
        raise ValueError("Invalid effective-count floor")
    equal = np.ones(assets) / assets
    rows = []
    for index, matrix in enumerate(covariance):
        scaled = matrix / budget[index]
        reference = float(equal @ scaled @ equal)
        record = {
            "index": index,
            "minimum": minimum,
            "variance_budget": float(budget[index]),
            "equal_weight_variance_to_budget": reference,
            "budget_relaxed": False,
        }
        if reference <= 1 + 1e-10:
            rows.append({**record, "status": "feasible_equal_weight_witness"})
            continue
        weights = cp.Variable(assets)
        constraints = [weights >= 0, cp.sum(weights) == 1]
        if minimum is not None:
            constraints.append(cp.sum_squares(weights) <= 1 / minimum)
        problem = cp.Problem(cp.Minimize(cp.quad_form(weights, cp.psd_wrap(scaled))), constraints)
        try:
            problem.solve(solver="CLARABEL", warm_start=False, tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
        except cp.error.SolverError as error:
            rows.append({**record, "status": "numerically_undetermined", "error": str(error)})
            continue
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
            rows.append({**record, "status": "numerically_undetermined", "solver_status": problem.status})
            continue
        candidate = np.asarray(weights.value)
        nonnegative = np.asarray(constraints[0].dual_value)
        equality = np.asarray(constraints[1].dual_value).item()
        concentration = np.asarray(constraints[2].dual_value).item() if minimum is not None else 0.0
        upper = float(candidate @ scaled @ candidate)
        # The quadratic Lagrangian infimum is a lower bound for every feasible w.
        # No primal weights or covariance entries are clipped to create a witness.
        lower = None
        # Any dual-feasible multipliers give a lower bound, even if the numerical
        # optimizer did not certify its own optimality. This never accepts an
        # inaccurate portfolio allocation: preflight tests only feasibility.
        if np.isfinite(nonnegative).all() and np.isfinite(equality) and (nonnegative >= 0).all() and concentration >= 0:
            linear = equality * np.ones(assets) - nonnegative
            lower = float(
                -0.25 * linear @ np.linalg.solve(scaled + concentration * np.eye(assets), linear)
                - equality
                - (concentration / minimum if minimum is not None else 0)
            )
        valid = (
            np.isfinite(candidate).all()
            and candidate.min() >= -1e-8
            and abs(candidate.sum() - 1) <= 1e-7
            and (minimum is None or candidate @ candidate <= 1 / minimum + 1e-7)
        )
        if valid and upper <= 1 + 1e-6:
            status = "feasible_optimized_witness"
        elif lower is not None and lower > 1 + 1e-6:
            status = "infeasible_dual_certificate"
        else:
            status = "numerically_undetermined"
        rows.append(
            {
                **record,
                "status": status,
                "solver_status": problem.status,
                "minimum_variance_upper_to_budget": upper,
                "minimum_variance_lower_to_budget": lower,
            }
        )
    return rows


def allocate_checked(
    covariance: np.ndarray, budget: np.ndarray, minimum: int | None, penalty: float, scores: np.ndarray
) -> dict:
    """Run the repaired allocator only after every monthly policy passes preflight.

    Args:
        covariance: Complete 24 by 15 by 15 decimal-squared risk forecast.
        budget: Positive 24-element variance ceilings, never silently increased.
        minimum: Optional effective-count floor.
        penalty: Unchanged concentration-penalty coefficient.
        scores: Dimensionless allocation scores with shape (24, 15).

    Returns:
        Preflight records and either all allocations or an explicit blocked policy.
        No partial performance, substitute portfolio or relaxed constraints.
    """
    from stable_allocator import StableScoreAllocator

    allocator = StableScoreAllocator(covariance, budget, minimum, penalty)
    if scores.shape != (24, 15) or not np.isfinite(scores).all():
        raise ValueError("Complete finite scores required")
    checks = preflight(covariance, budget, minimum)
    if not all(row["status"].startswith("feasible_") for row in checks):
        return {"status": "blocked_preflight", "preflight": checks, "allocations": None}
    return {"status": "allocated", "preflight": checks, "allocations": allocator.allocate(scores)}
