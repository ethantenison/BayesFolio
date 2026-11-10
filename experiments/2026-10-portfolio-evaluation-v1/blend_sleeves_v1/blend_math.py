"""Pure cached-prediction blends and dimension-independent strict allocation.

Research-only numerical functions; no fitting, historical outcome access or I/O.
Means are decimal discounted-excess returns, covariance and budgets are decimal
squared, allocation scores are dimensionless. Imported by run_blends.py.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from scipy.stats import rankdata


def _validate(covariance: np.ndarray, budget: np.ndarray) -> None:
    if covariance.ndim != 3 or covariance.shape[-1] != covariance.shape[-2]:
        raise ValueError("Expected months by assets by assets covariance")
    if budget.shape != covariance.shape[:1] or not np.isfinite(budget).all() or not (budget > 0).all():
        raise ValueError("Complete positive variance budgets required")
    if not np.isfinite(covariance).all() or not np.allclose(
        covariance, covariance.swapaxes(-1, -2), rtol=1e-10, atol=1e-14
    ):
        raise ValueError("Finite symmetric covariance required; no repair")
    np.linalg.cholesky(covariance)


def _matrix_function(matrix: np.ndarray, function: str) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    if function == "log":
        if (values <= 0).any():
            raise ValueError("SPD required; eigenvalues are never clipped")
        values = np.log(values)
    elif function == "exp":
        values = np.exp(values)
    else:
        raise ValueError("Unknown matrix function")
    return (vectors * values[..., None, :]) @ vectors.swapaxes(-1, -2)


def blend_panel(broad: dict, compact: dict, alpha: np.ndarray, method: str, beta: float) -> dict:
    """Blend predictive means and SPD covariances on identical asset/time axes.

    Args:
        broad: Arrays named mean and covariance, with optional leading seed axis.
        compact: Matched arrays in the same decimal units.
        alpha: One compact-model mean fraction per asset, between zero and one.
        method: arithmetic, precision, log, correlation or mixture.
        beta: Compact covariance fraction between zero and one.

    Returns:
        Mean and covariance arrays. Mixture uses the exact two-component first two
        moments only when all alpha values equal beta. Its Gaussian NLL is a
        moment-matched approximation, not the mixture density or a single GP posterior.
    """
    left, right = broad["mean"], compact["mean"]
    a, b = broad["covariance"], compact["covariance"]
    if left.shape != right.shape or a.shape != b.shape or a.shape != (*left.shape, left.shape[-1]):
        raise ValueError("Matched mean/covariance axes required")
    if alpha.shape != left.shape[-1:] or not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
        raise ValueError("One valid compact fraction per asset required")
    if not 0 <= beta <= 1 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Finite forecasts and valid covariance fraction required")
    _validate(a.reshape(-1, a.shape[-1], a.shape[-1]), np.ones(a.size // a.shape[-1] ** 2))
    _validate(b.reshape(-1, b.shape[-1], b.shape[-1]), np.ones(b.size // b.shape[-1] ** 2))
    mean = (1 - alpha) * left + alpha * right
    if method == "arithmetic":
        covariance = (1 - beta) * a + beta * b
    elif method == "precision":
        covariance = np.linalg.inv((1 - beta) * np.linalg.inv(a) + beta * np.linalg.inv(b))
    elif method == "log":
        covariance = _matrix_function(
            (1 - beta) * _matrix_function(a, "log") + beta * _matrix_function(b, "log"), "exp"
        )
    elif method == "correlation":
        sa, sb = np.sqrt(np.diagonal(a, axis1=-2, axis2=-1)), np.sqrt(np.diagonal(b, axis1=-2, axis2=-1))
        ra, rb = a / (sa[..., :, None] * sa[..., None, :]), b / (sb[..., :, None] * sb[..., None, :])
        standard = (1 - beta) * sa + beta * sb
        covariance = ((1 - beta) * ra + beta * rb) * standard[..., :, None] * standard[..., None, :]
    elif method == "mixture":
        if not np.allclose(alpha, beta, rtol=0, atol=1e-14):
            raise ValueError("Mixture covariance requires a single common mixture fraction")
        difference = right - left
        covariance = (
            (1 - beta) * a + beta * b + beta * (1 - beta) * (difference[..., :, None] * difference[..., None, :])
        )
    else:
        raise ValueError(f"Unknown covariance method {method}")
    # Roundoff symmetrization is part of the matrix-function calculation, not SPD repair.
    covariance = (covariance + covariance.swapaxes(-1, -2)) / 2
    _validate(covariance.reshape(-1, a.shape[-1], a.shape[-1]), np.ones(a.size // a.shape[-1] ** 2))
    return {"mean": mean, "covariance": covariance}


def allocation_scores(
    broad: np.ndarray, compact: np.ndarray, mean: np.ndarray, alpha: np.ndarray, method: str
) -> np.ndarray:
    """Create rank allocation coefficients without mislabeling ranks as returns.

    Args:
        broad: Broad forecast means, (..., assets), decimal returns.
        compact: Matched compact forecast means.
        mean: Actual blended return forecasts for forecast scoring.
        alpha: Per-asset compact fraction.
        method: mean ranks blended means; borda blends source percentile ranks.

    Returns:
        Dimensionless coefficients with the same axes. Borda does not rerank its
        blended coefficients; its magnitudes express model agreement.
    """
    count = mean.shape[-1]
    if method == "mean":
        return rankdata(mean, axis=-1) / count
    if method == "borda":
        return ((1 - alpha) * rankdata(broad, axis=-1) + alpha * rankdata(compact, axis=-1)) / count
    raise ValueError("Unknown decision blend")


def _solve(problem: cp.Problem) -> list[dict]:
    attempts = []
    for step, regularization in [(0.9, 1e-12), (0.8, 0.0), (0.95, 1e-12), (0.99, 1e-12)]:
        settings = {"max_step_fraction": step, "static_regularization_enable": regularization > 0}
        if regularization:
            settings["static_regularization_constant"] = regularization
        try:
            problem.solve(
                solver="CLARABEL", warm_start=False, tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9, **settings
            )
            attempts.append({"solver": "CLARABEL", "status": problem.status, **settings})
        except cp.error.SolverError as error:
            attempts.append({"solver": "CLARABEL", "status": "solver_error", "error": str(error), **settings})
        if attempts[-1]["status"] not in {"solver_error", "optimal_inaccurate"}:
            break
    if attempts[-1]["status"] in {"solver_error", "optimal_inaccurate"}:
        try:
            problem.solve(solver="SCS", warm_start=False, eps_abs=1e-10, eps_rel=1e-10, max_iters=100000)
            attempts.append({"solver": "SCS", "status": problem.status})
        except cp.error.SolverError as error:
            attempts.append({"solver": "SCS", "status": "solver_error", "error": str(error)})
    if attempts[-1]["status"] != cp.OPTIMAL:
        raise ValueError(f"No strict optimal solve: {attempts}")
    return attempts


def _record(weights: np.ndarray, matrix: np.ndarray, budget: float, minimum: int, index: int) -> dict:
    concentration, variance = float(weights @ weights), float(weights @ matrix @ weights)
    if (
        not np.isfinite(weights).all()
        or weights.min() < -1e-8
        or abs(weights.sum() - 1) > 1e-7
        or concentration > 1 / minimum + 1e-7
        or variance > budget * (1 + 1e-6)
    ):
        raise ValueError("Raw weights violate the unchanged combined portfolio constraints")
    return {
        "index": index,
        "status": "solved",
        "weights": weights.tolist(),
        "minimum_effective": minimum,
        "variance_budget": float(budget),
        "predicted_variance": variance,
        "effective_count": 1 / concentration,
        "largest_weight": float(weights.max()),
        "ETF_position_cap": None,
        "fallback_used": False,
        "constraint_relaxation": False,
    }


def joint_allocate(
    covariance: np.ndarray,
    budget: np.ndarray,
    scores: np.ndarray,
    minimum: int,
    equity_indices: list[int] | None = None,
) -> list[dict]:
    """Solve strict full-portfolio controls, optionally fixing equity capital at 80%.

    Args:
        covariance: Months by assets by assets risk, decimal squared.
        budget: Monthly positive original full-portfolio variance ceilings.
        scores: Dimensionless monthly objective coefficients.
        minimum: Combined effective asset floor.
        equity_indices: If present, constrain their combined capital to 0.8.

    Returns:
        Raw feasible monthly weights and numerical attempts; failures raise.
    """
    _validate(covariance, budget)
    if scores.shape != covariance.shape[:2] or not np.isfinite(scores).all():
        raise ValueError("Complete finite scores required")
    records = []
    for index, (matrix, limit, coefficients) in enumerate(zip(covariance, budget, scores, strict=True)):
        weights = cp.Variable(len(coefficients))
        factor = np.linalg.cholesky(matrix / limit).T
        constraints = [weights >= 0, cp.sum(weights) == 1, cp.norm(factor @ weights) <= 1]
        constraints.append(cp.norm(weights) <= 1 / np.sqrt(minimum))
        if equity_indices is not None:
            constraints.append(cp.sum(weights[equity_indices]) == 0.8)
        objective = (coefficients - coefficients.mean()) @ weights - 1e-8 * cp.sum_squares(weights)
        problem = cp.Problem(cp.Maximize(objective), constraints)
        attempts = _solve(problem)
        raw = np.asarray(weights.value).copy()
        if equity_indices is not None and abs(raw[equity_indices].sum() - 0.8) > 1e-7:
            raise ValueError("Fixed equity capital failed validation")
        records.append({**_record(raw, matrix, float(limit), minimum, index), "numerical_attempts": attempts})
    return records


def sleeve_allocate(
    covariance: np.ndarray,
    budget: np.ndarray,
    scores: np.ndarray,
    minimum: int,
    equity_indices: list[int],
) -> list[dict]:
    """Solve equity and bond sleeves separately with a shared concentration price.

    Args:
        covariance: Full monthly risk including cross-sleeve covariance.
        budget: Original full-portfolio variance ceiling for each month.
        scores: Within-sleeve percentile ranks on the full ordered asset axis.
        minimum: Effective-count floor for the COMBINED portfolio only.
        equity_indices: Equity asset positions; remaining positions are fixed income.

    Returns:
        Full portfolios, normalized separate sleeve weights, dual certificates and
        cross-sleeve risk. Each sleeve has volatility at most sqrt(budget), which
        conservatively guarantees combined variance at most budget by Cauchy-Schwarz.
        No arbitrary sleeve asset floors or independent-model assumption is used.
        A scalar dual price enforces the combined concentration floor without a
        full-vector allocation solve; price search uses no realized returns.
    """
    _validate(covariance, budget)
    if scores.shape != covariance.shape[:2] or not np.isfinite(scores).all():
        raise ValueError("Complete finite scores required")
    fixed_indices = [i for i in range(scores.shape[1]) if i not in equity_indices]
    groups, shares = [equity_indices, fixed_indices], [0.8, 0.2]
    records = []
    for index, (matrix, limit, coefficients) in enumerate(zip(covariance, budget, scores, strict=True)):
        price = cp.Parameter(nonneg=True)
        problems, variables = [], []
        for indices, fraction in zip(groups, shares, strict=True):
            weights = cp.Variable(len(indices))
            local = matrix[np.ix_(indices, indices)]
            constraints = [
                weights >= 0,
                cp.sum(weights) == 1,
                cp.norm(np.linalg.cholesky(local / limit).T @ weights) <= 1,
            ]
            centered = coefficients[indices] - coefficients[indices].mean()
            objective = fraction * centered @ weights - (1e-8 + price) * fraction**2 * cp.sum_squares(weights)
            problems.append(cp.Problem(cp.Maximize(objective), constraints))
            variables.append(weights)

        def at(value: float) -> tuple[np.ndarray, list[dict]]:
            price.value = value
            combined, attempts = np.zeros(len(coefficients)), []
            for indices, fraction, problem, variable in zip(groups, shares, problems, variables, strict=True):
                attempts.extend(_solve(problem))
                raw = np.asarray(variable.value)
                if raw.min() < -1e-8 or abs(raw.sum() - 1) > 1e-7:
                    raise ValueError("Invalid raw sleeve weights; no normalization")
                if raw @ matrix[np.ix_(indices, indices)] @ raw > limit * (1 + 1e-6):
                    raise ValueError("Sleeve risk budget violated")
                combined[indices] = fraction * raw
            return combined, attempts

        candidate, attempts = at(0.0)
        calls, multiplier, gap = 1, 0.0, 0.0
        if candidate @ candidate > 1 / minimum:
            low, high = 0.0, 1.0
            for _ in range(30):
                candidate, attempts = at(high)
                calls += 1
                if candidate @ candidate <= 1 / minimum:
                    break
                high *= 2
            else:
                raise ValueError("Separate risk constraints and combined concentration have no found feasible point")
            for _ in range(60):
                midpoint = (low + high) / 2
                trial, trial_attempts = at(midpoint)
                calls += 1
                if trial @ trial > 1 / minimum:
                    low = midpoint
                else:
                    high, candidate, attempts = midpoint, trial, trial_attempts
                gap = high * (1 / minimum - float(candidate @ candidate))
                if gap <= 2e-9 and high - low <= 1e-5:
                    break
            multiplier = high
            if gap > 5e-8:
                raise ValueError("Concentration dual certificate not accurate enough")
        record = _record(candidate, matrix, float(limit), minimum, index)
        eq, fi = candidate[equity_indices], candidate[fixed_indices]
        records.append(
            {
                **record,
                "numerical_attempts": attempts,
                "concentration_price": multiplier,
                "concentration_dual_gap": gap,
                "separate_sleeve_solver_calls": calls * 2,
                "equity_sleeve_weights": (eq / 0.8).tolist(),
                "fixed_income_sleeve_weights": (fi / 0.2).tolist(),
                "equity_variance_contribution": float(eq @ matrix[np.ix_(equity_indices, equity_indices)] @ eq),
                "fixed_income_variance_contribution": float(fi @ matrix[np.ix_(fixed_indices, fixed_indices)] @ fi),
                "cross_sleeve_variance_contribution": float(
                    2 * eq @ matrix[np.ix_(equity_indices, fixed_indices)] @ fi
                ),
                "sleeve_budget_policy": "each_normalized_sleeve_variance_le_original_full_budget",
            }
        )
    return records
