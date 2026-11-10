"""Sample frozen joint Gaussian GP forecasts and apply the July Riskfolio policy.

Isolated research numerics only: no GP imports, fitting, I/O or historical-return
estimation. Means/scenarios are decimal discounted excess returns; covariance is
decimal squared and weights are fractions. Riskfolio's hist estimators operate
on simulated GP scenarios, not observed market returns.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import pandas as pd
import riskfolio as rp


def sample_joint(mean: np.ndarray, covariance: np.ndarray, *, seed: int, month: str, count: int = 5000) -> np.ndarray:
    """Draw reproducible joint Gaussian scenarios without changing the forecast.

    Args:
        mean: (assets,) saved current GP predictive means, decimal target units.
        covariance: (assets, assets) observed predictive covariance including noise.
        seed: Existing GP initialization seed, used only as a scenario RNG key.
        month: ISO forecast-origin date, another scheduling-independent RNG key.
        count: Number of independent Gaussian draws; historical policy uses 5000.

    Returns:
        (count, assets) joint scenarios with the supplied population distribution.
        No moment matching, eigenvalue clipping, diagonalization or jitter occurs.
    """
    if mean.ndim != 1 or covariance.shape != (len(mean), len(mean)) or count < 2:
        raise ValueError("Aligned vector/matrix and at least two scenarios required")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Finite frozen GP moments required")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("Symmetric predictive covariance required")
    factor = np.linalg.cholesky(covariance)
    generator = np.random.default_rng(np.random.SeedSequence([20261024, seed, int(month.replace("-", ""))]))
    return mean + generator.standard_normal((count, len(mean))) @ factor.T


def audit_weights(weights: np.ndarray) -> dict[str, float]:
    """Reject violations of the old policy without clipping or normalizing.

    Args:
        weights: One finite long-only weight vector with full-investment budget.

    Returns:
        Budget residual, largest weight, effective assets and minimum weight.
        Effective assets must be at least 10 and no ETF may exceed 25 percent.
    """
    if weights.ndim != 1 or not np.isfinite(weights).all():
        raise ValueError("Finite one-dimensional weights required")
    residual = abs(float(weights.sum()) - 1)
    concentration = float(weights @ weights)
    if weights.min() < -1e-8 or residual > 1e-7 or weights.max() > 0.25 + 1e-8:
        raise ValueError("Long-only, full-investment or 25-percent-cap violation")
    if concentration > 0.1 + 1e-8:
        raise ValueError("Minimum effective assets 10 violated")
    return {
        "budget_residual": residual,
        "max_weight": float(weights.max()),
        "effective_assets": 1 / concentration,
        "minimum_weight": float(weights.min()),
    }


def allocate_scenarios(scenarios: np.ndarray, assets: list[str]) -> dict[str, Any]:
    """Call real Riskfolio with the old economic settings and strict solver audits.

    Args:
        scenarios: Complete (draws, assets) joint GP decimal-return scenarios.
        assets: Exact current ETF column order; no helper ETFs are removed.

    Returns:
        Accepted weights, raw solver weights and numerical-attempt diagnostics.
        Policy: Classic/CVaR/Sharpe, alpha 0.5, arithmetic mean, rf zero, hist/hist,
        long-only, full investment, effective assets >=10, ETF cap 25 percent.
        Only numerical solver tolerances may be retried; economic constraints
        never change. Riskfolio internally takes absolute values/normalizes its
        output; raw solver feasibility and negligible alteration are audited.

    Raises:
        RuntimeError: No strictly optimal feasible solution; never uses fallback.
    """
    if scenarios.ndim != 2 or scenarios.shape[1] != len(assets) or len(set(assets)) != len(assets):
        raise ValueError("Scenario columns must match the complete unique ETF order")
    if len(assets) < 10 or len(scenarios) < 2 or not np.isfinite(scenarios).all():
        raise ValueError("At least ten ETFs and complete finite scenarios required")
    attempts = []
    settings = [{}, {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10, "max_iter": 1000}]
    for parameters in settings:
        started = time.monotonic()
        portfolio = rp.Portfolio(returns=pd.DataFrame(scenarios, columns=assets), nea=10)
        portfolio.alpha, portfolio.upperlng, portfolio.lowerlng = 0.5, 0.25, 0.0
        portfolio.card = None
        portfolio.solvers = ["CLARABEL"]
        portfolio.sol_params = {"CLARABEL": parameters}
        portfolio.assets_stats(method_mu="hist", method_cov="hist")
        captured = []
        original_solve = cp.Problem.solve

        def tracked_solve(problem: cp.Problem, *args: Any, **kwargs: Any) -> Any:
            captured.append(problem)
            return original_solve(problem, *args, **kwargs)

        record: dict[str, Any] = {"solver": "CLARABEL", "parameters": parameters}
        try:
            with patch.object(cp.Problem, "solve", tracked_solve):
                result = portfolio.optimization(model="Classic", rm="CVaR", obj="Sharpe", rf=0.0, hist=True)
            if len(captured) != 1 or captured[0].status != cp.OPTIMAL:
                raise ValueError(f"Strict optimal status required: {[problem.status for problem in captured]}")
            if result is None or result.index.tolist() != assets:
                raise ValueError("Riskfolio returned missing or reordered ETF weights")
            problem = captured[0]
            variables = problem.variables()
            vectors = [variable for variable in variables if variable.shape == (len(assets), 1)]
            scalars = [variable for variable in variables if variable.shape == (1, 1)]
            # Riskfolio 7.2.1 declares w then k, before its auxiliary CVaR scalars.
            if len(vectors) != 1 or not scalars:
                raise ValueError("Cannot identify the unmodified Riskfolio solver weights")
            scale = min(scalars, key=lambda variable: variable.id)
            if scale.value is None or vectors[0].value is None or float(np.asarray(scale.value).item()) <= 0:
                raise ValueError("Invalid homogeneous Riskfolio weight scale")
            raw = np.asarray(vectors[0].value).ravel() / float(np.asarray(scale.value).item())
            weights = result.iloc[:, 0].to_numpy(dtype=float)
            raw_audit, accepted_audit = audit_weights(raw), audit_weights(weights)
            difference = float(np.max(np.abs(weights - raw)))
            if difference > 1e-8:
                raise ValueError("Riskfolio output alteration exceeds numerical tolerance")
            record.update(
                status=problem.status,
                raw_weight_audit=raw_audit,
                returned_weight_audit=accepted_audit,
                package_output_max_alteration=difference,
                solve_seconds=problem.solver_stats.solve_time,
                iterations=problem.solver_stats.num_iters,
                wall_seconds=time.monotonic() - started,
            )
            attempts.append(record)
            return {"weights": weights, "raw_weights": raw, "attempts": attempts, "audit": accepted_audit}
        except (ValueError, RuntimeError, cp.error.SolverError) as error:
            record.update(error=f"{type(error).__name__}: {error}", wall_seconds=time.monotonic() - started)
            attempts.append(record)
    raise RuntimeError(f"Riskfolio failed without changing constraints: {attempts}")
