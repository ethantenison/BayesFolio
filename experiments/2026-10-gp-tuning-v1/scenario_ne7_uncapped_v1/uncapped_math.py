"""Explicit NE7, uncapped signed-Sharpe numerics; no portfolio or return fallback.

Pure research numerical adapter reusing the tested explicit-policy implementation.
Returns are decimal, covariance decimal squared and weights fractions. NE7 and
long-only/full investment remain; a bound of one is mathematically redundant,
not an ETF position cap. Bounded numerical retries change only solver step size
or linear-system regularization, never constraints, objective or tolerances.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import cvxpy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scenario_ne7_v1"))
import ne7_math  # noqa: E402
from ne7_math import Policy, audit_weights  # noqa: E402


def strict_solve(problem: cp.Problem) -> dict:
    """Solve one unchanged conic problem with bounded, audited numeric retries.

    Args:
        problem: Already-built dimensionless conic Sharpe or mean-sign problem.

    Returns:
        Strict optimal status, primal/dual residuals and gap, constraint audit,
        and every numerical attempt. Reduced-accuracy outcomes are rejected.

    Raises:
        RuntimeError: No full-accuracy feasible certified solve; no weights reused.
    """
    attempts = []
    for extra in [{}, {"max_step_fraction": .9}, {"max_step_fraction": .8},
                  {"max_step_fraction": .95, "static_regularization_constant": 1e-12},
                  {"max_step_fraction": .9, "static_regularization_enable": False}]:
        with warnings.catch_warnings(record=True) as caught:
            try:
                problem.solve(solver="CLARABEL", warm_start=False, tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                              tol_feas=1e-9, max_iter=1000, **extra)
            except cp.error.SolverError as error:
                attempts.append({"settings": extra, "status": "solver_error", "error": str(error)})
                continue
        record = {"settings": extra, "status": problem.status, "warnings": [str(w.message) for w in caught]}
        attempts.append(record)
        if problem.status != cp.OPTIMAL:
            continue
        solution = problem._solver_cache["CLARABEL"].get_solution()
        certificate = {"status": problem.status, "primal_residual": float(solution.r_prim),
                       "dual_residual": float(solution.r_dual),
                       "duality_gap": float(solution.obj_val - solution.obj_val_dual),
                       "dual_upper_bound": -float(solution.obj_val_dual),
                       "iterations": int(problem.solver_stats.num_iters),
                       "largest_constraint_violation": max(float(np.max(c.violation())) for c in problem.constraints)}
        if (certificate["primal_residual"] <= 1e-8 and certificate["dual_residual"] <= 1e-8
                and abs(certificate["duality_gap"]) <= 1e-8
                and certificate["largest_constraint_violation"] <= 1e-8):
            return {**certificate, "numerical_attempts": attempts}
        record["rejected_certificate"] = certificate
    raise RuntimeError(f"No strict certified conic solve under identical policy: {attempts}")


def allocate_scenarios(scenarios: np.ndarray, assets: list[str], *, seed: int,
                       policy: Policy = Policy(7, 1)) -> dict:
    """Allocate unchanged scenarios, with NE7 and no position cap by default.

    Args:
        scenarios: (5000,15) finite decimal excess-return draws, either sign.
        assets: Exact fifteen unique ETF column names.
        seed: Deterministic optimizer-start seed.
        policy: Explicit constraints; default NE7 with no nonredundant cap.
            A capped policy is permitted only for recorded diagnostic comparisons.

    Returns:
        Raw audited weights, signed Sharpe and numerical route/certificates.
        No variance ceiling, CVaR, return floor, clipping or replacement weights.
        Use one allocation per process, not concurrent threads.
    """
    with patch.object(ne7_math, "_strict_solve", strict_solve):
        result = ne7_math.allocate_scenarios(scenarios, assets, seed=seed, policy=policy)
    audit_weights(result["weights"], policy)
    return result
