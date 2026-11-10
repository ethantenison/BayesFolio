"""Equivalent cone formulation for the sealed study's portfolio allocations.

Research-only numerical repair; no forecasts, covariances, budgets, objective
preferences or acceptance tolerances are changed. Returns are decimals and
covariance/variance budgets are decimal squared. Import after adding the sealed
approved_scope_v1 directory to sys.path; see run_repair.py for usage.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import study_common  # noqa: F401  # Establish the sealed framework's research import paths.
from study_math import ScoreAllocator


class StableScoreAllocator(ScoreAllocator):
    """Solve the identical investment problem using explicitly factored cones.

    Inherits the frozen input validation and retains strict solver tolerances and
    raw-weight acceptance checks. On the fully invested simplex, centering scores
    subtracts only a constant. Positive objective scaling changes no maximizer.

    Attributes:
        objective_scale: Positive numerical scale, not a preference parameter.
    """

    def __init__(self, covariance: np.ndarray, budget: np.ndarray, minimum: int | None, penalty: float) -> None:
        super().__init__(covariance, budget, minimum, penalty)
        self.objective_scale = 4.0 if penalty == 0.25 else 1.0
        self.problems = []
        for index, matrix in enumerate(covariance):
            weights, scores = cp.Variable(15), cp.Parameter(15)
            # If A = L L.T, w.T A w = ||L.T w||_2**2 exactly.
            factor = np.linalg.cholesky(matrix / budget[index]).T
            constraints = [weights >= 0, cp.sum(weights) == 1, cp.norm(factor @ weights, 2) <= 1]
            if minimum is not None:
                constraints.append(cp.norm(weights, 2) <= 1 / np.sqrt(minimum))
            centered = scores - cp.sum(scores) / 15
            objective = self.objective_scale * (centered @ weights - penalty * cp.sum_squares(weights))
            problem = cp.Problem(cp.Maximize(objective), constraints)
            if not problem.is_dpp():
                raise ValueError("Equivalent formulation must remain DPP")
            self.problems.append((problem, weights, scores))

    def allocate(self, scores: np.ndarray) -> list[dict]:
        """Apply unchanged strict acceptance and attach numerical diagnostics.

        Args:
            scores: Finite dimensionless coefficients with shape (24, 15).

        Returns:
            All monthly records, including raw accepted weights or explicit nulls.
        """
        if scores.shape != (24, 15) or not np.isfinite(scores).all():
            raise ValueError("Incomplete objective scores")
        records = []
        for index, (problem, variable, parameter) in enumerate(self.problems):
            parameter.value = scores[index]
            record = {
                "index": index,
                "weights": None,
                "minimum_effective": self.minimum,
                "ETF_position_cap": None,
                "fallback_used": False,
                "constraint_relaxation": False,
                "variance_budget": float(self.budget[index]),
                "formulation": "factored_norm_centered_objective_v2",
                "objective_scale": self.objective_scale,
            }
            attempts = []
            # Numerical retries solve the SAME problem, never a substitute portfolio.
            # Infeasible problems are not retried and inaccurate weights are not used.
            for step, regularization in [(0.9, 1e-12), (0.8, 0.0), (0.95, 1e-12), (0.99, 1e-12)]:
                settings = {"max_step_fraction": step, "static_regularization_enable": regularization > 0}
                if regularization > 0:
                    settings["static_regularization_constant"] = regularization
                try:
                    problem.solve(
                        solver="CLARABEL",
                        warm_start=False,
                        enforce_dpp=True,
                        tol_gap_abs=1e-9,
                        tol_gap_rel=1e-9,
                        tol_feas=1e-9,
                        **settings,
                    )
                except cp.error.SolverError as error:
                    attempts.append({**settings, "status": "solver_error", "error": str(error)})
                    continue
                attempts.append({**settings, "status": problem.status, "iterations": problem.solver_stats.num_iters})
                if problem.status != cp.OPTIMAL_INACCURATE:
                    break
            record["numerical_attempts"] = attempts
            if attempts[-1]["status"] in {"optimal_inaccurate", "solver_error"}:
                try:
                    problem.solve(
                        solver="SCS", warm_start=False, enforce_dpp=True, eps_abs=1e-10, eps_rel=1e-10, max_iters=100000
                    )
                    attempts.append(
                        {"solver": "SCS", "status": problem.status, "iterations": problem.solver_stats.num_iters}
                    )
                except cp.error.SolverError as error:
                    attempts.append({"solver": "SCS", "status": "solver_error", "error": str(error)})
            if attempts[-1]["status"] == "solver_error":
                records.append({**record, "status": "solver_error"})
                continue
            record.update(solver_status=problem.status, solver_iterations=problem.solver_stats.num_iters)
            if problem.status != cp.OPTIMAL or variable.value is None:
                records.append({**record, "status": str(problem.status)})
                continue
            weights = np.asarray(variable.value).copy()
            squared = float(weights @ weights)
            variance = float(weights @ self.covariance[index] @ weights)
            if (
                not np.isfinite(weights).all()
                or weights.min() < -1e-8
                or abs(weights.sum() - 1) > 1e-7
                or variance > self.budget[index] * (1 + 1e-6)
                or (self.minimum is not None and squared > 1 / self.minimum + 1e-7)
            ):
                records.append({**record, "status": "constraint_violation"})
                continue
            records.append(
                {
                    **record,
                    "status": "solved",
                    "weights": weights.tolist(),
                    "predicted_variance": variance,
                    "effective_count": 1 / squared,
                    "largest_weight": float(weights.max()),
                    "scaled_objective": float(problem.value),
                }
            )
        return records
