"""Rank utility with continuously penalized observed-return portfolio variance.

Research-only numerical extension of the sealed strict allocator. No I/O,
realized returns, GP fitting or calibration. Scores and utility are dimensionless;
covariance and the common reference variance are decimal-return squared.
Usage: imported by run_rank_risk.py; zero risk aversion preserves the old problem.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from feasibility import preflight
from stable_allocator import StableScoreAllocator


class RankRiskAllocator(StableScoreAllocator):
    """Extend the existing objective without duplicating its numerical solver.

    Attributes:
        risk_aversion: Nonnegative penalty on variance divided by the unchanged
            monthly budget. The denominator is common across covariance sources.
        concentration_penalty: Original tiny tie-breaking coefficient, not risk.
    """

    def __init__(
        self,
        covariance: np.ndarray,
        budget: np.ndarray,
        minimum: int | None,
        penalty: float,
        risk_aversion: float,
    ) -> None:
        if not np.isfinite(risk_aversion) or risk_aversion < 0:
            raise ValueError("Risk aversion must be finite and nonnegative")
        super().__init__(covariance, budget, minimum, penalty)
        self.risk_aversion = float(risk_aversion)
        self.concentration_penalty = float(penalty)
        if risk_aversion == 0:
            return  # Preserve the exact previous CVXPY problem and solver path.
        for index, (original, weights, scores) in enumerate(self.problems):
            factor = np.linalg.cholesky(covariance[index] / budget[index]).T
            objective = original.objective.expr - self.objective_scale * risk_aversion * cp.sum_squares(
                factor @ weights
            )
            problem = cp.Problem(cp.Maximize(objective), original.constraints)
            if not problem.is_dpp():
                raise ValueError("Rank-risk problem must remain reusable")
            self.problems[index] = (problem, weights, scores)

    def allocate(self, scores: np.ndarray) -> list[dict]:
        """Solve the declared utility and retain strict raw-weight acceptance.

        Args:
            scores: Finite dimensionless rank coefficients, shape (24, 15).

        Returns:
            All month records, including explicit failures without substitute
            weights. Successful records include independently recomputed utility.
        """
        records = super().allocate(scores)
        for record in records:
            record.update(risk_aversion=self.risk_aversion, formulation="rank_risk_factored_norm_v1")
            if record["status"] != "solved":
                continue
            index = record["index"]
            weights = np.asarray(record["weights"])
            rank_score = float(scores[index] @ weights)
            normalized_variance = record["predicted_variance"] / record["variance_budget"]
            utility = (
                rank_score
                - self.risk_aversion * normalized_variance
                - self.concentration_penalty * float(weights @ weights)
            )
            reconstructed = record["scaled_objective"] / self.objective_scale + scores[index].mean()
            if not np.isclose(utility, reconstructed, rtol=0, atol=1e-8):
                raise ValueError("Solver objective and independently recomputed rank-risk utility disagree")
            record.update(
                weighted_rank_score=rank_score,
                normalized_predicted_variance=float(normalized_variance),
                risk_penalty_value=float(self.risk_aversion * normalized_variance),
                rank_risk_utility=float(utility),
            )
        return records


def allocate_rank_risk_checked(
    covariance: np.ndarray,
    budget: np.ndarray,
    minimum: int | None,
    penalty: float,
    scores: np.ndarray,
    *,
    risk_aversion: float,
) -> dict:
    """Preflight the unchanged policy before solving rank-risk allocations.

    Args:
        covariance: Shape (24, 15, 15), decimal-return squared, including noise.
        budget: Common training-only variance scale and ceiling, shape (24,).
        minimum: Effective asset floor; seven in this study.
        penalty: Original concentration tie-breaker, 1e-8 in this study.
        scores: Dimensionless coefficients, shape (24, 15).
        risk_aversion: Nonnegative continuous risk penalty.

    Returns:
        Preflight and all allocation records, or an explicit blocked policy.
        Failed or inaccurate allocations are never replaced or relaxed.
    """
    allocator = RankRiskAllocator(covariance, budget, minimum, penalty, risk_aversion)
    if scores.shape != (24, 15) or not np.isfinite(scores).all():
        raise ValueError("Complete finite scores required")
    checks = preflight(covariance, budget, minimum)
    if not all(row["status"].startswith("feasible_") for row in checks):
        return {"status": "blocked_preflight", "preflight": checks, "allocations": None}
    return {"status": "allocated", "preflight": checks, "allocations": allocator.allocate(scores)}
