"""Result contracts for reviewed monthly portfolio decisions."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from bayesfolio.contracts.base import SchemaName, VersionedContract
from bayesfolio.contracts.results.backtest import WalkForwardBacktestResult
from bayesfolio.contracts.results.optimize import OptimizeResult


class MonthlyPortfolioDecisionResult(VersionedContract):
    """Auditable disposition of monthly candidate portfolio weights.

    Attributes:
        schema: Stable schema identifier.
        schema_version: Semantic version of this contract.
        strategy_id: Strategy identifier shared by candidate and backtest.
        candidate_run_label: Run label of the reviewed monthly candidate.
        as_of_date: Candidate portfolio decision date.
        decision: Explicit approval or rejection.
        reviewer: Human or committee identifier responsible for the decision.
        reviewed_at: Timestamp of the decision.
        rationale: Explanation recorded with the decision.
        backtest: Walk-forward evidence reviewed for the strategy.
        benchmarks: Baseline results reviewed on the same evaluation grid.
        candidate_weights: Candidate optimizer weights that were reviewed.
        finalized_weights: Approved weights, or ``None`` after rejection.
    """

    schema: Literal[SchemaName.MONTHLY_PORTFOLIO_DECISION_RESULT] = SchemaName.MONTHLY_PORTFOLIO_DECISION_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    strategy_id: str = Field(min_length=1)
    candidate_run_label: str = Field(min_length=1)
    as_of_date: date
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1)
    reviewed_at: AwareDatetime
    rationale: str = Field(min_length=1)
    backtest: WalkForwardBacktestResult
    benchmarks: list[WalkForwardBacktestResult]
    candidate_weights: OptimizeResult
    finalized_weights: OptimizeResult | None

    @model_validator(mode="after")
    def validate_decision_payload(self) -> "MonthlyPortfolioDecisionResult":
        """Keep approval state and exposed final weights consistent."""

        if self.decision == "approved":
            if not self.benchmarks:
                raise ValueError("approved decisions require at least one benchmark backtest")
            if self.finalized_weights != self.candidate_weights:
                raise ValueError("approved finalized_weights must equal candidate_weights")
        elif self.finalized_weights is not None:
            raise ValueError("rejected decisions cannot contain finalized_weights")
        return self
