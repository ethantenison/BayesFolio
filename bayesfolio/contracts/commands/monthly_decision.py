"""Contracts for explicit review of monthly candidate portfolio weights."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from bayesfolio.contracts.base import SchemaName, VersionedContract
from bayesfolio.contracts.results.backtest import WalkForwardBacktestResult
from bayesfolio.contracts.results.monthly_portfolio import MonthlyPortfolioResult


class FinalizeMonthlyPortfolioCommand(VersionedContract):
    """Record a human decision about a backtested monthly candidate.

    Attributes:
        schema: Stable schema identifier.
        schema_version: Semantic version of this contract.
        candidate: Monthly optimizer output under review.
        backtest: Walk-forward evidence for the candidate's strategy.
        benchmarks: Walk-forward baseline results on the same evaluation grid.
        decision: Explicit human approval or rejection.
        reviewer: Human or committee identifier responsible for the decision.
        reviewed_at: Timestamp of the decision.
        rationale: Non-empty explanation of the decision.
    """

    schema: Literal[SchemaName.MONTHLY_PORTFOLIO_FINALIZE_COMMAND] = SchemaName.MONTHLY_PORTFOLIO_FINALIZE_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    candidate: MonthlyPortfolioResult
    backtest: WalkForwardBacktestResult
    benchmarks: list[WalkForwardBacktestResult] = Field(default_factory=list)
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1)
    reviewed_at: AwareDatetime
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_review_timing(self) -> "FinalizeMonthlyPortfolioCommand":
        """Require the review to occur no earlier than the candidate date."""

        if self.reviewed_at.date() < self.candidate.as_of_date:
            raise ValueError("reviewed_at cannot precede the candidate as_of_date")
        if self.decision == "approved" and not self.benchmarks:
            raise ValueError("approved decisions require at least one benchmark backtest")
        return self
