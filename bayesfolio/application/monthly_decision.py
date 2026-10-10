"""Application entry point for recording a reviewed monthly decision."""

from __future__ import annotations

from bayesfolio.contracts.commands.monthly_decision import FinalizeMonthlyPortfolioCommand
from bayesfolio.contracts.results.monthly_decision import MonthlyPortfolioDecisionResult
from bayesfolio.engine.automation.monthly_decision import finalize_monthly_portfolio as _finalize


def finalize_monthly_portfolio(
    command: FinalizeMonthlyPortfolioCommand,
) -> MonthlyPortfolioDecisionResult:
    """Record an explicit final decision for monthly candidate weights.

    Args:
        command: Candidate output, walk-forward evidence, and human review.

    Returns:
        Auditable approved or rejected portfolio decision.
    """

    return _finalize(command)
