"""Pure review gate between monthly candidate weights and final weights."""

from __future__ import annotations

from bayesfolio.contracts.commands.monthly_decision import FinalizeMonthlyPortfolioCommand
from bayesfolio.contracts.results.monthly_decision import MonthlyPortfolioDecisionResult


def finalize_monthly_portfolio(
    command: FinalizeMonthlyPortfolioCommand,
) -> MonthlyPortfolioDecisionResult:
    """Finalize or reject candidate weights using aligned review evidence.

    This function does not decide whether a strategy is financially suitable.
    It enforces that the candidate and backtest identify the same strategy, that
    evaluation evidence was observable strictly before the candidate's as-of
    date, that benchmarks use the same grid, and that only an explicit human
    approval releases finalized weights.

    Args:
        command: Candidate, walk-forward evidence, and explicit human decision.

    Returns:
        Auditable decision result. Rejections contain no finalized weights.

    Raises:
        ValueError: If strategy identities differ, benchmark grids do not match,
            or any evidence is realized on or after the candidate as-of date.
    """

    candidate = command.candidate
    backtest = command.backtest
    if candidate.strategy_id != backtest.strategy_id:
        raise ValueError(
            "Candidate and backtest strategy_id values must match; "
            f"candidate={candidate.strategy_id!r}, backtest={backtest.strategy_id!r}."
        )

    evaluation_grid = [(period.decision_date, period.realization_date) for period in backtest.periods]
    benchmark_ids: set[str] = set()
    for benchmark in command.benchmarks:
        if benchmark.strategy_id == candidate.strategy_id:
            raise ValueError("Benchmark strategy_id must differ from the candidate strategy_id.")
        if benchmark.strategy_id in benchmark_ids:
            raise ValueError(f"Duplicate benchmark strategy_id: {benchmark.strategy_id!r}.")
        benchmark_grid = [(period.decision_date, period.realization_date) for period in benchmark.periods]
        if benchmark_grid != evaluation_grid:
            raise ValueError("Candidate and benchmark backtests must use the same evaluation grid.")
        benchmark_ids.add(benchmark.strategy_id)

    all_periods = [*backtest.periods]
    for benchmark in command.benchmarks:
        all_periods.extend(benchmark.periods)
    latest_realization_date = max(period.realization_date for period in all_periods)
    if latest_realization_date >= candidate.as_of_date:
        raise ValueError(
            "Backtest evidence must be observable strictly before the candidate as_of_date; "
            f"latest_realization_date={latest_realization_date.isoformat()}, "
            f"as_of_date={candidate.as_of_date.isoformat()}."
        )

    finalized_weights = candidate.optimize_result if command.decision == "approved" else None
    return MonthlyPortfolioDecisionResult(
        strategy_id=candidate.strategy_id,
        candidate_run_label=candidate.run_label,
        as_of_date=candidate.as_of_date,
        decision=command.decision,
        reviewer=command.reviewer,
        reviewed_at=command.reviewed_at,
        rationale=command.rationale,
        backtest=backtest,
        benchmarks=command.benchmarks,
        candidate_weights=candidate.optimize_result,
        finalized_weights=finalized_weights,
    )
