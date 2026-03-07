from __future__ import annotations

from bayesfolio.contracts.commands.backtest import BacktestCommand
from bayesfolio.contracts.results.backtest import BacktestResult


def command_to_engine_inputs(command: BacktestCommand) -> dict[str, object]:
    """Map a BacktestCommand to keyword arguments for the backtest runner.

    Args:
        command: The backtest command contract.

    Returns:
        Dict of engine keyword arguments (optimize_result, start_date, end_date).
    """
    return {
        "optimize_result": command.optimize_result,
        "start_date": command.start_date,
        "end_date": command.end_date,
    }


def engine_output_to_result(
    cumulative_return: float,
    annualized_return: float,
    annualized_volatility: float,
    sharpe_ratio: float,
    max_drawdown: float,
    calmar_ratio: float,
    sortino_ratio: float,
) -> BacktestResult:
    """Map raw backtest runner output to a BacktestResult contract.

    Args:
        cumulative_return: Total return as decimal (0.10 = 10%).
        annualized_return: Annualized return as decimal (0.10 = 10%).
        annualized_volatility: Annualized volatility as decimal (0.15 = 15%).
        sharpe_ratio: Sharpe ratio (dimensionless).
        max_drawdown: Maximum drawdown as decimal (negative or zero).
        calmar_ratio: Calmar ratio (dimensionless).
        sortino_ratio: Sortino ratio (dimensionless).

    Returns:
        BacktestResult contract.
    """
    return BacktestResult(
        cumulative_return=cumulative_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        sortino_ratio=sortino_ratio,
    )
