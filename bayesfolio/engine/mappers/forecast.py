from __future__ import annotations

from bayesfolio.contracts.commands.forecast import ForecastCommand
from bayesfolio.contracts.results.forecast import ForecastResult


def command_to_engine_inputs(command: ForecastCommand) -> dict[str, object]:
    """Map a ForecastCommand to keyword arguments for the GP forecast engine.

    Args:
        command: The forecast command contract.

    Returns:
        Dict of engine keyword arguments (tickers, horizon_days, seed).
    """
    return {
        "tickers": command.tickers,
        "horizon_days": command.horizon_days,
        "seed": command.seed,
    }


def engine_output_to_result(
    asset_order: list[str],
    mean: list[float],
    covariance: list[list[float]],
) -> ForecastResult:
    """Map raw GP forecast engine output to a ForecastResult contract.

    Args:
        asset_order: Tickers in canonical order.
        mean: Predicted mean returns as decimals (0.02 = 2%).
        covariance: Predicted covariance matrix as decimals squared.

    Returns:
        ForecastResult contract.
    """
    return ForecastResult(asset_order=asset_order, mean=mean, covariance=covariance)
