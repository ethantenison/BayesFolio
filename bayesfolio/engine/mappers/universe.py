from __future__ import annotations

from bayesfolio.contracts.commands.universe import UniverseCommand
from bayesfolio.contracts.ui.universe import UniverseRecord


def command_to_engine_inputs(command: UniverseCommand) -> dict[str, object]:
    """Map a UniverseCommand to keyword arguments for the universe loader.

    Args:
        command: The universe command contract.

    Returns:
        Dict of engine keyword arguments (tickers, start_date, end_date, return_unit).
    """
    return {
        "tickers": command.tickers,
        "start_date": command.start_date,
        "end_date": command.end_date,
        "return_unit": command.return_unit,
    }


def engine_output_to_result(
    asset_order: list[str],
    n_observations: int,
    return_unit: str = "decimal",
) -> UniverseRecord:
    """Map raw universe loader output to a UniverseRecord contract.

    Args:
        asset_order: Tickers in canonical order.
        n_observations: Number of time observations.
        return_unit: Unit of returns; 'decimal' or 'percent_points'.

    Returns:
        UniverseRecord contract.
    """
    return UniverseRecord(
        asset_order=asset_order,
        n_observations=n_observations,
        return_unit=return_unit,
    )
