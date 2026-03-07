from __future__ import annotations

from bayesfolio.contracts.commands.optimize import OptimizeCommand
from bayesfolio.contracts.results.optimize import OptimizeResult


def command_to_engine_inputs(command: OptimizeCommand) -> dict[str, object]:
    """Map an OptimizeCommand to keyword arguments for the engine optimizer.

    Args:
        command: The optimization command contract.

    Returns:
        Dict of engine keyword arguments
        (objective, risk_measure, min_weight, max_weight, nea, hist).
    """
    return {
        "objective": command.objective,
        "risk_measure": command.risk_measure,
        "min_weight": command.min_weight,
        "max_weight": command.max_weight,
        "nea": command.nea,
        "hist": command.hist,
    }


def engine_output_to_result(asset_order: list[str], weights: list[float]) -> OptimizeResult:
    """Map raw engine optimizer output to an OptimizeResult contract.

    Args:
        asset_order: Tickers in canonical order.
        weights: Portfolio weights as decimals (0.02 = 2%). Must sum to 1.0.

    Returns:
        OptimizeResult contract.
    """
    return OptimizeResult(asset_order=asset_order, weights=weights)
