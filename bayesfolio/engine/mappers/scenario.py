from __future__ import annotations

from bayesfolio.contracts.commands.scenario import ScenarioCommand


def command_to_engine_inputs(command: ScenarioCommand) -> dict[str, object]:
    """Map a ScenarioCommand to keyword arguments for the scenario sampler.

    Args:
        command: The scenario command contract.

    Returns:
        Dict of engine keyword arguments (asset_order, n_scenarios, values, return_unit, seed).
    """
    return {
        "asset_order": command.asset_order,
        "n_scenarios": command.n_scenarios,
        "values": command.values,
        "return_unit": command.return_unit,
        "seed": command.seed,
    }


def engine_output_to_result(command: ScenarioCommand) -> ScenarioCommand:
    """Return the scenario command as the result (scenario panel IS the output).

    Args:
        command: The validated scenario command contract.

    Returns:
        The same ScenarioCommand, which contains the scenario panel data.
    """
    return command
