from __future__ import annotations

from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class ScenarioCommand(VersionedContract):
    """Command to generate scenario return panel from forecast.

    Attributes:
        asset_order: List of ticker symbols.
        n_scenarios: Number of Monte Carlo scenarios to generate. Must be >= 1.
        values: Scenario matrix; shape [n_scenarios, n_assets]; returns as decimals.
        return_unit: Unit of returns; always 'decimal'.
        seed: RNG seed for deterministic scenario generation.
    """

    schema: Literal[SchemaName.SCENARIO_COMMAND] = SchemaName.SCENARIO_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    asset_order: list[str]
    n_scenarios: int = Field(ge=1)
    values: list[list[float]]
    return_unit: str = Field(default="decimal")
    seed: int = 42
