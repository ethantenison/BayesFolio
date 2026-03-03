from __future__ import annotations

from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class ScenarioUiInput(VersionedContract):
    """UI input for scenario-based risk analysis.

    Attributes:
        n_scenarios: Number of scenarios to show.
        return_unit: Unit of returns; 'decimal' or 'percent_points'.
    """

    schema: Literal[SchemaName.SCENARIO_UI_INPUT] = SchemaName.SCENARIO_UI_INPUT
    schema_version: Literal["0.1.0"] = "0.1.0"
    n_scenarios: int = Field(default=1000, ge=1)
    return_unit: str = Field(default="decimal")
