from __future__ import annotations

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class ScenarioUiInput(VersionedContract):
    """UI input for scenario-based risk analysis.

    Attributes:
        n_scenarios: Number of scenarios to show.
        return_unit: Unit of returns; 'decimal' or 'percent_points'.
    """

    schema: SchemaName = Field(default=SchemaName.SCENARIO_UI_INPUT, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
    n_scenarios: int = Field(default=1000, ge=1)
    return_unit: str = Field(default="decimal")
