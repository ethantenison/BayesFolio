from __future__ import annotations

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class OptimizeCommand(VersionedContract):
    """Command to run portfolio optimization."""

    schema: SchemaName = Field(default=SchemaName.OPTIMIZE_COMMAND, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
    objective: str
    risk_measure: str
    min_weight: float = 0.0
    max_weight: float = 0.35
    hist: bool = True
