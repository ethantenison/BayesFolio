from __future__ import annotations

from typing import Literal

from bayesfolio.contracts.base import SchemaName, VersionedContract


class OptimizeCommand(VersionedContract):
    """Command to run portfolio optimization."""

    schema: Literal[SchemaName.OPTIMIZE_COMMAND] = SchemaName.OPTIMIZE_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    objective: str
    risk_measure: str
    min_weight: float = 0.0
    max_weight: float = 0.35
    hist: bool = True
