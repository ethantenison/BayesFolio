from __future__ import annotations

from typing import Literal

from bayesfolio.contracts.base import SchemaName, VersionedContract


class OptimizeCommand(VersionedContract):
    """Command to run portfolio optimization.

    Attributes:
        objective: Optimization objective (for example ``Sharpe``).
        risk_measure: Risk measure identifier (for example ``CVaR``).
        model: Riskfolio model identifier (for example ``Classic``).
        rf: Risk-free rate in decimal units for the same return period.
        kelly: Optional Kelly mode (``approx`` or ``exact``).
        min_weight: Minimum long-only asset weight as decimal.
        max_weight: Maximum long-only asset weight as decimal (Riskfolio ``upperlng`` equivalent).
        nea: Target number of assets (Riskfolio cardinality helper parameter).
        hist: Whether to use historical estimation mode in Riskfolio.
    """

    schema: Literal[SchemaName.OPTIMIZE_COMMAND] = SchemaName.OPTIMIZE_COMMAND
    schema_version: Literal["0.1.0"] = "0.1.0"
    objective: str
    risk_measure: str
    model: str = "Classic"
    rf: float = 0.0
    kelly: str | None = None
    min_weight: float = 0.0
    max_weight: float = 0.35
    nea: int = 6
    hist: bool = True
