from __future__ import annotations

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class BeliefsCommand(VersionedContract):
    """Structured prior beliefs over expected return, volatility, and correlation.

    Attributes:
        expected_return: Per-ticker expected return as decimal (0.02 = 2%).
        volatility: Per-ticker annualized volatility as decimal (0.15 = 15%).
        pairwise_correlation: Pairwise correlation matrix (dimensionless, [-1, 1]).
        return_unit: Unit of returns; always 'decimal'.
    """

    schema: SchemaName = Field(default=SchemaName.BELIEFS_COMMAND, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
    expected_return: dict[str, float] = Field(default_factory=dict)
    volatility: dict[str, float] = Field(default_factory=dict)
    pairwise_correlation: dict[str, dict[str, float]] = Field(default_factory=dict)
    return_unit: str = Field(default="decimal")
