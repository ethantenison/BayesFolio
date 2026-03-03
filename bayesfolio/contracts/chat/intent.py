from __future__ import annotations

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class ParsedIntent(VersionedContract):
    """Parsed user intent extracted from natural language.

    Attributes:
        objective: Portfolio optimization objective (e.g. 'Sharpe').
        risk_measure: Risk measure (e.g. 'CVaR').
        long_only: Whether to restrict to long-only positions.
        min_weight: Minimum position weight as decimal.
        max_weight: Maximum position weight as decimal.
    """

    schema: SchemaName = Field(default=SchemaName.CHAT_INTENT_PARSED, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
    objective: str = Field(default="Sharpe")
    risk_measure: str = Field(default="CVaR")
    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float = 0.35
