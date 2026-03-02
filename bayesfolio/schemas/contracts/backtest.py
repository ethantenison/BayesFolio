from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from bayesfolio.schemas.common import SchemaMetadata


class BacktestResult(BaseModel):
    """Backtest summary transport object with key portfolio metrics."""

    metadata: SchemaMetadata = Field(default_factory=SchemaMetadata)
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float

    model_config = ConfigDict(extra="forbid")
