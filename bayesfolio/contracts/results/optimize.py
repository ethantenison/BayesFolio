from __future__ import annotations

from pydantic import Field

from bayesfolio.contracts.base import SchemaName, VersionedContract


class OptimizeResult(VersionedContract):
    """Optimized portfolio weights in stable asset order.

    Attributes:
        asset_order: List of ticker symbols in the order weights correspond to.
        weights: Portfolio weights as decimals (0.02 = 2%). Sum should equal 1.0.
    """

    schema: SchemaName = Field(default=SchemaName.OPTIMIZE_RESULT, const=True)
    schema_version: str = Field(default="0.1.0", const=True)
    asset_order: list[str]
    weights: list[float]
