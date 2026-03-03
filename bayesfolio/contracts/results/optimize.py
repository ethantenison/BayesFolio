from __future__ import annotations

from typing import Literal

from bayesfolio.contracts.base import SchemaName, VersionedContract


class OptimizeResult(VersionedContract):
    """Optimized portfolio weights in stable asset order.

    Attributes:
        asset_order: List of ticker symbols in the order weights correspond to.
        weights: Portfolio weights as decimals (0.02 = 2%). Sum should equal 1.0.
    """

    schema: Literal[SchemaName.OPTIMIZE_RESULT] = SchemaName.OPTIMIZE_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    asset_order: list[str]
    weights: list[float]
