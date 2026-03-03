from __future__ import annotations

import pandas as pd

from bayesfolio.schemas.contracts.universe import UniverseRequest, UniverseSnapshot


def build_universe_snapshot(returns: pd.DataFrame, request: UniverseRequest) -> UniverseSnapshot:
    """Create a stable universe snapshot from aligned returns data."""

    return UniverseSnapshot(
        metadata=request.metadata,
        asset_order=list(returns.columns),
        n_observations=int(len(returns)),
        return_unit=request.return_unit,
    )
