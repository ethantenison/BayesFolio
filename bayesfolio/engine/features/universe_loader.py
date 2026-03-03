from __future__ import annotations

import pandas as pd

from bayesfolio.contracts.commands.universe import UniverseCommand
from bayesfolio.contracts.ui.universe import UniverseRecord


def build_universe_snapshot(returns: pd.DataFrame, request: UniverseCommand) -> UniverseRecord:
    """Create a stable universe snapshot from aligned returns data."""

    return UniverseRecord(
        asset_order=list(returns.columns),
        n_observations=int(len(returns)),
        return_unit=request.return_unit,
    )
