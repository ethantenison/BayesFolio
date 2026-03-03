from __future__ import annotations

import numpy as np
import pandas as pd

from bayesfolio.contracts.commands.optimize import OptimizeCommand
from bayesfolio.contracts.commands.scenario import ScenarioCommand
from bayesfolio.contracts.results.optimize import OptimizeResult


def optimize_from_scenarios(
    scenarios: ScenarioCommand,
    request: OptimizeCommand,
) -> OptimizeResult:
    """Build simple normalized weights from scenario expected returns.

    This adapter preserves the schema-first boundary while allowing downstream
    replacement with full Riskfolio optimization.
    """

    panel = pd.DataFrame(scenarios.values, columns=scenarios.asset_order)
    mean_returns = panel.mean(axis=0).to_numpy(dtype=float)

    raw = np.clip(mean_returns, a_min=request.min_weight, a_max=request.max_weight)
    if float(raw.sum()) <= 0:
        raw = np.ones_like(raw, dtype=float)
    weights = raw / float(raw.sum())

    return OptimizeResult(
        asset_order=scenarios.asset_order,
        weights=weights.astype(float).tolist(),
    )
