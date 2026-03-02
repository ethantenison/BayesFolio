from __future__ import annotations

import numpy as np
import pandas as pd

from bayesfolio.schemas.common import SchemaMetadata
from bayesfolio.schemas.contracts.optimize import OptimizationRequest, OptimizationResult
from bayesfolio.schemas.contracts.scenarios import ScenarioPanel


def optimize_from_scenarios(
    scenarios: ScenarioPanel,
    request: OptimizationRequest,
) -> OptimizationResult:
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

    return OptimizationResult(
        metadata=SchemaMetadata(**request.metadata.model_dump()),
        asset_order=scenarios.asset_order,
        weights=weights.astype(float).tolist(),
    )
