from __future__ import annotations

import numpy as np
import pandas as pd
import riskfolio as rp

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


def optimize_from_historical_returns(
    returns: pd.DataFrame,
    request: OptimizeCommand,
) -> OptimizeResult:
    """Optimize portfolio weights from historical returns using Riskfolio.

    Args:
        returns: Historical periodic returns with columns as ticker symbols.
            Returns must be decimal units (0.02 = 2%).
        request: Optimization command with objective/risk settings.

    Returns:
        OptimizeResult with stable ``asset_order`` and decimal weights.

    Raises:
        ValueError: If returns are empty or contain fewer than 2 valid assets
            after cleaning.
    """

    if returns.empty:
        msg = "Historical returns are empty."
        raise ValueError(msg)

    if returns.shape[1] < 2:
        msg = "Historical returns must include at least 2 assets."
        raise ValueError(msg)

    clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    clean_returns = clean_returns.apply(pd.to_numeric, errors="coerce")
    clean_returns = clean_returns.dropna(axis=0, how="any")

    if clean_returns.shape[1] < 2:
        msg = "No valid asset columns remain after cleaning historical returns."
        raise ValueError(msg)

    asset_order = clean_returns.columns.tolist()
    weights_array: np.ndarray

    try:
        portfolio = rp.Portfolio(returns=clean_returns)
        portfolio.assets_stats(method_mu="hist", method_cov="hist")

        weights_df = portfolio.optimization(
            model="Classic",
            rm=request.risk_measure,
            obj=request.objective,
            rf=0,
            l=1,
            hist=request.hist,
        )

        if weights_df is None or weights_df.empty:
            raise RuntimeError("Riskfolio returned empty weights.")

        weights_array = np.ravel(weights_df.to_numpy(dtype=float))
    except Exception:
        weights_array = np.ones(len(asset_order), dtype=float)

    weights_array = np.clip(weights_array, a_min=0.0, a_max=None)
    total = float(weights_array.sum())
    if total <= 0:
        weights_array = np.ones(len(asset_order), dtype=float)
        total = float(weights_array.sum())

    normalized = weights_array / total
    return OptimizeResult(asset_order=asset_order, weights=normalized.astype(float).tolist())
