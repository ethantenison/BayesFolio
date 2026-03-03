from __future__ import annotations

import pandas as pd

from bayesfolio.engine.backtest.runner import run_weighted_backtest
from bayesfolio.engine.forecast.forecast_adapter import build_forecast_payload
from bayesfolio.engine.asset_allocation.riskfolio_adapter import optimize_from_scenarios
from bayesfolio.engine.report.assembler import assemble_report
from bayesfolio.engine.scenarios.sampler import sample_joint_scenarios
from bayesfolio.schemas.common import SchemaMetadata
from bayesfolio.schemas.contracts.optimize import OptimizationRequest
from bayesfolio.schemas.contracts.report import ReportBundle


def run_schema_first_pipeline(
    asset_order: list[str],
    mean,
    covariance,
    realized_returns: pd.DataFrame,
    objective: str = "Sharpe",
    risk_measure: str = "CVaR",
    n_scenarios: int = 500,
    seed: int = 1,
) -> ReportBundle:
    """Run minimal end-to-end schema-first pipeline from forecast to report."""

    metadata = SchemaMetadata()
    forecast = build_forecast_payload(asset_order, mean, covariance, metadata=metadata)
    scenarios = sample_joint_scenarios(forecast=forecast, n_scenarios=n_scenarios, seed=seed)
    opt_request = OptimizationRequest(metadata=metadata, objective=objective, risk_measure=risk_measure)
    optimization = optimize_from_scenarios(scenarios=scenarios, request=opt_request)
    backtest = run_weighted_backtest(realized_returns=realized_returns, optimization=optimization)
    return assemble_report(backtest)
