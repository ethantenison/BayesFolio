from __future__ import annotations

import pandas as pd

from bayesfolio.contracts.commands.optimize import OptimizeCommand
from bayesfolio.contracts.results.features import FeaturesDatasetResult
from bayesfolio.contracts.results.report import ReportResult
from bayesfolio.engine.asset_allocation.riskfolio_adapter import optimize_from_scenarios
from bayesfolio.engine.backtest.runner import run_weighted_backtest
from bayesfolio.engine.forecast.forecast_adapter import build_forecast_payload
from bayesfolio.engine.report.assembler import assemble_report
from bayesfolio.engine.scenarios.sampler import sample_joint_scenarios


def run_schema_first_pipeline(
    asset_order: list[str],
    mean,
    covariance,
    realized_returns: pd.DataFrame,
    features_result: FeaturesDatasetResult | None = None,
    objective: str = "Sharpe",
    risk_measure: str = "CVaR",
    n_scenarios: int = 500,
    seed: int = 1,
) -> ReportResult:
    """Run minimal end-to-end schema-first pipeline from forecast to report.

    Args:
        asset_order: Ordered asset identifiers for forecast/scenario dimensions.
        mean: Forecast mean vector aligned to ``asset_order``.
        covariance: Forecast covariance matrix aligned to ``asset_order``.
        realized_returns: Realized returns used for backtest evaluation.
        features_result: Optional features dataset result. When provided,
            model-free market diagnostics are forwarded into report assembly.
        objective: Optimization objective passed to optimizer command.
        risk_measure: Risk measure passed to optimizer command.
        n_scenarios: Number of sampled scenarios for optimization.
        seed: Random seed for scenario sampling reproducibility.

    Returns:
        ReportResult with headline backtest metrics and optional
        ``market_structure`` diagnostics.

    Example:
        Default behavior (no feature diagnostics):
            ``run_schema_first_pipeline(..., realized_returns=returns_df)``

        With diagnostics from a prior feature build:
            ``run_schema_first_pipeline(..., realized_returns=returns_df, features_result=features_result)``
    """

    forecast = build_forecast_payload(asset_order, mean, covariance)
    scenarios = sample_joint_scenarios(forecast=forecast, n_scenarios=n_scenarios, seed=seed)
    opt_request = OptimizeCommand(objective=objective, risk_measure=risk_measure)
    optimization = optimize_from_scenarios(scenarios=scenarios, request=opt_request)
    backtest = run_weighted_backtest(realized_returns=realized_returns, optimization=optimization)
    return assemble_report(backtest_result=backtest, features_result=features_result)
