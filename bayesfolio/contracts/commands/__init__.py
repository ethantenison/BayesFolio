"""Command contracts for engine execution inputs.

This package exports versioned command schemas consumed at pipeline boundaries.
"""

from bayesfolio.contracts.commands.backtest import WalkForwardBacktestCommand, WalkForwardPeriod
from bayesfolio.contracts.commands.brokerage import BuildRebalancePlanCommand
from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.contracts.commands.gp_planner import GPPlannerRequest
from bayesfolio.contracts.commands.monthly_decision import FinalizeMonthlyPortfolioCommand
from bayesfolio.contracts.commands.monthly_portfolio import MonthlyPortfolioCommand
from bayesfolio.contracts.commands.performance import PerformancePeriod, TrackPortfolioPerformanceCommand
from bayesfolio.contracts.commands.portfolio_adjustment import (
    AdjustmentModelConfig,
    AssetForecastEvidence,
    GeneratePortfolioAdjustmentCommand,
    PortfolioAdjustmentPolicy,
    PortfolioRiskEvidence,
    ResearchEvidence,
    RetrievalRecord,
)

__all__ = [
    "BuildFeaturesDatasetCommand",
    "BuildRebalancePlanCommand",
    "AdjustmentModelConfig",
    "AssetForecastEvidence",
    "FinalizeMonthlyPortfolioCommand",
    "GeneratePortfolioAdjustmentCommand",
    "GPPlannerRequest",
    "MonthlyPortfolioCommand",
    "PerformancePeriod",
    "PortfolioAdjustmentPolicy",
    "PortfolioRiskEvidence",
    "ResearchEvidence",
    "RetrievalRecord",
    "TrackPortfolioPerformanceCommand",
    "WalkForwardBacktestCommand",
    "WalkForwardPeriod",
]
