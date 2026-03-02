"""Schemas-first transport contracts for the BayesFolio pipeline."""

from bayesfolio.schemas.contracts.backtest import BacktestResult
from bayesfolio.schemas.contracts.beliefs import PriorBeliefs
from bayesfolio.schemas.contracts.forecast import ForecastPayload
from bayesfolio.schemas.contracts.intent import OptimizationIntent
from bayesfolio.schemas.contracts.optimize import OptimizationRequest, OptimizationResult
from bayesfolio.schemas.contracts.report import ReportBundle
from bayesfolio.schemas.contracts.scenarios import ScenarioPanel
from bayesfolio.schemas.contracts.universe import UniverseRequest, UniverseSnapshot

__all__ = [
    "BacktestResult",
    "ForecastPayload",
    "OptimizationIntent",
    "OptimizationRequest",
    "OptimizationResult",
    "PriorBeliefs",
    "ReportBundle",
    "ScenarioPanel",
    "UniverseRequest",
    "UniverseSnapshot",
]
