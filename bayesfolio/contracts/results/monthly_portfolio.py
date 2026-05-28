"""Result contracts for monthly BayesFolio portfolio automation."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import ContractModel, VersionedContract
from bayesfolio.contracts.results.features import FeaturesDatasetResult
from bayesfolio.contracts.results.forecast import ForecastResult
from bayesfolio.contracts.results.gp_workflow import GPWorkflowResult
from bayesfolio.contracts.results.optimize import OptimizeResult
from bayesfolio.contracts.results.report import ArtifactPointer


class MonthlyPredictionRecord(ContractModel):
    """One asset-level prediction summary for the target month."""

    asset: str
    prediction: float
    uncertainty: float
    score: float
    eligible_for_final_portfolio: bool


class MonthlyRunArtifacts(ContractModel):
    """Pointers to persisted monthly automation artifacts."""

    feature_dataset: ArtifactPointer | None = None
    scenarios_csv: ArtifactPointer | None = None
    predictions_csv: ArtifactPointer | None = None
    weights_csv: ArtifactPointer | None = None
    summary_json: ArtifactPointer | None = None


class MonthlyPortfolioResult(VersionedContract):
    """Final result contract for end-to-end monthly automation."""

    schema: Literal["bayesfolio.monthly_portfolio.result"] = "bayesfolio.monthly_portfolio.result"
    schema_version: Literal["0.1.0"] = "0.1.0"
    as_of_date: date
    run_label: str
    training_universe: list[str]
    final_portfolio_universe: list[str]
    helper_assets_only: list[str] = Field(default_factory=list)
    features_result: FeaturesDatasetResult
    gp_workflow_result: GPWorkflowResult
    forecast_result: ForecastResult
    optimize_result: OptimizeResult
    top_predictions: list[MonthlyPredictionRecord] = Field(default_factory=list)
    artifacts: MonthlyRunArtifacts = Field(default_factory=MonthlyRunArtifacts)
    diagnostics: list[str] = Field(default_factory=list)
