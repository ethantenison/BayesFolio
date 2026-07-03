"""Contracts for monthly BayesFolio portfolio automation.

These schemas define the config boundary for end-to-end monthly portfolio
execution. They are intended to support deterministic automation, artifact
logging, and scheduled execution.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from bayesfolio.contracts.base import VersionedContract
from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.core.settings import RiskfolioConfig


class MonthlyForecastConfig(VersionedContract):
    """Configuration for the monthly GP forecast stage.

    Attributes:
        schema: Contract schema identifier.
        schema_version: Semantic schema version.
        target_column: Forecast target column in decimal return units.
        task_column: Asset identifier column.
        time_feature_columns: Ordered time feature columns.
        etf_feature_columns: Ordered ETF feature columns.
        macro_feature_columns: Ordered macro feature columns.
        rank: Multitask rank passed into the GP builder.
        min_inferred_noise_level: Initial likelihood noise floor.
        seed: Optional deterministic seed.
        posterior_scenario_count: Number of posterior scenarios to sample.
    """

    schema: Literal["bayesfolio.monthly_portfolio.forecast_config"] = "bayesfolio.monthly_portfolio.forecast_config"
    schema_version: Literal["0.1.0"] = "0.1.0"
    target_column: str = "y_excess_lead"
    task_column: str = "asset_id"
    time_feature_columns: list[str] = Field(default_factory=lambda: ["t_index"])
    etf_feature_columns: list[str] = Field(
        default_factory=lambda: [
            "lag_y_excess_lead",
            "baspread",
            "ret_kurt",
            "chmom",
            "mom12m",
            "mom36m",
            "cs_mom_rank",
            "max_dd_6m",
            "ma_signal",
            "ret_autocorr",
            "vol_z",
        ]
    )
    macro_feature_columns: list[str] = Field(
        default_factory=lambda: [
            "hy_spread",
            "hy_spread_chg_1p",
            "hy_spread_z_12p",
            "vix_slope",
            "vix_ts_z_12p",
            "vix",
            "spy_flow_z_12p",
            "spy_ret",
            "erp",
            "cpi_chg_12p",
            "cpi_chg_1p",
            "copper_ret",
            "oil_ret",
            "gold_crude_ratio",
            "pct_above_50dma",
            "em_fx_ret",
        ]
    )
    rank: int | None = Field(default=5, ge=1)
    min_inferred_noise_level: float = Field(default=5e-3, gt=0.0)
    seed: int | None = None
    posterior_scenario_count: int = Field(default=5000, ge=1)

    @property
    def allowed_feature_columns(self) -> list[str]:
        """Return the exact feature order used by the deterministic GP builder."""

        return [*self.time_feature_columns, *self.etf_feature_columns, *self.macro_feature_columns]


class MonthlyPortfolioArtifactsConfig(VersionedContract):
    """Artifact naming and storage configuration for monthly runs.

    Attributes:
        schema: Contract schema identifier.
        schema_version: Semantic schema version.
        artifact_root: Root directory for persisted outputs.
        run_label: Optional stable run label.
        save_predictions_csv: Whether to persist ranking predictions.
        save_scenarios_csv: Whether to persist posterior scenarios.
        save_weights_csv: Whether to persist final weights.
        save_summary_json: Whether to persist the final contract result.
        mlflow_tracking_uri: Optional MLflow tracking URI.
        mlflow_experiment_name: Optional MLflow experiment name.
    """

    schema: Literal["bayesfolio.monthly_portfolio.artifacts_config"] = "bayesfolio.monthly_portfolio.artifacts_config"
    schema_version: Literal["0.1.0"] = "0.1.0"
    artifact_root: str = "artifacts/monthly_portfolios"
    run_label: str | None = None
    save_predictions_csv: bool = True
    save_scenarios_csv: bool = True
    save_weights_csv: bool = True
    save_summary_json: bool = True
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None


class MonthlyPortfolioCommand(VersionedContract):
    """Top-level command contract for a monthly portfolio automation run.

    Attributes:
        schema: Contract schema identifier.
        schema_version: Semantic schema version.
        as_of_date: Effective month-end date for the run.
        training_universe: Universe used to build features and fit the GP.
        final_portfolio_universe: Optional investable subset for final weights.
        helper_assets_only: Assets allowed in training but excluded from final weights.
        feature_dataset: Feature dataset build command.
        forecast: Monthly forecast-stage configuration.
        riskfolio: Riskfolio optimization configuration.
        artifacts: Artifact/logging configuration.
    """

    schema: Literal["bayesfolio.monthly_portfolio.command"] = "bayesfolio.monthly_portfolio.command"
    schema_version: Literal["0.1.0"] = "0.1.0"
    as_of_date: date
    training_universe: list[str] = Field(min_length=2)
    final_portfolio_universe: list[str] | None = None
    helper_assets_only: list[str] = Field(default_factory=list)
    feature_dataset: BuildFeaturesDatasetCommand
    forecast: MonthlyForecastConfig
    riskfolio: RiskfolioConfig
    artifacts: MonthlyPortfolioArtifactsConfig = Field(default_factory=MonthlyPortfolioArtifactsConfig)

    @model_validator(mode="after")
    def validate_universe_relationships(self) -> "MonthlyPortfolioCommand":
        """Ensure training/final universes and helper assets are consistent."""

        training_set = set(self.training_universe)
        if set(self.helper_assets_only) - training_set:
            raise ValueError("helper_assets_only must be a subset of training_universe")

        final_universe = self.final_portfolio_universe
        if final_universe is None:
            final_universe = [ticker for ticker in self.training_universe if ticker not in set(self.helper_assets_only)]
            self.final_portfolio_universe = final_universe

        final_set = set(final_universe)
        if final_set - training_set:
            raise ValueError("final_portfolio_universe must be a subset of training_universe")
        if final_set & set(self.helper_assets_only):
            raise ValueError("helper_assets_only cannot appear in final_portfolio_universe")
        if len(final_universe) < 2:
            raise ValueError("final_portfolio_universe must contain at least 2 assets")
        return self
