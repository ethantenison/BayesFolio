"""End-to-end monthly portfolio automation workflow.

This module turns the current notebook-style monthly workflow into a reusable,
config-driven engine function with structured artifacts and reproducibility
logging.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import StratifiedStandardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from bayesfolio.contracts.commands.monthly_portfolio import MonthlyPortfolioCommand
from bayesfolio.contracts.commands.optimize import OptimizeCommand
from bayesfolio.contracts.results.forecast import ForecastResult
from bayesfolio.contracts.results.gp_workflow import (
    GPFitValidationSummary,
    GPPlannerResponse,
    GPWorkflowResult,
    NormalizationStats,
    ResolvedFeatureBlock,
)
from bayesfolio.contracts.results.monthly_portfolio import (
    MonthlyPortfolioResult,
    MonthlyPredictionRecord,
    MonthlyRunArtifacts,
)
from bayesfolio.contracts.results.report import ArtifactPointer
from bayesfolio.engine.asset_allocation.riskfolio_adapter import optimize_from_historical_returns
from bayesfolio.engine.automation.monthly_gp_defaults import build_default_monthly_gp_configs
from bayesfolio.engine.features import build_features_dataset, make_default_feature_providers
from bayesfolio.engine.features.gp_data_prep import prepare_multitask_gp_data_with_task_feature
from bayesfolio.engine.forecast.gp.multitask_builder import build_multitask_gp
from bayesfolio.io.artifact_store import ParquetArtifactStore
from bayesfolio.io.fingerprints import sha256_digest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonthlyPortfolioRuntimeDependencies:
    """Dependency bundle for monthly automation runtime wiring."""

    feature_providers: object
    artifact_store: ParquetArtifactStore


@dataclass(frozen=True)
class FixedMonthlyGPArtifacts:
    """Artifacts produced by the deterministic monthly GP fit."""

    result: GPWorkflowResult
    model: object
    likelihood: object
    normalized_train_x: torch.Tensor
    train_y: torch.Tensor
    task_map: dict[str, int]
    outcome_transform: StratifiedStandardize


def build_default_dependencies(command: MonthlyPortfolioCommand) -> MonthlyPortfolioRuntimeDependencies:
    """Build default providers and artifact store for monthly automation."""

    return MonthlyPortfolioRuntimeDependencies(
        feature_providers=make_default_feature_providers(cache_root="artifacts/cache"),
        artifact_store=ParquetArtifactStore(base_dir=command.artifacts.artifact_root),
    )


def run_monthly_portfolio(
    command: MonthlyPortfolioCommand,
    *,
    dependencies: MonthlyPortfolioRuntimeDependencies | None = None,
) -> MonthlyPortfolioResult:
    """Execute the end-to-end monthly portfolio workflow."""

    runtime = dependencies or build_default_dependencies(command)
    run_label = command.artifacts.run_label or f"monthly_portfolio_{command.as_of_date.isoformat()}"
    logger.info("Starting monthly portfolio run %s for %s", run_label, command.as_of_date.isoformat())

    diagnostics: list[str] = []
    persisted_artifacts = MonthlyRunArtifacts()

    if command.artifacts.mlflow_tracking_uri:
        mlflow.set_tracking_uri(command.artifacts.mlflow_tracking_uri)
    if command.artifacts.mlflow_experiment_name:
        mlflow.set_experiment(command.artifacts.mlflow_experiment_name)

    with mlflow.start_run(run_name=run_label):
        mlflow.log_dict(command.model_dump(mode="json"), "contracts/monthly_portfolio_command.json")
        git_sha = _get_git_sha()
        if git_sha is not None:
            mlflow.log_param("git_sha", git_sha)

        features_result = build_features_dataset(
            command=command.feature_dataset,
            providers=runtime.feature_providers,
            artifact_store=runtime.artifact_store,
        )
        persisted_artifacts.feature_dataset = _feature_artifact_to_report_pointer(features_result.artifact)
        mlflow.log_dict(features_result.model_dump(mode="json"), "contracts/features_result.json")

        features_df = pd.read_parquet(features_result.artifact.uri)
        feature_columns = command.forecast.allowed_feature_columns
        selected_columns = [*feature_columns, command.forecast.target_column, command.forecast.task_column]
        training_df = features_df[selected_columns].dropna(subset=[command.forecast.target_column]).copy()
        scoring_df = features_df[selected_columns].copy()

        gp_artifacts = _run_fixed_monthly_gp(training_df=training_df, command=command)
        mlflow.log_dict(gp_artifacts.result.model_dump(mode="json"), "contracts/gp_workflow_result.json")

        scenario_frame, prediction_records, forecast_result = _build_monthly_forecasts(
            scoring_df=scoring_df,
            gp_artifacts=gp_artifacts,
            scenario_count=command.forecast.posterior_scenario_count,
            target_column=command.forecast.target_column,
            task_column=command.forecast.task_column,
            final_portfolio_universe=command.final_portfolio_universe or [],
            helper_assets_only=command.helper_assets_only,
        )
        mlflow.log_dict(forecast_result.model_dump(mode="json"), "contracts/forecast_result.json")

        final_scenarios = scenario_frame.loc[:, command.final_portfolio_universe]
        optimize_result = optimize_from_historical_returns(
            final_scenarios,
            request=OptimizeCommand(
                objective=command.riskfolio.obj,
                risk_measure=command.riskfolio.rm,
                model=command.riskfolio.model,
                rf=command.riskfolio.rf,
                max_weight=command.riskfolio.upperlng,
                nea=command.riskfolio.nea,
                hist=command.riskfolio.hist,
            ),
        )
        mlflow.log_dict(optimize_result.model_dump(mode="json"), "contracts/optimize_result.json")

        output_dir = Path(command.artifacts.artifact_root) / run_label
        output_dir.mkdir(parents=True, exist_ok=True)

        if command.artifacts.save_scenarios_csv:
            scenarios_path = output_dir / f"{run_label}_posterior_scenarios.csv"
            scenario_frame.to_csv(scenarios_path, index=False)
            persisted_artifacts.scenarios_csv = _local_file_to_artifact_pointer(scenarios_path, "csv")
        if command.artifacts.save_predictions_csv:
            predictions_path = output_dir / f"{run_label}_predictions.csv"
            predictions_frame = pd.DataFrame([record.model_dump(mode="json") for record in prediction_records])
            predictions_frame.to_csv(predictions_path, index=False)
            persisted_artifacts.predictions_csv = _local_file_to_artifact_pointer(predictions_path, "csv")
        if command.artifacts.save_weights_csv:
            weights_path = output_dir / f"{run_label}_weights.csv"
            weights_frame = pd.DataFrame({"asset": optimize_result.asset_order, "weight": optimize_result.weights})
            weights_frame.to_csv(weights_path, index=False)
            persisted_artifacts.weights_csv = _local_file_to_artifact_pointer(weights_path, "csv")

        result = MonthlyPortfolioResult(
            as_of_date=command.as_of_date,
            run_label=run_label,
            training_universe=command.training_universe,
            final_portfolio_universe=command.final_portfolio_universe or [],
            helper_assets_only=command.helper_assets_only,
            features_result=features_result,
            gp_workflow_result=gp_artifacts.result,
            forecast_result=forecast_result,
            optimize_result=optimize_result,
            top_predictions=prediction_records,
            artifacts=persisted_artifacts,
            diagnostics=diagnostics,
        )

        if command.artifacts.save_summary_json:
            summary_path = output_dir / f"{run_label}_summary.json"
            summary_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
            persisted_artifacts.summary_json = _local_file_to_artifact_pointer(summary_path, "json")
            result = result.model_copy(update={"artifacts": persisted_artifacts})

        mlflow.log_dict(result.model_dump(mode="json"), "contracts/monthly_portfolio_result.json")
        logger.info("Completed monthly portfolio run %s", run_label)
        return result


def _run_fixed_monthly_gp(*, training_df: pd.DataFrame, command: MonthlyPortfolioCommand) -> FixedMonthlyGPArtifacts:
    feature_columns = command.forecast.allowed_feature_columns
    feature_index = {name: idx for idx, name in enumerate(feature_columns)}
    covar_config, mean_config = build_default_monthly_gp_configs(
        forecast_config=command.forecast,
        feature_index=feature_index,
    )

    train_x_raw, train_y, task_map = prepare_multitask_gp_data_with_task_feature(
        training_df,
        target_col=command.forecast.target_column,
        asset_col=command.forecast.task_column,
        drop_cols=[],
        dtype=torch.float32,
    )
    train_x, normalization = _normalize_train_x(train_x_raw, feature_columns)

    task_feature_idx = train_x.shape[-1] - 1
    all_task_values = train_x[:, task_feature_idx].to(torch.long).unique(sorted=True)
    outcome_transform = StratifiedStandardize(
        stratification_idx=task_feature_idx,
        all_task_values=all_task_values,
        batch_shape=train_y.shape[:-2],
    )

    if command.forecast.seed is not None:
        torch.manual_seed(command.forecast.seed)

    model = build_multitask_gp(
        train_X=train_x,
        train_Y=train_y,
        task_feature=-1,
        covar_config=covar_config,
        mean_config=mean_config,
        rank=command.forecast.rank,
        min_inferred_noise_level=command.forecast.min_inferred_noise_level,
        outcome_transform=outcome_transform,
        input_transform=None,
    )

    model.train()
    likelihood = model.likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_mll(mll)

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        posterior = model.posterior(train_x)
        if posterior.mean.shape[-2] != train_x.shape[0]:
            raise RuntimeError("Posterior mean row count does not match training rows")

    resolved_blocks = [
        ResolvedFeatureBlock(
            name="time",
            variable_type="time",
            feature_names=command.forecast.time_feature_columns,
            dims=[feature_index[column] for column in command.forecast.time_feature_columns],
        ),
        ResolvedFeatureBlock(
            name="etf",
            variable_type="etf",
            feature_names=command.forecast.etf_feature_columns,
            dims=[feature_index[column] for column in command.forecast.etf_feature_columns],
        ),
        ResolvedFeatureBlock(
            name="macro",
            variable_type="macro",
            feature_names=command.forecast.macro_feature_columns,
            dims=[feature_index[column] for column in command.forecast.macro_feature_columns],
        ),
    ]

    result = GPWorkflowResult(
        planner_client_status="deterministic_default",
        planner_response=GPPlannerResponse(
            planner_status="ok",
            instruction_mode="explicit_constraints",
            dataset_assumptions=[
                "Monthly automation uses the fixed April/May 2026 multitask GP architecture by default."
            ],
            selected_design=None,
        ),
        target_column=command.forecast.target_column,
        task_column=command.forecast.task_column,
        feature_columns=feature_columns,
        resolved_blocks=resolved_blocks,
        mean_config=mean_config.model_dump(mode="json"),
        covar_config=covar_config.model_dump(mode="json"),
        normalization=normalization,
        fit_validation=GPFitValidationSummary(
            build_success=True,
            fit_success=True,
            prediction_success=True,
            attempt_count=1,
            min_inferred_noise_level=command.forecast.min_inferred_noise_level,
        ),
        final_status="ok",
        diagnostics=[f"Fixed monthly GP architecture applied with rank={command.forecast.rank}."],
    )

    return FixedMonthlyGPArtifacts(
        result=result,
        model=model,
        likelihood=likelihood,
        normalized_train_x=train_x,
        train_y=train_y,
        task_map=task_map,
        outcome_transform=outcome_transform,
    )


def _build_monthly_forecasts(
    *,
    scoring_df: pd.DataFrame,
    gp_artifacts,
    scenario_count: int,
    target_column: str,
    task_column: str,
    final_portfolio_universe: list[str],
    helper_assets_only: list[str],
) -> tuple[pd.DataFrame, list[MonthlyPredictionRecord], ForecastResult]:
    model = gp_artifacts.model
    normalized_x, asset_cols = _prepare_scoring_tensor(
        scoring_df=scoring_df,
        gp_artifacts=gp_artifacts,
        target_column=target_column,
        task_column=task_column,
    )

    with torch.no_grad():
        posterior = model.posterior(normalized_x, observation_noise=True)
        pred_mean = posterior.mean.squeeze(-1).detach().cpu()
        pred_var = posterior.variance.squeeze(-1).clamp_min(0.0).detach().cpu()
        pred_std = pred_var.sqrt()
        scenario_samples = posterior.rsample(torch.Size([scenario_count])).squeeze(-1).detach().cpu()

    scenario_frame = pd.DataFrame(scenario_samples.numpy(), columns=asset_cols)
    prediction_frame = pd.DataFrame(
        {
            "asset": asset_cols,
            "prediction": pred_mean.numpy(),
            "uncertainty": pred_std.numpy(),
        }
    )
    prediction_frame["score"] = prediction_frame["prediction"] / prediction_frame["uncertainty"].replace(0.0, pd.NA)
    prediction_frame["score"] = prediction_frame["score"].fillna(0.0)
    eligible_assets = set(final_portfolio_universe)
    helper_assets = set(helper_assets_only)
    prediction_frame["eligible_for_final_portfolio"] = prediction_frame["asset"].isin(
        eligible_assets
    ) & ~prediction_frame["asset"].isin(helper_assets)
    prediction_frame = prediction_frame.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)

    prediction_records = [MonthlyPredictionRecord(**record) for record in prediction_frame.to_dict(orient="records")]
    scenario_for_forecast = scenario_frame.loc[:, final_portfolio_universe]
    forecast_result = ForecastResult(
        asset_order=final_portfolio_universe,
        mean=scenario_for_forecast.mean(axis=0).astype(float).tolist(),
        covariance=scenario_for_forecast.cov().astype(float).values.tolist(),
    )
    return scenario_frame, prediction_records, forecast_result


def _prepare_scoring_tensor(
    *,
    scoring_df: pd.DataFrame,
    gp_artifacts: FixedMonthlyGPArtifacts,
    target_column: str,
    task_column: str,
) -> tuple[torch.Tensor, list[str]]:
    feature_columns = gp_artifacts.result.feature_columns
    frame = scoring_df[[*feature_columns, target_column, task_column]].copy()
    latest_rows = frame.groupby(task_column, sort=False).tail(1).reset_index(drop=True)
    task_map = gp_artifacts.task_map
    row_task_ids = [task_map[asset] for asset in latest_rows[task_column].tolist()]

    x = torch.zeros((len(latest_rows), len(feature_columns) + 1), dtype=gp_artifacts.normalized_train_x.dtype)
    for idx, feature in enumerate(feature_columns):
        min_value = gp_artifacts.result.normalization.mins[feature]
        range_value = gp_artifacts.result.normalization.ranges[feature]
        normalized_values = ((latest_rows[feature] - min_value) / range_value).astype(float).tolist()
        x[:, idx] = torch.tensor(normalized_values, dtype=x.dtype)
    x[:, -1] = torch.tensor(row_task_ids, dtype=x.dtype)
    asset_cols = latest_rows[task_column].tolist()
    return x, asset_cols


def _normalize_train_x(train_x: torch.Tensor, feature_columns: list[str]) -> tuple[torch.Tensor, NormalizationStats]:
    normalized = train_x.clone()
    non_task_indices = list(range(len(feature_columns)))
    mins = train_x[:, non_task_indices].amin(dim=0)
    maxs = train_x[:, non_task_indices].amax(dim=0)
    ranges = (maxs - mins).clamp_min(1e-12)
    normalized[:, non_task_indices] = (train_x[:, non_task_indices] - mins) / ranges
    normalization = NormalizationStats(
        feature_names=feature_columns,
        mins={name: float(value) for name, value in zip(feature_columns, mins.tolist(), strict=True)},
        maxs={name: float(value) for name, value in zip(feature_columns, maxs.tolist(), strict=True)},
        ranges={name: float(value) for name, value in zip(feature_columns, ranges.tolist(), strict=True)},
    )
    return normalized, normalization


def _feature_artifact_to_report_pointer(artifact) -> ArtifactPointer:
    return ArtifactPointer(
        path=artifact.uri,
        artifact_format=artifact.format,
        digest=artifact.fingerprint,
        byte_size=0,
    )


def _local_file_to_artifact_pointer(path: Path, artifact_format: str) -> ArtifactPointer:
    data = path.read_bytes()
    return ArtifactPointer(
        path=str(path),
        artifact_format=artifact_format,
        digest=sha256_digest(data),
        byte_size=len(data),
    )


def _get_git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
