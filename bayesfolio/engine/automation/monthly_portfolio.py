"""End-to-end monthly portfolio automation workflow.

This module owns pure monthly computation and sequencing. Providers, feature
storage, tracking, artifact persistence, and source-revision discovery arrive
through injected dependencies assembled by the application layer. Feature rows
are aligned strictly before/on the requested as-of date, return targets use
decimal units, and the accepted monthly multitask GP is fit in double precision.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import gpytorch
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
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
from bayesfolio.engine.automation.monthly_ports import MonthlyPortfolioRunIO
from bayesfolio.engine.features.dataset_builder import (
    ArtifactStoreProtocol,
    FeatureProviders,
    build_features_dataset,
)
from bayesfolio.engine.forecast.gp.multitask_builder import (
    CovarModuleConfig,
    MeanModuleConfig,
    build_multitask_gp,
)
from bayesfolio.engine.forecast.gp.time_varying_kernel import build_time_varying_kernel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonthlyPortfolioRuntimeDependencies:
    """Dependency bundle for monthly automation runtime wiring.

    Attributes:
        feature_providers: External ETF, macro, and return-label providers.
        artifact_store: Feature-dataset parquet persistence implementation.
        run_io: Runtime tracking and artifact I/O adapter. ``None`` is accepted
            only for legacy construction and must be completed by the
            application composition root before engine execution.
    """

    feature_providers: FeatureProviders
    artifact_store: ArtifactStoreProtocol
    run_io: MonthlyPortfolioRunIO | None = None


@dataclass(frozen=True)
class FixedMonthlyGPArtifacts:
    """Artifacts produced by the deterministic monthly GP fit.

    Attributes:
        result: Structured description and validation status of the GP fit.
        model: Fitted multitask GP model.
        likelihood: Likelihood associated with the fitted model.
        train_x: Raw double-precision training design tensor. The model-owned
            BoTorch input transform normalizes non-task columns.
        train_y: Training target tensor in decimal-return units.
        task_map: Asset identifier to task-index mapping.
        outcome_transform: Task-aware target standardization transform.
    """

    result: GPWorkflowResult
    model: object
    likelihood: object
    train_x: torch.Tensor
    train_y: torch.Tensor
    task_map: dict[str, int]
    outcome_transform: StratifiedStandardize


def build_default_dependencies(command: MonthlyPortfolioCommand) -> MonthlyPortfolioRuntimeDependencies:
    """Build default monthly dependencies through the application layer.

    Args:
        command: Validated monthly workflow configuration.

    Returns:
        Complete runtime dependencies.

    Warns:
        DeprecationWarning: Always. Import the builder from
            :mod:`bayesfolio.application` instead.
    """

    warnings.warn(
        "bayesfolio.engine.automation.build_default_dependencies is deprecated; "
        "use bayesfolio.application.build_default_monthly_portfolio_dependencies",
        DeprecationWarning,
        stacklevel=2,
    )
    from bayesfolio.application.monthly_portfolio import build_default_monthly_portfolio_dependencies

    return build_default_monthly_portfolio_dependencies(command)


def run_monthly_portfolio(
    command: MonthlyPortfolioCommand,
    *,
    dependencies: MonthlyPortfolioRuntimeDependencies | None = None,
) -> MonthlyPortfolioResult:
    """Run the monthly workflow through the compatibility entry point.

    Args:
        command: Validated monthly workflow configuration.
        dependencies: Optional custom or legacy dependency bundle.

    Returns:
        Structured monthly result containing forecasts, weights, and artifacts.

    Warns:
        DeprecationWarning: Always. The canonical entry point now lives in
            :mod:`bayesfolio.application`.
    """

    warnings.warn(
        "bayesfolio.engine.automation.run_monthly_portfolio is deprecated; "
        "use bayesfolio.application.run_monthly_portfolio",
        DeprecationWarning,
        stacklevel=2,
    )
    from bayesfolio.application.monthly_portfolio import run_monthly_portfolio as run_application_workflow

    return run_application_workflow(command, dependencies=dependencies)


def execute_monthly_portfolio(
    command: MonthlyPortfolioCommand,
    *,
    dependencies: MonthlyPortfolioRuntimeDependencies,
) -> MonthlyPortfolioResult:
    """Execute monthly business sequencing with fully injected dependencies.

    Args:
        command: Validated monthly workflow configuration.
        dependencies: Complete provider, store, and runtime-I/O dependencies.

    Returns:
        Structured monthly result containing forecasts, weights, and artifacts.

    Raises:
        ValueError: If the application layer did not supply ``run_io``.
    """

    runtime = dependencies
    if runtime.run_io is None:
        msg = "Monthly engine execution requires an injected run_io adapter."
        raise ValueError(msg)
    run_io = runtime.run_io
    run_label = command.artifacts.run_label or f"monthly_portfolio_{command.as_of_date.isoformat()}"
    logger.info("Starting monthly portfolio run %s for %s", run_label, command.as_of_date.isoformat())

    diagnostics: list[str] = []
    persisted_artifacts = MonthlyRunArtifacts()

    with run_io.start_run(run_name=run_label):
        run_io.log_contract("monthly_portfolio_command", command)
        run_io.log_source_revision()

        features_result = build_features_dataset(
            command=command.feature_dataset,
            providers=runtime.feature_providers,
            artifact_store=runtime.artifact_store,
        )
        persisted_artifacts.feature_dataset = _feature_artifact_to_report_pointer(features_result.artifact)
        run_io.log_contract("features_result", features_result)

        features_df = run_io.load_parquet(features_result.artifact.uri)
        training_df, scoring_df = _prepare_as_of_feature_frames(features_df=features_df, command=command)

        gp_artifacts = _run_fixed_monthly_gp(training_df=training_df, command=command)
        run_io.log_contract("gp_workflow_result", gp_artifacts.result)

        scenario_frame, prediction_records, forecast_result = _build_monthly_forecasts(
            scoring_df=scoring_df,
            gp_artifacts=gp_artifacts,
            scenario_count=command.forecast.posterior_scenario_count,
            target_column=command.forecast.target_column,
            task_column=command.forecast.task_column,
            final_portfolio_universe=command.final_portfolio_universe or [],
            helper_assets_only=command.helper_assets_only,
        )
        run_io.log_contract("forecast_result", forecast_result)

        final_scenarios = scenario_frame.loc[:, command.final_portfolio_universe]
        optimize_result = optimize_from_historical_returns(
            final_scenarios,
            request=OptimizeCommand(
                objective=command.riskfolio.obj,
                risk_measure=command.riskfolio.rm,
                model=command.riskfolio.model,
                rf=command.riskfolio.rf,
                alpha=command.riskfolio.alpha,
                max_weight=command.riskfolio.upperlng,
                nea=command.riskfolio.nea,
                hist=command.riskfolio.hist,
            ),
        )
        run_io.log_contract("optimize_result", optimize_result)

        if command.artifacts.save_scenarios_csv:
            persisted_artifacts.scenarios_csv = run_io.save_csv(
                scenario_frame,
                f"{run_label}/{run_label}_posterior_scenarios.csv",
            )
        if command.artifacts.save_predictions_csv:
            predictions_frame = pd.DataFrame([record.model_dump(mode="json") for record in prediction_records])
            persisted_artifacts.predictions_csv = run_io.save_csv(
                predictions_frame,
                f"{run_label}/{run_label}_predictions.csv",
            )
        if command.artifacts.save_weights_csv:
            weights_frame = pd.DataFrame({"asset": optimize_result.asset_order, "weight": optimize_result.weights})
            persisted_artifacts.weights_csv = run_io.save_csv(
                weights_frame,
                f"{run_label}/{run_label}_weights.csv",
            )

        result = MonthlyPortfolioResult(
            strategy_id=command.strategy_id,
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
            persisted_artifacts.summary_json = run_io.save_json_contract(
                result,
                f"{run_label}/{run_label}_summary.json",
            )
            result = result.model_copy(update={"artifacts": persisted_artifacts})

        run_io.log_contract("monthly_portfolio_result", result)
        logger.info("Completed monthly portfolio run %s", run_label)
        return result


def _prepare_as_of_feature_frames(
    *,
    features_df: pd.DataFrame,
    command: MonthlyPortfolioCommand,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align training and scoring panels without future information."""

    forecast = command.forecast
    date_column = "date"
    selected_columns = [
        date_column,
        *forecast.allowed_feature_columns,
        forecast.target_column,
        forecast.task_column,
    ]
    missing_columns = sorted(set(selected_columns) - set(features_df.columns))
    if missing_columns:
        raise ValueError(f"Feature artifact is missing required columns: {missing_columns}")

    panel = features_df.loc[:, selected_columns].copy()
    panel[date_column] = pd.to_datetime(panel[date_column])
    panel[forecast.task_column] = pd.Categorical(
        panel[forecast.task_column],
        categories=command.training_universe,
        ordered=True,
    )
    panel = panel[panel[forecast.task_column].notna()].sort_values(
        [date_column, forecast.task_column],
        kind="stable",
    )

    as_of = pd.Timestamp(command.as_of_date)
    training_df = panel[(panel[date_column] < as_of) & panel[forecast.target_column].notna()].copy()
    scoring_df = panel[panel[date_column] == as_of].copy()
    if training_df.empty:
        raise ValueError(f"No labeled training rows exist before as_of_date={command.as_of_date.isoformat()}.")
    if scoring_df.empty:
        raise ValueError(f"No scoring rows exist at as_of_date={command.as_of_date.isoformat()}.")

    scoring_counts = scoring_df.groupby(forecast.task_column, observed=True).size()
    duplicate_assets = scoring_counts[scoring_counts != 1].index.astype(str).tolist()
    missing_assets = [asset for asset in command.training_universe if asset not in scoring_counts.index]
    if duplicate_assets or missing_assets:
        raise ValueError(
            "Scoring panel must contain exactly one row per training-universe asset; "
            f"missing={missing_assets}, duplicate_or_extra={duplicate_assets}."
        )

    return training_df, scoring_df


def _prepare_training_tensors(
    *,
    training_df: pd.DataFrame,
    command: MonthlyPortfolioCommand,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Convert the ordered monthly panel to historical-parity tensors."""

    forecast = command.forecast
    feature_columns = forecast.allowed_feature_columns
    task_map = {asset: index for index, asset in enumerate(command.training_universe)}
    feature_values = training_df.loc[:, feature_columns].to_numpy(dtype=float)
    target_values = training_df[forecast.target_column].to_numpy(dtype=float)
    task_values = training_df[forecast.task_column].map(task_map).to_numpy(dtype=float).reshape(-1, 1)
    if not np.isfinite(feature_values).all():
        raise ValueError("Training features must be finite after cleaning.")
    if not np.isfinite(target_values).all():
        raise ValueError("Training targets must be finite decimal returns.")
    if not np.isfinite(task_values).all():
        raise ValueError("Training panel contains an asset outside training_universe.")

    train_x = torch.tensor(
        np.concatenate([feature_values, task_values], axis=1),
        dtype=torch.float64,
    )
    train_y = torch.tensor(target_values, dtype=torch.float64).unsqueeze(-1)
    return train_x, train_y, task_map


def _run_fixed_monthly_gp(*, training_df: pd.DataFrame, command: MonthlyPortfolioCommand) -> FixedMonthlyGPArtifacts:
    feature_columns = command.forecast.allowed_feature_columns
    feature_index = {name: idx for idx, name in enumerate(feature_columns)}
    covar_config, mean_config = build_default_monthly_gp_configs(
        forecast_config=command.forecast,
        feature_index=feature_index,
    )

    train_x, train_y, task_map = _prepare_training_tensors(training_df=training_df, command=command)
    normalization = _normalization_stats(train_x, feature_columns)

    task_feature_idx = train_x.shape[-1] - 1
    all_task_values = train_x[:, task_feature_idx].to(torch.long).unique(sorted=True)
    outcome_transform = StratifiedStandardize(
        stratification_idx=task_feature_idx,
        all_task_values=all_task_values,
        batch_shape=train_y.shape[:-2],
    )

    model, likelihood = _build_and_fit_monthly_model(
        train_x=train_x,
        train_y=train_y,
        command=command,
        covar_config=covar_config,
        mean_config=mean_config,
        outcome_transform=outcome_transform,
    )

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
            dataset_assumptions=["Monthly automation uses the accepted July 2026 multitask GP architecture."],
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
        diagnostics=[
            "Accepted monthly GP architecture applied: E + M + T + ET + MT + EMT.",
            f"rank={command.forecast.rank}; dtype=float64; maxiter={command.forecast.maxiter}; "
            f"time_modulation={command.forecast.time_modulation}; seed={command.forecast.seed}.",
        ],
    )

    return FixedMonthlyGPArtifacts(
        result=result,
        model=model,
        likelihood=likelihood,
        train_x=train_x,
        train_y=train_y,
        task_map=task_map,
        outcome_transform=outcome_transform,
    )


def _build_and_fit_monthly_model(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    command: MonthlyPortfolioCommand,
    covar_config: CovarModuleConfig,
    mean_config: MeanModuleConfig,
    outcome_transform: StratifiedStandardize,
) -> tuple[Any, Any]:
    """Build and fit the accepted model without leaking Torch default dtype."""

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(command.forecast.seed)
        input_transform = Normalize(
            d=train_x.shape[-1],
            indices=list(range(len(command.forecast.allowed_feature_columns))),
        )
        model = build_multitask_gp(
            train_X=train_x,
            train_Y=train_y,
            task_feature=-1,
            covar_config=covar_config,
            mean_config=mean_config,
            rank=command.forecast.rank,
            min_inferred_noise_level=command.forecast.min_inferred_noise_level,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            task_covar_prior=None,
        )
        if command.forecast.time_modulation == "lengthscale_only":
            data_kernel, task_kernel = model.covar_module.kernels
            model.covar_module = (
                build_time_varying_kernel(data_kernel, time_feature_index=0, target="lengthscale") * task_kernel
            )

        model.train()
        likelihood = model.likelihood
        mll = ExactMarginalLogLikelihood(likelihood, model)
        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={"options": {"maxiter": command.forecast.maxiter}},
        )
        return model, likelihood
    finally:
        torch.set_default_dtype(previous_dtype)


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
    scoring_x, asset_cols = _prepare_scoring_tensor(
        scoring_df=scoring_df,
        gp_artifacts=gp_artifacts,
        target_column=target_column,
        task_column=task_column,
    )

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(scoring_x, observation_noise=True)
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
    latest_rows = scoring_df[[*feature_columns, target_column, task_column]].copy().reset_index(drop=True)
    task_map = gp_artifacts.task_map
    row_task_ids = [task_map[asset] for asset in latest_rows[task_column].tolist()]

    x = torch.zeros((len(latest_rows), len(feature_columns) + 1), dtype=gp_artifacts.train_x.dtype)
    for idx, feature in enumerate(feature_columns):
        x[:, idx] = torch.tensor(latest_rows[feature].to_numpy(dtype=float), dtype=x.dtype)
    x[:, -1] = torch.tensor(row_task_ids, dtype=x.dtype)
    asset_cols = latest_rows[task_column].tolist()
    return x, asset_cols


def _normalization_stats(train_x: torch.Tensor, feature_columns: list[str]) -> NormalizationStats:
    non_task_indices = list(range(len(feature_columns)))
    mins = train_x[:, non_task_indices].amin(dim=0)
    maxs = train_x[:, non_task_indices].amax(dim=0)
    ranges = (maxs - mins).clamp_min(1e-12)
    return NormalizationStats(
        feature_names=feature_columns,
        mins={name: float(value) for name, value in zip(feature_columns, mins.tolist(), strict=True)},
        maxs={name: float(value) for name, value in zip(feature_columns, maxs.tolist(), strict=True)},
        ranges={name: float(value) for name, value in zip(feature_columns, ranges.tolist(), strict=True)},
    )


def _feature_artifact_to_report_pointer(artifact) -> ArtifactPointer:
    return ArtifactPointer(
        path=artifact.uri,
        artifact_format=artifact.format,
        digest=artifact.fingerprint,
        byte_size=artifact.byte_size,
    )
