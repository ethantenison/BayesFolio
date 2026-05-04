"""Planner-driven multitask Gaussian Process workflow orchestration.

This module orchestrates planner-backed Gaussian Process design selection,
compilation into active multitask builder configs, tensor preparation,
normalization, model fitting, prediction validation, and deterministic repair.
It does not perform file or network I/O beyond the delegated planner client.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.outcome import StratifiedStandardize
from gpytorch.likelihoods import HadamardGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from bayesfolio.contracts.commands.gp_planner import DatasetSummaryItem, GPPlannerRequest
from bayesfolio.contracts.results.gp_workflow import (
    GPFitValidationSummary,
    GPRepairAttempt,
    GPWorkflowResult,
    NormalizationStats,
    ResolvedFeatureBlock,
)
from bayesfolio.engine.agent.gp_planner import plan_gp_design_with_status
from bayesfolio.engine.agent.prompts import DEFAULT_GP_PLANNER_PROMPT
from bayesfolio.engine.features.gp_data_prep import prepare_multitask_gp_data_with_task_feature
from bayesfolio.engine.forecast.gp.feature_catalog import get_non_task_feature_columns, validate_feature_groups
from bayesfolio.engine.forecast.gp.multitask_builder import (
    CovarModuleConfig,
    InteractionPolicy,
    KernelBlockConfig,
    KernelBlockRole,
    LengthscalePolicy,
    MeanKind,
    MeanModuleConfig,
    build_multitask_gp,
    default_covar_config_for_non_task_dims,
)
from bayesfolio.engine.forecast.gp.planner_compiler import CompiledPlannerDesign, compile_planner_response


@dataclass(frozen=True)
class PlannedGPWorkflowOptions:
    """Runtime options for the planner-driven multitask GP workflow.

    Attributes:
        planner_prompt: System prompt supplied to the planner client.
        planner_timeout_seconds: HTTP timeout for the planner call.
        require_live_planner: If True, fail fast unless planner status is
            ``ok`` from a live endpoint response.
        dtype: Torch dtype for tensor creation.
        rank: Multitask rank passed to ``build_multitask_gp``.
        min_inferred_noise_level: Initial minimum inferred noise floor.
        max_repair_steps: Maximum number of deterministic repair actions.
        seed: Optional deterministic torch seed.
    """

    planner_prompt: str = DEFAULT_GP_PLANNER_PROMPT
    planner_timeout_seconds: float = 20.0
    require_live_planner: bool = False
    dtype: torch.dtype = torch.float64
    rank: int | None = 1
    min_inferred_noise_level: float = 5e-3
    max_repair_steps: int = 8
    seed: int | None = None


def run_planned_multitask_gp_from_dataframe(
    *,
    df: pd.DataFrame,
    input_columns: list[str],
    output_columns: list[str],
    instruction_text: str,
    feature_groups: dict[str, list[str]] | None = None,
    options: PlannedGPWorkflowOptions | None = None,
) -> PlannedMultitaskGPArtifacts:
    """Run planner-driven multitask GP directly from input/output column lists.

    Args:
        df: Input dataframe with all modeling columns.
        input_columns: Ordered usable non-task feature columns.
        output_columns: Two-column list in order ``[target_column,
            task_column]``.
        instruction_text: Free-form GP instruction text and prior beliefs.
        feature_groups: Optional semantic feature-group mapping. When omitted,
            a single ``inputs`` group is built from ``input_columns``.
        options: Optional workflow runtime options.

    Returns:
        Engine-facing planner-driven GP artifacts.

    Raises:
        ValueError: If required columns are missing or output columns do not
            contain exactly ``[target_column, task_column]``.
    """

    if len(output_columns) != 2:
        raise ValueError("output_columns must contain exactly [target_column, task_column]")

    target_column, task_column = output_columns
    required_columns = [*input_columns, target_column, task_column]
    missing = [name for name in required_columns if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataframe columns: {missing}")

    deduped_inputs = list(dict.fromkeys(input_columns))
    resolved_groups = feature_groups if feature_groups is not None else {"inputs": deduped_inputs}

    return run_planned_multitask_gp_workflow(
        df=df,
        target_column=target_column,
        task_column=task_column,
        feature_groups=resolved_groups,
        allowed_feature_columns=deduped_inputs,
        instruction_text=instruction_text,
        options=options,
    )


@dataclass(frozen=True)
class PlannedMultitaskGPArtifacts:
    """Engine-facing artifacts returned by the planner-driven GP workflow.

    Attributes:
        result: Structured workflow result contract.
        model: Final fitted multitask GP model, when fitting succeeded.
        likelihood: Final likelihood associated with the fitted model.
        normalized_train_x: Normalized training design matrix.
        train_y: Training targets.
        task_map: Mapping from task identifier values to integer task indices.
        outcome_transform: Outcome transform used during fitting.
    """

    result: GPWorkflowResult
    model: MultiTaskGP | None
    likelihood: HadamardGaussianLikelihood | None
    normalized_train_x: torch.Tensor
    train_y: torch.Tensor
    task_map: dict[object, int]
    outcome_transform: StratifiedStandardize


def run_planned_multitask_gp_workflow(
    *,
    df: pd.DataFrame,
    target_column: str,
    task_column: str,
    feature_groups: dict[str, list[str]] | None = None,
    allowed_feature_columns: list[str] | None = None,
    instruction_text: str,
    options: PlannedGPWorkflowOptions | None = None,
) -> PlannedMultitaskGPArtifacts:
    """Run the end-to-end planner-driven multitask GP workflow.

    Args:
        df: Input dataframe containing target, task, and feature columns.
        target_column: Target return column in decimal units.
        task_column: Task or asset identifier column.
        feature_groups: Optional semantic feature-group mapping.
        allowed_feature_columns: Optional explicit whitelist of non-task
            feature columns.
        instruction_text: Free-form GP instruction text and prior beliefs.
        options: Optional workflow runtime options.

    Returns:
        Engine-facing workflow artifacts including the structured result
        contract and the fitted model when successful.

    Raises:
        ValueError: If required dataframe columns are missing or if task values
            are null.
    """

    runtime_options = options or PlannedGPWorkflowOptions()
    if runtime_options.seed is not None:
        torch.manual_seed(runtime_options.seed)

    if df[task_column].isnull().any():
        raise ValueError(f"Task column {task_column} contains null values")

    feature_columns = get_non_task_feature_columns(
        df,
        target_column=target_column,
        task_column=task_column,
        allowed_feature_columns=allowed_feature_columns,
    )
    normalized_groups = validate_feature_groups(
        feature_groups or {},
        feature_columns=feature_columns,
        task_column=task_column,
        target_column=target_column,
    )
    planner_request = _build_planner_request(
        df=df,
        target_column=target_column,
        task_column=task_column,
        feature_columns=feature_columns,
        feature_groups=normalized_groups,
        allowed_feature_columns=allowed_feature_columns,
        instruction_text=instruction_text,
        planner_prompt=runtime_options.planner_prompt,
    )
    planner_response, planner_client_status = plan_gp_design_with_status(
        request=planner_request,
        timeout_seconds=runtime_options.planner_timeout_seconds,
    )
    if runtime_options.require_live_planner and planner_client_status != "ok":
        raise RuntimeError(f"Live planner was required but unavailable. Planner status: {planner_client_status}.")

    compiled = compile_planner_response(
        planner_response,
        feature_columns=feature_columns,
        feature_groups=normalized_groups,
        instruction_text=instruction_text,
    )

    training_df = df[[*feature_columns, target_column, task_column]].copy()
    train_x, train_y, task_map = prepare_multitask_gp_data_with_task_feature(
        training_df,
        target_col=target_column,
        asset_col=task_column,
        drop_cols=[],
        dtype=runtime_options.dtype,
    )
    normalized_train_x, normalization = _normalize_train_x(train_x, feature_columns)
    outcome_transform = _build_outcome_transform(normalized_train_x, train_y)

    diagnostics: list[str] = []
    repair_attempts: list[GPRepairAttempt] = []
    current_compiled = compiled
    current_noise_floor = runtime_options.min_inferred_noise_level
    build_success = False
    fit_success = False
    prediction_success = False
    model: MultiTaskGP | None = None
    likelihood: HadamardGaussianLikelihood | None = None
    final_status = "fit_failed"
    failure_stage = "build"
    attempt_count = 0

    repair_actions = _repair_actions()[: runtime_options.max_repair_steps]
    repair_index = 0

    while True:
        attempt_count += 1
        try:
            model = build_multitask_gp(
                train_X=normalized_train_x,
                train_Y=train_y,
                task_feature=-1,
                covar_config=current_compiled.covar_config,
                mean_config=current_compiled.mean_config,
                rank=runtime_options.rank,
                min_inferred_noise_level=current_noise_floor,
                outcome_transform=outcome_transform,
                input_transform=None,
            )
            build_success = True
            failure_stage = "fit"
        except Exception as exc:
            diagnostics.append(f"Build attempt {attempt_count} failed: {exc}")
            model = None
        else:
            try:
                model.train()
                likelihood = model.likelihood
                mll = ExactMarginalLogLikelihood(likelihood, model)
                fit_gpytorch_mll(mll)
                fit_success = True
                failure_stage = "prediction"
            except Exception as exc:
                diagnostics.append(f"Fit attempt {attempt_count} failed: {exc}")
                likelihood = None
            else:
                try:
                    model.eval()
                    likelihood.eval()
                    with torch.no_grad():
                        posterior = model.posterior(normalized_train_x)
                        if posterior.mean.shape[-2] != normalized_train_x.shape[0]:
                            raise RuntimeError("Posterior mean row count does not match training rows")
                    prediction_success = True
                    final_status = "ok"
                    break
                except Exception as exc:
                    diagnostics.append(f"Prediction attempt {attempt_count} failed: {exc}")
                    failure_stage = "prediction"

        if repair_index >= len(repair_actions):
            final_status = {
                "build": "build_failed",
                "fit": "fit_failed",
                "prediction": "prediction_failed",
            }[failure_stage]
            break

        repair_applied = False
        while repair_index < len(repair_actions) and not repair_applied:
            step_number = repair_index + 1
            action = repair_actions[repair_index]
            repair_index += 1
            (
                updated_compiled,
                updated_noise_floor,
                repair_attempt,
                repair_applied,
            ) = _apply_repair_action(
                action=action,
                step_number=step_number,
                compiled=current_compiled,
                feature_columns=feature_columns,
                current_noise_floor=current_noise_floor,
            )
            repair_attempts.append(repair_attempt)
            if repair_applied:
                current_compiled = updated_compiled
                current_noise_floor = updated_noise_floor

        if not repair_applied:
            final_status = {
                "build": "build_failed",
                "fit": "fit_failed",
                "prediction": "prediction_failed",
            }[failure_stage]
            break

    result = GPWorkflowResult(
        planner_client_status=planner_client_status,
        planner_response=current_compiled.planner_response,
        target_column=target_column,
        task_column=task_column,
        feature_columns=feature_columns,
        resolved_blocks=current_compiled.resolved_blocks,
        mean_config=current_compiled.mean_config.model_dump(mode="json"),
        covar_config=current_compiled.covar_config.model_dump(mode="json"),
        normalization=normalization,
        fit_validation=GPFitValidationSummary(
            build_success=build_success,
            fit_success=fit_success,
            prediction_success=prediction_success,
            attempt_count=attempt_count,
            min_inferred_noise_level=current_noise_floor,
        ),
        repair_attempts=repair_attempts,
        final_status=final_status,
        diagnostics=diagnostics,
    )
    return PlannedMultitaskGPArtifacts(
        result=result,
        model=model if prediction_success else None,
        likelihood=likelihood if prediction_success else None,
        normalized_train_x=normalized_train_x,
        train_y=train_y,
        task_map=task_map,
        outcome_transform=outcome_transform,
    )


def _build_planner_request(
    *,
    df: pd.DataFrame,
    target_column: str,
    task_column: str,
    feature_columns: list[str],
    feature_groups: dict[str, list[str]],
    allowed_feature_columns: list[str] | None,
    instruction_text: str,
    planner_prompt: str,
) -> GPPlannerRequest:
    """Build the planner request contract from workflow inputs.

    Args:
        df: Input dataframe.
        target_column: Target column name.
        task_column: Task column name.
        feature_columns: Ordered usable non-task feature columns.
        feature_groups: Validated feature-group mapping.
        allowed_feature_columns: Optional explicit feature whitelist.
        instruction_text: Free-form planner instruction text.
        planner_prompt: Planner system prompt.

    Returns:
        Planner request contract.
    """

    return GPPlannerRequest(
        target_column=target_column,
        task_column=task_column,
        user_instruction_text=instruction_text,
        feature_groups=feature_groups,
        allowed_feature_columns=allowed_feature_columns,
        dataset_summary=_build_dataset_summary_items(df, feature_columns, task_column),
        planner_prompt=planner_prompt,
    )


def _build_dataset_summary_items(
    df: pd.DataFrame,
    feature_columns: list[str],
    task_column: str,
) -> list[DatasetSummaryItem]:
    """Build a compact dataset summary passed to the planner.

    Args:
        df: Input dataframe.
        feature_columns: Ordered usable non-task feature columns.
        task_column: Task column name.

    Returns:
        Structured dataset summary items.
    """

    task_count = int(df[task_column].nunique())
    row_count = int(len(df))
    feature_count = len(feature_columns)
    ratio = float(row_count / feature_count) if feature_count else 0.0
    missing_rate = float(df[feature_columns].isnull().mean().mean()) if feature_columns else 0.0
    return [
        DatasetSummaryItem(key="row_count", value=row_count, description="Number of rows in the modeling dataframe."),
        DatasetSummaryItem(key="task_count", value=task_count, description="Number of unique tasks or assets."),
        DatasetSummaryItem(key="feature_count", value=feature_count, description="Number of usable non-task features."),
        DatasetSummaryItem(
            key="sample_to_feature_ratio",
            value=ratio,
            description="Rows divided by usable non-task feature count.",
        ),
        DatasetSummaryItem(
            key="mean_feature_missing_rate",
            value=missing_rate,
            description="Average feature missing rate before workflow validation.",
        ),
    ]


def _normalize_train_x(train_x: torch.Tensor, feature_columns: list[str]) -> tuple[torch.Tensor, NormalizationStats]:
    """Apply min-max normalization to non-task GP inputs.

    Args:
        train_x: Training design matrix including task feature column.
        feature_columns: Ordered non-task feature names.

    Returns:
        Tuple of normalized training matrix and normalization statistics.
    """

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


def _build_outcome_transform(train_x: torch.Tensor, train_y: torch.Tensor) -> StratifiedStandardize:
    """Create the multitask outcome transform for the workflow.

    Args:
        train_x: Training design matrix including task feature column.
        train_y: Training targets.

    Returns:
        Stratified standardization transform keyed by task value.
    """

    task_feature = train_x.shape[-1] - 1
    all_task_values = train_x[:, task_feature].to(torch.long).unique(sorted=True)
    return StratifiedStandardize(
        stratification_idx=task_feature,
        all_task_values=all_task_values,
        observed_task_values=train_x[:, task_feature].to(torch.long),
        batch_shape=train_y.shape[:-2],
    )


def _repair_actions() -> list[str]:
    """Return the ordered deterministic repair actions for the workflow."""

    return [
        "increase_noise_floor",
        "remove_interactions",
        "hierarchical_to_additive",
        "remove_periodic",
        "remove_rq",
        "reduce_ard",
        "simplify_mean",
        "collapse_to_conservative_matern",
    ]


def _apply_repair_action(
    *,
    action: str,
    step_number: int,
    compiled: CompiledPlannerDesign,
    feature_columns: list[str],
    current_noise_floor: float,
) -> tuple[CompiledPlannerDesign, float, GPRepairAttempt, bool]:
    """Apply one deterministic repair action to the compiled GP design.

    Args:
        action: Repair action identifier.
        step_number: Ordered repair step number.
        compiled: Current compiled planner design.
        feature_columns: Ordered non-task feature columns.
        current_noise_floor: Current minimum inferred noise floor.

    Returns:
        Updated compiled design, updated noise floor, repair-attempt record, and
        a boolean indicating whether the action materially changed the state.
    """

    explicit = compiled.explicit_context
    covar_config = compiled.covar_config.model_copy(deep=True)
    mean_config = compiled.mean_config.model_copy(deep=True)
    resolved_blocks = [block.model_copy(deep=True) for block in compiled.resolved_blocks]

    if action == "increase_noise_floor":
        next_floor = _next_noise_floor(current_noise_floor)
        if next_floor == current_noise_floor:
            return (
                compiled,
                current_noise_floor,
                GPRepairAttempt(
                    step=step_number,
                    action=action,
                    status="no_change",
                    detail="Noise floor already at the maximum deterministic fallback value.",
                ),
                False,
            )
        return (
            compiled,
            next_floor,
            GPRepairAttempt(
                step=step_number,
                action=action,
                status="applied",
                detail=f"Raised min_inferred_noise_level to {next_floor}.",
            ),
            True,
        )

    if action == "remove_interactions":
        if explicit.lock_interactions:
            return (
                compiled,
                current_noise_floor,
                GPRepairAttempt(
                    step=step_number,
                    action=action,
                    status="blocked_by_explicit_instruction",
                    detail="Explicit interaction policy prevents removing interactions.",
                ),
                False,
            )
        changed = covar_config.interaction_policy is not InteractionPolicy.NONE or bool(
            covar_config.custom_interactions
        )
        covar_config.interaction_policy = InteractionPolicy.NONE
        covar_config.custom_interactions = []
        if not changed:
            return (
                compiled,
                current_noise_floor,
                GPRepairAttempt(
                    step=step_number,
                    action=action,
                    status="no_change",
                    detail="No interactions were present to remove.",
                ),
                False,
            )
        updated = _rebuild_compiled(compiled, covar_config, mean_config, resolved_blocks)
        attempt = GPRepairAttempt(
            step=step_number,
            action=action,
            status="applied",
            detail="Removed policy-generated and custom interactions.",
        )
        return updated, current_noise_floor, attempt, True

    if action == "hierarchical_to_additive":
        if explicit.lock_global_structure:
            return (
                compiled,
                current_noise_floor,
                GPRepairAttempt(
                    step=step_number,
                    action=action,
                    status="blocked_by_explicit_instruction",
                    detail="Explicit global structure prevents hierarchical-to-additive fallback.",
                ),
                False,
            )
        if covar_config.global_structure.value != "hierarchical":
            return (
                compiled,
                current_noise_floor,
                GPRepairAttempt(
                    step=step_number,
                    action=action,
                    status="no_change",
                    detail="Covariance structure is already non-hierarchical.",
                ),
                False,
            )
        covar_config.global_structure = covar_config.global_structure.ADDITIVE
        covar_config.interaction_policy = InteractionPolicy.NONE
        covar_config.custom_interactions = []
        updated = _rebuild_compiled(compiled, covar_config, mean_config, resolved_blocks)
        attempt = GPRepairAttempt(
            step=step_number,
            action=action,
            status="applied",
            detail="Changed global structure to additive and removed interactions.",
        )
        return updated, current_noise_floor, attempt, True

    if action in {"remove_periodic", "remove_rq"}:
        target_kind = "periodic" if action == "remove_periodic" else "rq"
        if _is_component_kind_locked(explicit, target_kind):
            return (
                compiled,
                current_noise_floor,
                GPRepairAttempt(
                    step=step_number,
                    action=action,
                    status="blocked_by_explicit_instruction",
                    detail=f"Explicit instruction preserves {target_kind} components.",
                ),
                False,
            )
        changed = False
        kept_blocks: list[KernelBlockConfig] = []
        kept_resolved: list[ResolvedFeatureBlock] = []
        for block, resolved in zip(covar_config.blocks, resolved_blocks, strict=True):
            kept_components = [component for component in block.components if component.kind.value != target_kind]
            if len(kept_components) != len(block.components):
                changed = True
            if kept_components:
                block.components = kept_components
                kept_blocks.append(block)
                kept_resolved.append(resolved)
        if not changed:
            return (
                compiled,
                current_noise_floor,
                GPRepairAttempt(
                    step=step_number,
                    action=action,
                    status="no_change",
                    detail=f"No {target_kind} components were present to remove.",
                ),
                False,
            )
        if not kept_blocks:
            return (
                compiled,
                current_noise_floor,
                GPRepairAttempt(
                    step=step_number,
                    action=action,
                    status="skipped",
                    detail=f"Removing all {target_kind} components would eliminate every covariance block.",
                ),
                False,
            )
        covar_config.blocks = kept_blocks
        covar_config.custom_interactions = [
            interaction
            for interaction in covar_config.custom_interactions
            if all(block_name in {block.name for block in kept_blocks} for block_name in interaction.blocks)
        ]
        updated = _rebuild_compiled(compiled, covar_config, mean_config, kept_resolved)
        attempt = GPRepairAttempt(
            step=step_number,
            action=action,
            status="applied",
            detail=f"Removed {target_kind} components from covariance blocks.",
        )
        return updated, current_noise_floor, attempt, True

    if action == "reduce_ard":
        candidate = _highest_risk_ard_block(covar_config, resolved_blocks, explicit)
        if candidate is None:
            status = "blocked_by_explicit_instruction" if _any_ard_locked(explicit) else "no_change"
            detail = (
                "All ARD-enabled blocks are locked by explicit instructions."
                if status == "blocked_by_explicit_instruction"
                else "No ARD-enabled blocks were available for simplification."
            )
            return (
                compiled,
                current_noise_floor,
                GPRepairAttempt(
                    step=step_number,
                    action=action,
                    status=status,
                    detail=detail,
                ),
                False,
            )
        for component in candidate.components:
            if hasattr(component, "ard"):
                component.ard = False
        updated = _rebuild_compiled(compiled, covar_config, mean_config, resolved_blocks)
        attempt = GPRepairAttempt(
            step=step_number,
            action=action,
            status="applied",
            detail=f"Disabled ARD on block {candidate.name}.",
        )
        return updated, current_noise_floor, attempt, True

    if action == "simplify_mean":
        if explicit.lock_mean_kind:
            return (
                compiled,
                current_noise_floor,
                GPRepairAttempt(
                    step=step_number,
                    action=action,
                    status="blocked_by_explicit_instruction",
                    detail="Explicit mean instruction prevents simplifying the mean module.",
                ),
                False,
            )
        if mean_config.kind is MeanKind.MULTITASK_CONSTANT:
            return (
                compiled,
                current_noise_floor,
                GPRepairAttempt(
                    step=step_number,
                    action=action,
                    status="no_change",
                    detail="Mean module is already multitask_constant.",
                ),
                False,
            )
        mean_config.kind = MeanKind.MULTITASK_CONSTANT
        mean_config.input_size = None
        updated = _rebuild_compiled(compiled, covar_config, mean_config, resolved_blocks)
        attempt = GPRepairAttempt(
            step=step_number,
            action=action,
            status="applied",
            detail="Simplified mean module to multitask_constant.",
        )
        return updated, current_noise_floor, attempt, True

    if action == "collapse_to_conservative_matern":
        if explicit.all_inputs_block is None and explicit.block_overrides:
            return (
                compiled,
                current_noise_floor,
                GPRepairAttempt(
                    step=step_number,
                    action=action,
                    status="blocked_by_explicit_instruction",
                    detail="Explicit block-level kernel instructions prevent collapsing to a single fallback block.",
                ),
                False,
            )
        collapsed_covar = default_covar_config_for_non_task_dims(
            list(range(len(feature_columns))),
            policy=LengthscalePolicy.ADAPTIVE,
        )
        collapsed_blocks = [
            ResolvedFeatureBlock(
                name="features",
                variable_type=KernelBlockRole.GENERIC.value,
                feature_names=feature_columns,
                dims=list(range(len(feature_columns))),
            )
        ]
        collapsed_mean = mean_config.model_copy(update={"kind": MeanKind.MULTITASK_CONSTANT, "input_size": None})
        updated = _rebuild_compiled(compiled, collapsed_covar, collapsed_mean, collapsed_blocks)
        attempt = GPRepairAttempt(
            step=step_number,
            action=action,
            status="applied",
            detail="Collapsed covariance to one conservative Matérn block across all non-task features.",
        )
        return updated, current_noise_floor, attempt, True

    return (
        compiled,
        current_noise_floor,
        GPRepairAttempt(
            step=step_number,
            action=action,
            status="skipped",
            detail="Unknown repair action.",
        ),
        False,
    )


def _rebuild_compiled(
    compiled: CompiledPlannerDesign,
    covar_config: CovarModuleConfig,
    mean_config: MeanModuleConfig,
    resolved_blocks: list[ResolvedFeatureBlock],
) -> CompiledPlannerDesign:
    """Rebuild a compiled planner design after deterministic repair changes."""

    planner_response = compiled.planner_response.model_copy(deep=True)
    design = planner_response.selected_design
    if design is not None:
        design.mean_config.kind = mean_config.kind.value
        design.covar_config = design.covar_config.model_copy(
            update={
                "blocks": [
                    {
                        "name": block.name,
                        "variable_type": block.variable_type.value,
                        "source_groups": [],
                        "source_features": resolved.feature_names,
                        "block_structure": block.block_structure.value,
                        "use_outputscale": block.use_outputscale,
                        "components": [],
                    }
                    for block, resolved in zip(covar_config.blocks, resolved_blocks, strict=True)
                ],
                "global_structure": covar_config.global_structure.value,
                "interaction_policy": covar_config.interaction_policy.value,
                "custom_interactions": [
                    interaction.model_dump(mode="json") for interaction in covar_config.custom_interactions
                ],
            }
        )
    return CompiledPlannerDesign(
        mean_config=mean_config,
        covar_config=covar_config,
        resolved_blocks=resolved_blocks,
        explicit_context=compiled.explicit_context,
        planner_response=planner_response,
    )


def _next_noise_floor(current_noise_floor: float) -> float:
    """Return the next deterministic noise-floor value."""

    schedule = [5e-4, 5e-3, 1e-2, 5e-2]
    for value in schedule:
        if value > current_noise_floor:
            return value
    return current_noise_floor


def _is_component_kind_locked(explicit_context, kind: str) -> bool:
    """Return whether a component kind is protected by explicit instructions."""

    return any(kind in lock.component_kinds for lock in explicit_context.locked_blocks.values())


def _any_ard_locked(explicit_context) -> bool:
    """Return whether any block has explicit ARD locking."""

    return any(lock.ard_locked for lock in explicit_context.locked_blocks.values())


def _highest_risk_ard_block(
    covar_config: CovarModuleConfig,
    resolved_blocks: list[ResolvedFeatureBlock],
    explicit_context,
) -> KernelBlockConfig | None:
    """Return the highest-dimensional block eligible for ARD simplification."""

    resolved_lookup = {block.name: block for block in resolved_blocks}
    candidates: list[tuple[int, KernelBlockConfig]] = []
    for block in covar_config.blocks:
        lock = explicit_context.locked_blocks.get(block.name)
        if lock is not None and lock.ard_locked:
            continue
        if not any(getattr(component, "ard", False) for component in block.components):
            continue
        candidates.append((len(resolved_lookup[block.name].dims), block))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]
