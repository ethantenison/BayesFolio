"""Planner output and workflow result contracts for multitask GP execution.

This module defines result-layer schemas for planner-selected Gaussian Process
designs and the end-to-end workflow execution summary. Contracts are data-only
and intended for cross-boundary exchange.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from bayesfolio.contracts.base import ContractModel, SchemaName, VersionedContract


class ScoreReason(ContractModel):
    """Numeric score and rationale for one planner objective.

    Attributes:
        score: Integer score on a 1-5 scale.
        reason: Short explanation for the assigned score.
    """

    score: int = Field(ge=1, le=5)
    reason: str


class PlannerObjectiveAssessment(ContractModel):
    """Planner evaluation of the selected design against key objectives.

    Attributes:
        fidelity_to_user_priors: Alignment with explicit instructions or
            softer prior beliefs.
        fit_stability: Estimated ability to fit and predict reliably.
        interpretability: Estimated explainability of the selected structure.
        expressiveness: Estimated nonlinear modeling capacity.
        repairability: Estimated ease of deterministic fallback repair.
    """

    fidelity_to_user_priors: ScoreReason
    fit_stability: ScoreReason
    interpretability: ScoreReason
    expressiveness: ScoreReason
    repairability: ScoreReason


class PlannerLengthscalePolicySpec(ContractModel):
    """Planner-selected lengthscale policy specification.

    Attributes:
        policy: Lengthscale policy name.
        manual: Optional manual prior parameters when the planner selects a
            manual log-normal policy.
    """

    policy: Literal["botorch_standard", "adaptive", "manual_lognormal"]
    manual: dict[str, float] | None = None


class PlannerPeriodPriorSpec(ContractModel):
    """Planner-selected periodic prior parameters.

    Attributes:
        p0: Prior median of the periodic length in normalized feature units.
        cv: Coefficient of variation for the period-length prior.
    """

    p0: float
    cv: float = 0.5


class PlannerKernelComponentSpec(ContractModel):
    """Planner-selected kernel component definition.

    Attributes:
        kind: Kernel family used by the active multitask builder.
        ard: Whether ARD lengthscales are requested.
        use_outputscale: Whether the component should be wrapped in a
            ``ScaleKernel`` when used directly as an additive term.
        matern_nu: Optional Matérn smoothness parameter.
        lengthscale_policy: Selected lengthscale prior policy.
        period_prior: Optional periodic prior configuration.
    """

    kind: Literal["matern", "rbf", "rq", "periodic", "linear"]
    ard: bool | None = None
    use_outputscale: bool = False
    matern_nu: float | None = None
    lengthscale_policy: PlannerLengthscalePolicySpec | None = None
    period_prior: PlannerPeriodPriorSpec | None = None


class PlannerBlockSpec(ContractModel):
    """Planner-selected covariance block definition.

    Attributes:
        name: Unique block name.
        variable_type: Semantic role of the block.
        source_groups: Feature-group names used to resolve this block.
        source_features: Explicit feature names used to resolve this block.
        block_structure: Whether components are added or multiplied.
        use_outputscale: Whether to scale the whole block as a top-level term.
        components: Kernel components combined inside this block.
    """

    name: str
    variable_type: Literal["generic", "time", "etf", "macro", "categorical"] = "generic"
    source_groups: list[str] = Field(default_factory=list)
    source_features: list[str] = Field(default_factory=list)
    block_structure: Literal["additive", "product"] = "additive"
    use_outputscale: bool = False
    components: list[PlannerKernelComponentSpec] = Field(default_factory=list)


class PlannerInteractionSpec(ContractModel):
    """Planner-selected explicit interaction between named blocks.

    Attributes:
        blocks: Ordered block names multiplied together in the interaction.
        name: Optional human-readable label.
        use_outputscale: Whether to scale the interaction term.
    """

    blocks: list[str] = Field(min_length=2)
    name: str | None = None
    use_outputscale: bool = True


class PlannerMeanSpec(ContractModel):
    """Planner-selected mean-module specification.

    Attributes:
        kind: Mean kind used by the active multitask builder.
    """

    kind: Literal[
        "multitask_constant",
        "multitask_zero",
        "multitask_linear",
        "constant",
        "zero",
        "linear",
    ]


class PlannerCovarSpec(ContractModel):
    """Planner-selected covariance architecture specification.

    Attributes:
        blocks: Covariance blocks in planner-selected order.
        global_structure: Top-level block-combination strategy.
        interaction_policy: Auto-generated interaction policy.
        custom_interactions: Optional explicit interactions between blocks.
    """

    blocks: list[PlannerBlockSpec] = Field(default_factory=list)
    global_structure: Literal["additive", "hierarchical", "non_compositional"] = "additive"
    interaction_policy: Literal["none", "sparse", "temporal_only", "full", "custom"] = "none"
    custom_interactions: list[PlannerInteractionSpec] = Field(default_factory=list)


class PlannerSelectedDesign(ContractModel):
    """Structured GP design selected by the planner.

    Attributes:
        name: Short name for the design.
        design_intent: Short explanation of why the design was chosen.
        objective_assessment: Planner scores against the five workflow
            objectives.
        mean_config: Planner-selected mean configuration.
        covar_config: Planner-selected covariance configuration.
        explicit_instruction_trace: Trace of explicit instruction handling.
        why_this_design_wins: High-level reasons this design was selected.
        tradeoffs_accepted: Tradeoffs accepted when balancing objectives.
        adjustments_from_user_request: Deviations from user text that were
            required to stay within supported hard constraints.
        fit_risk: Estimated fit-risk category.
        fit_risk_reasons: Short list of risk explanations.
        repair_plan: Ordered deterministic fallback plan.
    """

    name: str
    design_intent: str
    objective_assessment: PlannerObjectiveAssessment
    mean_config: PlannerMeanSpec
    covar_config: PlannerCovarSpec
    explicit_instruction_trace: list[str] = Field(default_factory=list)
    why_this_design_wins: list[str] = Field(default_factory=list)
    tradeoffs_accepted: list[str] = Field(default_factory=list)
    adjustments_from_user_request: list[str] = Field(default_factory=list)
    fit_risk: Literal["low", "medium", "high"] = "medium"
    fit_risk_reasons: list[str] = Field(default_factory=list)
    repair_plan: list[str] = Field(default_factory=list)


class GPPlannerResponse(VersionedContract):
    """Planner result contract for a structured multitask GP design.

    Attributes:
        schema: Contract schema identifier.
        schema_version: Semantic schema version.
        planner_status: Planner status after response validation.
        clarification_questions: Questions emitted when the planner needs more
            information.
        dataset_assumptions: Assumptions the planner relied on.
        instruction_mode: Whether the planner acted under explicit hard
            constraints or softer preference optimization.
        selected_design: Final planner-selected GP design.
    """

    schema: Literal[SchemaName.GP_PLANNER_RESULT] = SchemaName.GP_PLANNER_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    planner_status: Literal["ok", "needs_clarification", "invalid_request"]
    clarification_questions: list[str] = Field(default_factory=list)
    dataset_assumptions: list[str] = Field(default_factory=list)
    instruction_mode: Literal["explicit_constraints", "preference_optimization"]
    selected_design: PlannerSelectedDesign | None = None


class ResolvedFeatureBlock(ContractModel):
    """Resolved feature names and dimensions for one covariance block.

    Attributes:
        name: Resolved block name.
        variable_type: Semantic role of the block.
        feature_names: Resolved non-task feature names in tensor order.
        dims: Integer feature indices in tensor order, excluding the task
            feature column.
    """

    name: str
    variable_type: str
    feature_names: list[str] = Field(default_factory=list)
    dims: list[int] = Field(default_factory=list)


class NormalizationStats(ContractModel):
    """Normalization statistics for non-task GP inputs.

    Attributes:
        feature_names: Ordered non-task feature names normalized by the
            workflow.
        mins: Minimum values used for min-max normalization by feature.
        maxs: Maximum values used for min-max normalization by feature.
        ranges: Clamped ranges used for normalization by feature.
    """

    feature_names: list[str] = Field(default_factory=list)
    mins: dict[str, float] = Field(default_factory=dict)
    maxs: dict[str, float] = Field(default_factory=dict)
    ranges: dict[str, float] = Field(default_factory=dict)


class GPFitValidationSummary(ContractModel):
    """Fit and prediction validation summary for the workflow.

    Attributes:
        build_success: Whether the model object was built successfully.
        fit_success: Whether hyperparameter fitting completed successfully.
        prediction_success: Whether the prediction smoke test completed
            successfully.
        attempt_count: Number of build/fit attempts performed.
        min_inferred_noise_level: Final minimum inferred noise floor used in
            the multitask likelihood.
    """

    build_success: bool = False
    fit_success: bool = False
    prediction_success: bool = False
    attempt_count: int = Field(ge=0, default=0)
    min_inferred_noise_level: float | None = None


class GPRepairAttempt(ContractModel):
    """One deterministic repair attempt applied by the workflow.

    Attributes:
        step: Ordered repair step number.
        action: Repair action identifier.
        status: Outcome of the repair step.
        detail: Human-readable description of what changed or why the step was
            skipped.
    """

    step: int = Field(ge=1)
    action: str
    status: Literal["applied", "skipped", "blocked_by_explicit_instruction", "no_change"]
    detail: str


class GPWorkflowResult(VersionedContract):
    """End-to-end workflow result for planner-driven multitask GP execution.

    Attributes:
        schema: Contract schema identifier.
        schema_version: Semantic schema version.
        planner_client_status: Planner client call status.
        planner_response: Validated planner response used by the workflow.
        target_column: Target column name in decimal-return units.
        task_column: Task identifier column name.
        feature_columns: Ordered non-task feature columns used to build
            training tensors.
        resolved_blocks: Resolved covariance blocks with feature names and
            dimensions.
        mean_config: Serialized mean configuration passed to the active builder.
        covar_config: Serialized covariance configuration passed to the active
            builder.
        normalization: Input normalization statistics.
        fit_validation: Fit and prediction validation summary.
        repair_attempts: Deterministic repair attempts applied by the workflow.
        final_status: Overall workflow terminal status.
        diagnostics: Additional diagnostic notes and failure messages.
    """

    schema: Literal[SchemaName.GP_WORKFLOW_RESULT] = SchemaName.GP_WORKFLOW_RESULT
    schema_version: Literal["0.1.0"] = "0.1.0"
    planner_client_status: str
    planner_response: GPPlannerResponse
    target_column: str
    task_column: str
    feature_columns: list[str] = Field(default_factory=list)
    resolved_blocks: list[ResolvedFeatureBlock] = Field(default_factory=list)
    mean_config: dict[str, object] = Field(default_factory=dict)
    covar_config: dict[str, object] = Field(default_factory=dict)
    normalization: NormalizationStats = Field(default_factory=NormalizationStats)
    fit_validation: GPFitValidationSummary = Field(default_factory=GPFitValidationSummary)
    repair_attempts: list[GPRepairAttempt] = Field(default_factory=list)
    final_status: Literal["ok", "planner_failed", "build_failed", "fit_failed", "prediction_failed"]
    diagnostics: list[str] = Field(default_factory=list)
