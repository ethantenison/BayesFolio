"""Pure compiler from planner output into active multitask GP builder configs.

This module enforces explicit GP instruction precedence, resolves planner
feature selections to tensor dimensions, and compiles planner contracts into
the active multitask builder configuration objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bayesfolio.contracts.results.gp_workflow import (
    GPPlannerResponse,
    PlannerBlockSpec,
    PlannerCovarSpec,
    PlannerKernelComponentSpec,
    ResolvedFeatureBlock,
)
from bayesfolio.engine.forecast.gp.feature_catalog import resolve_planner_blocks
from bayesfolio.engine.forecast.gp.multitask_builder import (
    BlockStructure,
    CovarModuleConfig,
    GlobalStructure,
    InteractionPolicy,
    KernelBlockConfig,
    KernelBlockRole,
    KernelInteractionConfig,
    LengthscalePolicy,
    LengthscalePolicyConfig,
    LengthscalePriorConfig,
    LinearKernelComponentConfig,
    MaternKernelComponentConfig,
    MeanKind,
    MeanModuleConfig,
    PeriodicKernelComponentConfig,
    PeriodLengthPriorConfig,
    RBFKernelComponentConfig,
    RQKernelComponentConfig,
)


@dataclass(frozen=True)
class BlockInstructionLock:
    """Deterministic explicit-instruction lock for one block.

    Attributes:
        component_kinds: Explicitly requested kernel kinds that must be
            preserved until a fallback is blocked.
        ard_locked: Whether ARD was explicitly fixed for the block.
    """

    component_kinds: frozenset[str] = frozenset()
    ard_locked: bool = False


@dataclass(frozen=True)
class ExplicitInstructionContext:
    """Parsed explicit GP instruction constraints.

    Attributes:
        all_inputs_block: Optional replacement block spanning all input
            variables.
        block_overrides: Block-specific explicit overrides keyed by block name.
        global_structure: Optional explicit global structure override.
        interaction_policy: Optional explicit interaction policy override.
        mean_kind: Optional explicit mean-kind override.
        locked_blocks: Explicit per-block repair locks.
        lock_global_structure: Whether global structure is explicitly fixed.
        lock_interactions: Whether interaction policy is explicitly fixed.
        lock_mean_kind: Whether mean kind is explicitly fixed.
        trace: Human-readable trace of recognized explicit instructions.
        adjustments: Required adjustments when a hard explicit request could not
            be honored literally.
    """

    all_inputs_block: PlannerBlockSpec | None = None
    block_overrides: dict[str, PlannerBlockSpec] = field(default_factory=dict)
    global_structure: GlobalStructure | None = None
    interaction_policy: InteractionPolicy | None = None
    mean_kind: MeanKind | None = None
    locked_blocks: dict[str, BlockInstructionLock] = field(default_factory=dict)
    lock_global_structure: bool = False
    lock_interactions: bool = False
    lock_mean_kind: bool = False
    trace: list[str] = field(default_factory=list)
    adjustments: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompiledPlannerDesign:
    """Compiled planner output ready for workflow execution.

    Attributes:
        mean_config: Active builder mean configuration.
        covar_config: Active builder covariance configuration.
        resolved_blocks: Resolved feature blocks with tensor dimensions.
        explicit_context: Parsed explicit instruction context used by repair.
        planner_response: Planner response after explicit enforcement.
    """

    mean_config: MeanModuleConfig
    covar_config: CovarModuleConfig
    resolved_blocks: list[ResolvedFeatureBlock]
    explicit_context: ExplicitInstructionContext
    planner_response: GPPlannerResponse


def compile_planner_response(
    planner_response: GPPlannerResponse,
    *,
    feature_columns: list[str],
    feature_groups: dict[str, list[str]],
    instruction_text: str,
) -> CompiledPlannerDesign:
    """Compile planner output into active multitask GP builder configs.

    Args:
        planner_response: Validated planner response contract.
        feature_columns: Ordered non-task feature columns.
        feature_groups: Validated feature-group mapping.
        instruction_text: Original free-form GP instruction text.

    Returns:
        Compiled active builder configuration bundle.

    Raises:
        ValueError: If no design is available or if the compiled design is
            invalid after explicit instruction enforcement.
    """

    design = planner_response.selected_design
    if design is None:
        raise ValueError("Planner response does not include a selected design")

    explicit_context = parse_explicit_instructions(instruction_text)
    effective_covar = _apply_explicit_overrides(
        covar_spec=design.covar_config,
        feature_columns=feature_columns,
        explicit_context=explicit_context,
    )
    mean_kind = explicit_context.mean_kind or MeanKind(design.mean_config.kind)

    resolved_blocks = resolve_planner_blocks(
        effective_covar.blocks,
        feature_columns=feature_columns,
        feature_groups=feature_groups,
    )
    covar_config = _compile_covar_spec(effective_covar, resolved_blocks)
    mean_config = MeanModuleConfig(kind=mean_kind)

    updated_response = planner_response.model_copy(deep=True)
    updated_design = updated_response.selected_design
    if updated_design is None:
        raise ValueError("Planner response unexpectedly lost selected design during compilation")
    updated_design.covar_config = effective_covar
    updated_design.mean_config.kind = mean_kind.value
    updated_design.explicit_instruction_trace.extend(explicit_context.trace)
    updated_design.adjustments_from_user_request.extend(explicit_context.adjustments)

    return CompiledPlannerDesign(
        mean_config=mean_config,
        covar_config=covar_config,
        resolved_blocks=resolved_blocks,
        explicit_context=explicit_context,
        planner_response=updated_response,
    )


def parse_explicit_instructions(instruction_text: str) -> ExplicitInstructionContext:
    """Parse deterministic explicit GP instructions from free-form text.

    Args:
        instruction_text: Free-form user GP instruction text.

    Returns:
        Parsed explicit instruction context used by compilation and repair.
    """

    lowered = re.sub(r"\s+", " ", instruction_text.strip().lower())
    context = ExplicitInstructionContext()
    trace: list[str] = []
    block_overrides: dict[str, PlannerBlockSpec] = {}
    locked_blocks: dict[str, BlockInstructionLock] = {}

    if "use additive global structure" in lowered:
        context = _replace_context(context, global_structure=GlobalStructure.ADDITIVE, lock_global_structure=True)
        trace.append("Explicit global structure constraint recognized: additive.")
    if "no interactions" in lowered:
        context = _replace_context(context, interaction_policy=InteractionPolicy.NONE, lock_interactions=True)
        trace.append("Explicit interaction constraint recognized: no interactions.")

    all_inputs_match = re.search(
        r"use (?:a |an )?matern(?: kernel)?(?:\s+(5/2|1/2|2\.5|0\.5))?(?: with ard)? for all input variables",
        lowered,
    )
    if all_inputs_match:
        matern_nu = _parse_matern_nu(all_inputs_match.group(1))
        all_inputs_block = PlannerBlockSpec(
            name="features",
            variable_type="generic",
            source_groups=[],
            source_features=[],
            block_structure="additive",
            use_outputscale=False,
            components=[
                PlannerKernelComponentSpec(
                    kind="matern",
                    ard="with ard" in all_inputs_match.group(0),
                    use_outputscale=True,
                    matern_nu=matern_nu,
                    lengthscale_policy={"policy": "adaptive"},
                )
            ],
        )
        locked_blocks[all_inputs_block.name] = BlockInstructionLock(
            component_kinds=frozenset({"matern"}),
            ard_locked=True,
        )
        context = _replace_context(
            context,
            all_inputs_block=all_inputs_block,
            locked_blocks=locked_blocks,
            global_structure=context.global_structure or GlobalStructure.ADDITIVE,
            interaction_policy=context.interaction_policy or InteractionPolicy.NONE,
        )
        trace.append("Explicit all-inputs Matérn constraint recognized and enforced.")

    for kind_left, kind_right, block_name in re.findall(
        r"use (?:a |an )?(linear|matern) plus (linear|matern)(?: kernel)? on (macro|etf|time)",
        lowered,
    ):
        components = [_make_component(kind_left), _make_component(kind_right)]
        override = PlannerBlockSpec(
            name=block_name,
            variable_type=block_name,
            source_groups=[block_name],
            source_features=[],
            block_structure="additive",
            use_outputscale=True,
            components=components,
        )
        block_overrides[block_name] = override
        locked_blocks[block_name] = BlockInstructionLock(
            component_kinds=frozenset(component.kind for component in components),
            ard_locked=any(component.ard is not None for component in components),
        )
        trace.append(f"Explicit block kernel combo recognized on {block_name}.")

    for nu_text, block_name in re.findall(r"matern(?: kernel)?\s*(5/2|1/2|2\.5|0\.5)? on (macro|etf|time)", lowered):
        existing_override = block_overrides.get(block_name)
        if existing_override is not None and len(existing_override.components) > 1:
            continue
        component = PlannerKernelComponentSpec(
            kind="matern",
            ard=True,
            use_outputscale=False,
            matern_nu=_parse_matern_nu(nu_text),
            lengthscale_policy={"policy": "adaptive"},
        )
        block_overrides[block_name] = PlannerBlockSpec(
            name=block_name,
            variable_type=block_name,
            source_groups=[block_name],
            source_features=[],
            block_structure="additive",
            use_outputscale=True,
            components=[component],
        )
        locked_blocks[block_name] = BlockInstructionLock(component_kinds=frozenset({"matern"}), ard_locked=True)
        trace.append(f"Explicit Matérn block constraint recognized on {block_name}.")

    if trace:
        context = _replace_context(context, block_overrides=block_overrides, locked_blocks=locked_blocks, trace=trace)
    return context


def _apply_explicit_overrides(
    *,
    covar_spec: PlannerCovarSpec,
    feature_columns: list[str],
    explicit_context: ExplicitInstructionContext,
) -> PlannerCovarSpec:
    """Apply parsed explicit instruction overrides to a planner covariance spec.

    Args:
        covar_spec: Planner-selected covariance specification.
        feature_columns: Ordered usable non-task feature columns.
        explicit_context: Parsed explicit instruction context.

    Returns:
        Effective planner covariance specification after explicit overrides.
    """

    effective = covar_spec.model_copy(deep=True)

    if explicit_context.all_inputs_block is not None:
        block = explicit_context.all_inputs_block.model_copy(deep=True)
        block.source_features = feature_columns.copy()
        effective.blocks = [block]
        effective.global_structure = (explicit_context.global_structure or GlobalStructure.ADDITIVE).value
        effective.interaction_policy = (explicit_context.interaction_policy or InteractionPolicy.NONE).value
        effective.custom_interactions = []
        return effective

    override_lookup = explicit_context.block_overrides
    if override_lookup:
        blocks_by_name = {block.name: block.model_copy(deep=True) for block in effective.blocks}
        for block_name, override in override_lookup.items():
            blocks_by_name[block_name] = override.model_copy(deep=True)
        ordered_names = [block.name for block in effective.blocks if block.name in blocks_by_name]
        for block_name in override_lookup:
            if block_name not in ordered_names:
                ordered_names.append(block_name)
        effective.blocks = [blocks_by_name[name] for name in ordered_names]

    if explicit_context.global_structure is not None:
        effective.global_structure = explicit_context.global_structure.value
    if explicit_context.interaction_policy is not None:
        effective.interaction_policy = explicit_context.interaction_policy.value
        if explicit_context.interaction_policy is InteractionPolicy.NONE:
            effective.custom_interactions = []

    return effective


def _compile_covar_spec(covar_spec: PlannerCovarSpec, resolved_blocks: list[ResolvedFeatureBlock]) -> CovarModuleConfig:
    """Compile an effective planner covariance spec into active builder config.

    Args:
        covar_spec: Effective planner covariance specification.
        resolved_blocks: Resolved feature blocks with concrete dimensions.

    Returns:
        Active builder covariance configuration.
    """

    dims_lookup = {block.name: block for block in resolved_blocks}
    blocks = [
        KernelBlockConfig(
            name=block_spec.name,
            variable_type=_map_variable_type(block_spec.variable_type),
            components=[
                _compile_component(component_spec, dims_lookup[block_spec.name].dims)
                for component_spec in block_spec.components
            ],
            block_structure=BlockStructure(block_spec.block_structure),
            use_outputscale=block_spec.use_outputscale,
        )
        for block_spec in covar_spec.blocks
    ]
    interactions = [
        KernelInteractionConfig(
            blocks=interaction.blocks,
            name=interaction.name,
            use_outputscale=interaction.use_outputscale,
        )
        for interaction in covar_spec.custom_interactions
    ]
    return CovarModuleConfig(
        blocks=blocks,
        global_structure=GlobalStructure(covar_spec.global_structure),
        interaction_policy=InteractionPolicy(covar_spec.interaction_policy),
        custom_interactions=interactions,
    )


def _compile_component(component_spec: PlannerKernelComponentSpec, dims: list[int]):
    """Compile one planner component into an active builder component config.

    Args:
        component_spec: Planner-selected component specification.
        dims: Resolved active dimensions for the enclosing block.

    Returns:
        Active multitask builder component configuration.
    """

    policy = _compile_lengthscale_policy(component_spec)
    if component_spec.kind == "matern":
        return MaternKernelComponentConfig(
            dims=dims,
            ard=True if component_spec.ard is None else component_spec.ard,
            matern_nu=component_spec.matern_nu or 2.5,
            use_outputscale=component_spec.use_outputscale,
            lengthscale_policy=policy,
        )
    if component_spec.kind == "rbf":
        return RBFKernelComponentConfig(
            dims=dims,
            ard=True if component_spec.ard is None else component_spec.ard,
            use_outputscale=component_spec.use_outputscale,
            lengthscale_policy=policy,
        )
    if component_spec.kind == "rq":
        return RQKernelComponentConfig(
            dims=dims,
            ard=True if component_spec.ard is None else component_spec.ard,
            use_outputscale=component_spec.use_outputscale,
            lengthscale_policy=policy,
        )
    if component_spec.kind == "periodic":
        period_prior = None
        if component_spec.period_prior is not None:
            period_prior = PeriodLengthPriorConfig(
                p0=component_spec.period_prior.p0,
                cv=component_spec.period_prior.cv,
            )
        return PeriodicKernelComponentConfig(
            dims=dims,
            ard=True if component_spec.ard is None else component_spec.ard,
            use_outputscale=component_spec.use_outputscale,
            lengthscale_policy=policy,
            period_prior=period_prior,
        )
    if component_spec.kind == "linear":
        return LinearKernelComponentConfig(
            dims=dims,
            use_outputscale=component_spec.use_outputscale,
        )
    raise ValueError(f"Unsupported planner kernel kind: {component_spec.kind}")


def _compile_lengthscale_policy(component_spec: PlannerKernelComponentSpec) -> LengthscalePolicyConfig:
    """Compile planner lengthscale policy into active builder policy config.

    Args:
        component_spec: Planner-selected component specification.

    Returns:
        Active builder lengthscale policy config.
    """

    if component_spec.lengthscale_policy is None:
        return LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE)

    policy = LengthscalePolicy(component_spec.lengthscale_policy.policy)
    manual = None
    if component_spec.lengthscale_policy.manual is not None:
        manual_values = component_spec.lengthscale_policy.manual
        manual = LengthscalePriorConfig(
            loc=float(manual_values["loc"]),
            scale=float(manual_values["scale"]),
            min_lengthscale=float(manual_values.get("min_lengthscale", 2.5e-2)),
            initial_value=float(manual_values["initial_value"]) if "initial_value" in manual_values else None,
        )
    return LengthscalePolicyConfig(policy=policy, manual=manual)


def _map_variable_type(variable_type: str) -> KernelBlockRole:
    """Map planner variable-type strings to active builder roles.

    Args:
        variable_type: Planner variable type string.

    Returns:
        Active builder block role.
    """

    return KernelBlockRole(variable_type)


def _make_component(kind: str) -> PlannerKernelComponentSpec:
    """Create a basic planner component spec from a kernel-kind string.

    Args:
        kind: Kernel kind string.

    Returns:
        Planner component specification.
    """

    if kind == "linear":
        return PlannerKernelComponentSpec(kind="linear", use_outputscale=False)
    return PlannerKernelComponentSpec(
        kind="matern",
        ard=True,
        use_outputscale=False,
        matern_nu=2.5,
        lengthscale_policy={"policy": "adaptive"},
    )


def _parse_matern_nu(raw_value: str | None) -> float:
    """Parse Matérn smoothness values from user text.

    Args:
        raw_value: Raw smoothness token from explicit text.

    Returns:
        Parsed Matérn smoothness value.
    """

    if raw_value in {None, "", "5/2", "2.5"}:
        return 2.5
    if raw_value in {"1/2", "0.5"}:
        return 0.5
    return float(raw_value)


def _replace_context(context: ExplicitInstructionContext, **updates: object) -> ExplicitInstructionContext:
    """Return a new explicit-instruction context with updated fields.

    Args:
        context: Existing explicit instruction context.
        **updates: Field updates.

    Returns:
        Updated explicit instruction context.
    """

    values = {
        "all_inputs_block": context.all_inputs_block,
        "block_overrides": context.block_overrides,
        "global_structure": context.global_structure,
        "interaction_policy": context.interaction_policy,
        "mean_kind": context.mean_kind,
        "locked_blocks": context.locked_blocks,
        "lock_global_structure": context.lock_global_structure,
        "lock_interactions": context.lock_interactions,
        "lock_mean_kind": context.lock_mean_kind,
        "trace": context.trace,
        "adjustments": context.adjustments,
    }
    values.update(updates)
    return ExplicitInstructionContext(**values)
