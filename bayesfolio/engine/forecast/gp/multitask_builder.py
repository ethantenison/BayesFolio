"""Configuration-driven builders for BoTorch MultiTaskGP modules.

This module provides a configuration layer for constructing ``MultiTaskGP``
mean and covariance modules with explicit prior, composition, and interaction
controls.

Boundary responsibility:
- Builds ``gpytorch`` mean and kernel modules and optionally instantiates a
  ``botorch.models.multitask.MultiTaskGP``.
- Does not perform I/O or business orchestration.

Units:
- This module configures model structure only; return units are inherited from
  caller-provided training data (BayesFolio convention is decimal returns,
  where ``0.02`` means 2%).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from math import log, sqrt
from typing import Annotated, Any, Literal, cast

import torch
from botorch.models.multitask import MultiTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import (
    Kernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    ProductKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
)
from gpytorch.likelihoods import HadamardGaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean, Mean, MultitaskMean, ZeroMean
from gpytorch.priors import LogNormalPrior
from pydantic import BaseModel, ConfigDict, Field, model_validator

SQRT2 = sqrt(2)
SQRT3 = sqrt(3)


class LengthscalePolicy(StrEnum):
    """Policy for lengthscale priors and constraints."""

    BOTORCH_STANDARD = "botorch_standard"
    ADAPTIVE = "adaptive"
    MANUAL_LOGNORMAL = "manual_lognormal"


class KernelKind(StrEnum):
    """Supported base kernels for configurable multitask covariance."""

    MATERN = "matern"
    RBF = "rbf"
    RQ = "rq"
    PERIODIC = "periodic"
    LINEAR = "linear"


class MeanKind(StrEnum):
    """Supported mean-module templates for MultiTaskGP."""

    MULTITASK_CONSTANT = "multitask_constant"
    MULTITASK_ZERO = "multitask_zero"
    MULTITASK_LINEAR = "multitask_linear"
    CONSTANT = "constant"
    ZERO = "zero"
    LINEAR = "linear"


class KernelBlockRole(StrEnum):
    """Semantic role of a covariance block.

    Attributes:
        GENERIC: Generic feature block with no special interaction semantics.
        TIME: Temporal block used by policies such as ``TEMPORAL_ONLY``.
        ETF: Asset-level feature block.
        MACRO: Macro feature block.
        CATEGORICAL: Categorical feature block.
    """

    GENERIC = "generic"
    TIME = "time"
    ETF = "etf"
    MACRO = "macro"
    CATEGORICAL = "categorical"


class BlockStructure(StrEnum):
    """How kernels inside a block are composed.

    Attributes:
        ADDITIVE: Sum component kernels inside the block.
        PRODUCT: Multiply component kernels inside the block.
    """

    ADDITIVE = "additive"
    PRODUCT = "product"


class GlobalStructure(StrEnum):
    """How blocks are combined at the top level.

    Attributes:
        ADDITIVE: Sum block kernels only.
        HIERARCHICAL: Sum block kernels and interaction kernels.
        NON_COMPOSITIONAL: Use a single block only.
    """

    ADDITIVE = "additive"
    HIERARCHICAL = "hierarchical"
    NON_COMPOSITIONAL = "non_compositional"


class InteractionPolicy(StrEnum):
    """Policy for auto-generated block interaction kernels.

    Attributes:
        NONE: No policy-generated interactions.
        SPARSE: Heuristic interactions based on block roles.
        TEMPORAL_ONLY: Interact temporal blocks with every other block.
        FULL: All pairwise block interactions.
        CUSTOM: Use only explicitly supplied custom interactions.
    """

    NONE = "none"
    SPARSE = "sparse"
    TEMPORAL_ONLY = "temporal_only"
    FULL = "full"
    CUSTOM = "custom"


class LengthscalePriorConfig(BaseModel):
    """Manual LogNormal prior and lower-bound configuration for lengthscale.

    Attributes:
        loc: LogNormal location parameter.
        scale: LogNormal scale parameter.
        min_lengthscale: Lower bound for constrained lengthscale values.
        initial_value: Optional explicit initial value for the transformed
            lengthscale parameter. If ``None``, uses the prior mode.
    """

    loc: float
    scale: float
    min_lengthscale: float = 2.5e-2
    initial_value: float | None = None

    model_config = ConfigDict(extra="forbid")


class LengthscalePolicyConfig(BaseModel):
    """Lengthscale-policy selection and optional manual parameters.

    Attributes:
        policy: Policy name.
        manual: Manual prior details used only when
            ``policy=MANUAL_LOGNORMAL``.
    """

    policy: LengthscalePolicy = LengthscalePolicy.BOTORCH_STANDARD
    manual: LengthscalePriorConfig | None = None

    model_config = ConfigDict(extra="forbid")


class PeriodLengthPriorConfig(BaseModel):
    """Adaptive period-length prior for periodic kernels.

    Attributes:
        p0: Prior median in normalized feature units.
        cv: Coefficient of variation controlling prior dispersion.
    """

    p0: float
    cv: float = 0.5

    model_config = ConfigDict(extra="forbid")


class BaseKernelComponentConfig(BaseModel):
    """Fields shared by all kernel component configurations.

    Attributes:
        kind: Base kernel type.
        dims: Active feature indices for this component, excluding task column.
        use_outputscale: If True, wraps this component in ``ScaleKernel`` when
            the component is used directly in an additive position. Product
            compositions ignore this flag and apply scaling only once at the
            outer product-term level.
    """

    kind: KernelKind
    dims: list[int]
    use_outputscale: bool = True

    model_config = ConfigDict(extra="forbid")


class MaternKernelComponentConfig(BaseKernelComponentConfig):
    """Kernel component configuration for ``MaternKernel``.

    Attributes:
        kind: Discriminator for a Matérn kernel.
        dims: Active feature indices for this component.
        use_outputscale: Whether to wrap the component in ``ScaleKernel`` when
            the component is used directly as an additive term.
        ard: Whether to use ARD lengthscales.
        matern_nu: Smoothness parameter for the Matérn kernel.
        lengthscale_policy: Lengthscale prior and lower-bound policy.
    """

    kind: Literal[KernelKind.MATERN] = KernelKind.MATERN
    ard: bool = True
    matern_nu: float = 2.5
    lengthscale_policy: LengthscalePolicyConfig = Field(default_factory=LengthscalePolicyConfig)


class RBFKernelComponentConfig(BaseKernelComponentConfig):
    """Kernel component configuration for ``RBFKernel``.

    Attributes:
        kind: Discriminator for an RBF kernel.
        dims: Active feature indices for this component.
        use_outputscale: Whether to wrap the component in ``ScaleKernel`` when
            the component is used directly as an additive term.
        ard: Whether to use ARD lengthscales.
        lengthscale_policy: Lengthscale prior and lower-bound policy.
    """

    kind: Literal[KernelKind.RBF] = KernelKind.RBF
    ard: bool = True
    lengthscale_policy: LengthscalePolicyConfig = Field(default_factory=LengthscalePolicyConfig)


class RQKernelComponentConfig(BaseKernelComponentConfig):
    """Kernel component configuration for ``RQKernel``.

    Attributes:
        kind: Discriminator for an RQ kernel.
        dims: Active feature indices for this component.
        use_outputscale: Whether to wrap the component in ``ScaleKernel`` when
            the component is used directly as an additive term.
        ard: Whether to use ARD lengthscales.
        lengthscale_policy: Lengthscale prior and lower-bound policy.
    """

    kind: Literal[KernelKind.RQ] = KernelKind.RQ
    ard: bool = True
    lengthscale_policy: LengthscalePolicyConfig = Field(default_factory=LengthscalePolicyConfig)


class PeriodicKernelComponentConfig(BaseKernelComponentConfig):
    """Kernel component configuration for ``PeriodicKernel``.

    Attributes:
        kind: Discriminator for a periodic kernel.
        dims: Active feature indices for this component.
        use_outputscale: Whether to wrap the component in ``ScaleKernel`` when
            the component is used directly as an additive term.
        ard: Whether to use ARD lengthscales.
        lengthscale_policy: Lengthscale prior and lower-bound policy.
        period_prior: Optional prior for period length in normalized units.
    """

    kind: Literal[KernelKind.PERIODIC] = KernelKind.PERIODIC
    ard: bool = True
    lengthscale_policy: LengthscalePolicyConfig = Field(default_factory=LengthscalePolicyConfig)
    period_prior: PeriodLengthPriorConfig | None = None


class LinearKernelComponentConfig(BaseKernelComponentConfig):
    """Kernel component configuration for ``LinearKernel``.

    Attributes:
        kind: Discriminator for a linear kernel.
        dims: Active feature indices for this component.
        use_outputscale: Whether to wrap the component in ``ScaleKernel`` when
            the component is used directly as an additive term.
    """

    kind: Literal[KernelKind.LINEAR] = KernelKind.LINEAR


KernelComponentConfig = Annotated[
    MaternKernelComponentConfig
    | RBFKernelComponentConfig
    | RQKernelComponentConfig
    | PeriodicKernelComponentConfig
    | LinearKernelComponentConfig,
    Field(discriminator="kind"),
]


class KernelTermConfig(BaseModel):
    """Legacy product term made from one or more kernel components.

    Attributes:
        components: Components multiplied together in order.
        use_outputscale: If True, wrap the final product in ``ScaleKernel``.
            When ``components`` contains multiple items, component-level
            outputscales are ignored and the product is wrapped once here.
    """

    components: list[KernelComponentConfig]
    use_outputscale: bool = False

    model_config = ConfigDict(extra="forbid")


class KernelBlockConfig(BaseModel):
    """A named covariance block used by the architecture-first builder.

    Attributes:
        name: Unique block identifier used by interaction definitions.
        variable_type: Semantic role used by interaction policies.
        components: Kernel components combined inside this block.
        block_structure: Whether components are added or multiplied.
        use_outputscale: If True, wrap the whole block kernel in
            ``ScaleKernel`` when used as a top-level additive term. When the
            block participates in an interaction product, its unscaled base
            kernel is used instead.
    """

    name: str
    variable_type: KernelBlockRole = KernelBlockRole.GENERIC
    components: list[KernelComponentConfig]
    block_structure: BlockStructure = BlockStructure.ADDITIVE
    use_outputscale: bool = False

    model_config = ConfigDict(extra="forbid")


class KernelInteractionConfig(BaseModel):
    """Explicit interaction term assembled from named blocks.

    Attributes:
        blocks: Ordered block names multiplied together in this interaction.
        name: Optional human-readable interaction label.
        use_outputscale: If True, wrap the final product interaction once in
            ``ScaleKernel``.
    """

    blocks: list[str]
    name: str | None = None
    use_outputscale: bool = True

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_blocks(self) -> KernelInteractionConfig:
        if len(self.blocks) < 2:
            raise ValueError("interaction blocks must contain at least two block names")
        return self


class CovarModuleConfig(BaseModel):
    """Covariance architecture for non-task features in ``MultiTaskGP``.

    Attributes:
        blocks: Architecture-first block definitions. Each block describes a
            named kernel expression over one or more feature subsets.
        global_structure: Top-level combination strategy for block kernels.
        interaction_policy: Policy used to auto-generate block interactions
            when ``global_structure=HIERARCHICAL``.
        custom_interactions: Explicit interaction products between named blocks.
            These are added in addition to policy-generated interactions unless
            ``interaction_policy=CUSTOM``.
        terms: Legacy additive-product terms kept temporarily for compatibility.
            New code should prefer ``blocks``.
    """

    blocks: list[KernelBlockConfig] = Field(default_factory=list)
    global_structure: GlobalStructure = GlobalStructure.ADDITIVE
    interaction_policy: InteractionPolicy = InteractionPolicy.NONE
    custom_interactions: list[KernelInteractionConfig] = Field(default_factory=list)
    terms: list[KernelTermConfig] | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_config_shape(self) -> CovarModuleConfig:
        has_terms = bool(self.terms)
        has_blocks = bool(self.blocks)
        if has_terms == has_blocks:
            raise ValueError("Exactly one of terms or blocks must be provided")
        if has_terms and self.custom_interactions:
            raise ValueError("custom_interactions are only supported with blocks")
        if has_terms and self.global_structure is not GlobalStructure.ADDITIVE:
            raise ValueError("legacy terms only support ADDITIVE global_structure")
        if has_terms and self.interaction_policy is not InteractionPolicy.NONE:
            raise ValueError("legacy terms do not support interaction policies")
        if has_blocks:
            block_names = [block.name for block in self.blocks]
            if len(block_names) != len(set(block_names)):
                raise ValueError("block names must be unique")
            known_blocks = set(block_names)
            for interaction in self.custom_interactions:
                missing = [name for name in interaction.blocks if name not in known_blocks]
                if missing:
                    raise ValueError(f"Unknown block names in interaction {interaction.blocks}: {missing}")
        return self


class MeanModuleConfig(BaseModel):
    """Mean-module configuration for ``MultiTaskGP``.

    Attributes:
        kind: Mean template to build.
        input_size: Non-task feature size for linear means.
        num_tasks: Number of tasks for multitask means.
    """

    kind: MeanKind = MeanKind.MULTITASK_CONSTANT
    input_size: int | None = None
    num_tasks: int | None = None

    model_config = ConfigDict(extra="forbid")


def make_period_length_prior(p0: float, cv: float = 0.5) -> LogNormalPrior:
    """Create a log-normal prior for period length.

    Args:
        p0: Prior median in normalized feature units.
        cv: Coefficient of variation.

    Returns:
        LogNormalPrior configured from ``p0`` and ``cv``.
    """

    sigma = sqrt(log(1 + cv**2))
    mu = log(p0)
    return LogNormalPrior(loc=mu, scale=sigma)


def _resolve_lengthscale_prior_and_constraint(
    *,
    policy: LengthscalePolicyConfig,
    ard_num_dims: int,
) -> tuple[LogNormalPrior, GreaterThan]:
    if ard_num_dims < 1:
        raise ValueError("ard_num_dims must be >= 1")

    if policy.policy == LengthscalePolicy.MANUAL_LOGNORMAL and policy.manual is None:
        raise ValueError("manual must be set when policy is MANUAL_LOGNORMAL")

    if policy.policy == LengthscalePolicy.BOTORCH_STANDARD:
        prior = LogNormalPrior(loc=SQRT2 + log(ard_num_dims) * 0.5, scale=SQRT3)
        minimum = 2.5e-2
        initial = prior.mode
    elif policy.policy == LengthscalePolicy.ADAPTIVE:
        prior = LogNormalPrior(loc=SQRT2 + log(ard_num_dims) * 0.5, scale=SQRT3)
        minimum = 2.5e-2 * sqrt(ard_num_dims)
        initial = prior.mode
    else:
        manual = policy.manual
        if manual is None:
            raise ValueError("manual lengthscale configuration is required")
        prior = LogNormalPrior(loc=manual.loc, scale=manual.scale)
        minimum = manual.min_lengthscale
        initial = prior.mode if manual.initial_value is None else manual.initial_value

    constraint = GreaterThan(minimum, initial_value=initial)
    return prior, constraint


def _sum_kernels(kernels: list[Kernel]) -> Kernel:
    if not kernels:
        raise ValueError("Cannot sum an empty kernel list")
    kernel_sum = kernels[0]
    for next_kernel in kernels[1:]:
        kernel_sum = kernel_sum + next_kernel
    return kernel_sum


def _unwrap_scale(kernel: Kernel) -> Kernel:
    current = kernel
    while isinstance(current, ScaleKernel):
        current = current.base_kernel
    return current


def _assert_no_inner_scales_in_products(kernel: Kernel, *, ctx: str = "sanity") -> None:
    """Assert that product-kernel children are not directly wrapped in ``ScaleKernel``."""

    def _walk(node: Kernel, path: str) -> None:
        if isinstance(node, ProductKernel):
            for child in node.kernels:
                if isinstance(child, ScaleKernel):
                    raise RuntimeError(
                        f"[{ctx}] Found ScaleKernel directly under ProductKernel at: {path} -> ProductKernel"
                    )
        kernels = getattr(node, "kernels", None)
        if kernels is not None:
            for index, child in enumerate(kernels):
                _walk(child, f"{path}/kernels[{index}]")
        elif isinstance(node, ScaleKernel):
            _walk(node.base_kernel, f"{path}/base_kernel")

    _walk(kernel, "kernel")


def _build_product_kernel(kernels: list[Kernel], *, batch_shape: torch.Size, use_outputscale: bool) -> Kernel:
    if not kernels:
        raise ValueError("Cannot multiply an empty kernel list")

    product = _unwrap_scale(kernels[0])
    for next_kernel in kernels[1:]:
        product = ProductKernel(product, _unwrap_scale(next_kernel))
    if use_outputscale:
        product = ScaleKernel(product, batch_shape=batch_shape)
    _assert_no_inner_scales_in_products(product, ctx="product")
    return product


def _build_kernel_component(
    component: KernelComponentConfig,
    *,
    batch_shape: torch.Size = torch.Size(),
    apply_outputscale: bool | None = None,
) -> Kernel:
    if isinstance(component, LinearKernelComponentConfig):
        kernel: Kernel = LinearKernel(active_dims=tuple(component.dims), ard_num_dims=None, batch_shape=batch_shape)
        should_scale = component.use_outputscale if apply_outputscale is None else apply_outputscale
        return ScaleKernel(kernel, batch_shape=batch_shape) if should_scale else kernel

    ard_num_dims = len(component.dims) if component.ard else 1
    prior, constraint = _resolve_lengthscale_prior_and_constraint(
        policy=component.lengthscale_policy,
        ard_num_dims=ard_num_dims,
    )

    if isinstance(component, MaternKernelComponentConfig):
        kernel = MaternKernel(
            nu=component.matern_nu,
            ard_num_dims=ard_num_dims,
            active_dims=tuple(component.dims),
            batch_shape=batch_shape,
            lengthscale_prior=prior,
            lengthscale_constraint=constraint,
        )
    elif isinstance(component, RBFKernelComponentConfig):
        kernel = RBFKernel(
            ard_num_dims=ard_num_dims,
            active_dims=tuple(component.dims),
            batch_shape=batch_shape,
            lengthscale_prior=prior,
            lengthscale_constraint=constraint,
        )
    elif isinstance(component, RQKernelComponentConfig):
        kernel = RQKernel(
            ard_num_dims=ard_num_dims,
            active_dims=tuple(component.dims),
            batch_shape=batch_shape,
            lengthscale_prior=prior,
            lengthscale_constraint=constraint,
        )
    elif isinstance(component, PeriodicKernelComponentConfig):
        period_length = component.period_prior.p0 if component.period_prior is not None else 0.25
        period_prior_cv = component.period_prior.cv if component.period_prior is not None else 0.5
        period_prior = make_period_length_prior(period_length, period_prior_cv)
        kernel = PeriodicKernel(
            ard_num_dims=ard_num_dims,
            active_dims=tuple(component.dims),
            batch_shape=batch_shape,
            period_length_prior=period_prior,
            period_length=period_length,
            lengthscale_prior=prior,
            lengthscale_constraint=constraint,
        )
    else:
        raise ValueError(f"Unsupported kernel kind: {component.kind}")

    should_scale = component.use_outputscale if apply_outputscale is None else apply_outputscale
    return ScaleKernel(kernel, batch_shape=batch_shape) if should_scale else kernel


def _build_legacy_covar_module(terms: list[KernelTermConfig], *, batch_shape: torch.Size) -> Kernel:
    if not terms:
        raise ValueError("config.terms cannot be empty")

    additive_terms: list[Kernel] = []
    for term in terms:
        if not term.components:
            raise ValueError("each term must have at least one component")
        if len(term.components) == 1:
            additive_terms.append(_build_kernel_component(term.components[0], batch_shape=batch_shape))
            continue

        built_components = [
            _build_kernel_component(component, batch_shape=batch_shape, apply_outputscale=False)
            for component in term.components
        ]
        additive_terms.append(
            _build_product_kernel(
                built_components,
                batch_shape=batch_shape,
                use_outputscale=term.use_outputscale,
            )
        )

    return _sum_kernels(additive_terms)


def _build_block_kernel(block: KernelBlockConfig, *, batch_shape: torch.Size, scaled: bool) -> Kernel:
    if not block.components:
        raise ValueError(f"block {block.name} must have at least one component")

    component_kernels: list[Kernel] = []
    for component in block.components:
        should_scale_component = (
            scaled and block.block_structure is BlockStructure.ADDITIVE and component.use_outputscale
        )
        component_kernels.append(
            _build_kernel_component(
                component,
                batch_shape=batch_shape,
                apply_outputscale=should_scale_component,
            )
        )

    if block.block_structure is BlockStructure.PRODUCT:
        return _build_product_kernel(
            component_kernels,
            batch_shape=batch_shape,
            use_outputscale=scaled and block.use_outputscale,
        )

    block_kernel = _sum_kernels(component_kernels)
    if scaled and block.use_outputscale:
        block_kernel = ScaleKernel(block_kernel, batch_shape=batch_shape)
    return block_kernel


def _dedupe_interactions(interactions: list[KernelInteractionConfig]) -> list[KernelInteractionConfig]:
    unique: list[KernelInteractionConfig] = []
    seen: set[tuple[str, ...]] = set()
    for interaction in interactions:
        key = tuple(sorted(dict.fromkeys(interaction.blocks)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(interaction)
    return unique


def _build_sparse_interactions(blocks: list[KernelBlockConfig]) -> list[KernelInteractionConfig]:
    interactions: list[KernelInteractionConfig] = []
    time_blocks = [block.name for block in blocks if block.variable_type is KernelBlockRole.TIME]
    continuous_like = {
        KernelBlockRole.GENERIC,
        KernelBlockRole.ETF,
        KernelBlockRole.MACRO,
    }
    categorical_blocks = [block.name for block in blocks if block.variable_type is KernelBlockRole.CATEGORICAL]

    for time_block in time_blocks:
        for block in blocks:
            if block.name != time_block:
                interactions.append(KernelInteractionConfig(blocks=[time_block, block.name]))

    for categorical_block in categorical_blocks:
        for block in blocks:
            if block.name == categorical_block:
                continue
            if block.variable_type in continuous_like:
                interactions.append(KernelInteractionConfig(blocks=[categorical_block, block.name]))

    return _dedupe_interactions(interactions)


def _build_policy_interactions(
    blocks: list[KernelBlockConfig],
    *,
    policy: InteractionPolicy,
) -> list[KernelInteractionConfig]:
    if policy in {InteractionPolicy.NONE, InteractionPolicy.CUSTOM}:
        return []

    if policy is InteractionPolicy.FULL:
        return [
            KernelInteractionConfig(blocks=[blocks[left].name, blocks[right].name])
            for left in range(len(blocks))
            for right in range(left + 1, len(blocks))
        ]

    if policy is InteractionPolicy.TEMPORAL_ONLY:
        time_blocks = [block.name for block in blocks if block.variable_type is KernelBlockRole.TIME]
        interactions: list[KernelInteractionConfig] = []
        for time_block in time_blocks:
            for block in blocks:
                if block.name != time_block:
                    interactions.append(KernelInteractionConfig(blocks=[time_block, block.name]))
        return _dedupe_interactions(interactions)

    if policy is InteractionPolicy.SPARSE:
        return _build_sparse_interactions(blocks)

    raise ValueError(f"Unsupported interaction policy: {policy}")


def _build_interaction_kernel(
    interaction: KernelInteractionConfig,
    *,
    block_lookup: dict[str, KernelBlockConfig],
    batch_shape: torch.Size,
) -> Kernel:
    block_kernels = [
        _build_block_kernel(block_lookup[name], batch_shape=batch_shape, scaled=False) for name in interaction.blocks
    ]
    return _build_product_kernel(
        block_kernels,
        batch_shape=batch_shape,
        use_outputscale=interaction.use_outputscale,
    )


def build_covar_module(config: CovarModuleConfig, batch_shape: torch.Size = torch.Size()) -> Kernel:
    """Build a covariance module from an architecture or legacy term config.

    Args:
        config: Covariance configuration.
        batch_shape: Optional GPyTorch batch shape.

    Returns:
        A ``gpytorch.kernels.Kernel`` usable as ``covar_module`` in
        ``MultiTaskGP``.
    """

    if config.terms is not None:
        return _build_legacy_covar_module(config.terms, batch_shape=batch_shape)

    if not config.blocks:
        raise ValueError("config.blocks cannot be empty")

    block_lookup = {block.name: block for block in config.blocks}
    block_kernels = [_build_block_kernel(block, batch_shape=batch_shape, scaled=True) for block in config.blocks]

    if config.global_structure is GlobalStructure.NON_COMPOSITIONAL:
        if len(block_kernels) != 1:
            raise ValueError("NON_COMPOSITIONAL structure requires exactly one block")
        return block_kernels[0]

    if config.global_structure is GlobalStructure.ADDITIVE:
        return _sum_kernels(block_kernels)

    interactions = _build_policy_interactions(config.blocks, policy=config.interaction_policy)
    interactions.extend(config.custom_interactions)
    interaction_kernels = [
        _build_interaction_kernel(interaction, block_lookup=block_lookup, batch_shape=batch_shape)
        for interaction in _dedupe_interactions(interactions)
    ]
    return _sum_kernels(block_kernels + interaction_kernels)


def build_mean_module(config: MeanModuleConfig) -> Mean:
    """Build a mean module for ``MultiTaskGP``.

    Args:
        config: Mean configuration.

    Returns:
        A ``gpytorch.means.Mean`` instance.
    """

    if config.kind == MeanKind.MULTITASK_CONSTANT:
        if config.num_tasks is None:
            raise ValueError("num_tasks is required for MULTITASK_CONSTANT")
        return MultitaskMean(ConstantMean(), num_tasks=config.num_tasks)

    if config.kind == MeanKind.MULTITASK_ZERO:
        if config.num_tasks is None:
            raise ValueError("num_tasks is required for MULTITASK_ZERO")
        return MultitaskMean(ZeroMean(), num_tasks=config.num_tasks)

    if config.kind == MeanKind.MULTITASK_LINEAR:
        if config.num_tasks is None or config.input_size is None:
            raise ValueError("num_tasks and input_size are required for MULTITASK_LINEAR")
        return MultitaskMean(LinearMean(config.input_size), num_tasks=config.num_tasks)

    if config.kind == MeanKind.CONSTANT:
        return ConstantMean()

    if config.kind == MeanKind.ZERO:
        return ZeroMean()

    if config.kind == MeanKind.LINEAR:
        if config.input_size is None:
            raise ValueError("input_size is required for LINEAR")
        return LinearMean(config.input_size)

    raise ValueError(f"Unsupported mean kind: {config.kind}")


def default_covar_config_for_non_task_dims(
    non_task_dims: Sequence[int],
    *,
    policy: LengthscalePolicy = LengthscalePolicy.BOTORCH_STANDARD,
) -> CovarModuleConfig:
    """Create a default one-block Matérn covariance over non-task dimensions.

    Args:
        non_task_dims: Non-task feature indices.
        policy: Lengthscale prior and constraint policy.

    Returns:
        CovarModuleConfig with one generic block.
    """

    return CovarModuleConfig(
        blocks=[
            KernelBlockConfig(
                name="features",
                variable_type=KernelBlockRole.GENERIC,
                components=[
                    MaternKernelComponentConfig(
                        dims=list(non_task_dims),
                        ard=True,
                        matern_nu=2.5,
                        use_outputscale=True,
                        lengthscale_policy=LengthscalePolicyConfig(policy=policy),
                    )
                ],
                block_structure=BlockStructure.ADDITIVE,
                use_outputscale=False,
            )
        ]
    )


def build_multitask_gp(
    *,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    task_feature: int,
    covar_config: CovarModuleConfig | None = None,
    mean_config: MeanModuleConfig | None = None,
    rank: int | None = 1,
    min_inferred_noise_level: float | None = None,
    outcome_transform: object | None = None,
    input_transform: object | None = None,
    validate_task_values: bool = True,
) -> MultiTaskGP:
    """Build a BoTorch ``MultiTaskGP`` with configurable mean and covariance.

    Args:
        train_X: Training design matrix including task feature column.
        train_Y: Training targets, shape ``n x 1`` or batch equivalent.
        task_feature: Index of task feature column in ``train_X``.
        covar_config: Optional covariance configuration. If ``None``, uses one
            generic Matérn block over non-task dimensions with BoTorch-standard
            priors.
        mean_config: Optional mean configuration. If ``None``, uses
            ``MULTITASK_CONSTANT``.
        rank: Task covariance rank.
        min_inferred_noise_level: Optional minimum inferred noise floor for the
            multitask likelihood. If ``None``, uses BoTorch defaults.
        outcome_transform: Optional BoTorch outcome transform.
        input_transform: Optional BoTorch input transform.
        validate_task_values: Whether ``MultiTaskGP`` validates task values.

    Returns:
        Configured ``MultiTaskGP`` instance.
    """

    task_feature_idx = task_feature if task_feature >= 0 else train_X.shape[-1] + task_feature
    non_task_dims = [i for i in range(train_X.shape[-1]) if i != task_feature_idx]

    if covar_config is None:
        covar_config = default_covar_config_for_non_task_dims(non_task_dims)
    covar_module = build_covar_module(covar_config, batch_shape=train_X.shape[:-2])

    num_tasks = int(train_X[..., task_feature_idx].to(torch.long).unique().numel())
    if mean_config is None:
        mean_config = MeanModuleConfig(kind=MeanKind.MULTITASK_CONSTANT, num_tasks=num_tasks)
    elif mean_config.num_tasks is None and mean_config.kind.startswith("multitask"):
        mean_config = mean_config.model_copy(update={"num_tasks": num_tasks})
    if mean_config.input_size is None and mean_config.kind in {MeanKind.MULTITASK_LINEAR, MeanKind.LINEAR}:
        mean_config = mean_config.model_copy(update={"input_size": len(non_task_dims)})
    mean_module = build_mean_module(mean_config)

    likelihood: HadamardGaussianLikelihood | None = None
    if min_inferred_noise_level is not None:
        noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
        likelihood = HadamardGaussianLikelihood(
            num_tasks=num_tasks,
            batch_shape=train_X.shape[:-2],
            task_feature_index=task_feature_idx,
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(min_inferred_noise_level, initial_value=noise_prior.mode),
        )

    outcome_t = cast(Any, outcome_transform)
    input_t = cast(Any, input_transform)
    return MultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=task_feature,
        covar_module=covar_module,
        mean_module=mean_module,
        likelihood=likelihood,
        rank=rank,
        outcome_transform=outcome_t,
        input_transform=input_t,
        validate_task_values=validate_task_values,
    )


__all__ = [
    "BlockStructure",
    "CovarModuleConfig",
    "GlobalStructure",
    "InteractionPolicy",
    "KernelBlockConfig",
    "KernelBlockRole",
    "KernelComponentConfig",
    "KernelInteractionConfig",
    "KernelKind",
    "KernelTermConfig",
    "LengthscalePolicy",
    "LengthscalePolicyConfig",
    "LengthscalePriorConfig",
    "LinearKernelComponentConfig",
    "MaternKernelComponentConfig",
    "MeanKind",
    "MeanModuleConfig",
    "PeriodicKernelComponentConfig",
    "PeriodLengthPriorConfig",
    "RBFKernelComponentConfig",
    "RQKernelComponentConfig",
    "build_covar_module",
    "build_mean_module",
    "build_multitask_gp",
    "default_covar_config_for_non_task_dims",
    "make_period_length_prior",
]
