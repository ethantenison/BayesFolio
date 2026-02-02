"""Gaussian Process Kernels"""

# ============================================================================
# Imports & Global Constants
# ============================================================================

from collections.abc import Callable
from enum import StrEnum
from math import exp, log, sqrt
from typing import Annotated, Any, Literal

import torch
from botorch.models.kernels import (
    CategoricalKernel,
    ExponentialDecayKernel,
    InfiniteWidthBNNKernel,
)
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import (
    Kernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    PiecewisePolynomialKernel,
    PolynomialKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
)
from gpytorch.priors import LogNormalPrior
from pydantic import BaseModel, Field


SQRT2 = sqrt(2)
SQRT3 = sqrt(3)


# ============================================================================
# Kernel Type Enums
# ============================================================================


class KernelType(StrEnum):
    """Supported Gaussian Process kernels."""

    # Stationary Kernels
    MATERN = "matern"
    RQ = "rq"
    RBF = "rbf"
    PERIODIC = "periodic"
    CATEGORICAL = "categorical"
    POLYNOMIAL = "polynomial"
    PIECEWISE_POLYNOMIAL = "piecewise_polynomial"

    # Non-Stationary Kernels
    EXPO_DECAY = "expodecay"
    LINEAR = "linear"
    INFINITE_WIDTH_BNN = "infinite_width_bnn"


class KernelVariableType(StrEnum):
    """High-level variable types used for kernel blocks."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"


class BlockStructure(StrEnum):
    """Within-block kernel structure."""

    JOINT = "joint"
    ADDITIVE = "additive"


class GlobalStructure(StrEnum):
    """Across-block kernel structure."""

    ADDITIVE = "additive"
    HIERARCHICAL = "hierarchical"
    NON_COMPOSITIONAL = "non_compositional"  # This is meant to just use a base kernel and no composition


class InteractionPolicy(StrEnum):
    """Strategy for cross-block kernel interactions."""

    NONE = "none"  # Pure additive - no cross-block interactions
    SPARSE = "sparse"  # Domain-informed heuristic interactions (default)
    TEMPORAL_ONLY = "temporal"  # Only temporal × other blocks
    FULL = "full"  # All pairwise interactions (expensive)


# ============================================================================
# Kernel Capability & Policy Tables
# ============================================================================

# Kernels that can be standalone in a kernel block.
ANCHOR_KERNELS_BY_TYPE = {
    KernelVariableType.CONTINUOUS: {
        KernelType.MATERN,
        KernelType.RQ,
    },
    KernelVariableType.TEMPORAL: {
        KernelType.MATERN,
        KernelType.EXPO_DECAY,
        KernelType.PERIODIC,
    },
    KernelVariableType.CATEGORICAL: {
        KernelType.CATEGORICAL,
    },
}

# Kernels that can be added as modifiers for specific kernel blocks.
MODIFIER_KERNELS_BY_TYPE = {
    KernelVariableType.CONTINUOUS: {
        KernelType.LINEAR,
        KernelType.PERIODIC,
    },
}


# ============================================================================
# Prior & Constraint Utilities
# ============================================================================


def make_period_length_prior(p0: float, cv: float = 0.5) -> LogNormalPrior:
    """Create a log-normal prior for periodic kernel period length."""
    sigma = sqrt(log(1 + cv**2))
    mu = log(p0)
    return LogNormalPrior(loc=mu, scale=sigma)


def default_period_prior(cv: float = 1.5) -> LogNormalPrior:
    """Default weak prior encouraging ~3–4 cycles on [0, 1]."""
    return make_period_length_prior(p0=0.25, cv=cv)


def make_period_length_prior2(p0: float, cv: float = 0.5) -> LogNormalPrior:
    sigma = sqrt(log(1 + cv**2))
    mu = log(p0)
    return LogNormalPrior(loc=mu, scale=sigma)


def make_dim_scaled_lengthscale_prior_and_constraint(
    ard_num_dims: int,
) -> tuple[LogNormalPrior, GreaterThan]:
    """Create dimension-scaled lengthscale prior and constraint."""
    ls_min = 2.5e-2 * sqrt(ard_num_dims)
    lengthscale_prior = LogNormalPrior(
        loc=SQRT2 + 0.5 * log(ard_num_dims),
        scale=SQRT3,
    )
    lengthscale_constraint = GreaterThan(
        ls_min,
        transform=None,
        initial_value=lengthscale_prior.mode,
    )
    return lengthscale_prior, lengthscale_constraint


# ============================================================================
# Kernel Configuration Schemas
# ============================================================================


class BaseKernelConfig(BaseModel):
    """Base class for all kernel configs."""

    kernel_type: KernelType
    ard: bool


class MaternKernelConfig(BaseKernelConfig):
    kernel_type: Literal[KernelType.MATERN]
    ard: bool = True
    nu: Literal[0.5, 1.5, 2.5] = 2.5


class RQKernelConfig(BaseKernelConfig):
    kernel_type: Literal[KernelType.RQ]
    ard: bool = True


class RBFKernelConfig(BaseKernelConfig):
    kernel_type: Literal[KernelType.RBF]
    ard: bool = True


class LinearKernelConfig(BaseKernelConfig):
    kernel_type: Literal[KernelType.LINEAR]
    ard: bool = False


class PeriodicKernelConfig(BaseKernelConfig):
    kernel_type: Literal[KernelType.PERIODIC]
    ard: bool = True
    period_prior_cv: float = 1.5
    period_median: float = 0.25


class PolynomialKernelConfig(BaseKernelConfig):
    kernel_type: Literal[KernelType.POLYNOMIAL]
    ard: bool = False
    power: int = 2


class PiecewisePolynomialKernelConfig(BaseKernelConfig):
    kernel_type: Literal[KernelType.PIECEWISE_POLYNOMIAL]
    ard: bool = True
    q: int = 2


class CategoricalKernelConfig(BaseKernelConfig):
    kernel_type: Literal[KernelType.CATEGORICAL]
    ard: bool = True


class ExpoDecayKernelConfig(BaseKernelConfig):
    kernel_type: Literal[KernelType.EXPO_DECAY]
    ard: bool = False


class InfiniteBNNKernelConfig(BaseKernelConfig):
    kernel_type: Literal[KernelType.INFINITE_WIDTH_BNN]
    depth: int = 2


# ============================================================================
# Low-level Kernel Builder Utilities
# ============================================================================


def _resolve_ard_num_dims(dims: list[int], ard: bool) -> int:
    """Resolve number of lengthscale parameters."""
    return len(dims) if ard else 1


# ============================================================================
# Kernel Builder Implementations
# ============================================================================


def build_matern_kernel(dims, cfg, *, batch_shape):
    ard_num_dims = _resolve_ard_num_dims(dims, cfg.ard)
    ls_prior, ls_constraint = make_dim_scaled_lengthscale_prior_and_constraint(ard_num_dims)
    return MaternKernel(
        nu=cfg.nu,
        ard_num_dims=ard_num_dims,
        active_dims=dims,
        batch_shape=batch_shape,
        lengthscale_prior=ls_prior,
        lengthscale_constraint=ls_constraint,
    )


def build_rbf_kernel(dims, cfg, *, batch_shape):
    ard_num_dims = _resolve_ard_num_dims(dims, cfg.ard)
    ls_prior, ls_constraint = make_dim_scaled_lengthscale_prior_and_constraint(ard_num_dims)
    return RBFKernel(
        ard_num_dims=ard_num_dims,
        active_dims=dims,
        batch_shape=batch_shape,
        lengthscale_prior=ls_prior,
        lengthscale_constraint=ls_constraint,
    )


def build_rq_kernel(dims, cfg, *, batch_shape):
    ard_num_dims = _resolve_ard_num_dims(dims, cfg.ard)
    ls_prior, ls_constraint = make_dim_scaled_lengthscale_prior_and_constraint(ard_num_dims)
    return RQKernel(
        ard_num_dims=ard_num_dims,
        active_dims=dims,
        batch_shape=batch_shape,
        lengthscale_prior=ls_prior,
        lengthscale_constraint=ls_constraint,
    )


def build_linear_kernel(dims, cfg, *, batch_shape):
    return LinearKernel(
        active_dims=dims,
        batch_shape=batch_shape,
        ard_num_dims=None,
    )


def build_periodic_kernel(dims, cfg, *, batch_shape):
    ard_num_dims = _resolve_ard_num_dims(dims, cfg.ard)
    ls_prior, ls_constraint = make_dim_scaled_lengthscale_prior_and_constraint(ard_num_dims)
    period_prior = make_period_length_prior(cfg.period_median, cfg.period_prior_cv)
    return PeriodicKernel(
        ard_num_dims=ard_num_dims,
        active_dims=dims,
        batch_shape=batch_shape,
        period_length=cfg.period_median,
        period_length_prior=period_prior,
        lengthscale_prior=ls_prior,
        lengthscale_constraint=ls_constraint,
    )


def build_polynomial_kernel(dims, cfg, *, batch_shape):
    ard_num_dims = _resolve_ard_num_dims(dims, cfg.ard)
    ls_prior, ls_constraint = make_dim_scaled_lengthscale_prior_and_constraint(ard_num_dims)
    return PolynomialKernel(
        power=cfg.power,
        active_dims=dims,
        ard_num_dims=ard_num_dims,
        batch_shape=batch_shape,
        offset_prior=None,
        offset_constraint=None,
        lengthscale_prior=ls_prior,
        lengthscale_constraint=ls_constraint,
    )


def build_piecewise_polynomial_kernel(dims, cfg, *, batch_shape):
    ard_num_dims = _resolve_ard_num_dims(dims, cfg.ard)
    ls_prior, ls_constraint = make_dim_scaled_lengthscale_prior_and_constraint(ard_num_dims)
    return PiecewisePolynomialKernel(
        q=cfg.q,
        ard_num_dims=ard_num_dims,
        active_dims=dims,
        batch_shape=batch_shape,
        lengthscale_prior=ls_prior,
        lengthscale_constraint=ls_constraint,
    )


def build_categorical_kernel(dims, cfg, *, batch_shape):
    ard_num_dims = _resolve_ard_num_dims(dims, cfg.ard)
    return CategoricalKernel(
        ard_num_dims=ard_num_dims,
        active_dims=dims,
        batch_shape=batch_shape,
        lengthscale_constraint=GreaterThan(1e-6),
    )


def build_expo_decay_kernel(dims, cfg, *, batch_shape):
    if len(dims) != 1:
        raise ValueError("EXPO_DECAY temporal kernel requires exactly one dimension.")
    ard_num_dims = _resolve_ard_num_dims(dims, cfg.ard)
    ls_prior_1d = LogNormalPrior(loc=log(0.25), scale=0.35)
    ls_constraint_1d = GreaterThan(0.05, initial_value=exp(log(0.25) - 0.35**2))
    power_prior = LogNormalPrior(loc=0.0, scale=0.25)
    offset_prior = LogNormalPrior(loc=-3.0, scale=0.35)
    offset_constraint = GreaterThan(1e-6, initial_value=exp(-3.0 - 0.35**2))
    return ExponentialDecayKernel(
        ard_num_dims=ard_num_dims,
        active_dims=dims,
        batch_shape=batch_shape,
        lengthscale_prior=ls_prior_1d,
        lengthscale_constraint=ls_constraint_1d,
        power_prior=power_prior,
        offset_prior=offset_prior,
        offset_constraint=offset_constraint,
    )


def build_infinite_width_bnn_kernel(dims, cfg, *, batch_shape):
    return InfiniteWidthBNNKernel(
        active_dims=dims,
        batch_shape=batch_shape,
        depth=cfg.depth,
    )


# ============================================================================
# Kernel Builder Registry
# ============================================================================

KERNEL_BUILDERS: dict[
    KernelType,
    Callable[[list[int], BaseKernelConfig, torch.Size], Kernel],
] = {
    KernelType.MATERN: lambda d, c, b: build_matern_kernel(d, c, batch_shape=b),
    KernelType.RQ: lambda d, c, b: build_rq_kernel(d, c, batch_shape=b),
    KernelType.RBF: lambda d, c, b: build_rbf_kernel(d, c, batch_shape=b),
    KernelType.LINEAR: lambda d, c, b: build_linear_kernel(d, c, batch_shape=b),
    KernelType.PERIODIC: lambda d, c, b: build_periodic_kernel(d, c, batch_shape=b),
    KernelType.CATEGORICAL: lambda d, c, b: build_categorical_kernel(d, c, batch_shape=b),
    KernelType.EXPO_DECAY: lambda d, c, b: build_expo_decay_kernel(d, c, batch_shape=b),
}


# ============================================================================
# Kernel Block & Architecture Schemas
# ============================================================================

KernelConfig = Annotated[
    (
        MaternKernelConfig
        | RQKernelConfig
        | RBFKernelConfig
        | LinearKernelConfig
        | PeriodicKernelConfig
        | ExpoDecayKernelConfig
        | CategoricalKernelConfig
        | InfiniteBNNKernelConfig
    ),
    Field(discriminator="kernel_type"),
]


class KernelBlockConfig(BaseModel):
    variable_type: KernelVariableType
    dims: list[int]
    block_structure: BlockStructure
    base_kernel: KernelConfig


class KernelArchitectureConfig(BaseModel):
    blocks: list[KernelBlockConfig]
    global_structure: GlobalStructure
    interaction_policy: InteractionPolicy = InteractionPolicy.SPARSE
    



# ============================================================================
# Kernel Composition Utilities
# ============================================================================


def _get_base_kernel_builder(kernel_type: KernelType):
    try:
        return KERNEL_BUILDERS[kernel_type]
    except KeyError as exc:
        raise ValueError(f"No kernel builder registered for {kernel_type}") from exc


def _sum_kernels(kernels: list[Kernel]) -> Kernel:
    if not kernels:
        raise ValueError("Cannot sum an empty kernel list.")
    out = kernels[0]
    for k in kernels[1:]:
        out = out + k
    return out


# ============================================================================
# Block-level Kernel Construction
# ============================================================================


def build_block_kernel(block: KernelBlockConfig, *, batch_shape: torch.Size) -> Kernel:
    builder = _get_base_kernel_builder(block.base_kernel.kernel_type)

    if block.block_structure is BlockStructure.JOINT:
        return ScaleKernel(
            builder(block.dims, block.base_kernel, batch_shape),
            batch_shape=batch_shape,
        )

    components = [
        ScaleKernel(
            builder([dim], block.base_kernel, batch_shape),
            batch_shape=batch_shape,
        )
        for dim in block.dims
    ]
    return _sum_kernels(components)


def _block_interaction_components(block: KernelBlockConfig, *, batch_shape: torch.Size) -> list[Kernel]:
    builder = _get_base_kernel_builder(block.base_kernel.kernel_type)
    if block.block_structure is BlockStructure.JOINT:
        return [builder(block.dims, block.base_kernel, batch_shape)]
    return [builder([d], block.base_kernel, batch_shape) for d in block.dims]


def _within_block_interactions(block: KernelBlockConfig, *, batch_shape: torch.Size) -> list[Kernel]:
    if block.block_structure is not BlockStructure.ADDITIVE:
        return []

    builder = _get_base_kernel_builder(block.base_kernel.kernel_type)
    components = [builder([d], block.base_kernel, batch_shape) for d in block.dims]

    interactions: list[Kernel] = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            interactions.append(ScaleKernel(components[i] * components[j], batch_shape=batch_shape))
    return interactions


# ============================================================================
# Architecture-level Kernel Construction
# ============================================================================


def _default_sparse_pairs(blocks: list[KernelBlockConfig]) -> list[tuple[int, int]]:
    """Domain-informed sparse interaction heuristic.

    Returns pairs of block indices that should have interaction terms.

    Heuristic rules:
    - Temporal variables interact with everything (time modulates all effects)
    - Continuous × Categorical (common in reaction optimization)
    - Chemical × Continuous (concentration/loading effects)
    - Avoids: Chemical × Categorical (usually independent), Chemical × Chemical
    """
    pairs = []
    temporal_idx = [i for i, b in enumerate(blocks) if b.variable_type == KernelVariableType.TEMPORAL]
    continuous_idx = [i for i, b in enumerate(blocks) if b.variable_type == KernelVariableType.CONTINUOUS]
    categorical_idx = [i for i, b in enumerate(blocks) if b.variable_type == KernelVariableType.CATEGORICAL]

    # Temporal interacts with everything (time modulates all effects)
    for t_idx in temporal_idx:
        for other in range(len(blocks)):
            if other != t_idx:
                pairs.append((min(t_idx, other), max(t_idx, other)))

    # Continuous × Categorical (important for reaction optimization)
    for c_idx in continuous_idx:
        for cat_idx in categorical_idx:
            pairs.append((min(c_idx, cat_idx), max(c_idx, cat_idx)))

    return list(set(pairs))  # Remove duplicates


def build_block_interactions(
    blocks: list[KernelBlockConfig],
    *,
    batch_shape: torch.Size,
    policy: InteractionPolicy = InteractionPolicy.SPARSE,
) -> list[Kernel]:
    """Build cross-block interaction kernels based on interaction policy.

    Args:
        blocks: List of kernel block configurations
        batch_shape: Batch shape for kernels
        policy: Interaction policy controlling which cross-block terms to include

    Returns:
        List of interaction kernel terms
    """
    if len(blocks) == 1:
        return _within_block_interactions(blocks[0], batch_shape=batch_shape)

    # Determine which block pairs should have interactions
    if policy == InteractionPolicy.NONE:
        return []

    elif policy == InteractionPolicy.TEMPORAL_ONLY:
        # Only interact temporal with other blocks
        temporal_idx = [i for i, b in enumerate(blocks) if b.variable_type == KernelVariableType.TEMPORAL]
        if not temporal_idx:
            return []  # No temporal variable, no interactions
        pairs = [(temporal_idx[0], j) for j in range(len(blocks)) if j != temporal_idx[0]]

    elif policy == InteractionPolicy.SPARSE:
        pairs = _default_sparse_pairs(blocks)

    elif policy == InteractionPolicy.FULL:
        # All pairwise interactions (original behavior)
        pairs = [(i, j) for i in range(len(blocks)) for j in range(i + 1, len(blocks))]

    else:
        raise ValueError(f"Unknown interaction policy: {policy}")

    # Cache kernel components to avoid redundant instantiation
    block_components_cache = [_block_interaction_components(block, batch_shape=batch_shape) for block in blocks]

    # Build interaction kernels for selected pairs
    interactions: list[Kernel] = []
    for i, j in pairs:
        for ki in block_components_cache[i]:
            for kj in block_components_cache[j]:
                interactions.append(ScaleKernel(ki * kj, batch_shape=batch_shape))

    return interactions


def build_kernel(cfg: KernelArchitectureConfig, *, batch_shape: torch.Size) -> Kernel:
    block_kernels = [build_block_kernel(b, batch_shape=batch_shape) for b in cfg.blocks]

    if cfg.global_structure is GlobalStructure.ADDITIVE:
        return _sum_kernels(block_kernels)

    if cfg.global_structure is GlobalStructure.NON_COMPOSITIONAL:
        if len(block_kernels) != 1:
            raise ValueError("Non-compositional structure requires exactly one block.")
        return block_kernels[0]

    interactions = build_block_interactions(
        cfg.blocks,
        batch_shape=batch_shape,
        policy=cfg.interaction_policy,
    )
    return _sum_kernels(block_kernels + interactions)


# ============================================================================
# Default Architecture Construction
# ============================================================================

DEFAULT_BASE_KERNEL_BY_TYPE = {
    KernelVariableType.CONTINUOUS: KernelType.MATERN,
    KernelVariableType.CATEGORICAL: KernelType.CATEGORICAL,
    KernelVariableType.TEMPORAL: KernelType.EXPO_DECAY,
}


def make_default_kernel_architecture(
    *,
    block_spec: list[dict[str, Any]],
) -> KernelArchitectureConfig:
    blocks: list[KernelBlockConfig] = []

    for spec in block_spec:
        var_type = spec["variable_type"]
        dims = spec["dims"]
        kernel_type = DEFAULT_BASE_KERNEL_BY_TYPE[var_type]

        if kernel_type is KernelType.MATERN:
            base_kernel = MaternKernelConfig(kernel_type=kernel_type, ard=True, nu=2.5)
        elif kernel_type is KernelType.RQ:
            base_kernel = RQKernelConfig(kernel_type=kernel_type, ard=True)
        elif kernel_type is KernelType.CATEGORICAL:
            base_kernel = CategoricalKernelConfig(kernel_type=kernel_type, ard=True)
        elif kernel_type is KernelType.EXPO_DECAY:
            base_kernel = ExpoDecayKernelConfig(kernel_type=kernel_type, ard=False)
        else:
            raise ValueError(f"No default kernel defined for {kernel_type}")

        blocks.append(
            KernelBlockConfig(
                variable_type=var_type,
                dims=dims,
                block_structure=BlockStructure.JOINT,
                base_kernel=base_kernel,
            )
        )

    return KernelArchitectureConfig(
        blocks=blocks,
        global_structure=GlobalStructure.ADDITIVE,
        interaction_policy=InteractionPolicy.SPARSE,  # Use sparse by default for efficiency
    )
