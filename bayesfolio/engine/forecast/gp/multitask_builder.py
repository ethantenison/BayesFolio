"""Configuration-driven builders for BoTorch MultiTaskGP modules.

This module provides a small configuration layer for constructing ``MultiTaskGP``
mean and covariance modules with explicit prior and constraint controls.

Boundary responsibility:
- Builds ``gpytorch`` mean / kernel modules and optionally instantiates a
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
from typing import Any, cast

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


class LengthscalePriorConfig(BaseModel):
    """Manual LogNormal prior / lower-bound configuration for lengthscale.

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

    @model_validator(mode="after")
    def _validate_manual_for_policy(self) -> LengthscalePolicyConfig:
        if self.policy == LengthscalePolicy.MANUAL_LOGNORMAL and self.manual is None:
            raise ValueError("manual must be set when policy is MANUAL_LOGNORMAL")
        return self


class PeriodLengthPriorConfig(BaseModel):
    """Adaptive period-length prior for periodic kernels.

    Attributes:
        p0: Prior median in normalized feature units.
        cv: Coefficient of variation controlling prior dispersion.
    """

    p0: float
    cv: float = 0.5

    model_config = ConfigDict(extra="forbid")


class KernelComponentConfig(BaseModel):
    """Single kernel component with explicit feature dimensions.

    Attributes:
        kind: Base kernel type.
        dims: Active feature indices for this component, excluding task column.
        ard: If True, use ARD over ``dims`` (where supported).
        matern_nu: Smoothness for Matern kernels.
        use_outputscale: If True, wrap this component in ``ScaleKernel``.
        lengthscale_policy: Lengthscale prior/constraint policy.
        period_prior: Optional period prior for periodic kernels.
    """

    kind: KernelKind
    dims: list[int]
    ard: bool = True
    matern_nu: float = 2.5
    use_outputscale: bool = True
    lengthscale_policy: LengthscalePolicyConfig = Field(default_factory=LengthscalePolicyConfig)
    period_prior: PeriodLengthPriorConfig | None = None

    model_config = ConfigDict(extra="forbid")


class KernelTermConfig(BaseModel):
    """A product term made from one or more kernel components.

    Attributes:
        components: Components multiplied together in order.
        use_outputscale: If True, wrap final product in ``ScaleKernel``.
    """

    components: list[KernelComponentConfig]
    use_outputscale: bool = False

    model_config = ConfigDict(extra="forbid")


class CovarModuleConfig(BaseModel):
    """Covariance architecture as an additive sum of product terms.

    Attributes:
        terms: Additive kernel terms. A term with one component is a plain
            component; a term with multiple components builds a product kernel.
    """

    terms: list[KernelTermConfig]

    model_config = ConfigDict(extra="forbid")


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


def _build_kernel_component(component: KernelComponentConfig, batch_shape: torch.Size = torch.Size()) -> Kernel:
    ard_num_dims = len(component.dims) if component.ard else 1

    if component.kind == KernelKind.LINEAR:
        kernel: Kernel = LinearKernel(active_dims=tuple(component.dims), ard_num_dims=None, batch_shape=batch_shape)
        return ScaleKernel(kernel, batch_shape=batch_shape) if component.use_outputscale else kernel

    prior, constraint = _resolve_lengthscale_prior_and_constraint(
        policy=component.lengthscale_policy,
        ard_num_dims=ard_num_dims,
    )

    if component.kind == KernelKind.MATERN:
        kernel = MaternKernel(
            nu=component.matern_nu,
            ard_num_dims=ard_num_dims,
            active_dims=tuple(component.dims),
            batch_shape=batch_shape,
            lengthscale_prior=prior,
            lengthscale_constraint=constraint,
        )
    elif component.kind == KernelKind.RBF:
        kernel = RBFKernel(
            ard_num_dims=ard_num_dims,
            active_dims=tuple(component.dims),
            batch_shape=batch_shape,
            lengthscale_prior=prior,
            lengthscale_constraint=constraint,
        )
    elif component.kind == KernelKind.RQ:
        kernel = RQKernel(
            ard_num_dims=ard_num_dims,
            active_dims=tuple(component.dims),
            batch_shape=batch_shape,
            lengthscale_prior=prior,
            lengthscale_constraint=constraint,
        )
    elif component.kind == KernelKind.PERIODIC:
        period_prior = make_period_length_prior(
            component.period_prior.p0 if component.period_prior is not None else 0.25,
            component.period_prior.cv if component.period_prior is not None else 0.5,
        )
        kernel = PeriodicKernel(
            ard_num_dims=ard_num_dims,
            active_dims=tuple(component.dims),
            batch_shape=batch_shape,
            period_length_prior=period_prior,
            period_length=period_prior.median,
            lengthscale_prior=prior,
            lengthscale_constraint=constraint,
        )
    else:
        raise ValueError(f"Unsupported kernel kind: {component.kind}")

    return ScaleKernel(kernel, batch_shape=batch_shape) if component.use_outputscale else kernel


def build_covar_module(config: CovarModuleConfig, batch_shape: torch.Size = torch.Size()) -> Kernel:
    """Build a covariance module from additive-product kernel terms.

    Args:
        config: Covariance configuration.
        batch_shape: Optional GPyTorch batch shape.

    Returns:
        A ``gpytorch.kernels.Kernel`` usable as ``covar_module`` in
        ``MultiTaskGP``.
    """

    if not config.terms:
        raise ValueError("config.terms cannot be empty")

    additive_terms: list[Kernel] = []
    for term in config.terms:
        if not term.components:
            raise ValueError("each term must have at least one component")

        built_components = [_build_kernel_component(c, batch_shape=batch_shape) for c in term.components]
        product: Kernel = built_components[0]
        for next_kernel in built_components[1:]:
            product = ProductKernel(product, next_kernel)
        if term.use_outputscale:
            product = ScaleKernel(product, batch_shape=batch_shape)
        additive_terms.append(product)

    kernel_sum = additive_terms[0]
    for next_term in additive_terms[1:]:
        kernel_sum = kernel_sum + next_term
    return kernel_sum


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
    """Create a default one-term Matern covariance over non-task dimensions.

    Args:
        non_task_dims: Non-task feature indices.
        policy: Lengthscale prior/constraint policy.

    Returns:
        CovarModuleConfig with one Matern component.
    """

    return CovarModuleConfig(
        terms=[
            KernelTermConfig(
                components=[
                    KernelComponentConfig(
                        kind=KernelKind.MATERN,
                        dims=list(non_task_dims),
                        ard=True,
                        matern_nu=2.5,
                        use_outputscale=True,
                        lengthscale_policy=LengthscalePolicyConfig(policy=policy),
                    )
                ]
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
    outcome_transform: object | None = None,
    input_transform: object | None = None,
    validate_task_values: bool = True,
) -> MultiTaskGP:
    """Build a BoTorch ``MultiTaskGP`` with configurable mean and covariance.

    Args:
        train_X: Training design matrix including task feature column.
        train_Y: Training targets, shape ``n x 1`` (or batch equivalent).
        task_feature: Index of task feature column in ``train_X``.
        covar_config: Optional covariance configuration. If ``None``, uses one
            Matern term over non-task dimensions with BoTorch-standard priors.
        mean_config: Optional mean configuration. If ``None``, uses
            ``MULTITASK_CONSTANT``.
        rank: Task covariance rank.
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
    outcome_t = cast(Any, outcome_transform)
    input_t = cast(Any, input_transform)
    return MultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        task_feature=task_feature,
        covar_module=covar_module,
        mean_module=mean_module,
        rank=rank,
        outcome_transform=outcome_t,
        input_transform=input_t,
        validate_task_values=validate_task_values,
    )


__all__ = [
    "CovarModuleConfig",
    "KernelComponentConfig",
    "KernelKind",
    "KernelTermConfig",
    "LengthscalePolicy",
    "LengthscalePolicyConfig",
    "LengthscalePriorConfig",
    "MeanKind",
    "MeanModuleConfig",
    "PeriodLengthPriorConfig",
    "build_covar_module",
    "build_mean_module",
    "build_multitask_gp",
    "default_covar_config_for_non_task_dims",
    "make_period_length_prior",
]
