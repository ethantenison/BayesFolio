from __future__ import annotations

from math import log
from typing import Any, cast

import pytest
import torch
from gpytorch.kernels import MaternKernel, PeriodicKernel, ScaleKernel
from gpytorch.means import MultitaskMean
from pydantic import ValidationError

from bayesfolio.engine.forecast.gp.multitask_builder import (
    CovarModuleConfig,
    KernelTermConfig,
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
    build_covar_module,
    build_multitask_gp,
)


def _unwrap_matern_from_scaled(kernel: ScaleKernel) -> MaternKernel:
    assert isinstance(kernel.base_kernel, MaternKernel)
    return kernel.base_kernel


def test_build_covar_module_legacy_policy_scales_min_constraint() -> None:
    config = CovarModuleConfig(
        terms=[
            KernelTermConfig(
                components=[
                    MaternKernelComponentConfig(
                        dims=[0, 1, 2, 3],
                        lengthscale_policy=LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE),
                    )
                ]
            )
        ]
    )

    kernel = build_covar_module(config)
    assert isinstance(kernel, ScaleKernel)

    matern = _unwrap_matern_from_scaled(kernel)
    lower_bound = float(cast(Any, matern).raw_lengthscale_constraint.lower_bound)
    expected = 2.5e-2 * (4**0.5)
    assert lower_bound == pytest.approx(expected)


def test_build_covar_module_manual_policy_uses_custom_prior() -> None:
    config = CovarModuleConfig(
        terms=[
            KernelTermConfig(
                components=[
                    RBFKernelComponentConfig(
                        dims=[0, 1],
                        lengthscale_policy=LengthscalePolicyConfig(
                            policy=LengthscalePolicy.MANUAL_LOGNORMAL,
                            manual=LengthscalePriorConfig(
                                loc=-0.8,
                                scale=0.35,
                                min_lengthscale=0.11,
                                initial_value=0.42,
                            ),
                        ),
                    )
                ]
            )
        ]
    )

    kernel = build_covar_module(config)
    base = cast(Any, kernel.base_kernel)

    assert float(base.raw_lengthscale_constraint.lower_bound) == pytest.approx(0.11)
    transformed = base.raw_lengthscale_constraint.transform(base.raw_lengthscale).mean().detach()
    assert float(transformed) == pytest.approx(0.42)


def test_rbf_component_rejects_matern_nu_field() -> None:
    with pytest.raises(ValidationError):
        RBFKernelComponentConfig(dims=[0, 1], matern_nu=2.5)


def test_build_multitask_gp_builds_modules_from_configs() -> None:
    train_x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 1.0, 0.0],
            [0.8, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_y = torch.tensor([[0.10], [0.12], [0.08], [0.11]], dtype=torch.double)

    covar_config = CovarModuleConfig(
        terms=[
            KernelTermConfig(
                components=[
                    MaternKernelComponentConfig(dims=[0]),
                    LinearKernelComponentConfig(dims=[1], use_outputscale=False),
                ],
                use_outputscale=True,
            )
        ]
    )
    mean_config = MeanModuleConfig(kind=MeanKind.MULTITASK_CONSTANT)

    model = build_multitask_gp(
        train_X=train_x,
        train_Y=train_y,
        task_feature=-1,
        covar_config=covar_config,
        mean_config=mean_config,
        rank=1,
    )

    assert isinstance(model.mean_module, MultitaskMean)
    assert model.num_tasks == 2
    assert model.train_inputs is not None
    assert model.train_inputs[0].shape == train_x.shape


def test_build_covar_module_periodic_component_uses_configured_period_length() -> None:
    config = CovarModuleConfig(
        terms=[
            KernelTermConfig(
                components=[
                    PeriodicKernelComponentConfig(
                        dims=[0],
                        period_prior=PeriodLengthPriorConfig(p0=0.33, cv=0.4),
                    )
                ]
            )
        ]
    )

    kernel = build_covar_module(config)
    assert isinstance(kernel, ScaleKernel)
    assert isinstance(kernel.base_kernel, PeriodicKernel)
    period_prior = kernel.base_kernel.period_length_prior
    assert period_prior is not None
    prior_loc = float(period_prior.loc.detach().view(-1)[0])
    assert prior_loc == pytest.approx(log(0.33), rel=1e-6)


def test_build_multitask_gp_applies_custom_noise_floor() -> None:
    train_x = torch.tensor(
        [
            [0.0, 0.0],
            [0.3, 0.0],
            [0.7, 1.0],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    train_y = torch.tensor([[0.10], [0.11], [0.09], [0.12]], dtype=torch.double)

    model = build_multitask_gp(
        train_X=train_x,
        train_Y=train_y,
        task_feature=-1,
        min_inferred_noise_level=5e-3,
    )

    lower_bound = float(cast(Any, model.likelihood.noise_covar).raw_noise_constraint.lower_bound)
    assert lower_bound == pytest.approx(5e-3)
