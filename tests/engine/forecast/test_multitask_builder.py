from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import MultitaskMean

from bayesfolio.engine.forecast.gp.multitask_builder import (
    CovarModuleConfig,
    KernelComponentConfig,
    KernelKind,
    KernelTermConfig,
    LengthscalePolicy,
    LengthscalePolicyConfig,
    LengthscalePriorConfig,
    MeanKind,
    MeanModuleConfig,
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
                    KernelComponentConfig(
                        kind=KernelKind.MATERN,
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
                    KernelComponentConfig(
                        kind=KernelKind.RBF,
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
                    KernelComponentConfig(kind=KernelKind.MATERN, dims=[0]),
                    KernelComponentConfig(kind=KernelKind.LINEAR, dims=[1], use_outputscale=False),
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
