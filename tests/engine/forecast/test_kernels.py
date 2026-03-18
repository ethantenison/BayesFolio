from __future__ import annotations

import torch
from gpytorch.kernels import AdditiveKernel, ProductKernel, ScaleKernel

from bayesfolio.engine.forecast.gp.kernels import (
    BlockStructure,
    GlobalStructure,
    InteractionPolicy,
    KernelArchitectureConfig,
    KernelBlockConfig,
    KernelType,
    KernelVariableType,
    MaternKernelConfig,
    PeriodicKernelConfig,
    RBFKernelConfig,
    build_kernel,
)


def test_build_kernel_routes_supported_architecture_through_active_builder() -> None:
    config = KernelArchitectureConfig(
        blocks=[
            KernelBlockConfig(
                variable_type=KernelVariableType.TEMPORAL,
                dims=[0],
                block_structure=BlockStructure.JOINT,
                base_kernel=PeriodicKernelConfig(kernel_type=KernelType.PERIODIC, ard=False),
            ),
            KernelBlockConfig(
                variable_type=KernelVariableType.CONTINUOUS,
                dims=[1],
                block_structure=BlockStructure.JOINT,
                base_kernel=MaternKernelConfig(kernel_type=KernelType.MATERN, ard=True, nu=0.5),
            ),
            KernelBlockConfig(
                variable_type=KernelVariableType.CONTINUOUS,
                dims=[2],
                block_structure=BlockStructure.JOINT,
                base_kernel=RBFKernelConfig(kernel_type=KernelType.RBF, ard=True),
            ),
        ],
        global_structure=GlobalStructure.HIERARCHICAL,
        interaction_policy=InteractionPolicy.TEMPORAL_ONLY,
    )

    kernel = build_kernel(config, batch_shape=torch.Size())

    assert isinstance(kernel, AdditiveKernel)
    assert len(kernel.kernels) == 5
    product_terms = [
        subkernel.base_kernel
        for subkernel in kernel.kernels
        if isinstance(subkernel, ScaleKernel) and isinstance(subkernel.base_kernel, ProductKernel)
    ]
    assert len(product_terms) == 2
    for product in product_terms:
        assert all(not isinstance(child, ScaleKernel) for child in product.kernels)


def test_build_kernel_preserves_within_block_interactions_for_single_additive_block() -> None:
    config = KernelArchitectureConfig(
        blocks=[
            KernelBlockConfig(
                variable_type=KernelVariableType.CONTINUOUS,
                dims=[0, 1, 2],
                block_structure=BlockStructure.ADDITIVE,
                base_kernel=MaternKernelConfig(kernel_type=KernelType.MATERN, ard=True, nu=2.5),
            )
        ],
        global_structure=GlobalStructure.HIERARCHICAL,
        interaction_policy=InteractionPolicy.NONE,
    )

    kernel = build_kernel(config, batch_shape=torch.Size())

    assert isinstance(kernel, AdditiveKernel)
    assert len(kernel.kernels) == 6
