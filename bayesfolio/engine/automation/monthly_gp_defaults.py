"""Deterministic default multitask GP configuration for monthly automation."""

from __future__ import annotations

from bayesfolio.contracts.commands.monthly_portfolio import MonthlyForecastConfig
from bayesfolio.engine.forecast.gp.multitask_builder import (
    BlockStructure,
    CovarModuleConfig,
    InteractionPolicy,
    KernelBlockConfig,
    KernelBlockRole,
    KernelInteractionConfig,
    LengthscalePolicy,
    LengthscalePolicyConfig,
    LinearKernelComponentConfig,
    MaternKernelComponentConfig,
    MeanKind,
    MeanModuleConfig,
    RQKernelComponentConfig,
)


def build_default_monthly_gp_configs(
    *,
    forecast_config: MonthlyForecastConfig,
    feature_index: dict[str, int],
) -> tuple[CovarModuleConfig, MeanModuleConfig]:
    """Build the fixed GP architecture used in recent monthly notebooks."""

    time_dims = [feature_index[column] for column in forecast_config.time_feature_columns]
    etf_dims = [feature_index[column] for column in forecast_config.etf_feature_columns]
    macro_dims = [feature_index[column] for column in forecast_config.macro_feature_columns]

    covar_config = CovarModuleConfig(
        blocks=[
            KernelBlockConfig(
                name="time",
                variable_type=KernelBlockRole.TIME,
                components=[
                    MaternKernelComponentConfig(
                        dims=time_dims,
                        matern_nu=0.5,
                        ard=True,
                        use_outputscale=True,
                        lengthscale_policy=LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE),
                    )
                ],
                block_structure=BlockStructure.ADDITIVE,
                use_outputscale=False,
            ),
            KernelBlockConfig(
                name="etf",
                variable_type=KernelBlockRole.ETF,
                components=[
                    MaternKernelComponentConfig(
                        dims=etf_dims,
                        matern_nu=0.5,
                        ard=True,
                        use_outputscale=True,
                        lengthscale_policy=LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE),
                    )
                ],
                block_structure=BlockStructure.ADDITIVE,
                use_outputscale=False,
            ),
            KernelBlockConfig(
                name="macro",
                variable_type=KernelBlockRole.MACRO,
                components=[
                    MaternKernelComponentConfig(
                        dims=macro_dims,
                        matern_nu=0.5,
                        ard=True,
                        use_outputscale=True,
                        lengthscale_policy=LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE),
                    ),
                    RQKernelComponentConfig(
                        dims=macro_dims,
                        ard=True,
                        use_outputscale=True,
                        lengthscale_policy=LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE),
                    ),
                    LinearKernelComponentConfig(
                        dims=macro_dims,
                        use_outputscale=True,
                    ),
                ],
                block_structure=BlockStructure.ADDITIVE,
                use_outputscale=False,
            ),
        ],
        global_structure="hierarchical",
        interaction_policy=InteractionPolicy.CUSTOM,
        custom_interactions=[
            KernelInteractionConfig(blocks=["time", "etf"], name="time_x_etf", use_outputscale=True),
            KernelInteractionConfig(blocks=["time", "macro"], name="time_x_macro", use_outputscale=True),
            KernelInteractionConfig(blocks=["macro", "etf"], name="macro_x_etf", use_outputscale=True),
        ],
    )
    mean_config = MeanModuleConfig(kind=MeanKind.MULTITASK_CONSTANT)
    return covar_config, mean_config
