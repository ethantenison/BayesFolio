"""Run first-round U.S. equity ETF GP/MTGP walk-forward tests.

This is a thin, experiment-scoped wrapper around the June portfolio
walk-forward runner. The source runner is reusable, but its task universe and
feature blocks are module-level constants, so this wrapper patches them before
delegating to the portfolio code.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENT_DIR = Path(__file__).resolve().parent
PORTFOLIO_RUNNER_PATH = REPO_ROOT / "experiments/2026-06-portfolio-optimization/run_monthly_optimization_walkforward.py"
DEFAULT_FEATURE_PATH = EXPERIMENT_DIR / "artifacts/us_equity_first_round_feature_candidates_20260705T145015Z.parquet"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "runs"

FIRST_ROUND_TICKERS = ["SPY", "MGK", "VTV", "IWM"]
TIME_COLS = ["t_index"]
ETF_COLS = [
    "lag_y_excess_lead",
    "lag2_y_excess_lead",
    "mom12m_skip1m",
    "mom6m",
    "cs_mom_rank",
    "trend_slope",
    "vol_1m",
    "vol_ratio_1m_3m",
    "vol_z",
    "vol_accel",
    "max_dd_3m",
    "ill_log",
    "dolvol_log",
    "turnover",
]
MACRO_COLS = [
    "vix",
    "vix_slope",
    "hy_spread_z_12p",
    "hy_spread_chg_1p",
    "spy_ret",
    "pct_above_50dma",
    "cpi_chg_12p",
    "term_spread",
]
INTL_MACRO_DOLLAR_FX = ["dxy", "em_fx_ret"]
INTL_MACRO_GLOBAL_RATES = ["tnote10y", "term_spread", "schp_ret", "y10_real_proxy", "breakeven_proxy"]
INTL_MACRO_COMMODITIES = ["oil_ret", "copper_ret", "gold_crude_ratio"]
INTL_MACRO_RISK_VOL = [
    "vix",
    "vix_slope",
    "vix_ts_chg_1p",
    "vix_ts_z_12p",
    "move_proxy",
    "dealer_gamma_proxy",
]
INTL_MACRO_BREADTH_FLOW = ["spy_ret", "erp", "rsp_spy_roc_1p", "pct_above_50dma", "spy_flow_z_12p"]
INTL_MACRO_CREDIT_INFLATION = [
    "credit_spread_chg_1p",
    "hy_spread_chg_1p",
    "hy_spread_z_12p",
    "cpi_chg_12p",
    "cpi_chg_1p",
]
INTL_MACRO_ALL = [
    *INTL_MACRO_DOLLAR_FX,
    *INTL_MACRO_GLOBAL_RATES,
    *INTL_MACRO_COMMODITIES,
    *INTL_MACRO_RISK_VOL,
    *INTL_MACRO_BREADTH_FLOW,
    *INTL_MACRO_CREDIT_INFLATION,
]
INTL_MACRO_DOLLAR_RATES = [*INTL_MACRO_DOLLAR_FX, *INTL_MACRO_GLOBAL_RATES]
INTL_MACRO_DOLLAR_COMMODITIES = [*INTL_MACRO_DOLLAR_FX, *INTL_MACRO_COMMODITIES]
INTL_MACRO_DOLLAR_RATES_COMMODITIES = [
    *INTL_MACRO_DOLLAR_FX,
    *INTL_MACRO_GLOBAL_RATES,
    *INTL_MACRO_COMMODITIES,
]
INTL_MACRO_RISK_BREADTH = [*INTL_MACRO_RISK_VOL, *INTL_MACRO_BREADTH_FLOW]
INTL_MACRO_EX_CREDIT = [
    *INTL_MACRO_DOLLAR_FX,
    *INTL_MACRO_GLOBAL_RATES,
    *INTL_MACRO_COMMODITIES,
    *INTL_MACRO_RISK_VOL,
    *INTL_MACRO_BREADTH_FLOW,
]
INPUT_COLUMNS = [*TIME_COLS, *ETF_COLS, *MACRO_COLS]
NLL_VARIANCE_FLOOR = 1e-12
NORMAL_COVERAGE_Z = {
    "coverage_50": 0.6744897501960817,
    "coverage_80": 1.2815515655446004,
    "coverage_95": 1.959963984540054,
}
FEATURE_SETS = {
    "target_history": {
        "etf": ["lag_y_excess_lead", "lag2_y_excess_lead"],
        "macro": [],
    },
    "momentum_trend": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": [],
    },
    "volatility_regime": {
        "etf": ["vol_1m", "vol_z", "vol_accel", "vol_ratio_1m_3m", "max_dd_3m"],
        "macro": [],
    },
    "liquidity": {
        "etf": ["ill_log", "dolvol_log", "turnover"],
        "macro": [],
    },
    "macro_risk": {
        "etf": [],
        "macro": [
            "vix",
            "vix_slope",
            "spy_ret",
            "pct_above_50dma",
            "hy_spread_chg_1p",
            "hy_spread_z_12p",
            "cpi_chg_12p",
            "term_spread",
        ],
    },
    "target_plus_momentum": {
        "etf": ["lag_y_excess_lead", "lag2_y_excess_lead", "mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": [],
    },
    "target_plus_volatility": {
        "etf": [
            "lag_y_excess_lead",
            "lag2_y_excess_lead",
            "vol_1m",
            "vol_z",
            "vol_accel",
            "vol_ratio_1m_3m",
            "max_dd_3m",
        ],
        "macro": [],
    },
    "target_plus_macro": {
        "etf": ["lag_y_excess_lead", "lag2_y_excess_lead"],
        "macro": [
            "vix",
            "vix_slope",
            "spy_ret",
            "pct_above_50dma",
            "hy_spread_chg_1p",
            "hy_spread_z_12p",
            "cpi_chg_12p",
            "term_spread",
        ],
    },
    "momentum_plus_volatility": {
        "etf": [
            "mom6m",
            "mom12m_skip1m",
            "cs_mom_rank",
            "trend_slope",
            "vol_1m",
            "vol_z",
            "vol_accel",
            "vol_ratio_1m_3m",
            "max_dd_3m",
        ],
        "macro": [],
    },
    "momentum_plus_macro": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": [
            "vix",
            "vix_slope",
            "spy_ret",
            "pct_above_50dma",
            "hy_spread_chg_1p",
            "hy_spread_z_12p",
            "cpi_chg_12p",
            "term_spread",
        ],
    },
    "momentum_plus_macro_market": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": ["vix", "vix_slope", "spy_ret", "pct_above_50dma"],
    },
    "intl_momentum_plus_dollar_fx": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": INTL_MACRO_DOLLAR_FX,
    },
    "intl_momentum_plus_global_rates": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": INTL_MACRO_GLOBAL_RATES,
    },
    "intl_momentum_plus_commodities": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": INTL_MACRO_COMMODITIES,
    },
    "intl_momentum_plus_risk_vol": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": INTL_MACRO_RISK_VOL,
    },
    "intl_momentum_plus_breadth_flow": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": INTL_MACRO_BREADTH_FLOW,
    },
    "intl_momentum_plus_credit_inflation": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": INTL_MACRO_CREDIT_INFLATION,
    },
    "intl_momentum_plus_dollar_rates": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": INTL_MACRO_DOLLAR_RATES,
    },
    "intl_momentum_plus_dollar_commodities": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": INTL_MACRO_DOLLAR_COMMODITIES,
    },
    "intl_momentum_plus_dollar_rates_commodities": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": INTL_MACRO_DOLLAR_RATES_COMMODITIES,
    },
    "intl_momentum_plus_risk_breadth": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": INTL_MACRO_RISK_BREADTH,
    },
    "intl_momentum_plus_ex_credit": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": INTL_MACRO_EX_CREDIT,
    },
    "intl_momentum_plus_full_macro": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": INTL_MACRO_ALL,
    },
    "intl_compact_expanded": {
        "etf": ETF_COLS,
        "macro": INTL_MACRO_ALL,
    },
    "momentum_plus_macro_credit_inflation": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": ["hy_spread_chg_1p", "hy_spread_z_12p", "cpi_chg_12p", "term_spread"],
    },
    "momentum_plus_macro_plus_volatility": {
        "etf": [
            "mom6m",
            "mom12m_skip1m",
            "cs_mom_rank",
            "trend_slope",
            "vol_1m",
            "vol_z",
            "vol_accel",
            "vol_ratio_1m_3m",
            "max_dd_3m",
        ],
        "macro": [
            "vix",
            "vix_slope",
            "spy_ret",
            "pct_above_50dma",
            "hy_spread_chg_1p",
            "hy_spread_z_12p",
            "cpi_chg_12p",
            "term_spread",
        ],
    },
    "momentum_plus_macro_plus_liquidity": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope", "ill_log", "dolvol_log", "turnover"],
        "macro": [
            "vix",
            "vix_slope",
            "spy_ret",
            "pct_above_50dma",
            "hy_spread_chg_1p",
            "hy_spread_z_12p",
            "cpi_chg_12p",
            "term_spread",
        ],
    },
    "momentum_plus_macro_plus_liquidity_volatility": {
        "etf": [
            "mom6m",
            "mom12m_skip1m",
            "cs_mom_rank",
            "trend_slope",
            "ill_log",
            "dolvol_log",
            "turnover",
            "vol_1m",
            "vol_z",
            "vol_accel",
            "vol_ratio_1m_3m",
            "max_dd_3m",
        ],
        "macro": [
            "vix",
            "vix_slope",
            "spy_ret",
            "pct_above_50dma",
            "hy_spread_chg_1p",
            "hy_spread_z_12p",
            "cpi_chg_12p",
            "term_spread",
        ],
    },
    "liquidity_plus_volatility": {
        "etf": ["ill_log", "dolvol_log", "turnover", "vol_1m", "vol_z", "vol_accel", "vol_ratio_1m_3m", "max_dd_3m"],
        "macro": [],
    },
    "compact_pruned": {
        "etf": ETF_COLS,
        "macro": MACRO_COLS,
    },
}
COVAR_GROUPING_PROPOSALS = [
    "default",
    "split_groups",
    "split_macro_structured_pairwise",
    "split_macro_market_nonlinear_credit_linear",
    "split_macro_credit_shrunk",
    "split_macro_no_etf_credit",
    "split_macro_cross_family_gate",
    "macro_split_market_rq_credit_linear",
    "macro_split_market_rq_linear_credit_matern15",
    "macro_split_market_matern_rq_credit_linear",
    "macro_split_market_matern_credit_rbf_linear",
    "macro_split_market_rbf_credit_matern25_linear",
    "macro_split_market_linear_credit_matern25",
    "macro_split_market_rq_credit_rbf",
    "macro_split_market_matern_credit_rq_linear",
    "macro_split_custom_pending",
    "macro_market_subsplit_custom_pending",
    "single_matern",
    "time_plus_matern_interaction",
]
MACRO_COMPONENT_PROPOSALS = ["default", "matern_linear", "matern_only", "linear_only"]


def load_portfolio_runner() -> Any:
    spec = importlib.util.spec_from_file_location("us_equity_portfolio_runner", PORTFOLIO_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load portfolio runner from {PORTFOLIO_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def patch_first_round_constants(
    runner: Any,
    *,
    mtgp_rank: int,
    feature_set: str,
    tickers: list[str] | None = None,
    custom_etf_cols: list[str] | None = None,
    custom_macro_cols: list[str] | None = None,
    covar_grouping_proposal: str = "default",
    macro_component_proposal: str = "default",
    include_time_index: bool = True,
    include_main_effects: bool = True,
    include_pairwise_interactions: bool = True,
    drop_etf_macro_interaction: bool = False,
    include_three_way_interaction: bool = False,
    include_time_etf_market_three_way: bool = False,
    include_time_etf_credit_three_way: bool = False,
    include_macro_market_credit_interaction: bool = False,
    include_macro_market_credit_etf_three_way: bool = False,
    include_time_macro_market_credit_three_way: bool = False,
    include_time_etf_market_credit_four_way: bool = False,
    macro_market_kernel_id: str | None = None,
    macro_market_ard: bool = True,
    macro_credit_inflation_kernel_id: str | None = None,
    macro_credit_inflation_ard: bool = True,
) -> None:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}. Expected one of: {sorted(FEATURE_SETS)}")
    feature_spec = FEATURE_SETS[feature_set]
    etf_cols = list(custom_etf_cols) if custom_etf_cols is not None else list(feature_spec["etf"])
    macro_cols = list(custom_macro_cols) if custom_macro_cols is not None else list(feature_spec["macro"])
    time_cols = list(TIME_COLS) if include_time_index else []
    family_tickers = list(tickers or FIRST_ROUND_TICKERS)
    task_exp = runner.task_exp
    task_exp.ETF_TICKERS = family_tickers
    task_exp.ASSET_GROUPS = {ticker: "us_equity" for ticker in family_tickers}
    task_exp.TIME_COLS = time_cols
    task_exp.ETF_COLS = etf_cols
    task_exp.MACRO_COLS = macro_cols
    task_exp.INPUT_COLUMNS = [*time_cols, *etf_cols, *macro_cols]
    task_exp.RANK = int(mtgp_rank)
    gp_builder = importlib.import_module("bayesfolio.engine.forecast.gp.multitask_builder")
    task_exp.RBFKernelComponentConfig = gp_builder.RBFKernelComponentConfig
    task_exp.PolynomialKernelComponentConfig = gp_builder.PolynomialKernelComponentConfig
    task_exp.SpectralMixtureKernelComponentConfig = gp_builder.SpectralMixtureKernelComponentConfig
    task_exp.SpectralDeltaKernelComponentConfig = gp_builder.SpectralDeltaKernelComponentConfig
    task_exp.build_covar_config = dynamic_covar_config(
        task_exp,
        time_cols=time_cols,
        etf_cols=etf_cols,
        macro_cols=macro_cols,
        covar_grouping_proposal=covar_grouping_proposal,
        macro_component_proposal=macro_component_proposal,
        include_main_effects=include_main_effects,
        include_pairwise_interactions=include_pairwise_interactions,
        drop_etf_macro_interaction=drop_etf_macro_interaction,
        include_three_way_interaction=include_three_way_interaction,
        include_time_etf_market_three_way=include_time_etf_market_three_way,
        include_time_etf_credit_three_way=include_time_etf_credit_three_way,
        include_macro_market_credit_interaction=include_macro_market_credit_interaction,
        include_macro_market_credit_etf_three_way=include_macro_market_credit_etf_three_way,
        include_time_macro_market_credit_three_way=include_time_macro_market_credit_three_way,
        include_time_etf_market_credit_four_way=include_time_etf_market_credit_four_way,
        macro_market_kernel_id=macro_market_kernel_id,
        macro_market_ard=macro_market_ard,
        macro_credit_inflation_kernel_id=macro_credit_inflation_kernel_id,
        macro_credit_inflation_ard=macro_credit_inflation_ard,
    )

    runner.HELPER_ASSETS = set()
    runner.PERIODS_PER_YEAR = 12
    us_static_weights = {
        "SPY": 0.45,
        "MGK": 0.15,
        "VTV": 0.20,
        "IWM": 0.15,
    }
    if set(family_tickers).issubset(us_static_weights):
        runner.SCHWAB_MODERATE_AGGRESSIVE_TARGET_WEIGHTS = {
            ticker: us_static_weights[ticker] for ticker in family_tickers
        }
    else:
        equal_static_weight = 0.95 / len(family_tickers)
        runner.SCHWAB_MODERATE_AGGRESSIVE_TARGET_WEIGHTS = {
            ticker: equal_static_weight for ticker in family_tickers
        }
    runner.SCHWAB_MODERATE_AGGRESSIVE_CASH_WEIGHT = 0.05


def dynamic_covar_config(
    task_exp: Any,
    *,
    time_cols: list[str],
    etf_cols: list[str],
    macro_cols: list[str],
    covar_grouping_proposal: str = "default",
    macro_component_proposal: str = "default",
    include_main_effects: bool = True,
    include_pairwise_interactions: bool = True,
    drop_etf_macro_interaction: bool = False,
    include_three_way_interaction: bool = False,
    include_time_etf_market_three_way: bool = False,
    include_time_etf_credit_three_way: bool = False,
    include_macro_market_credit_interaction: bool = False,
    include_macro_market_credit_etf_three_way: bool = False,
    include_time_macro_market_credit_three_way: bool = False,
    include_time_etf_market_credit_four_way: bool = False,
    macro_market_kernel_id: str | None = None,
    macro_market_ard: bool = True,
    macro_credit_inflation_kernel_id: str | None = None,
    macro_credit_inflation_ard: bool = True,
) -> Any:
    def build() -> Any:
        if covar_grouping_proposal not in COVAR_GROUPING_PROPOSALS:
            raise ValueError(
                f"Unknown covariance grouping proposal {covar_grouping_proposal!r}. "
                f"Expected one of: {COVAR_GROUPING_PROPOSALS}"
            )
        if macro_component_proposal not in MACRO_COMPONENT_PROPOSALS:
            raise ValueError(
                f"Unknown macro component proposal {macro_component_proposal!r}. "
                f"Expected one of: {MACRO_COMPONENT_PROPOSALS}"
            )
        idx = {name: i for i, name in enumerate([*time_cols, *etf_cols, *macro_cols])}
        adaptive = task_exp.LengthscalePolicyConfig(policy=task_exp.LengthscalePolicy.ADAPTIVE)
        blocks = []

        def matern_block(
            name: str,
            cols: list[str],
            role: Any,
            *,
            include_as_main_effect: bool = include_main_effects,
        ) -> Any | None:
            active_cols = [col for col in cols if col in idx]
            if not active_cols:
                return None
            return task_exp.KernelBlockConfig(
                name=name,
                variable_type=role,
                components=[
                    task_exp.MaternKernelComponentConfig(
                        dims=[idx[c] for c in active_cols],
                        matern_nu=0.5,
                        ard=True,
                        use_outputscale=True,
                        lengthscale_policy=adaptive,
                    )
                ],
                block_structure=task_exp.BlockStructure.ADDITIVE,
                use_outputscale=False,
                include_as_main_effect=include_as_main_effect,
            )

        def market_risk_block(
            name: str = "macro_market",
            *,
            include_as_main_effect: bool = include_main_effects,
        ) -> Any | None:
            active_cols = [col for col in ["vix", "vix_slope", "spy_ret", "pct_above_50dma"] if col in idx]
            if not active_cols:
                return None
            dims = [idx[c] for c in active_cols]
            return task_exp.KernelBlockConfig(
                name=name,
                variable_type=task_exp.KernelBlockRole.MACRO,
                components=[
                    task_exp.MaternKernelComponentConfig(
                        dims=dims,
                        matern_nu=0.5,
                        ard=True,
                        use_outputscale=True,
                        lengthscale_policy=adaptive,
                    ),
                    task_exp.LinearKernelComponentConfig(dims=dims, use_outputscale=True),
                ],
                block_structure=task_exp.BlockStructure.ADDITIVE,
                use_outputscale=False,
                include_as_main_effect=include_as_main_effect,
            )

        def credit_rates_block(
            name: str = "macro_credit_inflation",
            *,
            include_matern: bool = True,
            matern_nu: float = 1.5,
            include_as_main_effect: bool = include_main_effects,
        ) -> Any | None:
            active_cols = [
                col
                for col in ["hy_spread_chg_1p", "hy_spread_z_12p", "cpi_chg_12p", "term_spread"]
                if col in idx
            ]
            if not active_cols:
                return None
            dims = [idx[c] for c in active_cols]
            components = []
            if include_matern:
                components.append(
                    task_exp.MaternKernelComponentConfig(
                        dims=dims,
                        matern_nu=matern_nu,
                        ard=True,
                        use_outputscale=True,
                        lengthscale_policy=adaptive,
                    )
                )
            components.append(task_exp.LinearKernelComponentConfig(dims=dims, use_outputscale=True))
            return task_exp.KernelBlockConfig(
                name=name,
                variable_type=task_exp.KernelBlockRole.MACRO,
                components=components,
                block_structure=task_exp.BlockStructure.ADDITIVE,
                use_outputscale=False,
                include_as_main_effect=include_as_main_effect,
            )

        def macro_block_from_specs(
            name: str,
            cols: list[str],
            specs: list[str],
            *,
            ard: bool = True,
            include_as_main_effect: bool = include_main_effects,
        ) -> Any | None:
            active_cols = [col for col in cols if col in idx]
            if not active_cols:
                return None
            dims = [idx[c] for c in active_cols]
            components = []
            for spec in specs:
                if spec.startswith("matern"):
                    components.append(
                        task_exp.MaternKernelComponentConfig(
                            dims=dims,
                            matern_nu=float(spec.removeprefix("matern")),
                            ard=ard,
                            use_outputscale=True,
                            lengthscale_policy=adaptive,
                        )
                    )
                elif spec == "rq":
                    components.append(
                        task_exp.RQKernelComponentConfig(
                            dims=dims,
                            ard=ard,
                            use_outputscale=True,
                            lengthscale_policy=adaptive,
                        )
                    )
                elif spec == "rbf":
                    components.append(
                        task_exp.RBFKernelComponentConfig(
                            dims=dims,
                            ard=ard,
                            use_outputscale=True,
                            lengthscale_policy=adaptive,
                        )
                    )
                elif spec == "linear":
                    components.append(task_exp.LinearKernelComponentConfig(dims=dims, use_outputscale=True))
                elif spec.startswith("polynomial"):
                    components.append(
                        task_exp.PolynomialKernelComponentConfig(
                            dims=dims,
                            ard=ard,
                            power=int(spec.removeprefix("polynomial")),
                            use_outputscale=True,
                            lengthscale_policy=adaptive,
                        )
                    )
                elif spec.startswith("spectral_mixture"):
                    components.append(
                        task_exp.SpectralMixtureKernelComponentConfig(
                            dims=dims,
                            ard=ard,
                            num_mixtures=int(spec.removeprefix("spectral_mixture")),
                            use_outputscale=True,
                        )
                    )
                elif spec == "spectral_delta":
                    components.append(
                        task_exp.SpectralDeltaKernelComponentConfig(
                            dims=dims,
                            use_outputscale=True,
                        )
                    )
                else:
                    raise ValueError(f"Unknown macro kernel component spec {spec!r}")
            return task_exp.KernelBlockConfig(
                name=name,
                variable_type=task_exp.KernelBlockRole.MACRO,
                components=components,
                block_structure=task_exp.BlockStructure.ADDITIVE,
                use_outputscale=False,
                include_as_main_effect=include_as_main_effect,
            )

        def append_interaction(
            interactions: list[Any],
            block_names: set[str],
            left: str,
            right: str,
            *,
            name: str | None = None,
            use_outputscale: bool = True,
        ) -> None:
            if include_pairwise_interactions and {left, right}.issubset(block_names):
                interactions.append(
                    task_exp.KernelInteractionConfig(
                        blocks=[left, right],
                        name=name or f"{left}_x_{right}",
                        use_outputscale=use_outputscale,
                    )
                )

        macro_market_cols = ["vix", "vix_slope", "spy_ret", "pct_above_50dma"]
        macro_credit_cols = ["hy_spread_chg_1p", "hy_spread_z_12p", "cpi_chg_12p", "term_spread"]

        macro_kernel_specs_by_id = {
            "m05_ard_on": ["matern0.5"],
            "m15_ard_on": ["matern1.5"],
            "m25_ard_on": ["matern2.5"],
            "m05_lin_ard_on": ["matern0.5", "linear"],
            "m15_lin_ard_on": ["matern1.5", "linear"],
            "m25_lin_ard_on": ["matern2.5", "linear"],
            "linear_ard_on": ["linear"],
            "rbf_ard_on": ["rbf"],
            "rq_ard_on": ["rq"],
            "rbf_lin_ard_on": ["rbf", "linear"],
            "rq_lin_ard_on": ["rq", "linear"],
            "rq_m05_lin_ard_on": ["rq", "matern0.5", "linear"],
            "rq_poly2_ard_on": ["rq", "polynomial2"],
            "m05_ard_off": ["matern0.5"],
            "m15_ard_off": ["matern1.5"],
            "m25_ard_off": ["matern2.5"],
            "m05_lin_ard_off": ["matern0.5", "linear"],
            "m15_lin_ard_off": ["matern1.5", "linear"],
            "m25_lin_ard_off": ["matern2.5", "linear"],
            "polynomial_degree2": ["polynomial2"],
            "polynomial_degree3": ["polynomial3"],
            "spectral_mixture_q2": ["spectral_mixture2"],
            "spectral_mixture_q4": ["spectral_mixture4"],
            "spectral_delta": ["spectral_delta"],
        }

        if covar_grouping_proposal == "macro_split_custom_pending":
            if macro_market_kernel_id not in macro_kernel_specs_by_id:
                raise ValueError(f"Unknown macro market kernel id: {macro_market_kernel_id!r}")
            if macro_credit_inflation_kernel_id not in macro_kernel_specs_by_id:
                raise ValueError(f"Unknown macro credit/inflation kernel id: {macro_credit_inflation_kernel_id!r}")
            split_blocks = [
                matern_block("time", time_cols, task_exp.KernelBlockRole.TIME),
                matern_block("etf", etf_cols, task_exp.KernelBlockRole.ETF),
                macro_block_from_specs(
                    "macro_market",
                    macro_market_cols,
                    macro_kernel_specs_by_id[macro_market_kernel_id],
                    ard=macro_market_ard,
                ),
                macro_block_from_specs(
                    "macro_credit_inflation",
                    macro_credit_cols,
                    macro_kernel_specs_by_id[macro_credit_inflation_kernel_id],
                    ard=macro_credit_inflation_ard,
                ),
            ]
            blocks = [block for block in split_blocks if block is not None]
            block_names = {block.name for block in blocks}
            custom_interactions = []
            append_interaction(custom_interactions, block_names, "time", "etf")
            append_interaction(custom_interactions, block_names, "time", "macro_market")
            append_interaction(custom_interactions, block_names, "time", "macro_credit_inflation")
            append_interaction(custom_interactions, block_names, "etf", "macro_market")
            append_interaction(custom_interactions, block_names, "etf", "macro_credit_inflation")
            if include_macro_market_credit_interaction:
                append_interaction(
                    custom_interactions,
                    block_names,
                    "macro_market",
                    "macro_credit_inflation",
                    name="macro_market_x_macro_credit_inflation",
                )
            include_time_etf_market = include_three_way_interaction or include_time_etf_market_three_way
            include_time_etf_credit = include_three_way_interaction or include_time_etf_credit_three_way
            if include_time_etf_market and {"time", "etf", "macro_market"}.issubset(block_names):
                custom_interactions.append(
                    task_exp.KernelInteractionConfig(
                        blocks=["time", "etf", "macro_market"],
                        name="time_x_etf_x_macro_market",
                        use_outputscale=True,
                    )
                )
            if include_time_etf_credit and {"time", "etf", "macro_credit_inflation"}.issubset(block_names):
                custom_interactions.append(
                    task_exp.KernelInteractionConfig(
                        blocks=["time", "etf", "macro_credit_inflation"],
                        name="time_x_etf_x_macro_credit_inflation",
                        use_outputscale=True,
                    )
                )
            if include_macro_market_credit_etf_three_way and {
                "etf",
                "macro_market",
                "macro_credit_inflation",
            }.issubset(block_names):
                custom_interactions.append(
                    task_exp.KernelInteractionConfig(
                        blocks=["macro_market", "macro_credit_inflation", "etf"],
                        name="macro_market_x_macro_credit_inflation_x_etf",
                        use_outputscale=True,
                    )
                )
            if include_time_macro_market_credit_three_way and {
                "time",
                "macro_market",
                "macro_credit_inflation",
            }.issubset(block_names):
                custom_interactions.append(
                    task_exp.KernelInteractionConfig(
                        blocks=["time", "macro_market", "macro_credit_inflation"],
                        name="time_x_macro_market_x_macro_credit_inflation",
                        use_outputscale=True,
                    )
                )
            if include_time_etf_market_credit_four_way and {
                "time",
                "etf",
                "macro_market",
                "macro_credit_inflation",
            }.issubset(block_names):
                custom_interactions.append(
                    task_exp.KernelInteractionConfig(
                        blocks=["time", "etf", "macro_market", "macro_credit_inflation"],
                        name="time_x_etf_x_macro_market_x_macro_credit_inflation",
                        use_outputscale=True,
                    )
                )
            return task_exp.CovarModuleConfig(
                blocks=blocks,
                global_structure=task_exp.GlobalStructure.HIERARCHICAL,
                interaction_policy=task_exp.InteractionPolicy.CUSTOM,
                custom_interactions=custom_interactions,
            )

        if covar_grouping_proposal == "macro_market_subsplit_custom_pending":
            market_subsplit_specs_by_id = {
                "vol_rq_trend_linear": (["rq"], ["linear"]),
                "vol_m05_trend_linear": (["matern0.5"], ["linear"]),
                "vol_rq_trend_rq": (["rq"], ["rq"]),
                "vol_linear_trend_linear": (["linear"], ["linear"]),
                "vol_rq_trend_m05": (["rq"], ["matern0.5"]),
                "vol_rq_trend_m15": (["rq"], ["matern1.5"]),
                "vol_linear_trend_m05": (["linear"], ["matern0.5"]),
                "vol_linear_trend_m15": (["linear"], ["matern1.5"]),
                "vol_rq_lin_trend_m05": (["rq", "linear"], ["matern0.5"]),
                "vol_rq_lin_trend_m15": (["rq", "linear"], ["matern1.5"]),
                "vol_m05_trend_rq": (["matern0.5"], ["rq"]),
                "vol_m15_trend_rq": (["matern1.5"], ["rq"]),
                "vol_m15_trend_linear": (["matern1.5"], ["linear"]),
                "vol_m05_trend_rq_lin": (["matern0.5"], ["rq", "linear"]),
                "vol_m15_trend_rq_lin": (["matern1.5"], ["rq", "linear"]),
            }
            if macro_market_kernel_id not in market_subsplit_specs_by_id:
                raise ValueError(f"Unknown macro market sub-split kernel id: {macro_market_kernel_id!r}")
            if macro_credit_inflation_kernel_id not in macro_kernel_specs_by_id:
                raise ValueError(f"Unknown macro credit/inflation kernel id: {macro_credit_inflation_kernel_id!r}")
            vol_specs, trend_specs = market_subsplit_specs_by_id[macro_market_kernel_id]
            split_blocks = [
                matern_block("time", time_cols, task_exp.KernelBlockRole.TIME),
                matern_block("etf", etf_cols, task_exp.KernelBlockRole.ETF),
                macro_block_from_specs(
                    "macro_market_vol_regime",
                    ["vix", "vix_slope"],
                    vol_specs,
                    ard=macro_market_ard,
                ),
                macro_block_from_specs(
                    "macro_market_trend_breadth",
                    ["spy_ret", "pct_above_50dma"],
                    trend_specs,
                    ard=macro_market_ard,
                ),
                macro_block_from_specs(
                    "macro_credit_inflation",
                    macro_credit_cols,
                    macro_kernel_specs_by_id[macro_credit_inflation_kernel_id],
                    ard=macro_credit_inflation_ard,
                ),
            ]
            blocks = [block for block in split_blocks if block is not None]
            block_names = {block.name for block in blocks}
            market_block_names = [
                name
                for name in ["macro_market_vol_regime", "macro_market_trend_breadth"]
                if name in block_names
            ]
            custom_interactions = []
            append_interaction(custom_interactions, block_names, "time", "etf")
            append_interaction(custom_interactions, block_names, "time", "macro_credit_inflation")
            append_interaction(custom_interactions, block_names, "etf", "macro_credit_inflation")
            for market_name in market_block_names:
                append_interaction(custom_interactions, block_names, "time", market_name)
                append_interaction(custom_interactions, block_names, "etf", market_name)
                if include_macro_market_credit_interaction:
                    append_interaction(
                        custom_interactions,
                        block_names,
                        market_name,
                        "macro_credit_inflation",
                        name=f"{market_name}_x_macro_credit_inflation",
                    )
                if (include_three_way_interaction or include_time_etf_market_three_way) and {
                    "time",
                    "etf",
                    market_name,
                }.issubset(block_names):
                    custom_interactions.append(
                        task_exp.KernelInteractionConfig(
                            blocks=["time", "etf", market_name],
                            name=f"time_x_etf_x_{market_name}",
                            use_outputscale=True,
                        )
                    )
                if include_macro_market_credit_etf_three_way and {
                    "etf",
                    market_name,
                    "macro_credit_inflation",
                }.issubset(block_names):
                    custom_interactions.append(
                        task_exp.KernelInteractionConfig(
                            blocks=[market_name, "macro_credit_inflation", "etf"],
                            name=f"{market_name}_x_macro_credit_inflation_x_etf",
                            use_outputscale=True,
                        )
                    )
            if include_time_etf_credit_three_way and {
                "time",
                "etf",
                "macro_credit_inflation",
            }.issubset(block_names):
                custom_interactions.append(
                    task_exp.KernelInteractionConfig(
                        blocks=["time", "etf", "macro_credit_inflation"],
                        name="time_x_etf_x_macro_credit_inflation",
                        use_outputscale=True,
                    )
                )
            if include_time_macro_market_credit_three_way:
                for market_name in market_block_names:
                    if {"time", market_name, "macro_credit_inflation"}.issubset(block_names):
                        custom_interactions.append(
                            task_exp.KernelInteractionConfig(
                                blocks=["time", market_name, "macro_credit_inflation"],
                                name=f"time_x_{market_name}_x_macro_credit_inflation",
                                use_outputscale=True,
                            )
                        )
            return task_exp.CovarModuleConfig(
                blocks=blocks,
                global_structure=task_exp.GlobalStructure.HIERARCHICAL,
                interaction_policy=task_exp.InteractionPolicy.CUSTOM,
                custom_interactions=custom_interactions,
            )

        if covar_grouping_proposal == "single_matern":
            block = matern_block(
                "all_features",
                [*time_cols, *etf_cols, *macro_cols],
                task_exp.KernelBlockRole.GENERIC,
            )
            if block is not None:
                blocks.append(block)
            return task_exp.CovarModuleConfig(
                blocks=blocks,
                global_structure=task_exp.GlobalStructure.ADDITIVE,
                interaction_policy=task_exp.InteractionPolicy.NONE,
                custom_interactions=[],
            )

        if covar_grouping_proposal == "time_plus_matern_interaction":
            time_block = matern_block("time", time_cols, task_exp.KernelBlockRole.TIME)
            feature_block = matern_block(
                "features",
                [*etf_cols, *macro_cols],
                task_exp.KernelBlockRole.GENERIC,
            )
            blocks = [block for block in [time_block, feature_block] if block is not None]
            custom_interactions = []
            block_names = {block.name for block in blocks}
            if include_pairwise_interactions and {"time", "features"}.issubset(block_names):
                custom_interactions.append(
                    task_exp.KernelInteractionConfig(
                        blocks=["time", "features"],
                        name="time_x_features",
                        use_outputscale=True,
                    )
                )
            return task_exp.CovarModuleConfig(
                blocks=blocks,
                global_structure=task_exp.GlobalStructure.HIERARCHICAL,
                interaction_policy=task_exp.InteractionPolicy.CUSTOM,
                custom_interactions=custom_interactions,
            )

        macro_literature_specs = {
            "macro_split_market_rq_credit_linear": (["rq"], ["linear"], False),
            "macro_split_market_rq_linear_credit_matern15": (["rq", "linear"], ["matern1.5", "linear"], False),
            "macro_split_market_matern_rq_credit_linear": (["matern0.5", "rq"], ["linear"], False),
            "macro_split_market_matern_credit_rbf_linear": (["matern0.5", "linear"], ["rbf", "linear"], False),
            "macro_split_market_rbf_credit_matern25_linear": (["rbf", "linear"], ["matern2.5", "linear"], False),
            "macro_split_market_linear_credit_matern25": (["linear"], ["matern2.5", "linear"], False),
            "macro_split_market_rq_credit_rbf": (["rq"], ["rbf"], False),
            "macro_split_market_matern_credit_rq_linear": (["matern0.5", "linear"], ["rq", "linear"], False),
        }
        if covar_grouping_proposal in macro_literature_specs:
            market_specs, credit_specs, include_market_credit = macro_literature_specs[covar_grouping_proposal]
            split_blocks = [
                matern_block("time", time_cols, task_exp.KernelBlockRole.TIME),
                matern_block("etf", etf_cols, task_exp.KernelBlockRole.ETF),
                macro_block_from_specs("macro_market", macro_market_cols, market_specs),
                macro_block_from_specs("macro_credit_inflation", macro_credit_cols, credit_specs),
            ]
            blocks = [block for block in split_blocks if block is not None]
            block_names = {block.name for block in blocks}
            custom_interactions = []
            append_interaction(custom_interactions, block_names, "time", "etf")
            append_interaction(custom_interactions, block_names, "time", "macro_market")
            append_interaction(custom_interactions, block_names, "time", "macro_credit_inflation")
            append_interaction(custom_interactions, block_names, "etf", "macro_market")
            append_interaction(custom_interactions, block_names, "etf", "macro_credit_inflation")
            if include_market_credit:
                append_interaction(
                    custom_interactions,
                    block_names,
                    "macro_market",
                    "macro_credit_inflation",
                    name="macro_market_x_macro_credit_inflation",
                )
            return task_exp.CovarModuleConfig(
                blocks=blocks,
                global_structure=task_exp.GlobalStructure.HIERARCHICAL,
                interaction_policy=task_exp.InteractionPolicy.CUSTOM,
                custom_interactions=custom_interactions,
            )

        if covar_grouping_proposal.startswith("split_macro_"):
            split_blocks = [
                matern_block("time", time_cols, task_exp.KernelBlockRole.TIME),
                matern_block("etf", etf_cols, task_exp.KernelBlockRole.ETF),
                market_risk_block(),
            ]
            if covar_grouping_proposal == "split_macro_market_nonlinear_credit_linear":
                split_blocks.append(credit_rates_block(include_matern=False))
            elif covar_grouping_proposal == "split_macro_credit_shrunk":
                split_blocks.append(credit_rates_block(include_matern=False))
            else:
                split_blocks.append(credit_rates_block(include_matern=True, matern_nu=1.5))
            blocks = [block for block in split_blocks if block is not None]
            block_names = {block.name for block in blocks}
            custom_interactions = []
            append_interaction(custom_interactions, block_names, "time", "etf")
            append_interaction(custom_interactions, block_names, "time", "macro_market")
            append_interaction(custom_interactions, block_names, "time", "macro_credit_inflation")
            append_interaction(custom_interactions, block_names, "etf", "macro_market")
            if covar_grouping_proposal not in {
                "split_macro_no_etf_credit",
                "split_macro_cross_family_gate",
            }:
                append_interaction(
                    custom_interactions,
                    block_names,
                    "etf",
                    "macro_credit_inflation",
                    use_outputscale=covar_grouping_proposal != "split_macro_credit_shrunk",
                )
            if covar_grouping_proposal == "split_macro_cross_family_gate":
                append_interaction(
                    custom_interactions,
                    block_names,
                    "macro_market",
                    "macro_credit_inflation",
                    name="macro_market_x_macro_credit_inflation",
                )
            return task_exp.CovarModuleConfig(
                blocks=blocks,
                global_structure=task_exp.GlobalStructure.HIERARCHICAL,
                interaction_policy=task_exp.InteractionPolicy.CUSTOM,
                custom_interactions=custom_interactions,
            )

        if covar_grouping_proposal == "split_groups":
            split_specs = [
                ("time", time_cols, task_exp.KernelBlockRole.TIME),
                ("etf_momentum", ["mom6m", "mom12m_skip1m"], task_exp.KernelBlockRole.ETF),
                ("etf_rank_trend", ["cs_mom_rank", "trend_slope"], task_exp.KernelBlockRole.ETF),
                ("macro_market", ["vix", "vix_slope", "spy_ret", "pct_above_50dma"], task_exp.KernelBlockRole.MACRO),
                (
                    "macro_credit_inflation",
                    ["hy_spread_chg_1p", "hy_spread_z_12p", "cpi_chg_12p", "term_spread"],
                    task_exp.KernelBlockRole.MACRO,
                ),
            ]
            for name, cols, role in split_specs:
                block = matern_block(name, cols, role)
                if block is not None:
                    blocks.append(block)
            custom_interactions = []
            if include_pairwise_interactions:
                for left_index, left in enumerate(blocks):
                    for right in blocks[left_index + 1 :]:
                        custom_interactions.append(
                            task_exp.KernelInteractionConfig(
                                blocks=[left.name, right.name],
                                name=f"{left.name}_x_{right.name}",
                                use_outputscale=True,
                            )
                        )
            return task_exp.CovarModuleConfig(
                blocks=blocks,
                global_structure=task_exp.GlobalStructure.HIERARCHICAL,
                interaction_policy=task_exp.InteractionPolicy.CUSTOM,
                custom_interactions=custom_interactions,
            )

        if time_cols:
            blocks.append(
                task_exp.KernelBlockConfig(
                    name="time",
                    variable_type=task_exp.KernelBlockRole.TIME,
                    components=[
                        task_exp.MaternKernelComponentConfig(
                            dims=[idx[c] for c in time_cols],
                            matern_nu=0.5,
                            ard=True,
                            use_outputscale=True,
                            lengthscale_policy=adaptive,
                        )
                    ],
                    block_structure=task_exp.BlockStructure.ADDITIVE,
                    use_outputscale=False,
                    include_as_main_effect=include_main_effects,
                )
            )
        if etf_cols:
            blocks.append(
                task_exp.KernelBlockConfig(
                    name="etf",
                    variable_type=task_exp.KernelBlockRole.ETF,
                    components=[
                        task_exp.MaternKernelComponentConfig(
                            dims=[idx[c] for c in etf_cols],
                            matern_nu=0.5,
                            ard=True,
                            use_outputscale=True,
                            lengthscale_policy=adaptive,
                        )
                    ],
                    block_structure=task_exp.BlockStructure.ADDITIVE,
                    use_outputscale=False,
                    include_as_main_effect=include_main_effects,
                )
            )
        if macro_cols:
            macro_dims = [idx[c] for c in macro_cols]
            macro_components = []
            if macro_component_proposal in {"default", "matern_linear", "matern_only"}:
                macro_components.append(
                    task_exp.MaternKernelComponentConfig(
                        dims=macro_dims,
                        matern_nu=0.5,
                        ard=True,
                        use_outputscale=True,
                        lengthscale_policy=adaptive,
                    )
                )
            if macro_component_proposal == "default":
                macro_components.append(
                    task_exp.RQKernelComponentConfig(
                        dims=macro_dims,
                        ard=True,
                        use_outputscale=True,
                        lengthscale_policy=adaptive,
                    )
                )
            if macro_component_proposal in {"default", "matern_linear", "linear_only"}:
                macro_components.append(task_exp.LinearKernelComponentConfig(dims=macro_dims, use_outputscale=True))
            blocks.append(
                task_exp.KernelBlockConfig(
                    name="macro",
                    variable_type=task_exp.KernelBlockRole.MACRO,
                    components=macro_components,
                    block_structure=task_exp.BlockStructure.ADDITIVE,
                    use_outputscale=False,
                    include_as_main_effect=include_main_effects,
                )
            )
        custom_interactions = []
        block_names = {block.name for block in blocks}
        if include_pairwise_interactions and {"time", "etf"}.issubset(block_names):
            custom_interactions.append(
                task_exp.KernelInteractionConfig(blocks=["time", "etf"], name="time_x_etf", use_outputscale=True)
            )
        if include_pairwise_interactions and {"time", "macro"}.issubset(block_names):
            custom_interactions.append(
                task_exp.KernelInteractionConfig(blocks=["time", "macro"], name="time_x_macro", use_outputscale=True)
            )
        if include_pairwise_interactions and not drop_etf_macro_interaction and {"macro", "etf"}.issubset(block_names):
            custom_interactions.append(
                task_exp.KernelInteractionConfig(blocks=["macro", "etf"], name="macro_x_etf", use_outputscale=True)
            )
        if include_three_way_interaction and {"time", "macro", "etf"}.issubset(block_names):
            custom_interactions.append(
                task_exp.KernelInteractionConfig(
                    blocks=["time", "macro", "etf"],
                    name="time_x_macro_x_etf",
                    use_outputscale=True,
                )
            )
        return task_exp.CovarModuleConfig(
            blocks=blocks,
            global_structure=task_exp.GlobalStructure.HIERARCHICAL,
            interaction_policy=task_exp.InteractionPolicy.CUSTOM,
            custom_interactions=custom_interactions,
        )

    return build


def parse_optional_column_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    columns = [column.strip() for column in raw.split(",") if column.strip()]
    return columns


def parse_wrapper_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=False,
    )
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--mtgp-rank", type=int, default=2)
    parser.add_argument(
        "--tickers",
        default=",".join(FIRST_ROUND_TICKERS),
        help="Comma-separated ETF task universe for this experiment wrapper.",
    )
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="compact_pruned")
    parser.add_argument(
        "--custom-etf-cols",
        default=None,
        help="Comma-separated ETF/local feature columns overriding the selected --feature-set ETF columns.",
    )
    parser.add_argument(
        "--custom-macro-cols",
        default=None,
        help="Comma-separated macro feature columns overriding the selected --feature-set macro columns.",
    )
    parser.add_argument(
        "--covar-grouping-proposal",
        choices=COVAR_GROUPING_PROPOSALS,
        default="default",
        help="Experiment-scoped covariance grouping proposal for the selected feature set.",
    )
    parser.add_argument(
        "--macro-component-proposal",
        choices=MACRO_COMPONENT_PROPOSALS,
        default="default",
        help="Experiment-scoped macro kernel component proposal for the default grouped covariance.",
    )
    parser.add_argument("--exclude-time-index", action="store_true")
    parser.add_argument(
        "--interactions-only",
        action="store_true",
        help="Exclude main-effect block kernels and keep only configured block interaction kernels.",
    )
    parser.add_argument(
        "--include-three-way-interaction",
        action="store_true",
        help="Add a time x ETF/local x macro three-way interaction product to the covariance kernel.",
    )
    parser.add_argument(
        "--include-time-etf-market-three-way",
        action="store_true",
        help="Add a time x ETF x macro_market three-way interaction for custom split-macro kernels.",
    )
    parser.add_argument(
        "--include-time-etf-credit-three-way",
        action="store_true",
        help="Add a time x ETF x macro_credit_inflation three-way interaction for custom split-macro kernels.",
    )
    parser.add_argument(
        "--exclude-pairwise-interactions",
        action="store_true",
        help="Exclude pairwise block interactions while allowing main effects and/or the three-way interaction.",
    )
    parser.add_argument(
        "--drop-etf-macro-interaction",
        action="store_true",
        help="Drop only the ETF x macro interaction while keeping other pairwise interactions.",
    )
    parser.add_argument(
        "--include-macro-market-credit-interaction",
        action="store_true",
        help="Add a macro_market x macro_credit_inflation interaction for custom split-macro kernels.",
    )
    parser.add_argument(
        "--include-macro-market-credit-etf-three-way",
        action="store_true",
        help="Add a macro_market x macro_credit_inflation x ETF three-way interaction for custom split-macro kernels.",
    )
    parser.add_argument(
        "--include-time-macro-market-credit-three-way",
        action="store_true",
        help="Add a time x macro_market x macro_credit_inflation three-way interaction for custom split-macro kernels.",
    )
    parser.add_argument(
        "--include-time-etf-market-credit-four-way",
        action="store_true",
        help="Add a time x ETF x macro_market x macro_credit_inflation four-way interaction.",
    )
    parser.add_argument("--macro-market-kernel-id", default=None)
    parser.add_argument("--macro-market-ard", choices=["true", "false"], default="true")
    parser.add_argument("--macro-credit-inflation-kernel-id", default=None)
    parser.add_argument("--macro-credit-inflation-ard", choices=["true", "false"], default="true")
    parser.add_argument("-h", "--help", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def forecast_metric_rows(predictions: pd.DataFrame, *, group_col: str | None) -> list[dict[str, Any]]:
    grouped = [("overall", predictions)] if group_col is None else predictions.groupby(group_col, observed=True)
    rows: list[dict[str, Any]] = []
    for label, frame in grouped:
        y_true = frame["y_true"].to_numpy(dtype=float)
        y_pred = frame["y_pred"].to_numpy(dtype=float)
        y_std = np.clip(frame["y_std"].to_numpy(dtype=float), math.sqrt(NLL_VARIANCE_FLOOR), None)
        residual = y_true - y_pred
        variance = np.square(y_std)
        row: dict[str, Any] = {
            "asset_id": str(label),
            "n": int(len(frame)),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(y_pred - y_true)),
            "mean_y_true": float(np.mean(y_true)),
            "mean_y_pred": float(np.mean(y_pred)),
            "std_y_true": float(np.std(y_true, ddof=1)) if len(frame) > 1 else math.nan,
            "std_y_pred": float(np.std(y_pred, ddof=1)) if len(frame) > 1 else math.nan,
            "mean_pred_std": float(np.mean(y_std)),
            "mean_gaussian_nll": float(
                np.mean(0.5 * (np.log(2.0 * math.pi * variance) + np.square(residual) / variance))
            ),
            "standardized_residual_mean": float(np.mean(residual / y_std)),
            "standardized_residual_std": float(np.std(residual / y_std, ddof=1)) if len(frame) > 1 else math.nan,
        }
        for name, z_value in NORMAL_COVERAGE_Z.items():
            row[name] = float(np.mean(np.abs(residual) <= z_value * y_std))
        rows.append(row)
    return rows


def write_forecast_metrics(output_dir: Path) -> None:
    predictions_path = output_dir / "gp_window_predictions.csv"
    if not predictions_path.exists():
        return
    predictions = pd.read_csv(predictions_path)
    metric_columns = [
        "asset_id",
        "n",
        "rmse",
        "mae",
        "bias",
        "mean_y_true",
        "mean_y_pred",
        "std_y_true",
        "std_y_pred",
        "mean_pred_std",
        "mean_gaussian_nll",
        "standardized_residual_mean",
        "standardized_residual_std",
        *NORMAL_COVERAGE_Z,
    ]
    by_asset = pd.DataFrame(forecast_metric_rows(predictions, group_col="asset_id")).loc[:, metric_columns]
    overall = pd.DataFrame(forecast_metric_rows(predictions, group_col=None)).loc[:, metric_columns]
    by_asset.to_csv(output_dir / "forecast_metrics_by_asset.csv", index=False)
    overall.to_csv(output_dir / "forecast_metrics_overall.csv", index=False)


def write_wrapper_metadata(
    output_dir: Path,
    *,
    feature_set: str,
    tickers: list[str],
    custom_etf_cols: list[str] | None,
    custom_macro_cols: list[str] | None,
    covar_grouping_proposal: str,
    macro_component_proposal: str,
    mtgp_rank: int,
    include_time_index: bool,
    include_main_effects: bool,
    include_pairwise_interactions: bool,
    drop_etf_macro_interaction: bool,
    include_three_way_interaction: bool,
    include_time_etf_market_three_way: bool,
    include_time_etf_credit_three_way: bool,
    include_macro_market_credit_interaction: bool,
    include_macro_market_credit_etf_three_way: bool,
    include_time_macro_market_credit_three_way: bool,
    include_time_etf_market_credit_four_way: bool,
    macro_market_kernel_id: str | None,
    macro_market_ard: bool,
    macro_credit_inflation_kernel_id: str | None,
    macro_credit_inflation_ard: bool,
) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    feature_spec = FEATURE_SETS[feature_set]
    etf_cols = list(custom_etf_cols) if custom_etf_cols is not None else list(feature_spec["etf"])
    macro_cols = list(custom_macro_cols) if custom_macro_cols is not None else list(feature_spec["macro"])
    time_cols = list(TIME_COLS) if include_time_index else []
    manifest["us_equity_first_round_wrapper"] = {
        "feature_set": feature_set,
        "custom_etf_columns": custom_etf_cols,
        "custom_macro_columns": custom_macro_cols,
        "covar_grouping_proposal": covar_grouping_proposal,
        "macro_component_proposal": macro_component_proposal,
        "mtgp_rank": int(mtgp_rank),
        "first_round_tickers": tickers,
        "time_index_included": bool(include_time_index),
        "main_effects_included": bool(include_main_effects),
        "pairwise_interactions_included": bool(include_pairwise_interactions),
        "etf_macro_interaction_dropped": bool(drop_etf_macro_interaction),
        "interactions_only": not include_main_effects,
        "three_way_interaction_included": bool(include_three_way_interaction),
        "time_etf_market_three_way_included": bool(include_three_way_interaction or include_time_etf_market_three_way),
        "time_etf_credit_three_way_included": bool(include_three_way_interaction or include_time_etf_credit_three_way),
        "macro_market_credit_interaction_included": bool(include_macro_market_credit_interaction),
        "macro_market_credit_etf_three_way_included": bool(include_macro_market_credit_etf_three_way),
        "time_macro_market_credit_three_way_included": bool(include_time_macro_market_credit_three_way),
        "time_etf_market_credit_four_way_included": bool(include_time_etf_market_credit_four_way),
        "macro_market_kernel_id": macro_market_kernel_id,
        "macro_market_ard": bool(macro_market_ard),
        "macro_credit_inflation_kernel_id": macro_credit_inflation_kernel_id,
        "macro_credit_inflation_ard": bool(macro_credit_inflation_ard),
        "time_columns": time_cols,
        "etf_columns": etf_cols,
        "macro_columns": macro_cols,
        "input_columns": [*time_cols, *etf_cols, *macro_cols],
        "non_time_feature_count": len(etf_cols) + len(macro_cols),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> None:
    wrapper_args, remaining = parse_wrapper_args(argv)
    runner = load_portfolio_runner()
    if wrapper_args.help:
        print(__doc__)
        print("\nWrapper defaults:")
        print(f"  --feature-path {DEFAULT_FEATURE_PATH}")
        print(f"  --output-root {DEFAULT_OUTPUT_ROOT}")
        print("  --mtgp-rank 2")
        print("  --feature-set compact_pruned")
        print("\nDelegated runner help:\n")
        original_argv = sys.argv
        try:
            sys.argv = [str(PORTFOLIO_RUNNER_PATH), "--help"]
            runner.parse_args()
        finally:
            sys.argv = original_argv
        return

    include_time_index = not wrapper_args.exclude_time_index
    include_main_effects = not wrapper_args.interactions_only
    include_pairwise_interactions = not wrapper_args.exclude_pairwise_interactions
    tickers = [ticker.strip().upper() for ticker in wrapper_args.tickers.split(",") if ticker.strip()]
    if not tickers:
        raise ValueError("--tickers must contain at least one ETF ticker")
    custom_etf_cols = parse_optional_column_list(wrapper_args.custom_etf_cols)
    custom_macro_cols = parse_optional_column_list(wrapper_args.custom_macro_cols)
    patch_first_round_constants(
        runner,
        mtgp_rank=wrapper_args.mtgp_rank,
        feature_set=wrapper_args.feature_set,
        tickers=tickers,
        custom_etf_cols=custom_etf_cols,
        custom_macro_cols=custom_macro_cols,
        covar_grouping_proposal=wrapper_args.covar_grouping_proposal,
        macro_component_proposal=wrapper_args.macro_component_proposal,
        include_time_index=include_time_index,
        include_main_effects=include_main_effects,
        include_pairwise_interactions=include_pairwise_interactions,
        drop_etf_macro_interaction=wrapper_args.drop_etf_macro_interaction,
        include_three_way_interaction=wrapper_args.include_three_way_interaction,
        include_time_etf_market_three_way=wrapper_args.include_time_etf_market_three_way,
        include_time_etf_credit_three_way=wrapper_args.include_time_etf_credit_three_way,
        include_macro_market_credit_interaction=wrapper_args.include_macro_market_credit_interaction,
        include_macro_market_credit_etf_three_way=wrapper_args.include_macro_market_credit_etf_three_way,
        include_time_macro_market_credit_three_way=wrapper_args.include_time_macro_market_credit_three_way,
        include_time_etf_market_credit_four_way=wrapper_args.include_time_etf_market_credit_four_way,
        macro_market_kernel_id=wrapper_args.macro_market_kernel_id,
        macro_market_ard=wrapper_args.macro_market_ard == "true",
        macro_credit_inflation_kernel_id=wrapper_args.macro_credit_inflation_kernel_id,
        macro_credit_inflation_ard=wrapper_args.macro_credit_inflation_ard == "true",
    )
    run_id = wrapper_args.run_id or (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{wrapper_args.feature_set}_rank{wrapper_args.mtgp_rank}"
    )
    delegated_argv = [
        "--feature-path",
        str(wrapper_args.feature_path),
        "--output-dir",
        str(wrapper_args.output_root / run_id),
        "--run-id",
        run_id,
        "--periods-per-year",
        "12",
        "--rebalance-frequency-label",
        "monthly_bme",
        *remaining,
    ]
    original_argv = sys.argv
    try:
        sys.argv = [str(PORTFOLIO_RUNNER_PATH), *delegated_argv]
        args = runner.parse_args()
    finally:
        sys.argv = original_argv
    runner.run(args)
    if not args.preflight_only:
        write_wrapper_metadata(
            Path(args.output_dir),
            feature_set=wrapper_args.feature_set,
            tickers=tickers,
            custom_etf_cols=custom_etf_cols,
            custom_macro_cols=custom_macro_cols,
            covar_grouping_proposal=wrapper_args.covar_grouping_proposal,
            macro_component_proposal=wrapper_args.macro_component_proposal,
            mtgp_rank=wrapper_args.mtgp_rank,
            include_time_index=include_time_index,
            include_main_effects=include_main_effects,
            include_pairwise_interactions=include_pairwise_interactions,
            drop_etf_macro_interaction=wrapper_args.drop_etf_macro_interaction,
            include_three_way_interaction=wrapper_args.include_three_way_interaction,
            include_time_etf_market_three_way=wrapper_args.include_time_etf_market_three_way,
            include_time_etf_credit_three_way=wrapper_args.include_time_etf_credit_three_way,
            include_macro_market_credit_interaction=wrapper_args.include_macro_market_credit_interaction,
            include_macro_market_credit_etf_three_way=wrapper_args.include_macro_market_credit_etf_three_way,
            include_time_macro_market_credit_three_way=wrapper_args.include_time_macro_market_credit_three_way,
            include_time_etf_market_credit_four_way=wrapper_args.include_time_etf_market_credit_four_way,
            macro_market_kernel_id=wrapper_args.macro_market_kernel_id,
            macro_market_ard=wrapper_args.macro_market_ard == "true",
            macro_credit_inflation_kernel_id=wrapper_args.macro_credit_inflation_kernel_id,
            macro_credit_inflation_ard=wrapper_args.macro_credit_inflation_ard == "true",
        )
        write_forecast_metrics(Path(args.output_dir))


if __name__ == "__main__":
    main()
