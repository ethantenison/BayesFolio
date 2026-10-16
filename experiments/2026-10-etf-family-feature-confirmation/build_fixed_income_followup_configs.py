"""Build deterministic feature-level fixed-income experiment configurations.

Usage:
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/build_fixed_income_followup_configs.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
SELECTION = {
    "maximize": ["mean_ic", "sharpe"],
    "minimize": ["mean_gaussian_nll"],
    "sharpe_definition": "mean(monthly excess return) / sample_std(monthly excess return) * sqrt(12)",
    "diagnostic_only": ["cumulative_return", "rmse"],
    "turnover_excluded": True,
}
HYPERVOLUME = {
    "reference_point": {"mean_ic": 0.0, "mean_gaussian_nll": 0.0, "sharpe": 0.0},
    "objective_transform": {
        "mean_ic": "maximize",
        "mean_gaussian_nll": "negate_then_maximize",
        "sharpe": "maximize",
    },
    "use_configuration_means_across_seeds": True,
    "fixed_for_subsequent_rounds": True,
}
FIXED_DECISIONS = {
    "include_t_index": True,
    "cached_data_only": True,
    "production_budget_reference_excluded": True,
}
COMMON_ARGS = [
    "--mtgp-rank",
    "2",
    "--maxiter",
    "15",
    "--posterior-scenarios",
    "256",
    "--resume-scenarios",
    "--gp-experiment",
    "signed_lkj_eta_2",
    "--upperlng",
    "0.35",
    "--nea",
    "3",
    "--gp-riskfolio-rm",
    "CVaR",
    "--gp-riskfolio-obj",
    "MaxRet",
    "--scenario-workers",
    "1",
    "--torch-num-threads",
    "1",
]
TREND = ["lag_y_excess_lead", "chmom", "mom12m", "mom36m", "cs_mom_rank", "ma_signal"]
MARKET_RISK = ["vix_slope", "vix_ts_z_12m", "vix", "spy_flow_z_12m", "spy_ret", "erp", "pct_above_50dma"]
GLOBAL = ["copper_ret", "oil_ret", "gold_crude_ratio", "em_fx_ret"]
CREDIT_INFLATION = ["hy_spread", "hy_spread_chg_1m", "hy_spread_z_12m", "cpi_yoy", "cpi_mom"]
ALL_LOCAL = [
    "lag_y_excess_lead",
    "baspread",
    "ret_kurt",
    "chmom",
    "mom12m",
    "mom36m",
    "cs_mom_rank",
    "max_dd_6m",
    "ma_signal",
    "ret_autocorr",
    "vol_z",
]
ALL_MACRO = [
    "hy_spread",
    "hy_spread_chg_1m",
    "hy_spread_z_12m",
    "vix_slope",
    "vix_ts_z_12m",
    "vix",
    "spy_flow_z_12m",
    "spy_ret",
    "erp",
    "cpi_yoy",
    "cpi_mom",
    "copper_ret",
    "oil_ret",
    "gold_crude_ratio",
    "pct_above_50dma",
    "em_fx_ret",
]


def configuration(identifier: str, label: str, local: list[str], macro: list[str]) -> dict[str, Any]:
    """Return one fully resolved experiment configuration."""
    return {
        "id": identifier,
        "label": label,
        "expected_input_columns": ["t_index", *local, *macro],
        "args": [
            "--feature-set",
            "momentum_plus_macro_market",
            "--custom-etf-cols",
            ",".join(local),
            "--custom-macro-cols",
            ",".join(macro),
            "--covar-grouping-proposal",
            "default",
            "--macro-component-proposal",
            "default",
            "--time-modulation-mode",
            "neither",
        ],
    }


def base_config(family: str, phase: str, results_prefix: str) -> dict[str, Any]:
    """Return the shared deterministic family contract."""
    family_files = {
        "us_fixed_income": ("us_fixed_income", ["BND", "IEF", "LQD", "HYG"]),
        "international_fixed_income": ("international_fixed_income", ["BNDX", "VWOB", "HYEM"]),
    }
    stem, tickers = family_files[family]
    artifact_root = "experiments/2026-10-etf-family-feature-confirmation/artifacts"
    return {
        "schema": "bayesfolio.etf_family_experiment.v1",
        "phase": phase,
        "family": family,
        "runner": "experiments/2026-07-international-equity-etf-features/run_first_round_walkforward.py",
        "feature_path": f"{artifact_root}/{stem}_cached_inputs_20261016.parquet",
        "feature_manifest": f"{artifact_root}/{stem}_cached_inputs_20261016.manifest.json",
        "scored_date_file": f"{artifact_root}/{stem}_scored_dates_24.txt",
        "output_root": "experiments/2026-10-etf-family-feature-confirmation/runs",
        "snapshot_root": "experiments/2026-10-etf-family-feature-confirmation/executions",
        "results_prefix": results_prefix,
        "tickers": tickers,
        "default_seeds": [27, 127, 227],
        "common_args": COMMON_ARGS,
        "selection": SELECTION,
        "hypervolume": HYPERVOLUME,
        "fixed_decisions": FIXED_DECISIONS,
    }


def write_config(filename: str, payload: dict[str, Any]) -> None:
    """Write one reproducible JSON configuration."""
    (EXPERIMENT_DIR / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Build evidence-directed targeted-ablation and refinement configurations."""
    us = base_config("us_fixed_income", "targeted_ablation", "us_fixed_income_targeted_ablation")
    us["configurations"] = [
        configuration("usfi_trend_hy_level", "USFI-TREND+HY-LEVEL", TREND, ["hy_spread"]),
        configuration("usfi_trend_hy_change", "USFI-TREND+HY-CHANGE", TREND, ["hy_spread_chg_1m"]),
        configuration("usfi_trend_hy_z", "USFI-TREND+HY-Z", TREND, ["hy_spread_z_12m"]),
        configuration("usfi_trend_inflation", "USFI-TREND+INFLATION", TREND, ["cpi_yoy", "cpi_mom"]),
        configuration(
            "usfi_trend_hy_change_z",
            "USFI-TREND+HY-CHANGE-Z",
            TREND,
            ["hy_spread_chg_1m", "hy_spread_z_12m"],
        ),
        configuration(
            "usfi_trend_no_lag",
            "USFI-TREND-NO-LAG",
            ["chmom", "mom12m", "mom36m", "cs_mom_rank", "ma_signal"],
            [],
        ),
        configuration(
            "usfi_trend_no_mom36",
            "USFI-TREND-NO-MOM36",
            ["lag_y_excess_lead", "chmom", "mom12m", "cs_mom_rank", "ma_signal"],
            [],
        ),
    ]
    us["comparison_configurations"] = [
        configuration("usfi_trend", "USFI-TREND", TREND, []),
        configuration("usfi_trend_credit_inflation", "USFI-TREND+CREDIT-INFLATION", TREND, CREDIT_INFLATION),
    ]
    write_config("us_fixed_income_targeted_ablation_config.json", us)

    international = base_config(
        "international_fixed_income", "targeted_ablation", "international_fixed_income_targeted_ablation"
    )
    international["configurations"] = [
        configuration("intfi_trend_market_global", "INTFI-TREND+MARKET+GLOBAL", TREND, MARKET_RISK + GLOBAL),
        configuration("intfi_trend_vix", "INTFI-TREND+VIX", TREND, ["vix_slope", "vix_ts_z_12m", "vix"]),
        configuration(
            "intfi_trend_equity_risk",
            "INTFI-TREND+EQUITY-RISK",
            TREND,
            ["spy_flow_z_12m", "spy_ret", "erp", "pct_above_50dma"],
        ),
        configuration("intfi_trend_em_fx", "INTFI-TREND+EM-FX", TREND, ["em_fx_ret"]),
        configuration("intfi_trend_commodity_ret", "INTFI-TREND+COMMODITY-RET", TREND, ["copper_ret", "oil_ret"]),
        configuration("intfi_trend_gold_oil", "INTFI-TREND+GOLD-OIL", TREND, ["gold_crude_ratio"]),
        configuration(
            "intfi_trend_global_vix",
            "INTFI-TREND+GLOBAL+VIX",
            TREND,
            GLOBAL + ["vix_slope", "vix_ts_z_12m", "vix"],
        ),
        configuration(
            "intfi_trend_global_equity_risk",
            "INTFI-TREND+GLOBAL+EQUITY-RISK",
            TREND,
            GLOBAL + ["spy_flow_z_12m", "spy_ret", "erp", "pct_above_50dma"],
        ),
    ]
    international["comparison_configurations"] = [
        configuration("intfi_trend_market_risk", "INTFI-TREND+MARKET-RISK", TREND, MARKET_RISK),
        configuration("intfi_trend_global", "INTFI-TREND+GLOBAL", TREND, GLOBAL),
        configuration("intfi_trend_all_macro", "INTFI-TREND+ALL-MACRO", TREND, ALL_MACRO),
        configuration("intfi_all_match", "INTFI-ALL-MATCH", ALL_LOCAL, ALL_MACRO),
    ]
    write_config("international_fixed_income_targeted_ablation_config.json", international)

    us_refinement = base_config("us_fixed_income", "refinement", "us_fixed_income_feature_refinement")
    us_compact = ["lag_y_excess_lead", "chmom", "mom12m", "cs_mom_rank", "ma_signal"]
    us_refinement["configurations"] = [
        *[
            configuration(
                f"usfi_compact_no_{feature}",
                f"USFI-COMPACT-NO-{feature.upper().replace('_', '-')}",
                [column for column in us_compact if column != feature],
                [],
            )
            for feature in us_compact
        ],
        *[
            configuration(
                f"usfi_compact_add_{feature}",
                f"USFI-COMPACT+{feature.upper().replace('_', '-')}",
                [*us_compact, feature],
                [],
            )
            for feature in ["baspread", "ret_kurt", "max_dd_6m", "ret_autocorr", "vol_z"]
        ],
    ]
    us_refinement["comparison_configurations"] = [
        configuration("usfi_trend_no_mom36", "USFI-TREND-NO-MOM36", us_compact, []),
        configuration("usfi_trend", "USFI-TREND", TREND, []),
    ]
    write_config("us_fixed_income_feature_refinement_config.json", us_refinement)

    international_refinement = base_config(
        "international_fixed_income", "refinement", "international_fixed_income_feature_refinement"
    )
    international_refinement["configurations"] = [
        *[
            configuration(
                f"intfi_all_macro_no_{feature}",
                f"INTFI-ALL-MACRO-NO-{feature.upper().replace('_', '-')}",
                TREND,
                [column for column in ALL_MACRO if column != feature],
            )
            for feature in ALL_MACRO
        ],
        *[
            configuration(
                f"intfi_all_macro_add_{feature}",
                f"INTFI-ALL-MACRO+{feature.upper().replace('_', '-')}",
                [*TREND, feature],
                ALL_MACRO,
            )
            for feature in ["baspread", "ret_kurt", "max_dd_6m", "ret_autocorr", "vol_z"]
        ],
    ]
    international_refinement["comparison_configurations"] = [
        configuration("intfi_trend_all_macro", "INTFI-TREND+ALL-MACRO", TREND, ALL_MACRO),
        configuration("intfi_all_match", "INTFI-ALL-MATCH", ALL_LOCAL, ALL_MACRO),
        configuration("intfi_trend_market_risk", "INTFI-TREND+MARKET-RISK", TREND, MARKET_RISK),
        configuration("intfi_trend_global", "INTFI-TREND+GLOBAL", TREND, GLOBAL),
    ]
    write_config("international_fixed_income_feature_refinement_config.json", international_refinement)

    us_combinations = base_config("us_fixed_income", "refinement", "us_fixed_income_combination_refinement")
    us_combination_specs = [
        ("no_mom12_add_vol_z", "NO-MOM12+VOL-Z", ["mom12m"], ["vol_z"]),
        ("no_mom12_add_ret_autocorr", "NO-MOM12+RET-AUTOCORR", ["mom12m"], ["ret_autocorr"]),
        ("no_cs_add_vol_z", "NO-CS+VOL-Z", ["cs_mom_rank"], ["vol_z"]),
        ("no_cs_add_ret_autocorr", "NO-CS+RET-AUTOCORR", ["cs_mom_rank"], ["ret_autocorr"]),
        ("no_mom12_no_cs", "NO-MOM12-NO-CS", ["mom12m", "cs_mom_rank"], []),
        ("no_mom12_no_cs_add_vol_z", "NO-MOM12-NO-CS+VOL-Z", ["mom12m", "cs_mom_rank"], ["vol_z"]),
        (
            "no_mom12_no_cs_add_ret_autocorr",
            "NO-MOM12-NO-CS+RET-AUTOCORR",
            ["mom12m", "cs_mom_rank"],
            ["ret_autocorr"],
        ),
    ]
    us_combinations["configurations"] = [
        configuration(
            f"usfi_compact_{identifier}",
            f"USFI-COMPACT-{label}",
            [*[column for column in us_compact if column not in removed], *added],
            [],
        )
        for identifier, label, removed, added in us_combination_specs
    ]
    us_combinations["comparison_configurations"] = [
        configuration("usfi_trend", "USFI-TREND", TREND, []),
        configuration("usfi_trend_no_mom36", "USFI-TREND-NO-MOM36", us_compact, []),
        configuration("usfi_compact_no_chmom", "USFI-COMPACT-NO-CHMOM", [c for c in us_compact if c != "chmom"], []),
        configuration("usfi_compact_no_mom12m", "USFI-COMPACT-NO-MOM12M", [c for c in us_compact if c != "mom12m"], []),
        configuration(
            "usfi_compact_no_cs_mom_rank",
            "USFI-COMPACT-NO-CS-MOM-RANK",
            [c for c in us_compact if c != "cs_mom_rank"],
            [],
        ),
        configuration("usfi_compact_add_ret_autocorr", "USFI-COMPACT+RET-AUTOCORR", [*us_compact, "ret_autocorr"], []),
        configuration("usfi_compact_add_vol_z", "USFI-COMPACT+VOL-Z", [*us_compact, "vol_z"], []),
    ]
    write_config("us_fixed_income_combination_refinement_config.json", us_combinations)

    international_combinations = base_config(
        "international_fixed_income", "refinement", "international_fixed_income_combination_refinement"
    )
    promising_removals = ["hy_spread_chg_1m", "erp", "cpi_mom", "copper_ret"]
    removal_specs = [
        ("no_hy_change_no_erp", ["hy_spread_chg_1m", "erp"]),
        ("no_hy_change_no_cpi_mom", ["hy_spread_chg_1m", "cpi_mom"]),
        ("no_hy_change_no_copper", ["hy_spread_chg_1m", "copper_ret"]),
        ("no_erp_no_cpi_mom", ["erp", "cpi_mom"]),
        ("no_erp_no_copper", ["erp", "copper_ret"]),
        ("no_cpi_mom_no_copper", ["cpi_mom", "copper_ret"]),
        ("no_hy_change_no_erp_no_cpi_mom", ["hy_spread_chg_1m", "erp", "cpi_mom"]),
        ("no_hy_change_no_erp_no_cpi_mom_no_copper", promising_removals),
    ]
    international_combinations["configurations"] = [
        *[
            configuration(
                f"intfi_all_macro_{identifier}",
                f"INTFI-ALL-MACRO-{identifier.upper().replace('_', '-')}",
                TREND,
                [column for column in ALL_MACRO if column not in removed],
            )
            for identifier, removed in removal_specs
        ],
        *[
            configuration(
                f"intfi_all_match_no_{feature}",
                f"INTFI-ALL-MATCH-NO-{feature.upper().replace('_', '-')}",
                ALL_LOCAL,
                [column for column in ALL_MACRO if column != feature],
            )
            for feature in promising_removals
        ],
    ]
    international_combinations["comparison_configurations"] = [
        configuration("intfi_all_match", "INTFI-ALL-MATCH", ALL_LOCAL, ALL_MACRO),
        configuration("intfi_trend_market_risk", "INTFI-TREND+MARKET-RISK", TREND, MARKET_RISK),
        configuration("intfi_trend_global", "INTFI-TREND+GLOBAL", TREND, GLOBAL),
        *[
            configuration(
                f"intfi_all_macro_no_{feature}",
                f"INTFI-ALL-MACRO-NO-{feature.upper().replace('_', '-')}",
                TREND,
                [column for column in ALL_MACRO if column != feature],
            )
            for feature in promising_removals
        ],
    ]
    write_config("international_fixed_income_combination_refinement_config.json", international_combinations)


if __name__ == "__main__":
    main()
