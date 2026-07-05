"""Run first-round U.S. equity ETF GP/MTGP walk-forward tests.

This is a thin, experiment-scoped wrapper around the June portfolio
walk-forward runner. The source runner is reusable, but its task universe and
feature blocks are module-level constants, so this wrapper patches them before
delegating to the portfolio code.
"""

from __future__ import annotations

import argparse
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
        "macro": ["vix", "vix_slope", "spy_ret", "pct_above_50dma", "hy_spread_chg_1p", "hy_spread_z_12p", "cpi_chg_12p", "term_spread"],
    },
    "target_plus_momentum": {
        "etf": ["lag_y_excess_lead", "lag2_y_excess_lead", "mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": [],
    },
    "target_plus_volatility": {
        "etf": ["lag_y_excess_lead", "lag2_y_excess_lead", "vol_1m", "vol_z", "vol_accel", "vol_ratio_1m_3m", "max_dd_3m"],
        "macro": [],
    },
    "target_plus_macro": {
        "etf": ["lag_y_excess_lead", "lag2_y_excess_lead"],
        "macro": ["vix", "vix_slope", "spy_ret", "pct_above_50dma", "hy_spread_chg_1p", "hy_spread_z_12p", "cpi_chg_12p", "term_spread"],
    },
    "momentum_plus_volatility": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope", "vol_1m", "vol_z", "vol_accel", "vol_ratio_1m_3m", "max_dd_3m"],
        "macro": [],
    },
    "momentum_plus_macro": {
        "etf": ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"],
        "macro": ["vix", "vix_slope", "spy_ret", "pct_above_50dma", "hy_spread_chg_1p", "hy_spread_z_12p", "cpi_chg_12p", "term_spread"],
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


def load_portfolio_runner() -> Any:
    spec = importlib.util.spec_from_file_location("us_equity_portfolio_runner", PORTFOLIO_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load portfolio runner from {PORTFOLIO_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def patch_first_round_constants(runner: Any, *, mtgp_rank: int, feature_set: str) -> None:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}. Expected one of: {sorted(FEATURE_SETS)}")
    feature_spec = FEATURE_SETS[feature_set]
    etf_cols = list(feature_spec["etf"])
    macro_cols = list(feature_spec["macro"])
    task_exp = runner.task_exp
    task_exp.ETF_TICKERS = FIRST_ROUND_TICKERS
    task_exp.ASSET_GROUPS = {ticker: "us_equity" for ticker in FIRST_ROUND_TICKERS}
    task_exp.TIME_COLS = TIME_COLS
    task_exp.ETF_COLS = etf_cols
    task_exp.MACRO_COLS = macro_cols
    task_exp.INPUT_COLUMNS = [*TIME_COLS, *etf_cols, *macro_cols]
    task_exp.RANK = int(mtgp_rank)
    task_exp.build_covar_config = dynamic_covar_config(task_exp, etf_cols=etf_cols, macro_cols=macro_cols)

    runner.HELPER_ASSETS = set()
    runner.PERIODS_PER_YEAR = 12
    runner.SCHWAB_MODERATE_AGGRESSIVE_TARGET_WEIGHTS = {
        "SPY": 0.45,
        "MGK": 0.15,
        "VTV": 0.20,
        "IWM": 0.15,
    }
    runner.SCHWAB_MODERATE_AGGRESSIVE_CASH_WEIGHT = 0.05


def dynamic_covar_config(task_exp: Any, *, etf_cols: list[str], macro_cols: list[str]) -> Any:
    def build() -> Any:
        idx = {name: i for i, name in enumerate([*TIME_COLS, *etf_cols, *macro_cols])}
        adaptive = task_exp.LengthscalePolicyConfig(policy=task_exp.LengthscalePolicy.ADAPTIVE)
        blocks = [
            task_exp.KernelBlockConfig(
                name="time",
                variable_type=task_exp.KernelBlockRole.TIME,
                components=[
                    task_exp.MaternKernelComponentConfig(
                        dims=[idx[c] for c in TIME_COLS],
                        matern_nu=0.5,
                        ard=True,
                        use_outputscale=True,
                        lengthscale_policy=adaptive,
                    )
                ],
                block_structure=task_exp.BlockStructure.ADDITIVE,
                use_outputscale=False,
            )
        ]
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
                )
            )
        if macro_cols:
            macro_dims = [idx[c] for c in macro_cols]
            blocks.append(
                task_exp.KernelBlockConfig(
                    name="macro",
                    variable_type=task_exp.KernelBlockRole.MACRO,
                    components=[
                        task_exp.MaternKernelComponentConfig(
                            dims=macro_dims,
                            matern_nu=0.5,
                            ard=True,
                            use_outputscale=True,
                            lengthscale_policy=adaptive,
                        ),
                        task_exp.RQKernelComponentConfig(
                            dims=macro_dims,
                            ard=True,
                            use_outputscale=True,
                            lengthscale_policy=adaptive,
                        ),
                        task_exp.LinearKernelComponentConfig(dims=macro_dims, use_outputscale=True),
                    ],
                    block_structure=task_exp.BlockStructure.ADDITIVE,
                    use_outputscale=False,
                )
            )
        custom_interactions = []
        block_names = {block.name for block in blocks}
        if {"time", "etf"}.issubset(block_names):
            custom_interactions.append(
                task_exp.KernelInteractionConfig(blocks=["time", "etf"], name="time_x_etf", use_outputscale=True)
            )
        if {"time", "macro"}.issubset(block_names):
            custom_interactions.append(
                task_exp.KernelInteractionConfig(blocks=["time", "macro"], name="time_x_macro", use_outputscale=True)
            )
        if {"macro", "etf"}.issubset(block_names):
            custom_interactions.append(
                task_exp.KernelInteractionConfig(blocks=["macro", "etf"], name="macro_x_etf", use_outputscale=True)
            )
        return task_exp.CovarModuleConfig(
            blocks=blocks,
            global_structure=task_exp.GlobalStructure.HIERARCHICAL,
            interaction_policy=task_exp.InteractionPolicy.CUSTOM,
            custom_interactions=custom_interactions,
        )

    return build


def parse_wrapper_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=False,
    )
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--mtgp-rank", type=int, default=2)
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="compact_pruned")
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
            "mean_gaussian_nll": float(np.mean(0.5 * (np.log(2.0 * math.pi * variance) + np.square(residual) / variance))),
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
    metric_columns = ["asset_id", "n", "rmse", "mae", "bias", "mean_y_true", "mean_y_pred", "std_y_true", "std_y_pred", "mean_pred_std", "mean_gaussian_nll", "standardized_residual_mean", "standardized_residual_std", *NORMAL_COVERAGE_Z]
    by_asset = pd.DataFrame(forecast_metric_rows(predictions, group_col="asset_id")).loc[:, metric_columns]
    overall = pd.DataFrame(forecast_metric_rows(predictions, group_col=None)).loc[:, metric_columns]
    by_asset.to_csv(output_dir / "forecast_metrics_by_asset.csv", index=False)
    overall.to_csv(output_dir / "forecast_metrics_overall.csv", index=False)


def write_wrapper_metadata(output_dir: Path, *, feature_set: str, mtgp_rank: int) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    feature_spec = FEATURE_SETS[feature_set]
    manifest["us_equity_first_round_wrapper"] = {
        "feature_set": feature_set,
        "mtgp_rank": int(mtgp_rank),
        "first_round_tickers": FIRST_ROUND_TICKERS,
        "time_columns": TIME_COLS,
        "etf_columns": list(feature_spec["etf"]),
        "macro_columns": list(feature_spec["macro"]),
        "input_columns": [*TIME_COLS, *feature_spec["etf"], *feature_spec["macro"]],
        "non_time_feature_count": len(feature_spec["etf"]) + len(feature_spec["macro"]),
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

    patch_first_round_constants(runner, mtgp_rank=wrapper_args.mtgp_rank, feature_set=wrapper_args.feature_set)
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
        write_wrapper_metadata(Path(args.output_dir), feature_set=wrapper_args.feature_set, mtgp_rank=wrapper_args.mtgp_rank)
        write_forecast_metrics(Path(args.output_dir))


if __name__ == "__main__":
    main()
