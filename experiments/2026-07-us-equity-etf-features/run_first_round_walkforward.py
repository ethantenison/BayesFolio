"""Run first-round U.S. equity ETF GP/MTGP walk-forward tests.

This is a thin, experiment-scoped wrapper around the June portfolio
walk-forward runner. The source runner is reusable, but its task universe and
feature blocks are module-level constants, so this wrapper patches them before
delegating to the portfolio code.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def load_portfolio_runner() -> Any:
    spec = importlib.util.spec_from_file_location("us_equity_portfolio_runner", PORTFOLIO_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load portfolio runner from {PORTFOLIO_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def patch_first_round_constants(runner: Any, *, mtgp_rank: int) -> None:
    task_exp = runner.task_exp
    task_exp.ETF_TICKERS = FIRST_ROUND_TICKERS
    task_exp.ASSET_GROUPS = {ticker: "us_equity" for ticker in FIRST_ROUND_TICKERS}
    task_exp.TIME_COLS = TIME_COLS
    task_exp.ETF_COLS = ETF_COLS
    task_exp.MACRO_COLS = MACRO_COLS
    task_exp.INPUT_COLUMNS = INPUT_COLUMNS
    task_exp.RANK = int(mtgp_rank)

    runner.HELPER_ASSETS = set()
    runner.PERIODS_PER_YEAR = 12
    runner.SCHWAB_MODERATE_AGGRESSIVE_TARGET_WEIGHTS = {
        "SPY": 0.45,
        "MGK": 0.15,
        "VTV": 0.20,
        "IWM": 0.15,
    }
    runner.SCHWAB_MODERATE_AGGRESSIVE_CASH_WEIGHT = 0.05


def parse_wrapper_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=False,
    )
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--mtgp-rank", type=int, default=2)
    parser.add_argument("-h", "--help", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def main(argv: list[str] | None = None) -> None:
    wrapper_args, remaining = parse_wrapper_args(argv)
    runner = load_portfolio_runner()
    if wrapper_args.help:
        print(__doc__)
        print("\nWrapper defaults:")
        print(f"  --feature-path {DEFAULT_FEATURE_PATH}")
        print(f"  --output-root {DEFAULT_OUTPUT_ROOT}")
        print("  --mtgp-rank 2")
        print("\nDelegated runner help:\n")
        original_argv = sys.argv
        try:
            sys.argv = [str(PORTFOLIO_RUNNER_PATH), "--help"]
            runner.parse_args()
        finally:
            sys.argv = original_argv
        return

    patch_first_round_constants(runner, mtgp_rank=wrapper_args.mtgp_rank)
    run_id = wrapper_args.run_id or f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_first_round_rank{wrapper_args.mtgp_rank}"
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


if __name__ == "__main__":
    main()
