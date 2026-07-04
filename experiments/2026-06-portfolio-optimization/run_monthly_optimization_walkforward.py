"""Walk-forward monthly portfolio optimization from multitask GP scenarios.

Usage:
    poetry run python experiments/2026-06-portfolio-optimization/run_monthly_optimization_walkforward.py \
        --run-id 20260616_gp_scenario_portfolio --maxiter 50
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import riskfolio as rp
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import StratifiedStandardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_EXPERIMENT = REPO_ROOT / "experiments" / "2026-06-task-covariance" / "run_task_covariance_rollforward.py"
SPEC = importlib.util.spec_from_file_location("task_covariance_rollforward", SOURCE_EXPERIMENT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load source experiment from {SOURCE_EXPERIMENT}")
task_exp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = task_exp
SPEC.loader.exec_module(task_exp)

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_PATH = Path("/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_06.parquet")
OUTPUT_ROOT = EXPERIMENT_DIR / "outputs"

HELPER_ASSETS = {"MGK", "BND"}
STARTING_VALUE = 10_000.0
PERIODS_PER_YEAR = 12
SIGNED_EXPERIMENTS = {
    "signed_no_prior",
    "signed_lkj_eta_2",
    "signed_lkj_eta_2_turnover_blend",
    "signed_lkj_eta_2_lengthscale_floor",
    "signed_lkj_eta_2_task_noise_floor",
}
LENGTHSCALE_FLOOR_EXPERIMENTS = {"lengthscale_floor", "signed_lkj_eta_2_lengthscale_floor"}
OUTPUTSCALE_FLOOR_EXPERIMENTS = {"component_outputscale_floor", "positive_no_prior_outputscale_floor"}
TURNOVER_BLEND_EXPERIMENTS = {"turnover_blend", "signed_lkj_eta_2_turnover_blend"}
TASK_NOISE_FLOOR_EXPERIMENTS = {"signed_lkj_eta_2_task_noise_floor"}


@dataclass(frozen=True)
class StrategyResult:
    name: str
    weights: pd.DataFrame
    returns: pd.Series


@dataclass(frozen=True)
class RiskfolioOptimizationResult:
    weights: pd.Series
    status: str
    fallback_stage: str
    clean_asset_count: int
    clean_observation_count: int
    message: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--posterior-scenarios", type=int, default=5000)
    parser.add_argument(
        "--periods-per-year",
        type=float,
        default=PERIODS_PER_YEAR,
        help="Annualization factor for the realized rebalance horizon.",
    )
    parser.add_argument(
        "--rebalance-frequency-label",
        type=str,
        default="monthly",
        help="Human-readable cadence label stored in the manifest/report.",
    )
    parser.add_argument(
        "--train-months",
        type=int,
        default=None,
        help=(
            "If set, fit each rebalance window on only this many calendar months before the "
            "construction date. Evaluation dates are unchanged."
        ),
    )
    parser.add_argument(
        "--include-live-window",
        action="store_true",
        help="Append the unlabeled live tail date as a portfolio construction point with NaN realized return.",
    )
    parser.add_argument(
        "--drop-incomplete-feature-dates",
        action="store_true",
        help="Drop dates before the first month where all model input feature cells are complete.",
    )
    parser.add_argument(
        "--min-feature-date",
        type=str,
        default=None,
        help="If set, drop feature rows before this YYYY-MM-DD date before scoring windows are selected.",
    )
    parser.add_argument(
        "--min-scored-date",
        type=str,
        default=None,
        help="If set, keep full training history but only evaluate scored windows on/after this YYYY-MM-DD date.",
    )
    parser.add_argument(
        "--min-inferred-noise-level",
        type=float,
        default=5e-3,
        help="Scalar likelihood noise variance floor in the transformed/standardized target space.",
    )
    parser.add_argument("--upperlng", type=float, default=0.20)
    parser.add_argument("--nea", type=int, default=10)
    parser.add_argument("--historical-method-mu", type=str, default="ewma2")
    parser.add_argument("--historical-method-cov", type=str, default="gerber1")
    parser.add_argument(
        "--gp-experiment",
        choices=[
            "control",
            "positive_no_prior",
            "signed_no_prior",
            "signed_lkj_eta_2",
            "lengthscale_floor",
            "component_outputscale_floor",
            "positive_no_prior_outputscale_floor",
            "scenario_mean_scale",
            "turnover_blend",
            "signed_lkj_eta_2_turnover_blend",
            "signed_lkj_eta_2_lengthscale_floor",
            "signed_lkj_eta_2_task_noise_floor",
        ],
        default="control",
    )
    parser.add_argument(
        "--input-transform-mode",
        choices=["botorch_normalize", "june_manual_minmax"],
        default="botorch_normalize",
        help=(
            "Feature scaling mode. june_manual_minmax mirrors notebooks/20260601_portfolio.py: "
            "manual min-max scaling of non-task columns before fitting, then input_transform=None."
        ),
    )
    parser.add_argument(
        "--time-modulation-mode",
        choices=task_exp.TIME_MODULATION_MODES,
        default="lengthscale_only",
        help=(
            "Existing GPArchitect time-varying hyperparameter wrapper. "
            "Use 'neither' for the time-decay kernel experiments."
        ),
    )
    parser.add_argument(
        "--kernel-proposal",
        choices=["none", "global_time_gate", "forecast_recency_gate", "changepoint_mixture"],
        default="none",
        help="Optional time-decay kernel construction applied to the data covariance before fitting.",
    )
    parser.add_argument(
        "--kernel-half-life-months",
        type=float,
        default=36.0,
        help="Half-life used by global_time_gate and forecast_recency_gate.",
    )
    parser.add_argument(
        "--kernel-changepoint-date",
        type=str,
        default="2021-03-31",
        help="Calendar changepoint used by changepoint_mixture.",
    )
    parser.add_argument(
        "--kernel-changepoint-width-months",
        type=float,
        default=6.0,
        help="Sigmoid transition width used by changepoint_mixture.",
    )
    parser.add_argument(
        "--kernel-composition-proposal",
        choices=[
            "none",
            "e_plus_mt_plus_t_plus_emt",
            "et_plus_mt_plus_emt",
            "e_plus_m_plus_t_plus_et_plus_mt_plus_emt",
            "e_plus_m_plus_t_plus_em_plus_et_plus_mt_plus_emt",
            "e_plus_et_plus_mt_plus_emt",
            "t_plus_et_plus_mt_plus_emt",
            "et_plus_mt15_plus_emt",
            "et_plus_mt_plus_emt_t15",
            "t_plus_et_plus_mt15_plus_emt",
            "et_plus_mt_plus_em_plus_emt",
            "et_plus_mt_plus_em",
        ],
        default="none",
        help=(
            "Optional explicit E/M/T covariance composition. "
            "E is ETF Matern(0.5), M is macro Linear+Matern(0.5)+RQ, T is time Matern(0.5)."
        ),
    )
    parser.add_argument(
        "--mean-kind",
        choices=["multitask_constant", "multitask_linear"],
        default="multitask_constant",
        help="Mean module used by the multitask GP.",
    )
    parser.add_argument("--scenario-mean-scale", type=float, default=1.0)
    parser.add_argument("--turnover-blend", type=float, default=0.50)
    parser.add_argument("--lengthscale-floor", type=float, default=0.02)
    parser.add_argument("--outputscale-floor", type=float, default=0.01)
    parser.add_argument("--outputscale-prior-median", type=float, default=0.05)
    parser.add_argument("--outputscale-prior-sigma", type=float, default=0.75)
    parser.add_argument(
        "--recency-half-life-months",
        type=float,
        default=None,
        help=(
            "If set, use fixed observation noise with exponential recency decay. "
            "A row this many months old receives half the recent-row weight."
        ),
    )
    parser.add_argument(
        "--recency-base-noise-level",
        type=float,
        default=5e-3,
        help="Recent-row fixed observation-noise variance in StratifiedStandardize target units.",
    )
    parser.add_argument(
        "--recency-max-noise-multiplier",
        type=float,
        default=100.0,
        help="Maximum old-row noise multiplier relative to recency-base-noise-level.",
    )
    parser.add_argument(
        "--task-noise-floor-raw-std",
        type=float,
        default=0.005,
        help=(
            "Monthly raw-return noise std floor used by task-noise-floor experiments. "
            "The script converts this to each ETF's StratifiedStandardize scale per window."
        ),
    )
    return parser.parse_args()


def git_sha(*, short: bool = False) -> str:
    cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
    return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()


def git_dirty_summary() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    run_id = args.run_id or f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{git_sha(short=True)}"
    return OUTPUT_ROOT / "runs" / run_id


def historical_strategy_name(args: argparse.Namespace) -> str:
    return f"historical_y_{args.historical_method_mu}_{args.historical_method_cov}_riskfolio"


def build_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    df: pd.DataFrame,
    scored_dates: list[pd.Timestamp],
    live_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    final_universe = [asset for asset in task_exp.ETF_TICKERS if asset not in HELPER_ASSETS]
    variant_name = gp_variant_name(args.gp_experiment)
    is_signed = args.gp_experiment in SIGNED_EXPERIMENTS
    uses_lengthscale_floor = args.gp_experiment in LENGTHSCALE_FLOOR_EXPERIMENTS
    uses_turnover_blend = args.gp_experiment in TURNOVER_BLEND_EXPERIMENTS
    uses_task_noise_floor = args.gp_experiment in TASK_NOISE_FLOOR_EXPERIMENTS
    return {
        "schema": "bayesfolio.portfolio_optimization_walkforward.manifest.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git_sha": git_sha(short=False),
        "git_dirty_summary_at_start": git_dirty_summary(),
        "feature_path": str(args.feature_path),
        "feature_sha256": sha256_file(args.feature_path),
        "source_experiment": str(SOURCE_EXPERIMENT),
        "output_dir": str(output_dir),
        "data": {
            "rows": int(len(df)),
            "date_min": df["date"].min().date().isoformat(),
            "date_max": df["date"].max().date().isoformat(),
            "min_feature_date_filter": args.min_feature_date,
            "min_scored_date_filter": args.min_scored_date,
            "target_col": task_exp.TARGET_COL,
            "training_universe": task_exp.ETF_TICKERS,
            "helper_assets_fit_but_excluded": sorted(HELPER_ASSETS),
            "final_portfolio_universe": final_universe,
        },
        "training_history": {
            "mode": "rolling_calendar_months" if args.train_months is not None else "all_available_before_window",
            "train_months": args.train_months,
            "window_rule": (
                "date >= window_date - DateOffset(months=train_months) and date < window_date"
                if args.train_months is not None
                else "date < window_date"
            ),
        },
        "rebalance_dates": [date.date().isoformat() for date in scored_dates],
        "live_rebalance_date": live_date.date().isoformat() if live_date is not None else None,
        "portfolio_construction_dates": [
            date.date().isoformat() for date in [*scored_dates, *([live_date] if live_date is not None else [])]
        ],
        "portfolio": {
            "starting_value": STARTING_VALUE,
            "rebalance_frequency": args.rebalance_frequency_label,
            "periods_per_year": args.periods_per_year,
            "n_realized_rebalances": len(scored_dates),
            "n_portfolio_construction_dates": len(scored_dates) + int(live_date is not None),
            "strategies": [
                "gp_scenarios_riskfolio",
                historical_strategy_name(args),
                "equal_weight",
            ],
            "riskfolio_gp": {
                "model": "Classic",
                "rm": "CVaR",
                "obj": "Sharpe",
                "method_mu": "hist",
                "method_cov": "hist",
                "hist": True,
                "upperlng": args.upperlng,
                "nea": args.nea,
                "alpha": 0.5,
            },
            "riskfolio_historical": {
                "model": "Classic",
                "rm": "CVaR",
                "obj": "Sharpe",
                "method_mu": args.historical_method_mu,
                "method_cov": args.historical_method_cov,
                "hist": True,
                "upperlng": args.upperlng,
                "nea": args.nea,
            },
        },
        "modeling": {
            "experiment": args.gp_experiment,
            "variant": variant_name,
            "task_kernel": "IndexKernel" if is_signed else "PositiveIndexKernel",
            "task_covar_prior": None
            if variant_name in {"signed_no_prior", "positive_no_prior"}
            else ("BetaPrior(2.5, 1.5)"),
            "min_inferred_noise_level": args.min_inferred_noise_level,
            "time_modulation_mode": args.time_modulation_mode,
            "kernel_proposal": {
                "name": args.kernel_proposal,
                "half_life_months": args.kernel_half_life_months
                if args.kernel_proposal in {"global_time_gate", "forecast_recency_gate"}
                else None,
                "changepoint_date": args.kernel_changepoint_date
                if args.kernel_proposal == "changepoint_mixture"
                else None,
                "changepoint_width_months": args.kernel_changepoint_width_months
                if args.kernel_proposal == "changepoint_mixture"
                else None,
            },
            "kernel_composition_proposal": {
                "name": args.kernel_composition_proposal,
                "components": (
                    "E=Matern(0.5), M=Linear+Matern(0.5)+RQ, T=Matern(0.5)"
                    if args.kernel_composition_proposal != "none"
                    else None
                ),
            },
            "lengthscale_floor": args.lengthscale_floor if uses_lengthscale_floor else None,
            "outputscale_floor": args.outputscale_floor
            if args.gp_experiment in OUTPUTSCALE_FLOOR_EXPERIMENTS
            else None,
            "outputscale_prior_median": args.outputscale_prior_median
            if args.gp_experiment in OUTPUTSCALE_FLOOR_EXPERIMENTS
            else None,
            "outputscale_prior_sigma": args.outputscale_prior_sigma
            if args.gp_experiment in OUTPUTSCALE_FLOOR_EXPERIMENTS
            else None,
            "scenario_mean_scale": args.scenario_mean_scale if args.gp_experiment == "scenario_mean_scale" else 1.0,
            "turnover_blend": args.turnover_blend if uses_turnover_blend else None,
            "task_noise_floor": (
                {
                    "enabled": True,
                    "raw_monthly_std_floor": args.task_noise_floor_raw_std,
                    "standardized_noise_variance_floor": (
                        "raw_monthly_std_floor ** 2 divided by each ETF training-window target variance"
                    ),
                    "scope": "likelihood noise lower bound; per ETF task; recomputed each rebalance window",
                }
                if uses_task_noise_floor
                else {"enabled": False}
            ),
            "recency_observation_noise": (
                {
                    "enabled": True,
                    "half_life_months": args.recency_half_life_months,
                    "base_standardized_noise_variance": args.recency_base_noise_level,
                    "max_noise_multiplier": args.recency_max_noise_multiplier,
                    "scope": (
                        "fixed train_Yvar per row; raw variance equals ETF training target variance "
                        "times standardized noise level divided by exponential recency weight"
                    ),
                }
                if args.recency_half_life_months is not None
                else {"enabled": False}
            ),
            "rank": task_exp.RANK,
            "outcome_transform": "StratifiedStandardize by ETF task",
            "input_transform": input_transform_description(args.input_transform_mode),
            "posterior_scenarios_per_rebalance": args.posterior_scenarios,
            "maxiter": args.maxiter,
            "seed": args.seed,
            "mean_kind": args.mean_kind,
        },
    }


def gp_variant_name(experiment: str) -> str:
    if experiment in {"positive_no_prior", "positive_no_prior_outputscale_floor"}:
        return "positive_no_prior"
    if experiment == "signed_no_prior":
        return "signed_no_prior"
    if experiment in SIGNED_EXPERIMENTS:
        return "signed_lkj_eta_2"
    return "positive_beta_prior"


def input_transform_description(mode: str) -> str:
    if mode == "june_manual_minmax":
        return "June notebook manual min-max on non-task columns; input_transform=None"
    return "BoTorch Normalize on non-task feature columns"


def training_slice(df: pd.DataFrame, window_date: pd.Timestamp, train_months: int | None) -> pd.DataFrame:
    train_df = df[(df["date"] < window_date) & df[task_exp.TARGET_COL].notna()].copy()
    if train_months is None:
        return train_df
    cutoff = window_date - pd.DateOffset(months=train_months)
    return train_df[train_df["date"] >= cutoff].copy()


def apply_min_feature_date(df: pd.DataFrame, min_feature_date: str | None) -> pd.DataFrame:
    if min_feature_date is None:
        return df
    cutoff = pd.Timestamp(min_feature_date)
    return df[df["date"] >= cutoff].copy().reset_index(drop=True)


def apply_min_scored_date(scored_dates: list[pd.Timestamp], min_scored_date: str | None) -> list[pd.Timestamp]:
    if min_scored_date is None:
        return scored_dates
    cutoff = pd.Timestamp(min_scored_date)
    return [date for date in scored_dates if date >= cutoff]


def drop_incomplete_feature_dates(df: pd.DataFrame) -> pd.DataFrame:
    feature_na_by_date = df.groupby("date", observed=True)[task_exp.INPUT_COLUMNS].apply(
        lambda frame: int(frame.isna().sum().sum())
    )
    complete_dates = feature_na_by_date[feature_na_by_date == 0].index
    if complete_dates.empty:
        raise ValueError("No complete feature dates found in feature panel.")
    first_complete_date = complete_dates.min()
    return df[df["date"] >= first_complete_date].copy().reset_index(drop=True)


def optimize_riskfolio(
    returns: pd.DataFrame,
    *,
    method_mu: str,
    method_cov: str,
    upperlng: float,
    nea: int,
    fallback_weights: pd.Series | None = None,
    fallback_label: str = "equal_weight",
) -> RiskfolioOptimizationResult:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any").dropna(axis=0, how="any")
    clean_asset_count = int(clean.shape[1])
    clean_observation_count = int(clean.shape[0])
    if clean.shape[1] < 2 or clean.empty:
        weights = normalize_fallback_weights(fallback_weights, clean.columns)
        stage = fallback_label if fallback_weights is not None and not weights.empty else "equal_weight_last_resort"
        if weights.empty:
            weights = equal_weight(clean.columns.tolist())
        return RiskfolioOptimizationResult(
            weights=weights,
            status="fallback",
            fallback_stage=stage,
            clean_asset_count=clean_asset_count,
            clean_observation_count=clean_observation_count,
            message="Riskfolio input had fewer than two clean assets or no clean observations.",
        )

    attempts = [
        {"stage": "primary", "nea": max(1, min(int(nea), clean.shape[1] - 1)), "upperlng": upperlng},
        {"stage": "relaxed_no_nea", "nea": None, "upperlng": upperlng},
        {"stage": "relaxed_no_nea_cap35", "nea": None, "upperlng": max(upperlng, 0.35)},
    ]
    messages: list[str] = []
    for attempt in attempts:
        try:
            weights = solve_riskfolio(
                clean,
                method_mu=method_mu,
                method_cov=method_cov,
                upperlng=float(attempt["upperlng"]),
                nea=attempt["nea"],
            )
            return RiskfolioOptimizationResult(
                weights=weights,
                status="solved" if attempt["stage"] == "primary" else "relaxed_solved",
                fallback_stage=str(attempt["stage"]),
                clean_asset_count=clean_asset_count,
                clean_observation_count=clean_observation_count,
            )
        except Exception as exc:
            messages.append(f"{attempt['stage']}: {type(exc).__name__}: {exc}")

    weights = normalize_fallback_weights(fallback_weights, clean.columns)
    if not weights.empty:
        return RiskfolioOptimizationResult(
            weights=weights,
            status="fallback",
            fallback_stage=fallback_label,
            clean_asset_count=clean_asset_count,
            clean_observation_count=clean_observation_count,
            message="; ".join(messages),
        )
    return RiskfolioOptimizationResult(
        weights=equal_weight(clean.columns.tolist()),
        status="fallback",
        fallback_stage="equal_weight_last_resort",
        clean_asset_count=clean_asset_count,
        clean_observation_count=clean_observation_count,
        message="; ".join(messages),
    )


def solve_riskfolio(
    clean: pd.DataFrame,
    *,
    method_mu: str,
    method_cov: str,
    upperlng: float,
    nea: int | None,
) -> pd.Series:
    n_assets = clean.shape[1]
    portfolio = rp.Portfolio(returns=clean) if nea is None else rp.Portfolio(returns=clean, nea=nea)
    portfolio.upperlng = max(float(upperlng), 1.0 / n_assets)
    portfolio.lowerlng = 0.0
    portfolio.card = None
    portfolio.alpha = 0.5
    portfolio.assets_stats(method_mu=method_mu, method_cov=method_cov)
    weights_df = portfolio.optimization(
        model="Classic",
        rm="CVaR",
        obj="Sharpe",
        rf=0.0,
        hist=True,
    )
    if weights_df is None or weights_df.empty:
        raise RuntimeError("Riskfolio returned empty weights")
    weights = weights_df.iloc[:, 0].astype(float)
    weights = weights.reindex(clean.columns).fillna(0.0).clip(lower=0.0)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Riskfolio returned non-positive or non-finite total weight")
    return weights / total


def normalize_fallback_weights(weights: pd.Series | None, assets: pd.Index) -> pd.Series:
    if weights is None:
        return pd.Series(dtype=float)
    normalized = weights.reindex(assets).fillna(0.0).clip(lower=0.0).astype(float)
    total = float(normalized.sum())
    if not np.isfinite(total) or total <= 0:
        return pd.Series(dtype=float)
    return normalized / total


def equal_weight(assets: list[str] | pd.Index) -> pd.Series:
    assets = list(assets)
    if not assets:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(assets), index=assets, dtype=float)


def build_time_varying_lengthscale_floor_kernel(
    base_kernel: gpytorch.kernels.Kernel,
    time_feature_index: int,
    *,
    floor: float,
) -> gpytorch.kernels.Kernel:
    """Build the lengthscale-only wrapper with a positive effective floor."""

    class _Module(gpytorch.kernels.Kernel):
        has_lengthscale = False

        def __init__(self) -> None:
            super().__init__()
            self.base_kernel = base_kernel
            self.time_feature_index = time_feature_index
            self.floor = floor
            self.register_parameter("raw_tv_bias", torch.nn.Parameter(torch.zeros(1)))
            self.register_parameter("raw_tv_slope", torch.nn.Parameter(torch.zeros(1)))

        def _effective_lengthscale(self, x: torch.Tensor) -> torch.Tensor:
            t = x[..., self.time_feature_index]
            return torch.nn.functional.softplus(self.raw_tv_bias + self.raw_tv_slope * t) + self.floor

        def _warp_time(self, x: torch.Tensor) -> torch.Tensor:
            lengthscale = self._effective_lengthscale(x)
            x_warped = x.clone()
            x_warped[..., self.time_feature_index] = x[..., self.time_feature_index] / lengthscale
            return x_warped

        def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs: object) -> torch.Tensor:
            return self.base_kernel(self._warp_time(x1), self._warp_time(x2), **kwargs).to_dense()

    return _Module()


def apply_outputscale_floor_prior(
    model: torch.nn.Module,
    *,
    floor: float,
    prior_median: float,
    prior_sigma: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prior = LogNormalPrior(loc=float(np.log(prior_median)), scale=prior_sigma)
    constraint = GreaterThan(floor)
    for index, (path, module) in enumerate(model.named_modules()):
        if not isinstance(module, ScaleKernel):
            continue
        module.register_constraint("raw_outputscale", constraint)
        module.register_prior("floor_lognormal_outputscale_prior", prior, "outputscale")
        module.initialize(outputscale=max(prior_median, floor * 1.01))
        rows.append(
            {
                "scale_index": index,
                "module_path": path,
                "base_kernel_type": module.base_kernel.__class__.__name__,
                "outputscale_floor": floor,
                "outputscale_prior_median": prior_median,
                "outputscale_prior_sigma": prior_sigma,
            }
        )
    return rows


def apply_task_noise_floor(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    *,
    raw_std_floor: float,
) -> list[dict[str, Any]]:
    """Set task-specific likelihood noise floors in standardized target units."""

    if raw_std_floor <= 0:
        raise ValueError(f"task noise raw std floor must be positive, got {raw_std_floor}")
    if not hasattr(model, "likelihood") or not hasattr(model.likelihood, "noise_covar"):
        return []

    grouped_std = train_df.groupby("asset_id", observed=True)[task_exp.TARGET_COL].std(ddof=1)
    global_std = float(train_df[task_exp.TARGET_COL].std(ddof=1))
    if not np.isfinite(global_std) or global_std <= 0:
        global_std = raw_std_floor

    floors: list[float] = []
    rows: list[dict[str, Any]] = []
    for task_id, asset_id in enumerate(task_exp.ETF_TICKERS):
        train_std = float(grouped_std.get(asset_id, global_std))
        if not np.isfinite(train_std) or train_std <= 0:
            train_std = global_std
        standardized_variance_floor = float((raw_std_floor / max(train_std, 1e-12)) ** 2)
        floors.append(standardized_variance_floor)
        rows.append(
            {
                "task_id": task_id,
                "asset_id": asset_id,
                "raw_monthly_std_floor": raw_std_floor,
                "train_target_std": train_std,
                "standardized_noise_variance_floor": standardized_variance_floor,
                "standardized_noise_std_floor": float(math.sqrt(standardized_variance_floor)),
            }
        )

    floor_tensor = torch.tensor(floors, dtype=model.likelihood.noise.dtype, device=model.likelihood.noise.device)
    initial = torch.maximum(floor_tensor * 1.25, floor_tensor + 1e-6)
    model.likelihood.noise_covar.register_constraint(
        "raw_noise",
        GreaterThan(floor_tensor, initial_value=initial),
    )
    model.likelihood.noise_covar.initialize(noise=initial)
    return rows


def build_experiment_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_yvar: torch.Tensor | None,
    *,
    args: argparse.Namespace,
) -> Any:
    variant_name = gp_variant_name(args.gp_experiment)
    uses_lengthscale_floor = args.gp_experiment in LENGTHSCALE_FLOOR_EXPERIMENTS
    variant = task_exp.VARIANTS[variant_name]
    original_gp_builder = task_exp.build_multitask_gp

    def noise_floor_builder(**kwargs: object) -> Any:
        kwargs["min_inferred_noise_level"] = args.min_inferred_noise_level
        return original_gp_builder(**kwargs)

    task_exp.build_multitask_gp = noise_floor_builder
    try:
        if not uses_lengthscale_floor:
            return build_model_with_transform_mode(
                train_x,
                train_y,
                train_yvar,
                variant,
                input_transform_mode=args.input_transform_mode,
                time_modulation_mode=args.time_modulation_mode,
                kernel_composition_proposal=args.kernel_composition_proposal,
                mean_kind=args.mean_kind,
            )

        original_builder = task_exp.build_time_varying_kernel

        def floor_builder(
            base_kernel: gpytorch.kernels.Kernel,
            time_feature_index: int,
            target: str,
            **kwargs: object,
        ) -> gpytorch.kernels.Kernel:
            if target == "lengthscale":
                return build_time_varying_lengthscale_floor_kernel(
                    base_kernel,
                    time_feature_index,
                    floor=args.lengthscale_floor,
                )
            return original_builder(base_kernel, time_feature_index, target, **kwargs)

        task_exp.build_time_varying_kernel = floor_builder
        try:
            return build_model_with_transform_mode(
                train_x,
                train_y,
                train_yvar,
                variant,
                input_transform_mode=args.input_transform_mode,
                time_modulation_mode=args.time_modulation_mode,
                kernel_composition_proposal=args.kernel_composition_proposal,
                mean_kind=args.mean_kind,
            )
        finally:
            task_exp.build_time_varying_kernel = original_builder
    finally:
        task_exp.build_multitask_gp = original_gp_builder


def build_model_with_transform_mode(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_yvar: torch.Tensor | None,
    variant: Any,
    *,
    input_transform_mode: str,
    time_modulation_mode: str,
    kernel_composition_proposal: str = "none",
    mean_kind: str = "multitask_constant",
) -> Any:
    task_idx = train_x.shape[-1] - 1
    all_task_values = train_x[:, task_idx].to(torch.long).unique(sorted=True)
    input_transform = None
    if input_transform_mode == "botorch_normalize":
        input_transform = Normalize(
            d=train_x.shape[-1],
            indices=list(range(len(task_exp.INPUT_COLUMNS))),
        )
    elif input_transform_mode != "june_manual_minmax":
        raise ValueError(f"Unknown input transform mode: {input_transform_mode}")

    outcome_transform = StratifiedStandardize(
        stratification_idx=task_idx,
        all_task_values=all_task_values,
        batch_shape=train_y.shape[:-2],
    )
    add_tv_os_ls, builder = task_exp.time_modulation_builder(time_modulation_mode)
    original_builder = task_exp.multitask_builder.add_time_varying_os_ls
    covar_config = build_kernel_composition_config(kernel_composition_proposal)
    mean_config = task_exp.MeanModuleConfig(kind=task_exp.MeanKind(mean_kind))
    task_exp.multitask_builder.add_time_varying_os_ls = builder
    try:
        model = task_exp.build_multitask_gp(
            train_X=train_x,
            train_Y=train_y,
            train_Yvar=train_yvar,
            task_feature=task_exp.TASK_FEATURE,
            covar_config=covar_config,
            mean_config=mean_config,
            rank=task_exp.RANK,
            min_inferred_noise_level=5e-3,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            task_covar_prior=variant.task_covar_prior,
            add_tv_os_ls=add_tv_os_ls,
        )
    finally:
        task_exp.multitask_builder.add_time_varying_os_ls = original_builder
    if variant.task_kernel == "signed":
        task_exp.replace_with_signed_index_kernel(model, eta=variant.lkj_eta)
    return model


def build_kernel_composition_config(proposal: str) -> Any:
    config = task_exp.build_covar_config()
    if proposal == "none":
        return config

    extra_blocks = []
    if proposal == "e_plus_mt_plus_t_plus_emt":
        interactions = [
            task_exp.KernelInteractionConfig(blocks=["macro", "time"], name="macro_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(
                blocks=["etf", "macro", "time"],
                name="etf_x_macro_x_time",
                use_outputscale=True,
            ),
        ]
        main_effect_blocks = {"etf", "time"}
    elif proposal == "et_plus_mt_plus_emt":
        interactions = [
            task_exp.KernelInteractionConfig(blocks=["etf", "time"], name="etf_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(blocks=["macro", "time"], name="macro_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(
                blocks=["etf", "macro", "time"],
                name="etf_x_macro_x_time",
                use_outputscale=True,
            ),
        ]
        main_effect_blocks = set()
    elif proposal == "e_plus_m_plus_t_plus_et_plus_mt_plus_emt":
        interactions = [
            task_exp.KernelInteractionConfig(blocks=["etf", "time"], name="etf_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(blocks=["macro", "time"], name="macro_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(
                blocks=["etf", "macro", "time"],
                name="etf_x_macro_x_time",
                use_outputscale=True,
            ),
        ]
        main_effect_blocks = {"etf", "macro", "time"}
    elif proposal == "e_plus_m_plus_t_plus_em_plus_et_plus_mt_plus_emt":
        interactions = [
            task_exp.KernelInteractionConfig(blocks=["etf", "macro"], name="etf_x_macro", use_outputscale=True),
            task_exp.KernelInteractionConfig(blocks=["etf", "time"], name="etf_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(blocks=["macro", "time"], name="macro_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(
                blocks=["etf", "macro", "time"],
                name="etf_x_macro_x_time",
                use_outputscale=True,
            ),
        ]
        main_effect_blocks = {"etf", "macro", "time"}
    elif proposal == "e_plus_et_plus_mt_plus_emt":
        interactions = [
            task_exp.KernelInteractionConfig(blocks=["etf", "time"], name="etf_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(blocks=["macro", "time"], name="macro_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(
                blocks=["etf", "macro", "time"],
                name="etf_x_macro_x_time",
                use_outputscale=True,
            ),
        ]
        main_effect_blocks = {"etf"}
    elif proposal == "t_plus_et_plus_mt_plus_emt":
        interactions = [
            task_exp.KernelInteractionConfig(blocks=["etf", "time"], name="etf_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(blocks=["macro", "time"], name="macro_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(
                blocks=["etf", "macro", "time"],
                name="etf_x_macro_x_time",
                use_outputscale=True,
            ),
        ]
        main_effect_blocks = {"time"}
    elif proposal == "et_plus_mt15_plus_emt":
        extra_blocks = [_time_block_with_matern_nu(config, name="time_matern15", matern_nu=1.5)]
        interactions = [
            task_exp.KernelInteractionConfig(blocks=["etf", "time"], name="etf_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(
                blocks=["macro", "time_matern15"],
                name="macro_x_time_matern15",
                use_outputscale=True,
            ),
            task_exp.KernelInteractionConfig(
                blocks=["etf", "macro", "time"],
                name="etf_x_macro_x_time",
                use_outputscale=True,
            ),
        ]
        main_effect_blocks = set()
    elif proposal == "et_plus_mt_plus_emt_t15":
        extra_blocks = [_time_block_with_matern_nu(config, name="time_matern15", matern_nu=1.5)]
        interactions = [
            task_exp.KernelInteractionConfig(blocks=["etf", "time"], name="etf_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(blocks=["macro", "time"], name="macro_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(
                blocks=["etf", "macro", "time_matern15"],
                name="etf_x_macro_x_time_matern15",
                use_outputscale=True,
            ),
        ]
        main_effect_blocks = set()
    elif proposal == "t_plus_et_plus_mt15_plus_emt":
        extra_blocks = [_time_block_with_matern_nu(config, name="time_matern15", matern_nu=1.5)]
        interactions = [
            task_exp.KernelInteractionConfig(blocks=["etf", "time"], name="etf_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(
                blocks=["macro", "time_matern15"],
                name="macro_x_time_matern15",
                use_outputscale=True,
            ),
            task_exp.KernelInteractionConfig(
                blocks=["etf", "macro", "time"],
                name="etf_x_macro_x_time",
                use_outputscale=True,
            ),
        ]
        main_effect_blocks = {"time"}
    elif proposal == "et_plus_mt_plus_em_plus_emt":
        interactions = [
            task_exp.KernelInteractionConfig(blocks=["etf", "time"], name="etf_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(blocks=["macro", "time"], name="macro_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(blocks=["etf", "macro"], name="etf_x_macro", use_outputscale=True),
            task_exp.KernelInteractionConfig(
                blocks=["etf", "macro", "time"],
                name="etf_x_macro_x_time",
                use_outputscale=True,
            ),
        ]
        main_effect_blocks = set()
    elif proposal == "et_plus_mt_plus_em":
        interactions = [
            task_exp.KernelInteractionConfig(blocks=["etf", "time"], name="etf_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(blocks=["macro", "time"], name="macro_x_time", use_outputscale=True),
            task_exp.KernelInteractionConfig(blocks=["etf", "macro"], name="etf_x_macro", use_outputscale=True),
        ]
        main_effect_blocks = set()
    else:
        raise ValueError(f"Unknown kernel composition proposal: {proposal}")

    blocks = [
        block.model_copy(update={"include_as_main_effect": block.name in main_effect_blocks}) for block in config.blocks
    ]
    blocks.extend(extra_blocks)
    return task_exp.CovarModuleConfig(
        blocks=blocks,
        global_structure=task_exp.GlobalStructure.HIERARCHICAL,
        interaction_policy=task_exp.InteractionPolicy.CUSTOM,
        custom_interactions=interactions,
    )


def _time_block_with_matern_nu(config: Any, *, name: str, matern_nu: float) -> Any:
    time_block = next(block for block in config.blocks if block.name == "time")
    component = time_block.components[0].model_copy(update={"matern_nu": matern_nu})
    return time_block.model_copy(
        update={
            "name": name,
            "components": [component],
            "include_as_main_effect": False,
        }
    )


def months_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return float((end.year - start.year) * 12 + (end.month - start.month))


def normalized_time_context(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, float]:
    train_min = pd.Timestamp(train_df["date"].min())
    train_max = pd.Timestamp(train_df["date"].max())
    eval_date = pd.Timestamp(eval_df["date"].iloc[0])
    span_months = max(months_between(train_min, train_max), 1.0)
    eval_norm = months_between(train_min, eval_date) / span_months
    return {
        "train_start": train_min.date().isoformat(),
        "train_end": train_max.date().isoformat(),
        "eval_date": eval_date.date().isoformat(),
        "span_months": span_months,
        "eval_time_norm": eval_norm,
    }


def replace_data_kernel(model: torch.nn.Module, data_kernel: gpytorch.kernels.Kernel) -> None:
    if not hasattr(model, "covar_module") or not hasattr(model.covar_module, "kernels"):
        raise TypeError("Expected model.covar_module to expose ProductKernel-style kernels.")
    model.covar_module.kernels[0] = data_kernel


def build_fixed_matern_time_gate(*, half_life_months: float, span_months: float) -> tuple[MaternKernel, float]:
    if half_life_months <= 0:
        raise ValueError(f"kernel half-life must be positive, got {half_life_months}")
    half_life_norm = max(half_life_months / max(span_months, 1e-12), 1e-12)
    lengthscale = half_life_norm / math.log(2.0)
    gate = MaternKernel(nu=0.5, active_dims=[0])
    gate.lengthscale = torch.as_tensor(lengthscale, dtype=torch.float64)
    gate.raw_lengthscale.requires_grad_(False)
    return gate, float(lengthscale)


class ForecastRecencyGateKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        *,
        reference_time_norm: float,
        half_life_norm: float,
        time_feature_index: int = 0,
    ) -> None:
        super().__init__()
        if half_life_norm <= 0:
            raise ValueError(f"half_life_norm must be positive, got {half_life_norm}")
        self.base_kernel = base_kernel
        self.reference_time_norm = float(reference_time_norm)
        self.half_life_norm = float(half_life_norm)
        self.time_feature_index = time_feature_index

    def _weights(self, x: torch.Tensor) -> torch.Tensor:
        age = (self.reference_time_norm - x[..., self.time_feature_index]).clamp_min(0.0)
        return torch.pow(torch.as_tensor(0.5, dtype=x.dtype, device=x.device), age / self.half_life_norm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs: object) -> torch.Tensor:
        k_base = self.base_kernel(x1, x2, **kwargs).to_dense()
        s1 = self._weights(x1).sqrt()
        s2 = self._weights(x2).sqrt()
        return s1.unsqueeze(-1) * k_base * s2.unsqueeze(-2)


class ChangePointMixtureKernel(gpytorch.kernels.Kernel):
    has_lengthscale = False

    def __init__(
        self,
        old_kernel: gpytorch.kernels.Kernel,
        recent_kernel: gpytorch.kernels.Kernel,
        *,
        changepoint_norm: float,
        width_norm: float,
        time_feature_index: int = 0,
    ) -> None:
        super().__init__()
        if width_norm <= 0:
            raise ValueError(f"width_norm must be positive, got {width_norm}")
        self.old_kernel = old_kernel
        self.recent_kernel = recent_kernel
        self.changepoint_norm = float(changepoint_norm)
        self.width_norm = float(width_norm)
        self.time_feature_index = time_feature_index

    def _recent_weight(self, x: torch.Tensor) -> torch.Tensor:
        z = (x[..., self.time_feature_index] - self.changepoint_norm) / self.width_norm
        return torch.sigmoid(z)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs: object) -> torch.Tensor:
        s1 = self._recent_weight(x1)
        s2 = self._recent_weight(x2)
        k_recent = self.recent_kernel(x1, x2, **kwargs).to_dense()
        k_old = self.old_kernel(x1, x2, **kwargs).to_dense()
        return s1.unsqueeze(-1) * k_recent * s2.unsqueeze(-2) + (1.0 - s1).unsqueeze(-1) * k_old * (1.0 - s2).unsqueeze(
            -2
        )


def apply_kernel_proposal(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> dict[str, float | str] | None:
    if args.kernel_proposal == "none":
        return None
    if args.time_modulation_mode != "neither":
        raise ValueError(
            "Time-decay kernel experiments should use --time-modulation-mode neither "
            "to avoid the existing time-varying lengthscale/outputscale wrapper."
        )

    context = normalized_time_context(train_df, eval_df)
    data_kernel = model.covar_module.kernels[0]
    if args.kernel_proposal == "global_time_gate":
        gate, lengthscale_norm = build_fixed_matern_time_gate(
            half_life_months=float(args.kernel_half_life_months),
            span_months=float(context["span_months"]),
        )
        replace_data_kernel(model, gate * data_kernel)
        return {
            **context,
            "kernel_proposal": args.kernel_proposal,
            "half_life_months": float(args.kernel_half_life_months),
            "time_gate_lengthscale_norm": lengthscale_norm,
        }

    if args.kernel_proposal == "forecast_recency_gate":
        half_life_norm = max(float(args.kernel_half_life_months) / float(context["span_months"]), 1e-12)
        replace_data_kernel(
            model,
            ForecastRecencyGateKernel(
                data_kernel,
                reference_time_norm=float(context["eval_time_norm"]),
                half_life_norm=half_life_norm,
            ),
        )
        return {
            **context,
            "kernel_proposal": args.kernel_proposal,
            "half_life_months": float(args.kernel_half_life_months),
            "half_life_norm": half_life_norm,
        }

    if args.kernel_proposal == "changepoint_mixture":
        train_start = pd.Timestamp(context["train_start"])
        changepoint = pd.Timestamp(args.kernel_changepoint_date)
        changepoint_norm = months_between(train_start, changepoint) / float(context["span_months"])
        width_norm = max(float(args.kernel_changepoint_width_months) / float(context["span_months"]), 1e-12)
        replace_data_kernel(
            model,
            ChangePointMixtureKernel(
                copy.deepcopy(data_kernel),
                data_kernel,
                changepoint_norm=changepoint_norm,
                width_norm=width_norm,
            ),
        )
        return {
            **context,
            "kernel_proposal": args.kernel_proposal,
            "changepoint_date": changepoint.date().isoformat(),
            "changepoint_norm": float(changepoint_norm),
            "changepoint_width_months": float(args.kernel_changepoint_width_months),
            "changepoint_width_norm": float(width_norm),
        }

    raise ValueError(f"Unknown kernel proposal: {args.kernel_proposal}")


def prepare_window_tensors_for_run(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, float] | None]:
    train_x, train_y, eval_x, _, _, _ = task_exp.prepare_window_tensors(train_df, eval_df)
    train_yvar, recency_summary = build_recency_train_yvar(train_df, eval_df, args=args)
    if args.input_transform_mode != "june_manual_minmax":
        return train_x, train_y, eval_x, train_yvar, recency_summary

    non_task_dim_count = len(task_exp.INPUT_COLUMNS)
    combined = torch.cat([train_x[:, :non_task_dim_count], eval_x[:, :non_task_dim_count]], dim=0)
    mins = combined.amin(dim=0)
    ranges = (combined.amax(dim=0) - mins).clamp_min(1e-12)
    train_x = train_x.clone()
    eval_x = eval_x.clone()
    train_x[:, :non_task_dim_count] = (train_x[:, :non_task_dim_count] - mins) / ranges
    eval_x[:, :non_task_dim_count] = (eval_x[:, :non_task_dim_count] - mins) / ranges
    return train_x, train_y, eval_x, train_yvar, recency_summary


def build_recency_train_yvar(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> tuple[torch.Tensor | None, dict[str, float] | None]:
    if args.recency_half_life_months is None:
        return None, None
    half_life = float(args.recency_half_life_months)
    base_noise = float(args.recency_base_noise_level)
    max_multiplier = float(args.recency_max_noise_multiplier)
    if half_life <= 0:
        raise ValueError(f"recency half-life must be positive, got {half_life}")
    if base_noise <= 0:
        raise ValueError(f"recency base noise level must be positive, got {base_noise}")
    if max_multiplier < 1:
        raise ValueError(f"recency max noise multiplier must be >= 1, got {max_multiplier}")

    window_date = pd.Timestamp(eval_df["date"].iloc[0])
    dates = pd.to_datetime(train_df["date"])
    age_months = ((window_date.year - dates.dt.year) * 12 + (window_date.month - dates.dt.month)).astype(float)
    age_months = age_months.clip(lower=0.0)
    weights = np.power(0.5, age_months.to_numpy(dtype=float) / half_life)
    multipliers = np.minimum(1.0 / np.clip(weights, 1.0 / max_multiplier, 1.0), max_multiplier)

    task_variances = train_df.groupby("asset_id", observed=True)[task_exp.TARGET_COL].var(ddof=1)
    global_variance = float(train_df[task_exp.TARGET_COL].var(ddof=1))
    if not np.isfinite(global_variance) or global_variance <= 0:
        global_variance = 1.0
    asset_variance = train_df["asset_id"].map(task_variances).astype(float).fillna(global_variance).to_numpy()
    asset_variance = np.clip(asset_variance, 1e-12, None)
    raw_yvar = asset_variance * base_noise * multipliers
    train_yvar = torch.tensor(raw_yvar, dtype=torch.float64).unsqueeze(-1)

    effective_n = float(np.square(weights.sum()) / np.square(weights).sum())
    summary = {
        "recency_half_life_months": half_life,
        "recency_base_noise_level": base_noise,
        "recency_max_noise_multiplier": max_multiplier,
        "recency_weight_min": float(weights.min()),
        "recency_weight_mean": float(weights.mean()),
        "recency_weight_max": float(weights.max()),
        "recency_noise_multiplier_min": float(multipliers.min()),
        "recency_noise_multiplier_mean": float(multipliers.mean()),
        "recency_noise_multiplier_max": float(multipliers.max()),
        "recency_effective_n_rows": effective_n,
    }
    return train_yvar, summary


def collect_model_diagnostics(
    model: torch.nn.Module,
    *,
    eval_x: torch.Tensor,
    window_date: pd.Timestamp,
    experiment: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, module in model.named_modules():
        if isinstance(module, ScaleKernel):
            values = module.outputscale.detach().cpu().reshape(-1)
            rows.append(
                {
                    "experiment": experiment,
                    "date": window_date.date().isoformat(),
                    "diagnostic": "outputscale",
                    "module_path": path,
                    "module_type": module.base_kernel.__class__.__name__,
                    "mean": float(values.mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
        if hasattr(module, "lengthscale") and module.lengthscale is not None:
            values = module.lengthscale.detach().cpu().reshape(-1)
            rows.append(
                {
                    "experiment": experiment,
                    "date": window_date.date().isoformat(),
                    "diagnostic": "lengthscale",
                    "module_path": path,
                    "module_type": module.__class__.__name__,
                    "mean": float(values.mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
        if hasattr(module, "_effective_lengthscale"):
            with torch.no_grad():
                values = module._effective_lengthscale(eval_x).detach().cpu().reshape(-1)  # noqa: SLF001
            rows.append(
                {
                    "experiment": experiment,
                    "date": window_date.date().isoformat(),
                    "diagnostic": "time_varying_effective_lengthscale_eval",
                    "module_path": path,
                    "module_type": module.__class__.__name__,
                    "mean": float(values.mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    if hasattr(model, "likelihood") and hasattr(model.likelihood, "noise"):
        values = model.likelihood.noise.detach().cpu().reshape(-1)
        rows.append(
            {
                "experiment": experiment,
                "date": window_date.date().isoformat(),
                "diagnostic": "likelihood_noise",
                "module_path": "likelihood",
                "module_type": model.likelihood.__class__.__name__,
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return rows


def calibrate_scenario_means(
    scenarios: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    scale: float,
) -> pd.DataFrame:
    if scale == 1.0:
        return scenarios
    means = predictions.set_index("asset_id")["y_pred"].reindex(scenarios.columns).astype(float)
    return scenarios.add((scale - 1.0) * means, axis=1)


def blend_with_previous_weights(
    weights: pd.Series,
    previous_weights: pd.Series | None,
    *,
    blend: float,
) -> pd.Series:
    if previous_weights is None:
        return weights
    blend = float(np.clip(blend, 0.0, 1.0))
    combined = blend * weights + (1.0 - blend) * previous_weights.reindex(weights.index).fillna(0.0)
    combined = combined.clip(lower=0.0)
    total = float(combined.sum())
    return combined / total if total > 0 else weights


def fit_gp_window(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    seed: int,
    maxiter: int,
    posterior_scenarios: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any] | None]:
    train_x, train_y, eval_x, train_yvar, recency_summary = prepare_window_tensors_for_run(
        train_df,
        eval_df,
        args=args,
    )
    torch.manual_seed(seed)
    model = build_experiment_model(train_x, train_y, train_yvar, args=args)
    kernel_proposal_summary = apply_kernel_proposal(model, train_df, eval_df, args=args)
    outputscale_prior_rows: list[dict[str, Any]] = []
    task_noise_floor_rows: list[dict[str, Any]] = []
    if args.gp_experiment in OUTPUTSCALE_FLOOR_EXPERIMENTS:
        outputscale_prior_rows = apply_outputscale_floor_prior(
            model,
            floor=args.outputscale_floor,
            prior_median=args.outputscale_prior_median,
            prior_sigma=args.outputscale_prior_sigma,
        )
    if args.gp_experiment in TASK_NOISE_FLOOR_EXPERIMENTS:
        task_noise_floor_rows = apply_task_noise_floor(
            model,
            train_df,
            raw_std_floor=args.task_noise_floor_raw_std,
        )
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": maxiter}})
    model.eval()
    model.likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(eval_x, observation_noise=True)
        pred_mean = posterior.mean.squeeze(-1).detach().cpu().numpy()
        pred_std = posterior.variance.squeeze(-1).clamp_min(0.0).sqrt().detach().cpu().numpy()
        scenario_samples = posterior.rsample(torch.Size([posterior_scenarios])).squeeze(-1).detach().cpu().numpy()

    assets = eval_df["asset_id"].astype(str).tolist()
    scenarios = pd.DataFrame(scenario_samples, columns=assets)
    predictions = pd.DataFrame(
        {
            "date": pd.Timestamp(eval_df["date"].iloc[0]).date().isoformat(),
            "asset_id": assets,
            "y_true": eval_df[task_exp.TARGET_COL].to_numpy(dtype=float),
            "y_pred": pred_mean,
            "y_std": pred_std,
            "score": pred_mean / np.clip(pred_std, 1e-12, None),
        }
    )
    diagnostics = collect_model_diagnostics(
        model,
        eval_x=eval_x,
        window_date=pd.Timestamp(eval_df["date"].iloc[0]),
        experiment=args.gp_experiment,
    )
    if recency_summary is not None:
        diagnostics.append(
            {
                "experiment": args.gp_experiment,
                "date": pd.Timestamp(eval_df["date"].iloc[0]).date().isoformat(),
                "diagnostic": "recency_observation_noise",
                "module_path": "train_Yvar",
                "module_type": "FixedNoiseGaussianLikelihood",
                **recency_summary,
            }
        )
    if kernel_proposal_summary is not None:
        diagnostics.append(
            {
                "experiment": args.gp_experiment,
                "date": pd.Timestamp(eval_df["date"].iloc[0]).date().isoformat(),
                "diagnostic": "kernel_proposal",
                "module_path": "covar_module.kernels.0",
                "module_type": model.covar_module.kernels[0].__class__.__name__,
                **kernel_proposal_summary,
            }
        )
    for row in outputscale_prior_rows:
        row["date"] = pd.Timestamp(eval_df["date"].iloc[0]).date().isoformat()
        row["experiment"] = args.gp_experiment
    for row in task_noise_floor_rows:
        row["date"] = pd.Timestamp(eval_df["date"].iloc[0]).date().isoformat()
        row["experiment"] = args.gp_experiment
        row["diagnostic"] = "task_noise_floor"
        row["module_path"] = "likelihood.noise_covar"
        row["module_type"] = model.likelihood.noise_covar.__class__.__name__
    task_diag = None
    try:
        corr = task_exp.task_correlation(model)
        task_diag = task_exp.covariance_diagnostics(
            corr,
            variant=args.gp_experiment,
            window_date=pd.Timestamp(eval_df["date"].iloc[0]),
        )
    except Exception:
        task_diag = None
    return scenarios, predictions, [*diagnostics, *outputscale_prior_rows, *task_noise_floor_rows], task_diag


def realized_return(weights: pd.Series, eval_returns: pd.Series) -> float:
    aligned = weights.reindex(eval_returns.index).fillna(0.0)
    value = float(np.dot(aligned.to_numpy(dtype=float), eval_returns.to_numpy(dtype=float)))
    return value if np.isfinite(value) else math.nan


def information_coefficient(predictions: pd.DataFrame, final_universe: list[str]) -> float:
    frame = predictions[predictions["asset_id"].isin(final_universe)]
    if frame["y_pred"].nunique() < 2 or frame["y_true"].nunique() < 2:
        return math.nan
    return float(frame["y_pred"].corr(frame["y_true"], method="spearman"))


def performance_stats(
    returns: pd.Series,
    weights: pd.DataFrame,
    *,
    starting_value: float,
    periods_per_year: float,
) -> dict[str, float]:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return {
            "n_rebalances": 0.0,
            "cumulative_return": math.nan,
            "cagr": math.nan,
            "annualized_vol": math.nan,
            "sharpe": math.nan,
            "max_drawdown": math.nan,
            "terminal_value": math.nan,
            "mean_period_return": math.nan,
            "mean_monthly_return": math.nan,
            "hit_rate": math.nan,
            "avg_turnover": math.nan,
            "max_weight": math.nan,
        }

    equity = (1.0 + returns).cumprod()
    cumulative_return = float(equity.iloc[-1] - 1.0)
    years = len(returns) / periods_per_year
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else math.nan
    ann_vol = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = float(cagr / ann_vol) if ann_vol > 0 else math.nan
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    turnover = weights.diff().abs().sum(axis=1) / 2.0
    if len(turnover) > 0:
        turnover.iloc[0] = weights.iloc[0].abs().sum()

    return {
        "n_rebalances": float(len(returns)),
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "terminal_value": float(starting_value * equity.iloc[-1]),
        "mean_period_return": float(returns.mean()),
        "mean_monthly_return": float(returns.mean()),
        "hit_rate": float((returns > 0).mean()),
        "avg_turnover": float(turnover.mean()),
        "max_weight": float(weights.max(axis=1).max()),
    }


def plot_equity_and_drawdown(strategy_returns: dict[str, pd.Series], output_dir: Path) -> None:
    equity = pd.DataFrame({name: (1.0 + ret).cumprod() * STARTING_VALUE for name, ret in strategy_returns.items()})
    drawdown = equity / equity.cummax() - 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    equity.plot(ax=ax)
    ax.set_title("Portfolio Value from $10,000")
    ax.set_ylabel("Portfolio value")
    ax.set_xlabel("Rebalance date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    drawdown.plot(ax=ax)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Rebalance date")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "drawdown_curve.png", dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as Markdown without optional dependencies."""

    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = [str(column) for column in formatted.columns]
    rows = formatted.astype(str).values.tolist()
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def run(args: argparse.Namespace) -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = task_exp.load_features(args.feature_path)
    if args.drop_incomplete_feature_dates:
        df = drop_incomplete_feature_dates(df)
    df = apply_min_feature_date(df, args.min_feature_date)
    scored_dates, live_date = task_exp.scored_and_live_dates(df, args.max_windows)
    scored_dates = apply_min_scored_date(scored_dates, args.min_scored_date)
    construction_dates = [*scored_dates]
    if args.include_live_window and live_date is not None:
        construction_dates.append(live_date)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    final_universe = [asset for asset in task_exp.ETF_TICKERS if asset not in HELPER_ASSETS]

    manifest = build_manifest(
        args,
        output_dir,
        df,
        scored_dates,
        live_date if args.include_live_window else None,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    hist_strategy = historical_strategy_name(args)
    return_rows: list[dict[str, Any]] = []
    weight_frames: dict[str, list[pd.Series]] = {
        "gp_scenarios_riskfolio": [],
        hist_strategy: [],
        "equal_weight": [],
    }
    prediction_rows: list[pd.DataFrame] = []
    ic_rows: list[dict[str, Any]] = []
    model_diag_rows: list[dict[str, Any]] = []
    task_diag_rows: list[dict[str, Any]] = []
    train_history_rows: list[dict[str, Any]] = []
    riskfolio_diag_rows: list[dict[str, Any]] = []
    previous_gp_weights: pd.Series | None = None
    previous_hist_weights: pd.Series | None = None

    for window_index, window_date in enumerate(construction_dates):
        print(f"rebalance {window_date.date()}", flush=True)
        train_df = training_slice(df, window_date, args.train_months)
        eval_df = df[df["date"] == window_date].copy()
        train_history_rows.append(
            {
                "date": window_date.date().isoformat(),
                "train_months": args.train_months,
                "train_rows": int(len(train_df)),
                "train_date_min": train_df["date"].min().date().isoformat() if not train_df.empty else None,
                "train_date_max": train_df["date"].max().date().isoformat() if not train_df.empty else None,
                "unique_train_dates": int(train_df["date"].nunique()),
            }
        )
        eval_returns = (
            eval_df.set_index(eval_df["asset_id"].astype(str))[task_exp.TARGET_COL]
            .reindex(final_universe)
            .astype(float)
        )

        scenarios, predictions, model_diagnostics, task_diag = fit_gp_window(
            train_df,
            eval_df,
            args=args,
            seed=task_exp.stable_seed(args.seed, gp_variant_name(args.gp_experiment), window_index),
            maxiter=args.maxiter,
            posterior_scenarios=args.posterior_scenarios,
        )
        if args.gp_experiment == "scenario_mean_scale":
            scenarios = calibrate_scenario_means(scenarios, predictions, scale=args.scenario_mean_scale)
        prediction_rows.append(predictions)
        model_diag_rows.extend(model_diagnostics)
        if task_diag is not None:
            task_diag_rows.append(task_diag)
        scenarios.loc[:, final_universe].to_csv(
            output_dir / f"gp_scenarios_{window_date.date().isoformat()}.csv",
            index=False,
        )

        hist_panel = train_df.pivot(index="date", columns="asset_id", values=task_exp.TARGET_COL).reindex(
            columns=final_universe
        )
        hist_result = optimize_riskfolio(
            hist_panel,
            method_mu=args.historical_method_mu,
            method_cov=args.historical_method_cov,
            upperlng=args.upperlng,
            nea=args.nea,
            fallback_weights=previous_hist_weights,
            fallback_label="previous_historical_weights",
        )
        hist_weights = hist_result.weights.reindex(final_universe).fillna(0.0)
        previous_hist_weights = hist_weights
        gp_fallback_weights = previous_gp_weights if previous_gp_weights is not None else hist_weights
        gp_fallback_label = "previous_gp_weights" if previous_gp_weights is not None else "historical_y_ewma2_weights"
        gp_result = optimize_riskfolio(
            scenarios.loc[:, final_universe],
            method_mu="hist",
            method_cov="hist",
            upperlng=args.upperlng,
            nea=args.nea,
            fallback_weights=gp_fallback_weights,
            fallback_label=gp_fallback_label,
        )
        gp_weights = gp_result.weights.reindex(final_universe).fillna(0.0)
        if args.gp_experiment in TURNOVER_BLEND_EXPERIMENTS:
            gp_weights = blend_with_previous_weights(
                gp_weights,
                previous_gp_weights,
                blend=args.turnover_blend,
            )
        previous_gp_weights = gp_weights.reindex(final_universe).fillna(0.0)
        ew_weights = equal_weight(final_universe)
        riskfolio_diag_rows.extend(
            [
                {
                    "date": window_date.date().isoformat(),
                    "strategy": "gp_scenarios_riskfolio",
                    "status": gp_result.status,
                    "fallback_stage": gp_result.fallback_stage,
                    "clean_asset_count": gp_result.clean_asset_count,
                    "clean_observation_count": gp_result.clean_observation_count,
                    "message": gp_result.message,
                },
                {
                    "date": window_date.date().isoformat(),
                    "strategy": hist_strategy,
                    "status": hist_result.status,
                    "fallback_stage": hist_result.fallback_stage,
                    "clean_asset_count": hist_result.clean_asset_count,
                    "clean_observation_count": hist_result.clean_observation_count,
                    "message": hist_result.message,
                },
            ]
        )

        weights_by_strategy = {
            "gp_scenarios_riskfolio": gp_weights.reindex(final_universe).fillna(0.0),
            hist_strategy: hist_weights.reindex(final_universe).fillna(0.0),
            "equal_weight": ew_weights.reindex(final_universe).fillna(0.0),
        }
        gp_ic = information_coefficient(predictions, final_universe)
        ic_rows.append({"date": window_date.date().isoformat(), "strategy": "gp_scenarios_riskfolio", "ic": gp_ic})

        is_live_window = bool(live_date is not None and window_date == live_date)
        for strategy, weights in weights_by_strategy.items():
            weight_frames[strategy].append(pd.Series(weights, name=window_date))
            return_rows.append(
                {
                    "date": window_date.date().isoformat(),
                    "strategy": strategy,
                    "return": math.nan if is_live_window else realized_return(weights, eval_returns),
                    "gp_ic": gp_ic if strategy == "gp_scenarios_riskfolio" else math.nan,
                    "is_live_window": is_live_window,
                }
            )

    returns_df = pd.DataFrame(return_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    ic_df = pd.DataFrame(ic_rows)
    weights_output: list[pd.DataFrame] = []
    strategy_returns: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, Any]] = []

    for strategy, rows in weight_frames.items():
        weights = pd.DataFrame(rows)
        weights.index = pd.to_datetime(weights.index)
        weights.index.name = "date"
        weights = weights.reindex(columns=final_universe).fillna(0.0)
        weights_long = weights.reset_index().melt(id_vars="date", var_name="asset_id", value_name="weight")
        weights_long["strategy"] = strategy
        weights_output.append(weights_long)

        strategy_ret = (
            returns_df[returns_df["strategy"] == strategy]
            .assign(date=lambda frame: pd.to_datetime(frame["date"]))
            .set_index("date")["return"]
            .astype(float)
        )
        strategy_returns[strategy] = strategy_ret
        summary = {
            "strategy": strategy,
            **performance_stats(
                strategy_ret,
                weights,
                starting_value=STARTING_VALUE,
                periods_per_year=args.periods_per_year,
            ),
        }
        if strategy == "gp_scenarios_riskfolio":
            summary["mean_ic"] = float(ic_df["ic"].mean())
            summary["median_ic"] = float(ic_df["ic"].median())
        else:
            summary["mean_ic"] = math.nan
            summary["median_ic"] = math.nan
        summary_rows.append(summary)

    weights_df = pd.concat(weights_output, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    returns_df.to_csv(output_dir / "portfolio_returns.csv", index=False)
    weights_df.to_csv(output_dir / "portfolio_weights.csv", index=False)
    predictions_df.to_csv(output_dir / "gp_window_predictions.csv", index=False)
    ic_df.to_csv(output_dir / "gp_window_ic.csv", index=False)
    summary_df.to_csv(output_dir / "portfolio_summary.csv", index=False)
    pd.DataFrame(train_history_rows).to_csv(output_dir / "window_training_history.csv", index=False)
    pd.DataFrame(riskfolio_diag_rows).to_csv(output_dir / "riskfolio_optimization_diagnostics.csv", index=False)
    if model_diag_rows:
        pd.DataFrame(model_diag_rows).to_csv(output_dir / "model_diagnostics.csv", index=False)
    if task_diag_rows:
        pd.DataFrame(task_diag_rows).to_csv(output_dir / "task_covariance_diagnostics.csv", index=False)
    plot_equity_and_drawdown(strategy_returns, output_dir)

    report = [
        "# Portfolio Optimization Walk-Forward Report",
        "",
        f"Run directory: `{output_dir}`",
        (
            f"Portfolio construction dates: `{len(construction_dates)}` "
            f"{args.rebalance_frequency_label} windows from "
            f"`{construction_dates[0].date()}` to `{construction_dates[-1].date()}` "
            f"(`{len(scored_dates)}` realized)."
        ),
        f"Starting value: `${STARTING_VALUE:,.0f}`.",
        "",
        "## Strategy Summary",
        "",
        markdown_table(summary_df),
        "",
        "## Notes",
        "",
        "- `MGK` and `BND` were included in GP fitting and scenario generation, then excluded from final weights.",
        (
            f"- Training history: `{args.train_months}` rolling calendar months before each window."
            if args.train_months is not None
            else "- Training history: all available rows before each window."
        ),
        (
            f"- `gp_scenarios_riskfolio` experiment variant: `{args.gp_experiment}`. "
            "Control is positive beta task covariance, lengthscale-only time modulation, rank 5."
        ),
        (
            f"- `{hist_strategy}` ignores GP predictions and optimizes directly on historical "
            "`y_excess_lead`, Sharpe, and CVaR."
        ),
        "- `equal_weight` is the additional baseline requested for portfolio-run comparison.",
        "- IRA context: turnover is tracked as a stability diagnostic, not as a tax-cost veto.",
        "- No transaction costs, taxes, slippage, or liquidity filters are applied in this end-to-end check.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")

    print(summary_df.to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    run(parse_args())
