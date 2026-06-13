"""Roll-forward task-covariance experiment for June 2026 ETF multitask GPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import botorch
import gpytorch
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import StratifiedStandardize
from botorch.models.utils.priors import BetaPrior
from gpytorch.kernels import IndexKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LKJCovariancePrior, LogNormalPrior

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bayesfolio.engine.backtest.evaluate_asset_pricing import evaluate_asset_pricing  # noqa: E402
from bayesfolio.engine.backtest.portfolio_helpers import long_short_returns_topk, portfolio_stats  # noqa: E402
from bayesfolio.engine.forecast.gp.multitask_builder import (  # noqa: E402
    BlockStructure,
    CovarModuleConfig,
    GlobalStructure,
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
    build_multitask_gp,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_PATH = Path("/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_06.parquet")
OUTPUT_ROOT = EXPERIMENT_DIR / "outputs"

ETF_TICKERS = [
    "SPY",
    "MGK",
    "VTV",
    "IJR",
    "IWM",
    "VNQ",
    "VNQI",
    "VEA",
    "VWO",
    "VSS",
    "BND",
    "IEF",
    "BNDX",
    "LQD",
    "HYG",
    "EWX",
    "VWOB",
    "HYEM",
]

ASSET_GROUPS = {
    "SPY": "us_equity",
    "MGK": "us_equity",
    "VTV": "us_equity",
    "IJR": "us_equity",
    "IWM": "us_equity",
    "VNQ": "real_estate",
    "VNQI": "real_estate",
    "VEA": "intl_equity",
    "VWO": "intl_equity",
    "VSS": "intl_equity",
    "EWX": "intl_equity",
    "BND": "fixed_income",
    "IEF": "fixed_income",
    "BNDX": "fixed_income",
    "LQD": "credit",
    "HYG": "credit",
    "VWOB": "credit",
    "HYEM": "credit",
}

TIME_COLS = ["t_index"]
ETF_COLS = [
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
MACRO_COLS = [
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
INPUT_COLUMNS = [*TIME_COLS, *ETF_COLS, *MACRO_COLS]
TARGET_COL = "y_excess_lead"
TASK_FEATURE = -1
RANK = 5
TIME_DIMS = tuple(INPUT_COLUMNS.index(col) for col in TIME_COLS)


@dataclass(frozen=True)
class Variant:
    name: str
    task_kernel: str
    task_covar_prior: object | None
    lkj_eta: float | None = None


VARIANTS = {
    "historical_mean": Variant("historical_mean", "baseline", None),
    "positive_no_prior": Variant("positive_no_prior", "positive", None),
    "positive_beta_prior": Variant(
        "positive_beta_prior",
        "positive",
        BetaPrior(concentration1=2.5, concentration0=1.5),
    ),
    "signed_no_prior": Variant("signed_no_prior", "signed", None),
    "signed_lkj_eta_1": Variant("signed_lkj_eta_1", "signed", None, lkj_eta=1.0),
    "signed_lkj_eta_2": Variant("signed_lkj_eta_2", "signed", None, lkj_eta=2.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=75)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=27)
    return parser.parse_args()


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    keep_cols = [*INPUT_COLUMNS, "date", "asset_id", TARGET_COL]
    df = pd.read_parquet(path).loc[:, keep_cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["asset_id"] = pd.Categorical(df["asset_id"], categories=ETF_TICKERS, ordered=True)
    df = df[df["asset_id"].notna()].sort_values(["date", "asset_id"]).reset_index(drop=True)
    return df


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    run_id = args.run_id or f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{git_sha(short=True)}"
    return OUTPUT_ROOT / "runs" / run_id


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


def scored_and_live_dates(df: pd.DataFrame, max_windows: int | None) -> tuple[list[pd.Timestamp], pd.Timestamp | None]:
    counts = df.groupby("date", observed=True)[TARGET_COL].apply(lambda s: int(s.notna().sum()))
    scored_dates = counts[counts == len(ETF_TICKERS)].index.to_list()
    live_dates = counts[counts == 0].index.to_list()
    scored_dates = scored_dates[-12:]
    if max_windows is not None:
        scored_dates = scored_dates[-max_windows:]
    live_date = live_dates[-1] if live_dates else None
    return scored_dates, live_date


def build_covar_config() -> CovarModuleConfig:
    idx = {name: i for i, name in enumerate(INPUT_COLUMNS)}
    time_dims = [idx[c] for c in TIME_COLS]
    etf_dims = [idx[c] for c in ETF_COLS]
    macro_dims = [idx[c] for c in MACRO_COLS]
    adaptive = LengthscalePolicyConfig(policy=LengthscalePolicy.ADAPTIVE)

    return CovarModuleConfig(
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
                        lengthscale_policy=adaptive,
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
                        lengthscale_policy=adaptive,
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
                        lengthscale_policy=adaptive,
                    ),
                    RQKernelComponentConfig(
                        dims=macro_dims,
                        ard=True,
                        use_outputscale=True,
                        lengthscale_policy=adaptive,
                    ),
                    LinearKernelComponentConfig(dims=macro_dims, use_outputscale=True),
                ],
                block_structure=BlockStructure.ADDITIVE,
                use_outputscale=False,
            ),
        ],
        global_structure=GlobalStructure.HIERARCHICAL,
        interaction_policy=InteractionPolicy.CUSTOM,
        custom_interactions=[
            KernelInteractionConfig(blocks=["time", "etf"], name="time_x_etf", use_outputscale=True),
            KernelInteractionConfig(blocks=["time", "macro"], name="time_x_macro", use_outputscale=True),
            KernelInteractionConfig(blocks=["macro", "etf"], name="macro_x_etf", use_outputscale=True),
        ],
    )


def prepare_window_tensors(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    global_time_mins: torch.Tensor,
    global_time_ranges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int], torch.Tensor, torch.Tensor]:
    task_map = {asset: idx for idx, asset in enumerate(ETF_TICKERS)}
    train_x_raw = _frame_to_x(train_df, task_map)
    train_y = torch.tensor(train_df[TARGET_COL].to_numpy(dtype=float), dtype=torch.float64).unsqueeze(-1)
    eval_x_raw = _frame_to_x(eval_df, task_map)

    train_x = train_x_raw.clone()
    eval_x = eval_x_raw.clone()
    mins = train_x[:, : len(INPUT_COLUMNS)].amin(dim=0)
    maxs = train_x[:, : len(INPUT_COLUMNS)].amax(dim=0)
    ranges = (maxs - mins).clamp_min(1e-12)
    time_index = torch.tensor(TIME_DIMS, dtype=torch.long)
    mins[time_index] = global_time_mins.to(dtype=mins.dtype, device=mins.device)
    ranges[time_index] = global_time_ranges.to(dtype=ranges.dtype, device=ranges.device)
    train_x[:, : len(INPUT_COLUMNS)] = (train_x[:, : len(INPUT_COLUMNS)] - mins) / ranges
    eval_x[:, : len(INPUT_COLUMNS)] = (eval_x[:, : len(INPUT_COLUMNS)] - mins) / ranges
    return train_x, train_y, eval_x, task_map, mins, ranges


def global_time_scaling(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    time_values = torch.tensor(df.loc[:, TIME_COLS].to_numpy(dtype=float), dtype=torch.float64)
    mins = time_values.amin(dim=0)
    maxs = time_values.amax(dim=0)
    ranges = (maxs - mins).clamp_min(1e-12)
    return mins, ranges


def _frame_to_x(df: pd.DataFrame, task_map: dict[str, int]) -> torch.Tensor:
    features = df.loc[:, INPUT_COLUMNS].to_numpy(dtype=float)
    task_ids = df["asset_id"].astype(str).map(task_map).to_numpy(dtype=float).reshape(-1, 1)
    return torch.tensor(np.concatenate([features, task_ids], axis=1), dtype=torch.float64)


def historical_mean_predict(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    by_asset = train_df.groupby("asset_id", observed=True)[TARGET_COL]
    means = by_asset.mean()
    stds = by_asset.std(ddof=1)
    global_mean = float(train_df[TARGET_COL].mean())
    global_std = float(train_df[TARGET_COL].std(ddof=1))
    if not np.isfinite(global_std) or global_std <= 0:
        global_std = 1e-6
    eval_assets = eval_df["asset_id"].astype(str)
    y_pred = eval_assets.map(means).fillna(global_mean).to_numpy(dtype=float)
    y_std = eval_assets.map(stds).fillna(global_std).to_numpy(dtype=float)
    y_std = np.clip(y_std, 1e-6, None)
    return y_pred, y_std


def training_asset_means(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> np.ndarray:
    means = train_df.groupby("asset_id", observed=True)[TARGET_COL].mean()
    global_mean = float(train_df[TARGET_COL].mean())
    return eval_df["asset_id"].astype(str).map(means).fillna(global_mean).to_numpy(dtype=float)


def build_model(train_x: torch.Tensor, train_y: torch.Tensor, variant: Variant) -> Any:
    task_idx = train_x.shape[-1] - 1
    all_task_values = train_x[:, task_idx].to(torch.long).unique(sorted=True)
    outcome_transform = StratifiedStandardize(
        stratification_idx=task_idx,
        all_task_values=all_task_values,
        batch_shape=train_y.shape[:-2],
    )
    model = build_multitask_gp(
        train_X=train_x,
        train_Y=train_y,
        task_feature=TASK_FEATURE,
        covar_config=build_covar_config(),
        mean_config=MeanModuleConfig(kind=MeanKind.MULTITASK_CONSTANT),
        rank=RANK,
        min_inferred_noise_level=5e-3,
        outcome_transform=outcome_transform,
        input_transform=None,
        task_covar_prior=variant.task_covar_prior,
        add_tv_os_ls=True,
    )
    if variant.task_kernel == "signed":
        replace_with_signed_index_kernel(model, eta=variant.lkj_eta)
    return model


def replace_with_signed_index_kernel(model: Any, *, eta: float | None) -> None:
    data_kernel = model.covar_module.kernels[0]
    task_prior = None
    if eta is not None:
        sd_prior = LogNormalPrior(loc=0.0, scale=0.5)
        task_prior = LKJCovariancePrior(n=model.num_tasks, eta=eta, sd_prior=sd_prior)
    signed_task_kernel = IndexKernel(
        num_tasks=model.num_tasks,
        rank=RANK,
        prior=task_prior,
        active_dims=[model._task_feature],
    )
    model.covar_module = data_kernel * signed_task_kernel


def fit_and_predict(model: Any, eval_x: torch.Tensor, *, maxiter: int) -> tuple[np.ndarray, np.ndarray]:
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": maxiter}})
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(eval_x, observation_noise=True)
        pred_mean = posterior.mean.squeeze(-1).detach().cpu().numpy()
        pred_std = posterior.variance.squeeze(-1).clamp_min(0.0).sqrt().detach().cpu().numpy()
    return pred_mean, pred_std


def task_correlation(model: Any) -> pd.DataFrame:
    task_kernel = model.covar_module.kernels[1]
    task_x = torch.zeros(model.num_tasks, len(INPUT_COLUMNS) + 1, dtype=model.train_inputs[0].dtype)
    task_x[:, model._task_feature] = torch.arange(model.num_tasks, dtype=task_x.dtype)
    with torch.no_grad():
        cov = task_kernel(task_x).to_dense().detach().cpu().numpy()
    diag = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(diag, diag)
    return pd.DataFrame(corr, index=ETF_TICKERS, columns=ETF_TICKERS)


def covariance_diagnostics(corr: pd.DataFrame, *, variant: str, window_date: pd.Timestamp) -> dict[str, float | str]:
    values = corr.to_numpy(dtype=float)
    offdiag = values[~np.eye(values.shape[0], dtype=bool)]
    eigvals = np.linalg.eigvalsh(values)
    eigvals_sorted = np.sort(eigvals)[::-1]
    total = eigvals_sorted.sum()
    rows: dict[str, float | str] = {
        "variant": variant,
        "window_date": window_date.date().isoformat(),
        "corr_min": float(np.nanmin(offdiag)),
        "corr_max": float(np.nanmax(offdiag)),
        "corr_mean_offdiag": float(np.nanmean(offdiag)),
        "negative_corr_share": float(np.mean(offdiag < 0)),
        "eig1_share": float(eigvals_sorted[0] / total) if total else math.nan,
        "eig5_share": float(eigvals_sorted[:5].sum() / total) if total else math.nan,
    }
    within: list[float] = []
    between: list[float] = []
    for i, left in enumerate(corr.index):
        for j, right in enumerate(corr.columns):
            if j <= i:
                continue
            target = within if ASSET_GROUPS[left] == ASSET_GROUPS[right] else between
            target.append(float(corr.iloc[i, j]))
    rows["corr_mean_within_group"] = float(np.nanmean(within))
    rows["corr_mean_between_group"] = float(np.nanmean(between))
    return rows


def panelize(prediction_rows: list[dict[str, Any]], column: str) -> pd.DataFrame:
    df = pd.DataFrame(prediction_rows)
    panel = df.pivot(index="date", columns="asset_id", values=column)
    return panel.reindex(columns=ETF_TICKERS)


def window_scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    rmse = float(np.sqrt(np.nanmean(residual**2)))
    mae = float(np.nanmean(np.abs(residual)))
    denom = np.clip(y_std, 1e-12, None)
    nlpd = float(np.nanmean(0.5 * np.log(2 * np.pi * denom**2) + 0.5 * (residual / denom) ** 2))
    cover_1sd = float(np.nanmean(np.abs(residual) <= denom))
    cover_2sd = float(np.nanmean(np.abs(residual) <= 2 * denom))
    return {"rmse": rmse, "mae": mae, "nlpd": nlpd, "cover_1sd": cover_1sd, "cover_2sd": cover_2sd}


def stable_seed(base_seed: int, variant_name: str, window_index: int) -> int:
    variant_offset = sum((idx + 1) * ord(char) for idx, char in enumerate(variant_name))
    return int(base_seed + variant_offset + window_index)


def summarize_variant(variant: str, pred_rows: list[dict[str, Any]]) -> dict[str, float | str]:
    y_true = panelize(pred_rows, "y_true")
    y_pred = panelize(pred_rows, "y_pred")
    y_std = panelize(pred_rows, "y_std")
    y_true_resid = panelize(pred_rows, "y_true_resid")
    y_pred_resid = panelize(pred_rows, "y_pred_resid")
    pricing_metrics = evaluate_asset_pricing(y_true, y_pred)
    ls_returns = long_short_returns_topk(y_true, y_pred, k=5, q=None, min_assets=10)
    ls_returns.index = y_true.index
    ls_stats = portfolio_stats(ls_returns, periods_per_year=12)
    scalars = window_scalar_metrics(y_true.to_numpy(), y_pred.to_numpy(), y_std.to_numpy())
    residual_pricing_metrics = evaluate_asset_pricing(y_true_resid, y_pred_resid)
    residual_signal_mask = y_pred_resid.apply(lambda row: row.dropna().nunique() > 1, axis=1)
    if residual_signal_mask.any():
        residual_ls_returns = long_short_returns_topk(
            y_true_resid.loc[residual_signal_mask],
            y_pred_resid.loc[residual_signal_mask],
            k=5,
            q=None,
            min_assets=10,
        )
        residual_ls_returns.index = y_true_resid.loc[residual_signal_mask].index
        residual_ls_stats = portfolio_stats(residual_ls_returns, periods_per_year=12)
        residual_ls_mean = float(residual_ls_returns.mean())
        residual_ls_hit_rate = float((residual_ls_returns > 0).mean())
    else:
        residual_ls_stats = {
            "cum_return": math.nan,
            "ann_return": math.nan,
            "ann_vol": math.nan,
            "sharpe": math.nan,
            "max_drawdown": math.nan,
        }
        residual_ls_mean = math.nan
        residual_ls_hit_rate = math.nan
    residual_scalars = window_scalar_metrics(
        y_true_resid.to_numpy(),
        y_pred_resid.to_numpy(),
        y_std.to_numpy(),
    )
    result: dict[str, float | str] = {"variant": variant, "n_windows": float(len(y_true))}
    result.update(pricing_metrics)
    result.update(scalars)
    result.update({f"top_bottom_5_{k}": float(v) for k, v in ls_stats.items()})
    result["top_bottom_5_mean_monthly"] = float(ls_returns.mean())
    result["top_bottom_5_hit_rate"] = float((ls_returns > 0).mean())
    result.update({f"resid_{key}": value for key, value in residual_pricing_metrics.items()})
    result.update({f"resid_{key}": value for key, value in residual_scalars.items()})
    result.update({f"resid_top_bottom_5_{key}": float(value) for key, value in residual_ls_stats.items()})
    result["resid_top_bottom_5_mean_monthly"] = residual_ls_mean
    result["resid_top_bottom_5_hit_rate"] = residual_ls_hit_rate
    return result


def build_manifest(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    df: pd.DataFrame,
    scored_dates: list[pd.Timestamp],
    live_date: pd.Timestamp | None,
    variants: list[Variant],
    global_time_mins: torch.Tensor,
    global_time_ranges: torch.Tensor,
) -> dict[str, Any]:
    train_sizes = []
    for window_date in scored_dates:
        train_df = df[(df["date"] < window_date) & df[TARGET_COL].notna()]
        train_sizes.append(
            {
                "window_date": window_date.date().isoformat(),
                "train_rows": int(len(train_df)),
                "train_months": int(train_df["date"].nunique()),
                "train_start": train_df["date"].min().date().isoformat(),
                "train_end": train_df["date"].max().date().isoformat(),
            }
        )

    live_train = None
    if live_date is not None:
        live_train_df = df[(df["date"] < live_date) & df[TARGET_COL].notna()]
        live_train = {
            "window_date": live_date.date().isoformat(),
            "train_rows": int(len(live_train_df)),
            "train_months": int(live_train_df["date"].nunique()),
            "train_start": live_train_df["date"].min().date().isoformat(),
            "train_end": live_train_df["date"].max().date().isoformat(),
        }

    return {
        "schema": "bayesfolio.task_covariance_experiment.manifest.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git_sha": git_sha(short=False),
        "git_dirty_summary_at_start": git_dirty_summary(),
        "feature_path": str(args.feature_path),
        "feature_sha256": sha256_file(args.feature_path),
        "data": {
            "rows": int(len(df)),
            "date_min": df["date"].min().date().isoformat(),
            "date_max": df["date"].max().date().isoformat(),
            "asset_count": int(df["asset_id"].nunique()),
            "target_col": TARGET_COL,
            "input_columns": INPUT_COLUMNS,
        },
        "scored_dates": [date.date().isoformat() for date in scored_dates],
        "live_date": live_date.date().isoformat() if live_date is not None else None,
        "train_sizes": train_sizes,
        "live_train_size": live_train,
        "variants": [variant.name for variant in variants],
        "modeling": {
            "rank": RANK,
            "mean_config": MeanKind.MULTITASK_CONSTANT.value,
            "task_feature": TASK_FEATURE,
            "add_time_varying_lengthscale": True,
            "add_time_varying_outputscale": True,
            "scaling": (
                "global min-max for time input columns over the full feature frame; "
                "train-window min-max for remaining non-task input columns; "
                "apply same stats to eval rows"
            ),
            "time_scaling": {
                "columns": TIME_COLS,
                "mins": {col: float(value) for col, value in zip(TIME_COLS, global_time_mins, strict=True)},
                "ranges": {col: float(value) for col, value in zip(TIME_COLS, global_time_ranges, strict=True)},
            },
            "outcome_transform": "StratifiedStandardize by ETF task",
            "residualized_metrics": "subtract each ETF training-window historical mean from y_true and y_pred",
        },
        "runtime": {
            "seed": args.seed,
            "seed_policy": "stable per variant/window from base seed + variant name offset + window index",
            "maxiter": args.maxiter,
            "botorch": botorch.__version__,
            "gpytorch": gpytorch.__version__,
            "torch": torch.__version__,
        },
        "output_dir": str(output_dir),
    }


def run(args: argparse.Namespace) -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    variants = [VARIANTS[name] for name in args.variants]
    df = load_features(args.feature_path)
    global_time_mins, global_time_ranges = global_time_scaling(df)
    scored_dates, live_date = scored_and_live_dates(df, args.max_windows)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = build_manifest(
        args=args,
        output_dir=output_dir,
        df=df,
        scored_dates=scored_dates,
        live_date=live_date,
        variants=variants,
        global_time_mins=global_time_mins,
        global_time_ranges=global_time_ranges,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    pred_rows: list[dict[str, Any]] = []
    window_metric_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    live_rows: list[dict[str, Any]] = []

    for variant in variants:
        print(f"== {variant.name} ==", flush=True)
        variant_pred_rows: list[dict[str, Any]] = []
        for window_index, window_date in enumerate(scored_dates):
            print(f"  window {window_date.date()}", flush=True)
            train_df = df[(df["date"] < window_date) & df[TARGET_COL].notna()].copy()
            eval_df = df[df["date"] == window_date].copy()
            if variant.task_kernel == "baseline":
                y_pred, y_std = historical_mean_predict(train_df, eval_df)
            else:
                torch.manual_seed(stable_seed(args.seed, variant.name, window_index))
                train_x, train_y, eval_x, _, _, _ = prepare_window_tensors(
                    train_df,
                    eval_df,
                    global_time_mins=global_time_mins,
                    global_time_ranges=global_time_ranges,
                )
                model = build_model(train_x, train_y, variant)
                y_pred, y_std = fit_and_predict(model, eval_x, maxiter=args.maxiter)
                corr = task_correlation(model)
                diag_rows.append(covariance_diagnostics(corr, variant=variant.name, window_date=window_date))

            y_true = eval_df[TARGET_COL].to_numpy(dtype=float)
            window_metric_rows.append(
                {
                    "variant": variant.name,
                    "date": window_date.date().isoformat(),
                    **window_scalar_metrics(y_true, y_pred, y_std),
                }
            )
            train_means = training_asset_means(train_df, eval_df)
            for row, mean, std, train_mean in zip(
                eval_df.itertuples(index=False),
                y_pred,
                y_std,
                train_means,
                strict=True,
            ):
                record = {
                    "variant": variant.name,
                    "date": pd.Timestamp(row.date).date().isoformat(),
                    "asset_id": str(row.asset_id),
                    "train_mean": float(train_mean),
                    "y_true": float(row.y_excess_lead),
                    "y_pred": float(mean),
                    "y_std": float(std),
                    "y_true_resid": float(row.y_excess_lead - train_mean),
                    "y_pred_resid": float(mean - train_mean),
                }
                pred_rows.append(record)
                variant_pred_rows.append(record)

        if live_date is not None:
            print(f"  live {live_date.date()}", flush=True)
            train_df = df[(df["date"] < live_date) & df[TARGET_COL].notna()].copy()
            eval_df = df[df["date"] == live_date].copy()
            if variant.task_kernel == "baseline":
                y_pred, y_std = historical_mean_predict(train_df, eval_df)
            else:
                torch.manual_seed(stable_seed(args.seed, variant.name, len(scored_dates)))
                train_x, train_y, eval_x, _, _, _ = prepare_window_tensors(
                    train_df,
                    eval_df,
                    global_time_mins=global_time_mins,
                    global_time_ranges=global_time_ranges,
                )
                model = build_model(train_x, train_y, variant)
                y_pred, y_std = fit_and_predict(model, eval_x, maxiter=args.maxiter)
            train_means = training_asset_means(train_df, eval_df)
            for row, mean, std, train_mean in zip(
                eval_df.itertuples(index=False),
                y_pred,
                y_std,
                train_means,
                strict=True,
            ):
                live_rows.append(
                    {
                        "variant": variant.name,
                        "date": pd.Timestamp(row.date).date().isoformat(),
                        "asset_id": str(row.asset_id),
                        "train_mean": float(train_mean),
                        "y_pred": float(mean),
                        "y_std": float(std),
                        "score": float(mean / max(float(std), 1e-12)),
                        "y_pred_resid": float(mean - train_mean),
                        "resid_score": float((mean - train_mean) / max(float(std), 1e-12)),
                    }
                )

        summary_row = summarize_variant(variant.name, variant_pred_rows)
        pd.DataFrame([summary_row]).to_csv(
            output_dir / f"summary_{variant.name}.csv",
            index=False,
        )

    summary_rows = [
        summarize_variant(name, [r for r in pred_rows if r["variant"] == name])
        for name in args.variants
    ]
    summary = pd.DataFrame(summary_rows)
    pd.DataFrame(pred_rows).to_csv(output_dir / "window_predictions.csv", index=False)
    pd.DataFrame(window_metric_rows).to_csv(output_dir / "window_metrics.csv", index=False)
    summary.to_csv(output_dir / "variant_summary.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(output_dir / "task_covariance_diagnostics.csv", index=False)
    if live_rows:
        pd.DataFrame(live_rows).to_csv(output_dir / "live_june_predictions.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    run(parse_args())
