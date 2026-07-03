"""Walk-forward portfolio test with plug-in heteroskedastic observation noise.

The candidate keeps the existing signed multitask GP latent covariance and task
covariance, but replaces inferred homoskedastic/task likelihood noise with fixed
per-observation noise estimated by a separate rolling raw-return variance model.
"""

from __future__ import annotations

import argparse
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
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import StratifiedStandardize
from gpytorch.mlls import ExactMarginalLogLikelihood

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENT_DIR = Path(__file__).resolve().parent
PORTFOLIO_EXPERIMENT = (
    REPO_ROOT
    / "experiments"
    / "2026-06-portfolio-optimization"
    / "run_monthly_optimization_walkforward.py"
)
PORT_SPEC = importlib.util.spec_from_file_location("portfolio_walkforward", PORTFOLIO_EXPERIMENT)
if PORT_SPEC is None or PORT_SPEC.loader is None:
    raise RuntimeError(f"Unable to load portfolio experiment from {PORTFOLIO_EXPERIMENT}")
portfolio_exp = importlib.util.module_from_spec(PORT_SPEC)
sys.modules[PORT_SPEC.name] = portfolio_exp
PORT_SPEC.loader.exec_module(portfolio_exp)

task_exp = portfolio_exp.task_exp
multitask_builder = task_exp.multitask_builder

OUTPUT_ROOT = EXPERIMENT_DIR / "outputs"
DEFAULT_FEATURE_PATH = portfolio_exp.DEFAULT_FEATURE_PATH
BASELINE_RUN = (
    REPO_ROOT
    / "experiments"
    / "2026-06-portfolio-optimization"
    / "outputs"
    / "runs"
    / "20260616_signed_lkj_eta_2_portfolio"
)
PERIODS_PER_YEAR = portfolio_exp.PERIODS_PER_YEAR
STARTING_VALUE = portfolio_exp.STARTING_VALUE
HELPER_ASSETS = portfolio_exp.HELPER_ASSETS


@dataclass(frozen=True)
class NoiseEstimate:
    train_yvar_raw: np.ndarray
    eval_yvar_raw: np.ndarray
    diagnostics: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--posterior-scenarios", type=int, default=5000)
    parser.add_argument("--upperlng", type=float, default=0.20)
    parser.add_argument("--nea", type=int, default=10)
    parser.add_argument("--noise-lookback", type=int, default=6)
    parser.add_argument("--noise-min-periods", type=int, default=3)
    parser.add_argument("--noise-class-shrinkage", type=float, default=0.35)
    parser.add_argument(
        "--noise-source",
        choices=[
            "trailing_return",
            "residual_history",
            "residual_ewma",
            "residual_robust",
            "residual_shrinkage",
        ],
        default="trailing_return",
        help="Source used to estimate raw observation noise variance.",
    )
    parser.add_argument(
        "--noise-variance-scale",
        type=float,
        default=1.0,
        help="Multiplicative calibration applied to raw noise variances before floor/ceiling clipping.",
    )
    parser.add_argument("--noise-floor-raw-std", type=float, default=0.005)
    parser.add_argument("--noise-ceiling-raw-std", type=float, default=0.20)
    parser.add_argument(
        "--residual-ewma-half-life",
        type=float,
        default=3.0,
        help="Half-life, in prior residual windows, for residual_ewma variance estimates.",
    )
    parser.add_argument(
        "--residual-robust-quantile",
        type=float,
        default=0.90,
        help="Central winsorization quantile for residual_robust variance estimates.",
    )
    parser.add_argument(
        "--residual-shrinkage-prior-n",
        type=float,
        default=6.0,
        help="Prior sample size for residual_shrinkage adaptive asset-to-class variance shrinkage.",
    )
    parser.add_argument("--baseline-run", type=Path, default=BASELINE_RUN)
    parser.add_argument(
        "--residual-predictions-path",
        type=Path,
        default=None,
        help="Prediction CSV used by residual-history noise mode. Defaults to baseline-run/gp_window_predictions.csv.",
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


def add_asset_group(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["asset_group"] = output["asset_id"].astype(str).map(task_exp.ASSET_GROUPS).fillna("unknown")
    return output


def shifted_rolling_var(values: pd.Series, *, lookback: int, min_periods: int) -> pd.Series:
    return values.rolling(window=lookback, min_periods=min_periods).var(ddof=1).shift(1)


def estimate_noise(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    lookback: int,
    min_periods: int,
    class_shrinkage: float,
    variance_scale: float,
    floor_raw_std: float,
    ceiling_raw_std: float,
) -> NoiseEstimate:
    """Estimate raw-return noise variance from trailing returns only."""

    if lookback < 2:
        raise ValueError(f"noise lookback must be at least 2, got {lookback}")
    if not 0.0 <= class_shrinkage <= 1.0:
        raise ValueError(f"noise class shrinkage must be in [0, 1], got {class_shrinkage}")
    if variance_scale <= 0:
        raise ValueError(f"noise variance scale must be positive, got {variance_scale}")
    if floor_raw_std <= 0 or ceiling_raw_std <= floor_raw_std:
        raise ValueError("noise floor/ceiling raw std settings are invalid")

    target = task_exp.TARGET_COL
    train = add_asset_group(train_df).sort_values(["asset_id", "date"]).copy()
    global_var = float(train[target].var(ddof=1))
    floor_var = float(floor_raw_std**2)
    ceiling_var = float(ceiling_raw_std**2)
    if not np.isfinite(global_var) or global_var <= 0:
        global_var = floor_var
    global_var = float(np.clip(global_var, floor_var, ceiling_var))

    asset_roll = train.groupby("asset_id", observed=True)[target].transform(
        lambda series: shifted_rolling_var(series, lookback=lookback, min_periods=min_periods)
    )
    asset_expanding = train.groupby("asset_id", observed=True)[target].transform(
        lambda series: series.expanding(min_periods=min_periods).var(ddof=1).shift(1)
    )
    class_roll = train.groupby("asset_group", observed=True)[target].transform(
        lambda series: shifted_rolling_var(series, lookback=lookback * 3, min_periods=min_periods * 2)
    )
    class_expanding = train.groupby("asset_group", observed=True)[target].transform(
        lambda series: series.expanding(min_periods=min_periods * 2).var(ddof=1).shift(1)
    )

    asset_var = asset_roll.fillna(asset_expanding)
    class_var = class_roll.fillna(class_expanding)
    raw_var = (1.0 - class_shrinkage) * asset_var + class_shrinkage * class_var
    raw_var = raw_var.fillna(asset_var).fillna(class_var).fillna(global_var)
    raw_var = raw_var * variance_scale
    raw_var = raw_var.clip(lower=floor_var, upper=ceiling_var).astype(float)

    eval_rows: list[dict[str, Any]] = []
    eval_vars: list[float] = []
    train_for_eval = add_asset_group(train_df).sort_values(["asset_id", "date"])
    for _, row in eval_df.iterrows():
        asset_id = str(row["asset_id"])
        asset_group = task_exp.ASSET_GROUPS.get(asset_id, "unknown")
        asset_history = train_for_eval.loc[train_for_eval["asset_id"].astype(str) == asset_id, target].tail(lookback)
        class_history = train_for_eval.loc[train_for_eval["asset_group"] == asset_group, target].tail(lookback * 3)
        asset_var_eval = float(asset_history.var(ddof=1)) if len(asset_history) >= min_periods else math.nan
        class_var_eval = float(class_history.var(ddof=1)) if len(class_history) >= min_periods * 2 else math.nan
        if np.isfinite(asset_var_eval) and np.isfinite(class_var_eval):
            eval_var = (1.0 - class_shrinkage) * asset_var_eval + class_shrinkage * class_var_eval
        elif np.isfinite(asset_var_eval):
            eval_var = asset_var_eval
        elif np.isfinite(class_var_eval):
            eval_var = class_var_eval
        else:
            eval_var = global_var
        eval_var = float(np.clip(eval_var * variance_scale, floor_var, ceiling_var))
        eval_vars.append(eval_var)
        eval_rows.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "asset_id": asset_id,
                "asset_group": asset_group,
                "noise_role": "eval",
                "raw_noise_variance": eval_var,
                "raw_noise_std": math.sqrt(eval_var),
                "asset_trailing_variance": asset_var_eval,
                "class_trailing_variance": class_var_eval,
                "global_train_variance": global_var,
            }
        )

    train_diag = train.loc[:, ["date", "asset_id", "asset_group"]].copy()
    train_diag["date"] = pd.to_datetime(train_diag["date"]).dt.date.astype(str)
    train_diag["noise_role"] = "train"
    train_diag["raw_noise_variance"] = raw_var.to_numpy(dtype=float)
    train_diag["raw_noise_std"] = np.sqrt(train_diag["raw_noise_variance"].to_numpy(dtype=float))
    train_diag["asset_trailing_variance"] = asset_var.to_numpy(dtype=float)
    train_diag["class_trailing_variance"] = class_var.to_numpy(dtype=float)
    train_diag["global_train_variance"] = global_var
    diagnostics = pd.concat([train_diag, pd.DataFrame(eval_rows)], ignore_index=True)
    return NoiseEstimate(
        train_yvar_raw=raw_var.to_numpy(dtype=float).reshape(-1, 1),
        eval_yvar_raw=np.array(eval_vars, dtype=float).reshape(-1, 1),
        diagnostics=diagnostics,
    )


def load_residual_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"date", "asset_id", "y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Residual predictions file is missing columns: {sorted(missing)}")
    frame = frame.loc[:, ["date", "asset_id", "y_true", "y_pred"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["asset_id"] = frame["asset_id"].astype(str)
    frame["asset_group"] = frame["asset_id"].map(task_exp.ASSET_GROUPS).fillna("unknown")
    frame["residual"] = frame["y_true"].astype(float) - frame["y_pred"].astype(float)
    return frame.sort_values(["date", "asset_id"]).reset_index(drop=True)


def weighted_variance(values: pd.Series, *, half_life: float) -> float:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return math.nan
    if half_life <= 0:
        raise ValueError(f"EWMA half-life must be positive, got {half_life}")
    ages = np.arange(len(clean) - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / half_life)
    weights = weights / weights.sum()
    mean = float(np.sum(weights * clean.to_numpy(dtype=float)))
    variance = float(np.sum(weights * np.square(clean.to_numpy(dtype=float) - mean)))
    return variance if np.isfinite(variance) and variance > 0 else math.nan


def winsorized_variance(values: pd.Series, *, quantile: float) -> float:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return math.nan
    if not 0.5 < quantile < 1.0:
        raise ValueError(f"robust quantile must be in (0.5, 1.0), got {quantile}")
    lower = float(clean.quantile(1.0 - quantile))
    upper = float(clean.quantile(quantile))
    clipped = clean.clip(lower=lower, upper=upper)
    variance = float(clipped.var(ddof=1))
    return variance if np.isfinite(variance) and variance > 0 else math.nan


def estimate_residual_history_noise(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    residual_history: pd.DataFrame,
    *,
    window_date: pd.Timestamp,
    lookback: int,
    min_periods: int,
    class_shrinkage: float,
    variance_scale: float,
    floor_raw_std: float,
    ceiling_raw_std: float,
    method: str = "residual_history",
    ewma_half_life: float = 3.0,
    robust_quantile: float = 0.90,
    shrinkage_prior_n: float = 6.0,
) -> NoiseEstimate:
    """Estimate noise from prior out-of-sample baseline residuals."""

    fallback = estimate_noise(
        train_df,
        eval_df,
        lookback=lookback,
        min_periods=min_periods,
        class_shrinkage=class_shrinkage,
        variance_scale=variance_scale,
        floor_raw_std=floor_raw_std,
        ceiling_raw_std=ceiling_raw_std,
    )
    if variance_scale <= 0:
        raise ValueError(f"noise variance scale must be positive, got {variance_scale}")
    prior = residual_history[residual_history["date"] < window_date].copy()
    if prior.empty:
        diagnostics = fallback.diagnostics.copy()
        diagnostics["noise_model_fallback"] = "trailing_return_no_prior_residuals"
        return NoiseEstimate(fallback.train_yvar_raw, fallback.eval_yvar_raw, diagnostics)

    floor_var = float(floor_raw_std**2)
    ceiling_var = float(ceiling_raw_std**2)
    global_var = float(prior["residual"].var(ddof=1))
    if not np.isfinite(global_var) or global_var <= 0:
        global_var = float(np.nanmean(fallback.eval_yvar_raw))
    if not np.isfinite(global_var) or global_var <= 0:
        global_var = floor_var

    if method not in {"residual_history", "residual_ewma", "residual_robust", "residual_shrinkage"}:
        raise ValueError(f"unsupported residual noise method: {method}")
    if shrinkage_prior_n <= 0:
        raise ValueError(f"shrinkage prior sample size must be positive, got {shrinkage_prior_n}")

    recent = prior.groupby("asset_id", observed=True).tail(lookback)
    recent_class = prior.groupby("asset_group", observed=True).tail(lookback * 3)
    asset_counts = recent.groupby("asset_id", observed=True)["residual"].count()
    class_counts = recent_class.groupby("asset_group", observed=True)["residual"].count()

    if method == "residual_ewma":
        asset_vars = recent.groupby("asset_id", observed=True)["residual"].apply(
            lambda series: weighted_variance(series, half_life=ewma_half_life)
        )
        class_vars = recent_class.groupby("asset_group", observed=True)["residual"].apply(
            lambda series: weighted_variance(series, half_life=ewma_half_life)
        )
    elif method == "residual_robust":
        asset_vars = recent.groupby("asset_id", observed=True)["residual"].apply(
            lambda series: winsorized_variance(series, quantile=robust_quantile)
        )
        class_vars = recent_class.groupby("asset_group", observed=True)["residual"].apply(
            lambda series: winsorized_variance(series, quantile=robust_quantile)
        )
    else:
        asset_vars = recent.groupby("asset_id", observed=True)["residual"].var(ddof=1)
        class_vars = recent_class.groupby("asset_group", observed=True)["residual"].var(ddof=1)
    asset_vars = asset_vars.where(asset_counts >= min_periods)
    class_vars = class_vars.where(class_counts >= min_periods * 2)

    def variance_for_asset(asset_id: str) -> tuple[float, float, float, str, float, int, int]:
        asset_group = task_exp.ASSET_GROUPS.get(asset_id, "unknown")
        asset_var = float(asset_vars.get(asset_id, math.nan))
        class_var = float(class_vars.get(asset_group, math.nan))
        asset_n = int(asset_counts.get(asset_id, 0))
        class_n = int(class_counts.get(asset_group, 0))
        shrinkage_weight = math.nan
        if np.isfinite(asset_var) and np.isfinite(class_var):
            if method == "residual_shrinkage":
                shrinkage_weight = float(shrinkage_prior_n / (asset_n + shrinkage_prior_n))
            else:
                shrinkage_weight = class_shrinkage
            raw_var = (1.0 - shrinkage_weight) * asset_var + shrinkage_weight * class_var
            source = "asset_class_residual"
        elif np.isfinite(asset_var):
            raw_var = asset_var
            source = "asset_residual"
        elif np.isfinite(class_var):
            raw_var = class_var
            source = "class_residual"
        else:
            raw_var = global_var
            source = "global_residual"
        raw_var = float(np.clip(raw_var * variance_scale, floor_var, ceiling_var))
        return raw_var, asset_var, class_var, source, shrinkage_weight, asset_n, class_n

    train_rows: list[dict[str, Any]] = []
    train_vars: list[float] = []
    train = add_asset_group(train_df)
    for _, row in train.iterrows():
        asset_id = str(row["asset_id"])
        raw_var, asset_var, class_var, source, shrinkage_weight, asset_n, class_n = variance_for_asset(asset_id)
        train_vars.append(raw_var)
        train_rows.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "asset_id": asset_id,
                "asset_group": row["asset_group"],
                "noise_role": "train",
                "raw_noise_variance": raw_var,
                "raw_noise_std": math.sqrt(raw_var),
                "asset_residual_variance": asset_var,
                "class_residual_variance": class_var,
                "global_residual_variance": global_var,
                "noise_source_detail": source,
                "residual_noise_method": method,
                "residual_shrinkage_weight": shrinkage_weight,
                "asset_prior_residual_count": asset_n,
                "class_prior_residual_count": class_n,
                "residual_ewma_half_life": ewma_half_life if method == "residual_ewma" else math.nan,
                "residual_robust_quantile": robust_quantile if method == "residual_robust" else math.nan,
                "residual_shrinkage_prior_n": shrinkage_prior_n if method == "residual_shrinkage" else math.nan,
                "n_prior_residual_dates": int(prior["date"].nunique()),
            }
        )

    eval_rows: list[dict[str, Any]] = []
    eval_vars: list[float] = []
    eval_with_group = add_asset_group(eval_df)
    for _, row in eval_with_group.iterrows():
        asset_id = str(row["asset_id"])
        raw_var, asset_var, class_var, source, shrinkage_weight, asset_n, class_n = variance_for_asset(asset_id)
        eval_vars.append(raw_var)
        eval_rows.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "asset_id": asset_id,
                "asset_group": row["asset_group"],
                "noise_role": "eval",
                "raw_noise_variance": raw_var,
                "raw_noise_std": math.sqrt(raw_var),
                "asset_residual_variance": asset_var,
                "class_residual_variance": class_var,
                "global_residual_variance": global_var,
                "noise_source_detail": source,
                "residual_noise_method": method,
                "residual_shrinkage_weight": shrinkage_weight,
                "asset_prior_residual_count": asset_n,
                "class_prior_residual_count": class_n,
                "residual_ewma_half_life": ewma_half_life if method == "residual_ewma" else math.nan,
                "residual_robust_quantile": robust_quantile if method == "residual_robust" else math.nan,
                "residual_shrinkage_prior_n": shrinkage_prior_n if method == "residual_shrinkage" else math.nan,
                "n_prior_residual_dates": int(prior["date"].nunique()),
            }
        )
    diagnostics = pd.DataFrame([*train_rows, *eval_rows])
    diagnostics["noise_model_fallback"] = "none"
    return NoiseEstimate(
        train_yvar_raw=np.array(train_vars, dtype=float).reshape(-1, 1),
        eval_yvar_raw=np.array(eval_vars, dtype=float).reshape(-1, 1),
        diagnostics=diagnostics,
    )


def build_fixed_noise_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_yvar: torch.Tensor,
) -> Any:
    """Build the signed LKJ multitask GP with fixed observation noise."""

    task_idx = train_x.shape[-1] - 1
    all_task_values = train_x[:, task_idx].to(torch.long).unique(sorted=True)
    input_transform = Normalize(
        d=train_x.shape[-1],
        indices=list(range(len(task_exp.INPUT_COLUMNS))),
    )
    outcome_transform = StratifiedStandardize(
        stratification_idx=task_idx,
        all_task_values=all_task_values,
        batch_shape=train_y.shape[:-2],
    )
    add_tv_os_ls, builder = task_exp.time_modulation_builder("lengthscale_only")
    original_builder = multitask_builder.add_time_varying_os_ls
    multitask_builder.add_time_varying_os_ls = builder
    try:
        covar_module = multitask_builder.build_covar_module(
            task_exp.build_covar_config(),
            batch_shape=train_x.shape[:-2],
        )
        if add_tv_os_ls:
            covar_module = multitask_builder.add_time_varying_os_ls(covar_module)
    finally:
        multitask_builder.add_time_varying_os_ls = original_builder

    num_tasks = int(train_x[..., task_idx].to(torch.long).unique().numel())
    mean_module = multitask_builder.build_mean_module(
        task_exp.MeanModuleConfig(kind=task_exp.MeanKind.MULTITASK_CONSTANT, num_tasks=num_tasks)
    )
    model = MultiTaskGP(
        train_X=train_x,
        train_Y=train_y,
        task_feature=task_exp.TASK_FEATURE,
        train_Yvar=train_yvar,
        covar_module=covar_module,
        mean_module=mean_module,
        rank=task_exp.RANK,
        task_covar_prior=task_exp.VARIANTS["signed_lkj_eta_2"].task_covar_prior,
        outcome_transform=outcome_transform,
        input_transform=input_transform,
    )
    task_exp.replace_with_signed_index_kernel(model, eta=2.0)
    return model


def fit_gp_window(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    args: argparse.Namespace,
    seed: int,
    maxiter: int,
    posterior_scenarios: int,
    residual_history: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.DataFrame, dict[str, Any] | None]:
    window_date = pd.Timestamp(eval_df["date"].iloc[0])
    if args.noise_source.startswith("residual_"):
        if residual_history is None:
            raise ValueError(f"{args.noise_source} noise source requires residual_history")
        noise = estimate_residual_history_noise(
            train_df,
            eval_df,
            residual_history,
            window_date=window_date,
            lookback=args.noise_lookback,
            min_periods=args.noise_min_periods,
            class_shrinkage=args.noise_class_shrinkage,
            variance_scale=args.noise_variance_scale,
            floor_raw_std=args.noise_floor_raw_std,
            ceiling_raw_std=args.noise_ceiling_raw_std,
            method=args.noise_source,
            ewma_half_life=args.residual_ewma_half_life,
            robust_quantile=args.residual_robust_quantile,
            shrinkage_prior_n=args.residual_shrinkage_prior_n,
        )
    else:
        noise = estimate_noise(
            train_df,
            eval_df,
            lookback=args.noise_lookback,
            min_periods=args.noise_min_periods,
            class_shrinkage=args.noise_class_shrinkage,
            variance_scale=args.noise_variance_scale,
            floor_raw_std=args.noise_floor_raw_std,
            ceiling_raw_std=args.noise_ceiling_raw_std,
        )
    train_x, train_y, eval_x, _, _, _ = task_exp.prepare_window_tensors(train_df, eval_df)
    train_yvar = torch.tensor(noise.train_yvar_raw, dtype=torch.float64)
    eval_yvar = torch.tensor(noise.eval_yvar_raw.squeeze(-1), dtype=torch.float64)
    torch.manual_seed(seed)
    model = build_fixed_noise_model(train_x, train_y, train_yvar)
    model.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": maxiter}})
    model.eval()
    model.likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_posterior = model.posterior(eval_x, observation_noise=False)
        latent_mean_t = latent_posterior.mean.squeeze(-1)
        latent_var_t = latent_posterior.variance.squeeze(-1).clamp_min(0.0)
        latent_samples_t = latent_posterior.rsample(torch.Size([posterior_scenarios])).squeeze(-1)
        generator = torch.Generator(device=latent_samples_t.device)
        generator.manual_seed(seed + 10_000)
        noise_samples = torch.randn(
            latent_samples_t.shape,
            dtype=latent_samples_t.dtype,
            device=latent_samples_t.device,
            generator=generator,
        ) * eval_yvar.sqrt().to(latent_samples_t).unsqueeze(0)
        scenario_samples = (latent_samples_t + noise_samples).detach().cpu().numpy()
        pred_mean = latent_mean_t.detach().cpu().numpy()
        pred_std = (latent_var_t + eval_yvar.to(latent_var_t)).sqrt().detach().cpu().numpy()

    assets = eval_df["asset_id"].astype(str).tolist()
    scenarios = pd.DataFrame(scenario_samples, columns=assets)
    predictions = pd.DataFrame(
        {
            "date": pd.Timestamp(eval_df["date"].iloc[0]).date().isoformat(),
            "asset_id": assets,
            "y_true": eval_df[task_exp.TARGET_COL].to_numpy(dtype=float),
            "y_pred": pred_mean,
            "y_std": pred_std,
            "latent_y_std": latent_var_t.sqrt().detach().cpu().numpy(),
            "noise_y_std": np.sqrt(noise.eval_yvar_raw.squeeze(-1)),
            "score": pred_mean / np.clip(pred_std, 1e-12, None),
        }
    )
    diagnostics = portfolio_exp.collect_model_diagnostics(
        model,
        eval_x=eval_x,
        window_date=pd.Timestamp(eval_df["date"].iloc[0]),
        experiment="heteroskedastic_fixed_noise_signed_lkj_eta_2",
    )
    for row in diagnostics:
        row["noise_model"] = "rolling_asset_class_shrunk_variance"
    task_diag = None
    try:
        corr = task_exp.task_correlation(model)
        task_diag = task_exp.covariance_diagnostics(
            corr,
            variant="heteroskedastic_fixed_noise_signed_lkj_eta_2",
            window_date=pd.Timestamp(eval_df["date"].iloc[0]),
        )
    except Exception:
        task_diag = None
    return scenarios, predictions, diagnostics, noise.diagnostics, task_diag


def performance_stats(returns: pd.Series, weights: pd.DataFrame) -> dict[str, float]:
    return portfolio_exp.performance_stats(returns, weights, starting_value=STARTING_VALUE)


def write_comparison_plots(
    strategy_returns: dict[str, pd.Series],
    noise_diagnostics: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    portfolio_exp.plot_equity_and_drawdown(strategy_returns, output_dir)

    eval_noise = noise_diagnostics[noise_diagnostics["noise_role"] == "eval"].copy()
    eval_noise["date"] = pd.to_datetime(eval_noise["date"])
    fig, ax = plt.subplots(figsize=(10, 6))
    for group, group_df in eval_noise.groupby("asset_group"):
        series = group_df.groupby("date")["raw_noise_std"].mean()
        ax.plot(series.index, series.values, marker="o", label=group)
    ax.set_title("Predicted Eval Noise Std by Asset Group")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Monthly raw-return noise std")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "eval_noise_std_by_group.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predictions["noise_y_std"], (predictions["y_true"] - predictions["y_pred"]).abs(), alpha=0.75)
    ax.set_title("Predicted Noise Std vs Absolute Forecast Error")
    ax.set_xlabel("Predicted monthly noise std")
    ax.set_ylabel("Absolute forecast error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "noise_std_vs_abs_error.png", dpi=160)
    plt.close(fig)


def write_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    df: pd.DataFrame,
    scored_dates: list[pd.Timestamp],
) -> None:
    manifest = {
        "schema": "bayesfolio.heteroskedastic_noise_walkforward.manifest.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git_sha": git_sha(short=False),
        "git_dirty_summary_at_start": git_dirty_summary(),
        "feature_path": str(args.feature_path),
        "feature_sha256": sha256_file(args.feature_path),
        "source_portfolio_experiment": str(PORTFOLIO_EXPERIMENT),
        "baseline_run": str(args.baseline_run),
        "output_dir": str(output_dir),
        "decision_question": (
            "Can a separate rolling noise model improve uncertainty calibration while preserving "
            "the signed multitask GP latent covariance and task covariance used for scenario draws?"
        ),
        "candidate": {
            "model": "signed_lkj_eta_2_multitask_gp_with_fixed_heteroskedastic_noise",
            "latent_covariance": "existing June 2026 time/ETF/macro hierarchical multitask covariance",
            "task_covariance": "signed IndexKernel with LKJ eta=2 replacement",
            "noise_model": {
                "kind": args.noise_source,
                "lookback": args.noise_lookback,
                "min_periods": args.noise_min_periods,
                "class_shrinkage": args.noise_class_shrinkage,
                "variance_scale": args.noise_variance_scale,
                "floor_raw_std": args.noise_floor_raw_std,
                "ceiling_raw_std": args.noise_ceiling_raw_std,
                "residual_ewma_half_life": args.residual_ewma_half_life,
                "residual_robust_quantile": args.residual_robust_quantile,
                "residual_shrinkage_prior_n": args.residual_shrinkage_prior_n,
                "residual_predictions_path": str(args.residual_predictions_path)
                if args.residual_predictions_path is not None
                else str(args.baseline_run / "gp_window_predictions.csv"),
            },
        },
        "data": {
            "rows": int(len(df)),
            "date_min": df["date"].min().date().isoformat(),
            "date_max": df["date"].max().date().isoformat(),
            "target_col": task_exp.TARGET_COL,
            "training_universe": task_exp.ETF_TICKERS,
            "helper_assets_fit_but_excluded": sorted(HELPER_ASSETS),
        },
        "rebalance_dates": [date.date().isoformat() for date in scored_dates],
        "run_config": {
            "maxiter": args.maxiter,
            "seed": args.seed,
            "posterior_scenarios": args.posterior_scenarios,
            "upperlng": args.upperlng,
            "nea": args.nea,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_baseline_summary(path: Path) -> pd.DataFrame | None:
    summary_path = path / "portfolio_summary.csv"
    if not summary_path.exists():
        return None
    return pd.read_csv(summary_path).assign(run_label="baseline_signed_lkj_eta_2")


def run(args: argparse.Namespace) -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = task_exp.load_features(args.feature_path)
    scored_dates, _ = task_exp.scored_and_live_dates(df, args.max_windows)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    final_universe = [asset for asset in task_exp.ETF_TICKERS if asset not in HELPER_ASSETS]
    write_manifest(args, output_dir, df, scored_dates)
    residual_history = None
    if args.noise_source.startswith("residual_"):
        residual_predictions_path = args.residual_predictions_path or args.baseline_run / "gp_window_predictions.csv"
        residual_history = load_residual_history(residual_predictions_path)

    return_rows: list[dict[str, Any]] = []
    weight_frames: dict[str, list[pd.Series]] = {
        "gp_scenarios_riskfolio": [],
        "historical_y_ewma2_riskfolio": [],
        "equal_weight": [],
    }
    prediction_rows: list[pd.DataFrame] = []
    ic_rows: list[dict[str, Any]] = []
    model_diag_rows: list[dict[str, Any]] = []
    task_diag_rows: list[dict[str, Any]] = []
    noise_diag_rows: list[pd.DataFrame] = []

    for window_index, window_date in enumerate(scored_dates):
        print(f"rebalance {window_date.date()}", flush=True)
        train_df = df[(df["date"] < window_date) & df[task_exp.TARGET_COL].notna()].copy()
        eval_df = df[df["date"] == window_date].copy()
        eval_returns = (
            eval_df.set_index(eval_df["asset_id"].astype(str))[task_exp.TARGET_COL]
            .reindex(final_universe)
            .astype(float)
        )
        scenarios, predictions, diagnostics, noise_diagnostics, task_diag = fit_gp_window(
            train_df,
            eval_df,
            args=args,
            seed=task_exp.stable_seed(args.seed, "heteroskedastic_fixed_noise", window_index),
            maxiter=args.maxiter,
            posterior_scenarios=args.posterior_scenarios,
            residual_history=residual_history,
        )
        prediction_rows.append(predictions)
        model_diag_rows.extend(diagnostics)
        noise_diag_rows.append(noise_diagnostics.assign(window_date=window_date.date().isoformat()))
        if task_diag is not None:
            task_diag_rows.append(task_diag)
        scenarios.loc[:, final_universe].to_csv(
            output_dir / f"gp_scenarios_{window_date.date().isoformat()}.csv",
            index=False,
        )

        gp_weights = portfolio_exp.optimize_riskfolio(
            scenarios.loc[:, final_universe],
            method_mu="hist",
            method_cov="hist",
            upperlng=args.upperlng,
            nea=args.nea,
        )
        hist_panel = train_df.pivot(index="date", columns="asset_id", values=task_exp.TARGET_COL).reindex(
            columns=final_universe
        )
        hist_weights = portfolio_exp.optimize_riskfolio(
            hist_panel,
            method_mu="ewma2",
            method_cov="ewma2",
            upperlng=args.upperlng,
            nea=args.nea,
        )
        ew_weights = portfolio_exp.equal_weight(final_universe)
        weights_by_strategy = {
            "gp_scenarios_riskfolio": gp_weights.reindex(final_universe).fillna(0.0),
            "historical_y_ewma2_riskfolio": hist_weights.reindex(final_universe).fillna(0.0),
            "equal_weight": ew_weights.reindex(final_universe).fillna(0.0),
        }
        gp_ic = portfolio_exp.information_coefficient(predictions, final_universe)
        ic_rows.append({"date": window_date.date().isoformat(), "strategy": "gp_scenarios_riskfolio", "ic": gp_ic})
        for strategy, weights in weights_by_strategy.items():
            weight_frames[strategy].append(pd.Series(weights, name=window_date))
            return_rows.append(
                {
                    "date": window_date.date().isoformat(),
                    "strategy": strategy,
                    "return": portfolio_exp.realized_return(weights, eval_returns),
                    "gp_ic": gp_ic if strategy == "gp_scenarios_riskfolio" else math.nan,
                }
            )

    returns_df = pd.DataFrame(return_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    ic_df = pd.DataFrame(ic_rows)
    noise_df = pd.concat(noise_diag_rows, ignore_index=True)
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
        summary = {"strategy": strategy, **performance_stats(strategy_ret, weights)}
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
    noise_df.to_csv(output_dir / "noise_model_diagnostics.csv", index=False)
    if model_diag_rows:
        pd.DataFrame(model_diag_rows).to_csv(output_dir / "model_diagnostics.csv", index=False)
    if task_diag_rows:
        pd.DataFrame(task_diag_rows).to_csv(output_dir / "task_covariance_diagnostics.csv", index=False)
    write_comparison_plots(strategy_returns, noise_df, predictions_df, output_dir)

    baseline_summary = load_baseline_summary(args.baseline_run)
    report = [
        "# Heteroskedastic Noise Plug-In Walk-Forward",
        "",
        f"Run directory: `{output_dir}`",
        f"Baseline run: `{args.baseline_run}`",
        (
            f"Rebalances: `{len(scored_dates)}` monthly windows from "
            f"`{scored_dates[0].date()}` to `{scored_dates[-1].date()}`."
        ),
        "",
        "## Candidate",
        "",
        "- Latent model: existing signed LKJ eta=2 multitask GP with time/ETF/macro covariance.",
        "- Observation noise: fixed per-observation raw-return variance from a rolling asset/class noise model.",
        "- Scenario semantics: latent multitask posterior draw plus independent predicted observation noise.",
        "",
        "## Strategy Summary",
        "",
        portfolio_exp.markdown_table(summary_df),
        "",
    ]
    if baseline_summary is not None:
        baseline_gp = baseline_summary[baseline_summary["strategy"] == "gp_scenarios_riskfolio"]
        candidate_gp = summary_df[summary_df["strategy"] == "gp_scenarios_riskfolio"]
        comparison = pd.concat(
            [
                baseline_gp.assign(run_label="baseline_signed_lkj_eta_2"),
                candidate_gp.assign(run_label="heteroskedastic_fixed_noise"),
            ],
            ignore_index=True,
        )
        report.extend(["## Baseline Comparison", "", portfolio_exp.markdown_table(comparison), ""])
    report.extend(
        [
            "## Visuals",
            "",
            "- `equity_curve.png` and `drawdown_curve.png`: portfolio path diagnostics.",
            "- `eval_noise_std_by_group.png`: predicted next-window noise by asset group.",
            "- `noise_std_vs_abs_error.png`: whether larger predicted noise tracks larger absolute forecast errors.",
            "",
            "## Caveats",
            "",
            "- This is a first plug-in noise model, not a fully joint heteroskedastic GP.",
            "- Training noise uses trailing returns only; it does not yet use out-of-sample GP residuals.",
            "- Fixed noise treats noise estimates as known, so uncertainty in the noise model is not propagated.",
            "- No transaction costs, taxes, slippage, or liquidity filters are applied.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")

    print(summary_df.to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    run(parse_args())
