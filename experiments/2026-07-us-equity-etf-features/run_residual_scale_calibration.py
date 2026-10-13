"""Learn a residual-scale calibrator for U.S. equity MTGP forecasts.

This is an experiment-scoped probe. It keeps the `momentum_plus_macro` MTGP
mean forecast and learns a simple walk-forward residual-scale model from
liquidity/volatility features. The intent is to test heteroskedastic forecast
calibration before wiring anything into portfolio construction.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import warnings
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
from gpytorch.mlls import ExactMarginalLogLikelihood
from linear_operator.utils.warnings import NumericalWarning

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
WRAPPER_PATH = EXPERIMENT_DIR / "run_first_round_walkforward.py"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / "runs"
DEFAULT_RUN_ID = "residual_scale_calibration_20260705T1731Z"

ASSETS = ["SPY", "MGK", "VTV", "IWM"]
OUTER_DATES = [
    "2026-01-30",
    "2026-02-27",
    "2026-03-31",
    "2026-04-30",
    "2026-05-29",
    "2026-06-30",
]
LIQUIDITY_COLS = ["ill_log", "dolvol_log", "turnover"]
VOLATILITY_COLS = ["vol_1m", "vol_z", "vol_accel", "vol_ratio_1m_3m", "max_dd_3m"]
TARGET_COL = "y_excess_lead"
NORMAL_COVERAGE_Z = {
    "coverage_50": 0.6744897501960817,
    "coverage_80": 1.2815515655446004,
    "coverage_95": 1.959963984540054,
}


@dataclass(frozen=True)
class CalibratorSpec:
    name: str
    continuous_cols: tuple[str, ...]
    include_baseline_log_var: bool = False


CALIBRATORS = [
    CalibratorSpec("asset_only", ()),
    CalibratorSpec("liquidity_ridge", tuple(LIQUIDITY_COLS)),
    CalibratorSpec("volatility_ridge", tuple(VOLATILITY_COLS)),
    CalibratorSpec("liquidity_volatility_ridge", tuple([*LIQUIDITY_COLS, *VOLATILITY_COLS])),
    CalibratorSpec(
        "liquidity_volatility_plus_baseline_ridge",
        tuple([*LIQUIDITY_COLS, *VOLATILITY_COLS]),
        include_baseline_log_var=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-train-dates", type=int, default=24)
    parser.add_argument("--maxiter", type=int, default=15)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--ridge-alpha", type=float, default=2.0)
    parser.add_argument("--min-aleatoric-var", type=float, default=1e-6)
    parser.add_argument("--max-aleatoric-var", type=float, default=0.025)
    return parser.parse_args()


def load_wrapper() -> Any:
    spec = importlib.util.spec_from_file_location("us_equity_first_round_wrapper", WRAPPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load wrapper from {WRAPPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def git_dirty_summary() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip()


def load_feature_panel(wrapper: Any, runner: Any) -> pd.DataFrame:
    all_cols = list(
        dict.fromkeys(
            [
                *wrapper.TIME_COLS,
                *wrapper.ETF_COLS,
                *wrapper.MACRO_COLS,
                "date",
                "asset_id",
                runner.task_exp.TARGET_COL,
            ]
        )
    )
    df = pd.read_parquet(wrapper.DEFAULT_FEATURE_PATH).loc[:, all_cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["asset_id"] = pd.Categorical(df["asset_id"], categories=ASSETS, ordered=True)
    df = df[df["asset_id"].notna()].sort_values(["date", "asset_id"]).reset_index(drop=True)
    return df


def full_target_dates(df: pd.DataFrame) -> list[pd.Timestamp]:
    counts = df.groupby("date", observed=True)[TARGET_COL].apply(lambda s: int(s.notna().sum()))
    return list(counts[counts == len(ASSETS)].index)


def fit_momentum_macro_panel(
    df: pd.DataFrame,
    scored_dates: list[pd.Timestamp],
    *,
    runner: Any,
    maxiter: int,
    seed: int,
    seed_indices: dict[pd.Timestamp, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variant = runner.task_exp.VARIANTS["signed_lkj_eta_2"]
    for window_idx, date in enumerate(scored_dates):
        train_df = df[(df["date"] < date) & df[TARGET_COL].notna()].copy()
        eval_df = df[df["date"].eq(date)].copy()
        train_x, train_y, eval_x, _, _, _ = runner.task_exp.prepare_window_tensors(train_df, eval_df)
        stable_seed = runner.task_exp.stable_seed(seed, "signed_lkj_eta_2", seed_indices.get(date, window_idx))
        torch.manual_seed(stable_seed)
        model = runner.build_model_with_transform_mode(
            train_x,
            train_y,
            None,
            variant,
            input_transform_mode="botorch_normalize",
            time_modulation_mode="lengthscale_only",
            kernel_composition_proposal="none",
            mean_kind="multitask_constant",
        )
        model.train()
        model.likelihood.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": maxiter}})
        model.eval()
        model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent = model.posterior(eval_x, observation_noise=False)
            observed = model.posterior(eval_x, observation_noise=True)
            mean = observed.mean.squeeze(-1).detach().cpu().numpy()
            latent_var = latent.variance.squeeze(-1).clamp_min(0.0).detach().cpu().numpy()
            observed_var = observed.variance.squeeze(-1).clamp_min(0.0).detach().cpu().numpy()
        for i, (_, row) in enumerate(eval_df.reset_index(drop=True).iterrows()):
            out = {
                "date": date.date().isoformat(),
                "seed": int(stable_seed),
                "asset_id": str(row["asset_id"]),
                "y_true": float(row[TARGET_COL]),
                "y_pred": float(mean[i]),
                "latent_gp_var": float(latent_var[i]),
                "baseline_total_var": float(observed_var[i]),
                "baseline_y_std": float(math.sqrt(max(observed_var[i], 1e-12))),
            }
            for col in [*LIQUIDITY_COLS, *VOLATILITY_COLS]:
                out[col] = float(row[col])
            rows.append(out)
    panel = pd.DataFrame(rows)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["residual"] = panel["y_true"] - panel["y_pred"]
    panel["residual_sq"] = np.square(panel["residual"])
    panel["aleatoric_target_var"] = np.clip(panel["residual_sq"] - panel["latent_gp_var"], 1e-8, None)
    panel["baseline_total_std"] = np.sqrt(np.clip(panel["baseline_total_var"], 1e-12, None))
    return panel


def design_matrix(
    frame: pd.DataFrame,
    spec: CalibratorSpec,
    *,
    train_frame: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = [np.ones(len(frame))]
    names = ["intercept"]
    asset_values = frame["asset_id"].astype(str)
    for asset in ASSETS:
        columns.append((asset_values == asset).astype(float).to_numpy())
        names.append(f"asset_{asset}")
    for col in spec.continuous_cols:
        mean = float(train_frame[col].mean())
        std = float(train_frame[col].std(ddof=1))
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0
        columns.append((frame[col].to_numpy(dtype=float) - mean) / std)
        names.append(col)
    if spec.include_baseline_log_var:
        values = np.log(np.clip(frame["baseline_total_var"].to_numpy(dtype=float), 1e-12, None))
        train_values = np.log(np.clip(train_frame["baseline_total_var"].to_numpy(dtype=float), 1e-12, None))
        mean = float(np.mean(train_values))
        std = float(np.std(train_values, ddof=1))
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0
        columns.append((values - mean) / std)
        names.append("log_baseline_total_var")
    return np.column_stack(columns), names


def ridge_fit_predict(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    spec: CalibratorSpec,
    *,
    alpha: float,
    min_var: float,
    max_var: float,
) -> tuple[np.ndarray, dict[str, float]]:
    x_train, names = design_matrix(train_frame, spec, train_frame=train_frame)
    x_eval, _ = design_matrix(eval_frame, spec, train_frame=train_frame)
    target = np.log(np.clip(train_frame["aleatoric_target_var"].to_numpy(dtype=float), min_var, max_var))
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ target)
    pred = np.exp(x_eval @ coef)
    pred = np.clip(pred, min_var, max_var)
    diagnostics = {f"coef_{name}": float(value) for name, value in zip(names, coef, strict=True)}
    return pred, diagnostics


def metric_rows(predictions: pd.DataFrame, *, group_col: str | None = None) -> list[dict[str, Any]]:
    grouped = [("overall", predictions)] if group_col is None else predictions.groupby(group_col, observed=True)
    rows: list[dict[str, Any]] = []
    for label, frame in grouped:
        y_true = frame["y_true"].to_numpy(dtype=float)
        y_pred = frame["y_pred"].to_numpy(dtype=float)
        y_std = np.clip(frame["y_std"].to_numpy(dtype=float), math.sqrt(1e-12), None)
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


def evaluate_calibrators(
    panel: pd.DataFrame,
    *,
    outer_dates: list[pd.Timestamp],
    ridge_alpha: float,
    min_var: float,
    max_var: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    outer_set = {date.normalize() for date in outer_dates}
    for outer_date in outer_dates:
        train_frame = panel[panel["date"] < outer_date].copy()
        eval_frame = panel[panel["date"].dt.normalize().eq(outer_date.normalize())].copy()
        baseline = eval_frame.copy()
        baseline["candidate"] = "baseline_momentum_plus_macro"
        baseline["calibrated_aleatoric_var"] = np.clip(
            baseline["baseline_total_var"] - baseline["latent_gp_var"], min_var, max_var
        )
        baseline["y_std"] = baseline["baseline_total_std"]
        prediction_frames.append(baseline)
        for spec in CALIBRATORS:
            pred_noise_var, diagnostics = ridge_fit_predict(
                train_frame,
                eval_frame,
                spec,
                alpha=ridge_alpha,
                min_var=min_var,
                max_var=max_var,
            )
            calibrated = eval_frame.copy()
            calibrated["candidate"] = spec.name
            calibrated["calibrated_aleatoric_var"] = pred_noise_var
            calibrated["y_std"] = np.sqrt(np.clip(calibrated["latent_gp_var"] + pred_noise_var, 1e-12, None))
            prediction_frames.append(calibrated)
            diagnostic_rows.append(
                {
                    "outer_date": outer_date.date().isoformat(),
                    "candidate": spec.name,
                    "n_calibration_rows": int(len(train_frame)),
                    "calibrated_noise_var_mean": float(np.mean(pred_noise_var)),
                    "calibrated_noise_var_min": float(np.min(pred_noise_var)),
                    "calibrated_noise_var_max": float(np.max(pred_noise_var)),
                    **diagnostics,
                }
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions[predictions["date"].dt.normalize().isin(outer_set)].copy()
    overall_frames = []
    by_asset_frames = []
    for candidate, frame in predictions.groupby("candidate", observed=True):
        overall = pd.DataFrame(metric_rows(frame, group_col=None))
        overall.insert(0, "candidate", candidate)
        overall_frames.append(overall)
        by_asset = pd.DataFrame(metric_rows(frame, group_col="asset_id"))
        by_asset.insert(0, "candidate", candidate)
        by_asset_frames.append(by_asset)
    overall_metrics = pd.concat(overall_frames, ignore_index=True).sort_values("mean_gaussian_nll")
    by_asset_metrics = pd.concat(by_asset_frames, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    return predictions, overall_metrics, by_asset_metrics, diagnostics


def write_plots(output_dir: Path, metrics: pd.DataFrame) -> None:
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot = metrics.sort_values("mean_gaussian_nll")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(plot["candidate"], plot["mean_gaussian_nll"], color="#2563eb")
    ax.invert_yaxis()
    ax.set_xlabel("Mean Gaussian NLL (lower is better)")
    ax.set_title("Learned residual-scale calibrators vs momentum+macro baseline")
    for i, value in enumerate(plot["mean_gaussian_nll"]):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "learned_scale_nll_leaderboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=NumericalWarning)
    output_dir = args.output_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = load_wrapper()
    runner = wrapper.load_portfolio_runner()
    wrapper.patch_first_round_constants(runner, mtgp_rank=2, feature_set="momentum_plus_macro")
    df = load_feature_panel(wrapper, runner)
    all_dates = full_target_dates(df)
    outer_dates = [pd.Timestamp(date) for date in OUTER_DATES]
    first_outer = min(outer_dates)
    scored_dates = [
        date
        for date in all_dates
        if date <= max(outer_dates)
        and df[(df["date"] < date) & df[TARGET_COL].notna()]["date"].nunique() >= args.min_train_dates
    ]
    if not all(date in scored_dates for date in outer_dates):
        raise RuntimeError("Outer dates missing from generated scored date grid")
    calibration_dates = [date for date in scored_dates if date < first_outer]
    outer_seed_indices = {date: idx for idx, date in enumerate(outer_dates)}
    calibration_seed_indices = {date: idx for idx, date in enumerate(calibration_dates)}
    seed_indices = {**calibration_seed_indices, **outer_seed_indices}

    panel = fit_momentum_macro_panel(
        df,
        scored_dates,
        runner=runner,
        maxiter=args.maxiter,
        seed=args.seed,
        seed_indices=seed_indices,
    )
    panel.to_csv(output_dir / "momentum_plus_macro_oos_residual_panel.csv", index=False)

    predictions, overall, by_asset, diagnostics = evaluate_calibrators(
        panel,
        outer_dates=outer_dates,
        ridge_alpha=args.ridge_alpha,
        min_var=args.min_aleatoric_var,
        max_var=args.max_aleatoric_var,
    )
    predictions.to_csv(output_dir / "outer_calibrated_predictions.csv", index=False)
    overall.to_csv(output_dir / "forecast_metrics_overall.csv", index=False)
    by_asset.to_csv(output_dir / "forecast_metrics_by_asset.csv", index=False)
    diagnostics.to_csv(output_dir / "calibrator_diagnostics_by_outer_date.csv", index=False)
    write_plots(output_dir, overall)

    manifest = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "experiment_type": "learned_residual_scale_calibration",
        "feature_path": str(wrapper.DEFAULT_FEATURE_PATH),
        "mean_model_feature_set": "momentum_plus_macro",
        "outer_dates": [date.date().isoformat() for date in outer_dates],
        "calibration_dates": [date.date().isoformat() for date in calibration_dates],
        "n_calibration_dates": len(calibration_dates),
        "min_train_dates": args.min_train_dates,
        "maxiter": args.maxiter,
        "seed": args.seed,
        "seed_policy": (
            "task_exp.stable_seed(seed, signed_lkj_eta_2, local_window_index); "
            "outer dates use the same local 0..5 indices as Stage A"
        ),
        "ridge_alpha": args.ridge_alpha,
        "min_aleatoric_var": args.min_aleatoric_var,
        "max_aleatoric_var": args.max_aleatoric_var,
        "calibrators": [spec.__dict__ for spec in CALIBRATORS],
        "metrics": {
            "overall": "forecast_metrics_overall.csv",
            "by_asset": "forecast_metrics_by_asset.csv",
        },
        "plots": ["figures/learned_scale_nll_leaderboard.png"],
        "caveat": (
            "Forecast-only residual-scale calibration. The MTGP mean is refit for each scored date. "
            "Residual-scale calibrators are fit only on OOS residual rows before each outer date."
        ),
        "git_sha": git_sha(),
        "git_dirty_summary_at_start": git_dirty_summary(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(output_dir)
    metric_cols = ["candidate", "mean_gaussian_nll", "rmse", "coverage_80", "coverage_95", "mean_pred_std"]
    print(overall[metric_cols].to_string(index=False))


if __name__ == "__main__":
    main()
