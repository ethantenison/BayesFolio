"""Tune a single-output XGBoost model and compute SHAP diagnostics.

The model predicts `y_excess_lead` for the international equity ETF family as
one pooled regression problem. `asset_id` is kept as a native XGBoost
categorical feature so the model can learn ETF-specific effects without using a
multitask GP.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from optuna.samplers import NSGAIISampler
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting fallback
    plt = None


EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = EXPERIMENT_DIR / "artifacts/international_equity_custom_feature_candidates_20260707T182133Z.parquet"
RUNS_DIR = EXPERIMENT_DIR / "runs"
TARGET_COL = "y_excess_lead"
ASSET_COL = "asset_id"
DATE_COL = "date"
TIME_INDEX_COL = "time_idx"
SEED = 27

ETF_MOMENTUM_COLS = ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"]
MACRO_MARKET_COLS = ["vix", "vix_slope", "spy_ret", "pct_above_50dma"]
MACRO_BREADTH_FLOW_COLS = ["spy_ret", "erp", "rsp_spy_roc_1p", "pct_above_50dma", "spy_flow_z_12p"]
INTERNATIONAL_MACRO_COLS = [
    "vix",
    "vix3m",
    "vix_slope",
    "vix_ts_level",
    "vix_ts_chg_1p",
    "vix_ts_z_12p",
    "skew_proxy",
    "move_proxy",
    "dealer_gamma_proxy",
    "dxy",
    "em_fx",
    "em_fx_ret",
    "spy_ret",
    "erp",
    "rsp_spy",
    "rsp_spy_roc_1p",
    "pct_above_50dma",
    "spy_flow_z_12p",
    "tnote10y",
    "tbill3m",
    "term_spread",
    "y10_nominal",
    "schp_ret",
    "y10_real_proxy",
    "breakeven_proxy",
    "de10y",
    "jp10y",
    "uk10y",
    "cn10y",
    "credit_spread",
    "credit_spread_chg_1p",
    "hy_spread",
    "hy_spread_chg_1p",
    "hy_spread_z_12p",
    "cpi_chg_12p",
    "cpi_chg_1p",
    "oil",
    "copper",
    "gold",
    "oil_ret",
    "copper_ret",
    "gold_crude_ratio",
]


@dataclass(frozen=True)
class TimeFold:
    fold_id: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    n_train_dates: int
    n_valid_dates: int
    n_train_rows: int
    n_valid_rows: int


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-trials", type=int, default=400)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--min-train-dates", type=int, default=36)
    parser.add_argument("--valid-window-dates", type=int, default=4)
    parser.add_argument("--step-dates", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    parser.add_argument("--shap-pareto-limit", type=int, default=12)
    parser.add_argument("--shap-sample-size", type=int, default=256)
    parser.add_argument("--smoke-trials", type=int, default=None)
    parser.add_argument("--fixed-colsample-bylevel", type=float, default=None)
    parser.add_argument("--fixed-colsample-bynode", type=float, default=None)
    parser.add_argument("--l2-only", action="store_true")
    parser.add_argument("--learning-rate-upper", type=float, default=0.25)
    parser.add_argument("--gamma-upper", type=float, default=1.0)
    parser.add_argument("--min-child-weight-min", type=float, default=0.01)
    parser.add_argument("--min-child-weight-max", type=float, default=10.0)
    parser.add_argument("--min-child-weight-step", type=int, default=None)
    parser.add_argument("--fixed-max-delta-step", type=float, default=None)
    parser.add_argument("--max-cat-to-onehot-choices", type=int, nargs="+", default=None)
    parser.add_argument("--fixed-max-cat-threshold", type=int, default=None)
    parser.add_argument(
        "--include-time-index",
        action="store_true",
        help="Add a zero-based ordinal date index as a numeric model feature.",
    )
    return parser.parse_args()


def git_metadata() -> dict[str, str]:
    def run(command: list[str]) -> str:
        result = subprocess.run(command, cwd=EXPERIMENT_DIR.parents[1], check=False, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()

    return {
        "git_sha": run(["git", "rev-parse", "HEAD"]),
        "git_dirty_summary": run(["git", "status", "--short"]),
    }


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module in [np, pd, optuna, shap, xgb]:
        versions[module.__name__] = getattr(module, "__version__", "unknown")
    return versions


def load_dataset(*, include_time_index: bool) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = pd.read_parquet(ARTIFACT_PATH).sort_values([DATE_COL, ASSET_COL]).reset_index(drop=True)
    if include_time_index:
        sorted_dates = pd.Index(sorted(frame[DATE_COL].unique()))
        date_to_index = {date: index for index, date in enumerate(sorted_dates)}
        frame[TIME_INDEX_COL] = frame[DATE_COL].map(date_to_index).astype("float64")
    feature_cols = [
        ASSET_COL,
        *dedupe([*ETF_MOMENTUM_COLS, *MACRO_MARKET_COLS, *MACRO_BREADTH_FLOW_COLS, *INTERNATIONAL_MACRO_COLS]),
    ]
    if include_time_index:
        feature_cols.append(TIME_INDEX_COL)
    missing = [column for column in feature_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"Feature columns missing from artifact: {missing}")
    model_frame = frame[[DATE_COL, TARGET_COL, *feature_cols]].dropna(subset=[TARGET_COL]).copy()
    categories = sorted(model_frame[ASSET_COL].astype(str).unique())
    model_frame[ASSET_COL] = pd.Categorical(model_frame[ASSET_COL].astype(str), categories=categories)
    return model_frame, feature_cols, categories


def make_time_folds(
    frame: pd.DataFrame,
    *,
    min_train_dates: int,
    valid_window_dates: int,
    step_dates: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[TimeFold]]:
    dates = pd.Index(sorted(frame[DATE_COL].unique()))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    metadata: list[TimeFold] = []
    fold_id = 0
    start = min_train_dates
    while start + valid_window_dates <= len(dates):
        train_dates = dates[:start]
        valid_dates = dates[start : start + valid_window_dates]
        train_mask = frame[DATE_COL].isin(train_dates).to_numpy()
        valid_mask = frame[DATE_COL].isin(valid_dates).to_numpy()
        folds.append((train_mask, valid_mask))
        metadata.append(
            TimeFold(
                fold_id=fold_id,
                train_start=str(pd.Timestamp(train_dates[0]).date()),
                train_end=str(pd.Timestamp(train_dates[-1]).date()),
                valid_start=str(pd.Timestamp(valid_dates[0]).date()),
                valid_end=str(pd.Timestamp(valid_dates[-1]).date()),
                n_train_dates=len(train_dates),
                n_valid_dates=len(valid_dates),
                n_train_rows=int(train_mask.sum()),
                n_valid_rows=int(valid_mask.sum()),
            )
        )
        fold_id += 1
        start += step_dates
    if not folds:
        raise ValueError("No validation folds produced; adjust split settings")
    return folds, metadata


def suggest_params(
    trial: optuna.Trial,
    *,
    n_estimators: int,
    xgb_n_jobs: int,
    fixed_colsample_bylevel: float | None,
    fixed_colsample_bynode: float | None,
    l2_only: bool,
    learning_rate_upper: float,
    gamma_upper: float,
    min_child_weight_min: float,
    min_child_weight_max: float,
    min_child_weight_step: int | None,
    fixed_max_delta_step: float | None,
    max_cat_to_onehot_choices: list[int] | None,
    fixed_max_cat_threshold: int | None,
) -> dict[str, Any]:
    reg_alpha = 0.0
    reg_lambda = 0.0
    if l2_only:
        reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True)
    else:
        reg_type = trial.suggest_categorical("reg_type", ["l1", "l2"])
        if reg_type == "l1":
            reg_alpha = trial.suggest_float("reg_alpha", 1e-8, 20.0, log=True)
        else:
            reg_lambda = trial.suggest_float("reg_lambda", 1e-8, 20.0, log=True)
    colsample_bylevel = (
        fixed_colsample_bylevel
        if fixed_colsample_bylevel is not None
        else trial.suggest_float("colsample_bylevel", 0.6, 1.0)
    )
    colsample_bynode = (
        fixed_colsample_bynode
        if fixed_colsample_bynode is not None
        else trial.suggest_float("colsample_bynode", 0.6, 1.0)
    )
    if min_child_weight_step is not None:
        min_child_weight = trial.suggest_int(
            "min_child_weight",
            int(min_child_weight_min),
            int(min_child_weight_max),
            step=min_child_weight_step,
        )
    else:
        min_child_weight = trial.suggest_float("min_child_weight", min_child_weight_min, min_child_weight_max, log=True)
    max_delta_step = (
        fixed_max_delta_step if fixed_max_delta_step is not None else trial.suggest_float("max_delta_step", 0.0, 2.0)
    )
    max_cat_threshold = (
        fixed_max_cat_threshold
        if fixed_max_cat_threshold is not None
        else trial.suggest_int("max_cat_threshold", 4, 64, log=True)
    )
    max_cat_to_onehot = (
        trial.suggest_categorical("max_cat_to_onehot", max_cat_to_onehot_choices)
        if max_cat_to_onehot_choices is not None
        else trial.suggest_int("max_cat_to_onehot", 1, 8)
    )

    return {
        "n_estimators": n_estimators,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": SEED,
        "n_jobs": xgb_n_jobs,
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, learning_rate_upper, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": colsample_bylevel,
        "colsample_bynode": colsample_bynode,
        "min_child_weight": min_child_weight,
        "gamma": trial.suggest_float("gamma", 1e-8, gamma_upper, log=True),
        "max_delta_step": max_delta_step,
        "max_bin": trial.suggest_int("max_bin", 64, 512, log=True),
        "max_cat_to_onehot": max_cat_to_onehot,
        "max_cat_threshold": max_cat_threshold,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
    }


def mean_date_ic(valid: pd.DataFrame) -> float:
    ics: list[float] = []
    for _, group in valid.groupby(DATE_COL, observed=True):
        if group[TARGET_COL].nunique(dropna=True) < 2 or group["prediction"].nunique(dropna=True) < 2:
            continue
        value = spearmanr(group[TARGET_COL], group["prediction"]).statistic
        if not math.isnan(value):
            ics.append(float(value))
    return float(np.mean(ics)) if ics else 0.0


def hit_direction(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def evaluate_params(
    params: dict[str, Any],
    *,
    frame: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, float]:
    fold_metrics: list[dict[str, float]] = []
    for train_mask, valid_mask in folds:
        train = frame.loc[train_mask]
        valid = frame.loc[valid_mask].copy()
        model = xgb.XGBRegressor(**params)
        model.fit(train[feature_cols], train[TARGET_COL], verbose=False)
        preds = model.predict(valid[feature_cols])
        valid["prediction"] = preds
        fold_metrics.append(
            {
                "rmse": float(mean_squared_error(valid[TARGET_COL], preds) ** 0.5),
                "ic": mean_date_ic(valid),
                "hit_direction": hit_direction(valid[TARGET_COL].to_numpy(), preds),
            }
        )
    metrics = pd.DataFrame(fold_metrics)
    return (
        float(metrics["rmse"].mean()),
        float(metrics["ic"].mean()),
        float(metrics["hit_direction"].mean()),
    )


def objective_factory(
    *,
    frame: pd.DataFrame,
    feature_cols: list[str],
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_estimators: int,
    xgb_n_jobs: int,
    fixed_colsample_bylevel: float | None,
    fixed_colsample_bynode: float | None,
    l2_only: bool,
    learning_rate_upper: float,
    gamma_upper: float,
    min_child_weight_min: float,
    min_child_weight_max: float,
    min_child_weight_step: int | None,
    fixed_max_delta_step: float | None,
    max_cat_to_onehot_choices: list[int] | None,
    fixed_max_cat_threshold: int | None,
) -> Any:
    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        params = suggest_params(
            trial,
            n_estimators=n_estimators,
            xgb_n_jobs=xgb_n_jobs,
            fixed_colsample_bylevel=fixed_colsample_bylevel,
            fixed_colsample_bynode=fixed_colsample_bynode,
            l2_only=l2_only,
            learning_rate_upper=learning_rate_upper,
            gamma_upper=gamma_upper,
            min_child_weight_min=min_child_weight_min,
            min_child_weight_max=min_child_weight_max,
            min_child_weight_step=min_child_weight_step,
            fixed_max_delta_step=fixed_max_delta_step,
            max_cat_to_onehot_choices=max_cat_to_onehot_choices,
            fixed_max_cat_threshold=fixed_max_cat_threshold,
        )
        rmse, ic, hit = evaluate_params(params, frame=frame, feature_cols=feature_cols, folds=folds)
        trial.set_user_attr("rmse", rmse)
        trial.set_user_attr("mean_ic", ic)
        trial.set_user_attr("hit_direction", hit)
        trial.set_user_attr("params_resolved", params)
        return rmse, ic, hit

    return objective


def trials_frame(study: optuna.Study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pareto_numbers = {trial.number for trial in study.best_trials}
    for trial in study.trials:
        row: dict[str, Any] = {
            "trial_number": trial.number,
            "state": str(trial.state),
            "is_pareto": trial.number in pareto_numbers,
        }
        if trial.values is not None:
            row.update({"rmse": trial.values[0], "mean_ic": trial.values[1], "hit_direction": trial.values[2]})
        row.update(trial.params)
        resolved = trial.user_attrs.get("params_resolved")
        if isinstance(resolved, dict):
            row.update(
                {f"resolved_{key}": value for key, value in resolved.items() if key in {"reg_alpha", "reg_lambda"}}
            )
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_frame(study: optuna.Study) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.best_trials:
        row = {
            "trial_number": trial.number,
            "rmse": trial.values[0],
            "mean_ic": trial.values[1],
            "hit_direction": trial.values[2],
        }
        resolved = trial.user_attrs.get("params_resolved", {})
        row.update(resolved)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["rmse_rank"] = frame["rmse"].rank(method="min", ascending=True)
    frame["ic_rank"] = frame["mean_ic"].rank(method="min", ascending=False)
    frame["hit_rank"] = frame["hit_direction"].rank(method="min", ascending=False)
    frame["balanced_rank"] = frame[["rmse_rank", "ic_rank", "hit_rank"]].mean(axis=1)
    return frame.sort_values(["balanced_rank", "rmse"], ascending=[True, True]).reset_index(drop=True)


def fit_full_model(params: dict[str, Any], frame: pd.DataFrame, feature_cols: list[str]) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**params)
    model.fit(frame[feature_cols], frame[TARGET_COL], verbose=False)
    return model


def compute_shap_outputs(
    *,
    run_root: Path,
    pareto: pd.DataFrame,
    frame: pd.DataFrame,
    feature_cols: list[str],
    sample_size: int,
    shap_pareto_limit: int,
) -> None:
    shap_dir = run_root / "artifacts" / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    if pareto.empty:
        return
    rng = np.random.default_rng(SEED)
    if len(frame) > sample_size:
        sample_idx = np.sort(rng.choice(np.arange(len(frame)), size=sample_size, replace=False))
        shap_frame = frame.iloc[sample_idx].copy()
    else:
        shap_frame = frame.copy()

    selected = pareto.head(shap_pareto_limit).copy()
    main_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    param_keys = [
        "n_estimators",
        "objective",
        "eval_metric",
        "tree_method",
        "enable_categorical",
        "random_state",
        "n_jobs",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "colsample_bylevel",
        "colsample_bynode",
        "min_child_weight",
        "gamma",
        "max_delta_step",
        "max_bin",
        "max_cat_to_onehot",
        "max_cat_threshold",
        "reg_alpha",
        "reg_lambda",
    ]
    for solution_rank, solution in selected.iterrows():
        params = {key: solution[key] for key in param_keys if key in solution and pd.notna(solution[key])}
        for int_key in [
            "n_estimators",
            "random_state",
            "n_jobs",
            "max_depth",
            "max_bin",
            "max_cat_to_onehot",
            "max_cat_threshold",
        ]:
            if int_key in params:
                params[int_key] = int(params[int_key])
        params["enable_categorical"] = bool(params["enable_categorical"])
        model = fit_full_model(params, frame, feature_cols)
        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(explainer.shap_values(shap_frame[feature_cols]))
        main_importance = np.abs(shap_values).mean(axis=0)
        for feature, importance in zip(feature_cols, main_importance, strict=True):
            main_rows.append(
                {
                    "solution_rank": int(solution_rank),
                    "trial_number": int(solution["trial_number"]),
                    "feature": feature,
                    "mean_abs_shap": float(importance),
                }
            )
        interactions = np.asarray(explainer.shap_interaction_values(shap_frame[feature_cols]))
        interaction_importance = np.abs(interactions).mean(axis=0)
        for i, left in enumerate(feature_cols):
            for j in range(i + 1, len(feature_cols)):
                interaction_rows.append(
                    {
                        "solution_rank": int(solution_rank),
                        "trial_number": int(solution["trial_number"]),
                        "feature_a": left,
                        "feature_b": feature_cols[j],
                        "mean_abs_interaction": float(interaction_importance[i, j]),
                    }
                )

    main = pd.DataFrame(main_rows)
    interactions = pd.DataFrame(interaction_rows)
    main.to_csv(shap_dir / "shap_main_importance_by_pareto_solution.csv", index=False)
    interactions.to_csv(shap_dir / "shap_interaction_importance_by_pareto_solution.csv", index=False)
    main_agg = (
        main.groupby("feature", as_index=False)["mean_abs_shap"]
        .agg(["mean", "std", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    interaction_agg = (
        interactions.groupby(["feature_a", "feature_b"], as_index=False)["mean_abs_interaction"]
        .agg(["mean", "std", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    main_agg.to_csv(shap_dir / "shap_main_importance_aggregate.csv", index=False)
    interaction_agg.to_csv(shap_dir / "shap_interaction_importance_aggregate.csv", index=False)
    write_shap_plots(shap_dir, main_agg, interaction_agg)


def write_shap_plots(shap_dir: Path, main: pd.DataFrame, interactions: pd.DataFrame) -> None:
    if plt is None:
        return
    top_main = main.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_main["feature"], top_main["mean"])
    ax.set_title("Top Mean Absolute SHAP Features Across Pareto Models")
    ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout()
    fig.savefig(shap_dir / "top_shap_main_importance.png", dpi=160)
    plt.close(fig)

    top_inter = interactions.head(20).copy()
    top_inter["pair"] = top_inter["feature_a"] + " x " + top_inter["feature_b"]
    top_inter = top_inter.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_inter["pair"], top_inter["mean"])
    ax.set_title("Top Mean Absolute SHAP Interactions Across Pareto Models")
    ax.set_xlabel("Mean absolute SHAP interaction value")
    fig.tight_layout()
    fig.savefig(shap_dir / "top_shap_interaction_importance.png", dpi=160)
    plt.close(fig)


def write_report(run_root: Path, *, pareto: pd.DataFrame, trials: pd.DataFrame) -> None:
    shap_dir = run_root / "artifacts" / "shap"
    main_path = shap_dir / "shap_main_importance_aggregate.csv"
    interaction_path = shap_dir / "shap_interaction_importance_aggregate.csv"
    lines = [
        "# XGBoost SHAP Optuna Screen",
        "",
        f"Run root: `{run_root}`",
        "",
        f"Trials complete: `{(trials['state'] == 'TrialState.COMPLETE').sum()}` / `{len(trials)}`",
        f"Pareto solutions: `{len(pareto)}`",
        "",
        "Objectives: minimize RMSE, maximize date-level Spearman IC, maximize hit direction.",
        "",
    ]
    if not pareto.empty:
        display_cols = [
            "trial_number",
            "rmse",
            "mean_ic",
            "hit_direction",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "colsample_bylevel",
            "colsample_bynode",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        ]
        lines.extend(
            [
                "## Pareto Hyperparameters",
                "",
                "```text",
                pareto[display_cols].head(30).to_string(index=False),
                "```",
                "",
            ]
        )
    if main_path.exists():
        main = pd.read_csv(main_path)
        lines.extend(["## Top SHAP Main Effects", "", "```text", main.head(20).to_string(index=False), "```", ""])
    if interaction_path.exists():
        interactions = pd.read_csv(interaction_path)
        lines.extend(
            ["## Top SHAP Interactions", "", "```text", interactions.head(20).to_string(index=False), "```", ""]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            "- `manifest.json`",
            "- `resolved_config.json`",
            "- `study_trials.csv`",
            "- `pareto_solutions.csv`",
            "- `folds.csv`",
            "- `artifacts/shap/shap_main_importance_aggregate.csv`",
            "- `artifacts/shap/shap_interaction_importance_aggregate.csv`",
        ]
    )
    (run_root / "REPORT.md").write_text("\n".join(lines) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def main() -> None:
    args = parse_args()
    if args.smoke_trials is not None:
        args.n_trials = args.smoke_trials
    run_id = args.run_id or f"xgboost_shap_optuna_{datetime.now(UTC).strftime('%Y%m%dT%H%MZ')}"
    run_root = RUNS_DIR / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "artifacts").mkdir(parents=True, exist_ok=True)

    frame, feature_cols, categories = load_dataset(include_time_index=args.include_time_index)
    folds, fold_metadata = make_time_folds(
        frame,
        min_train_dates=args.min_train_dates,
        valid_window_dates=args.valid_window_dates,
        step_dates=args.step_dates,
    )
    pd.DataFrame([asdict(fold) for fold in fold_metadata]).to_csv(run_root / "folds.csv", index=False)
    resolved_config = {
        "run_id": run_id,
        "artifact_path": str(ARTIFACT_PATH),
        "target_col": TARGET_COL,
        "feature_cols": feature_cols,
        "include_time_index": args.include_time_index,
        "fixed_hyperparameters": {
            "colsample_bylevel": args.fixed_colsample_bylevel,
            "colsample_bynode": args.fixed_colsample_bynode,
            "max_delta_step": args.fixed_max_delta_step,
            "max_cat_threshold": args.fixed_max_cat_threshold,
            "reg_alpha": 0.0 if args.l2_only else None,
        },
        "time_index": {
            "column": TIME_INDEX_COL if args.include_time_index else None,
            "definition": "zero-based ordinal over sorted unique date values" if args.include_time_index else None,
        },
        "asset_categories": categories,
        "n_trials": args.n_trials,
        "n_estimators_fixed": args.n_estimators,
        "time_split": {
            "min_train_dates": args.min_train_dates,
            "valid_window_dates": args.valid_window_dates,
            "step_dates": args.step_dates,
            "n_folds": len(folds),
        },
        "objectives": ["rmse_minimize", "mean_date_spearman_ic_maximize", "hit_direction_maximize"],
        "search_space": {
            "max_depth": [2, 12],
            "learning_rate": [0.005, args.learning_rate_upper, "log"],
            "subsample": [0.6, 1.0],
            "colsample_bytree": [0.6, 1.0],
            "colsample_bylevel": args.fixed_colsample_bylevel
            if args.fixed_colsample_bylevel is not None
            else [0.6, 1.0],
            "colsample_bynode": args.fixed_colsample_bynode if args.fixed_colsample_bynode is not None else [0.6, 1.0],
            "min_child_weight": [
                args.min_child_weight_min,
                args.min_child_weight_max,
                f"int_step_{args.min_child_weight_step}",
            ]
            if args.min_child_weight_step is not None
            else [args.min_child_weight_min, args.min_child_weight_max, "log"],
            "gamma": [1e-8, args.gamma_upper, "log"],
            "max_delta_step": args.fixed_max_delta_step if args.fixed_max_delta_step is not None else [0.0, 2.0],
            "max_bin": [64, 512, "log_int"],
            "max_cat_to_onehot": args.max_cat_to_onehot_choices
            if args.max_cat_to_onehot_choices is not None
            else [1, 8],
            "max_cat_threshold": args.fixed_max_cat_threshold
            if args.fixed_max_cat_threshold is not None
            else [4, 64, "log_int"],
            "regularization": "L2 only: tune reg_lambda and fix reg_alpha=0"
            if args.l2_only
            else "sample exactly one of L1 reg_alpha or L2 reg_lambda; the other is fixed to 0",
        },
    }
    write_json(run_root / "resolved_config.json", resolved_config)
    write_json(
        run_root / "manifest.json",
        {
            "schema": "bayesfolio.international_xgboost_shap_optuna.v1",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "running",
            "run_root": str(run_root),
            "git": git_metadata(),
            "packages": package_versions(),
            **resolved_config,
        },
    )

    sampler = NSGAIISampler(seed=SEED, population_size=64)
    study = optuna.create_study(directions=["minimize", "maximize", "maximize"], sampler=sampler)
    study.optimize(
        objective_factory(
            frame=frame,
            feature_cols=feature_cols,
            folds=folds,
            n_estimators=args.n_estimators,
            xgb_n_jobs=args.xgb_n_jobs,
            fixed_colsample_bylevel=args.fixed_colsample_bylevel,
            fixed_colsample_bynode=args.fixed_colsample_bynode,
            l2_only=args.l2_only,
            learning_rate_upper=args.learning_rate_upper,
            gamma_upper=args.gamma_upper,
            min_child_weight_min=args.min_child_weight_min,
            min_child_weight_max=args.min_child_weight_max,
            min_child_weight_step=args.min_child_weight_step,
            fixed_max_delta_step=args.fixed_max_delta_step,
            max_cat_to_onehot_choices=args.max_cat_to_onehot_choices,
            fixed_max_cat_threshold=args.fixed_max_cat_threshold,
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )
    trials = trials_frame(study)
    pareto = pareto_frame(study)
    trials.to_csv(run_root / "study_trials.csv", index=False)
    pareto.to_csv(run_root / "pareto_solutions.csv", index=False)
    compute_shap_outputs(
        run_root=run_root,
        pareto=pareto,
        frame=frame,
        feature_cols=feature_cols,
        sample_size=args.shap_sample_size,
        shap_pareto_limit=args.shap_pareto_limit,
    )
    write_report(run_root, pareto=pareto, trials=trials)
    manifest = json.loads((run_root / "manifest.json").read_text())
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["trial_count"] = len(trials)
    manifest["pareto_count"] = len(pareto)
    manifest["report_path"] = str(run_root / "REPORT.md")
    write_json(run_root / "manifest.json", manifest)
    print(f"Wrote XGBoost SHAP Optuna run to {run_root}", flush=True)


if __name__ == "__main__":
    main()
