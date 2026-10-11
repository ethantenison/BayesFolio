"""Compare three monthly return targets with one accepted multitask-GP design.

The experiment evaluates adjusted total excess returns, price-only excess
returns, and separately modeled price/distribution components. Financial
returns are decimal values (``0.02`` means 2%). Generated evidence includes
prediction quality, posterior uncertainty calibration, and total-return
portfolio outcomes.

Usage:
    uv run --locked python \
      experiments/2026-10-return-target-comparison/run_return_target_comparison.py \
      --output-dir experiments/2026-10-return-target-comparison/evidence/20261011_three_target_comparison_final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import botorch
import gpytorch
import numpy as np
import pandas as pd
import riskfolio
import torch
import yfinance as yf
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import StratifiedStandardize
from gpytorch.mlls import ExactMarginalLogLikelihood

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bayesfolio.contracts.commands.backtest import WalkForwardBacktestCommand, WalkForwardPeriod  # noqa: E402
from bayesfolio.contracts.commands.monthly_portfolio import MonthlyForecastConfig  # noqa: E402
from bayesfolio.contracts.commands.optimize import OptimizeCommand  # noqa: E402
from bayesfolio.engine.asset_allocation.riskfolio_adapter import optimize_from_historical_returns  # noqa: E402
from bayesfolio.engine.automation.monthly_gp_defaults import build_default_monthly_gp_configs  # noqa: E402
from bayesfolio.engine.backtest.evaluate_asset_pricing import evaluate_asset_pricing  # noqa: E402
from bayesfolio.engine.backtest.runner import run_walk_forward_backtest  # noqa: E402
from bayesfolio.engine.forecast.gp.multitask_builder import build_multitask_gp  # noqa: E402
from bayesfolio.engine.forecast.gp.time_varying_kernel import build_time_varying_kernel  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_PATH = Path("/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_06.parquet")
ASSETS = ["SPY", "IJR", "IEF", "VEA", "BNDX"]
ASSET_CLASS = {
    "SPY": "us_large_cap_equity",
    "IJR": "us_small_cap_equity",
    "IEF": "us_fixed_income",
    "VEA": "developed_international_equity",
    "BNDX": "developed_international_fixed_income",
}
TIME_FEATURES = ["t_index"]
ETF_FEATURES = [
    "lag_target",
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
MACRO_FEATURES = [
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
FEATURE_COLUMNS = [*TIME_FEATURES, *ETF_FEATURES, *MACRO_FEATURES]
TARGET_COLUMNS = {
    "total": "target_total_excess",
    "price": "target_price_excess",
    "distribution": "target_distribution_contribution",
}
LAG_COLUMNS = {
    "total": "lag_total_excess",
    "price": "lag_price_excess",
    "distribution": "lag_distribution_contribution",
}
SETUPS = ["total_return", "price_only", "decomposed"]
INTERVAL_LEVELS = (0.50, 0.80, 0.95)
EVALUATION_START = pd.Timestamp("2024-05-31")
EVALUATION_END = pd.Timestamp("2026-04-30")


def parse_args() -> argparse.Namespace:
    """Parse deterministic experiment configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--input-panel", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=16638)
    parser.add_argument("--maxiter", type=int, default=75)
    parser.add_argument("--posterior-scenarios", type=int, default=5000)
    parser.add_argument("--min-inferred-noise-level", type=float, default=0.0025)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _download_market_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for asset in ASSETS:
        history = yf.Ticker(asset).history(
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=8)).date().isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=True,
            repair=False,
        )
        if history.empty:
            raise RuntimeError(f"Yahoo returned no market data for {asset}")
        history = history.reset_index()
        date_column = "Date" if "Date" in history.columns else history.columns[0]
        dates = pd.to_datetime(history[date_column])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        frame = pd.DataFrame(
            {
                "date": dates.dt.normalize(),
                "asset_id": asset,
                "close": history["Close"].to_numpy(dtype=float),
                "adj_close": history["Adj Close"].to_numpy(dtype=float),
                "dividends": history.get("Dividends", pd.Series(0.0, index=history.index)).to_numpy(dtype=float),
                "stock_splits": history.get("Stock Splits", pd.Series(0.0, index=history.index)).to_numpy(dtype=float),
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["asset_id", "date"]).reset_index(drop=True)


def _build_input_panel(feature_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    features = pd.read_parquet(feature_path).copy()
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    features = features[features["asset_id"].isin(ASSETS)].sort_values(["asset_id", "date"]).reset_index(drop=True)
    if features.empty:
        raise ValueError("The frozen feature artifact contains no selected assets")
    missing_assets = sorted(set(ASSETS) - set(features["asset_id"].astype(str)))
    if missing_assets:
        raise ValueError(f"The frozen feature artifact is missing assets: {missing_assets}")

    market = _download_market_data(features["date"].min(), features["date"].max())
    split_rows = market.loc[market["stock_splits"].abs() > 0]
    if not split_rows.empty:
        split_summary = split_rows[["asset_id", "date", "stock_splits"]].to_dict(orient="records")
        raise ValueError(
            "The selected window contains stock splits; this experiment intentionally refuses literal "
            "raw-close returns. "
            f"Implement and verify split-only adjustment before rerunning: {split_summary}"
        )

    panel = features.copy()
    panel["target_total_excess"] = panel["y_excess_lead"].astype(float)
    panel["target_price_excess"] = np.nan
    panel["target_distribution_contribution"] = np.nan
    panel["cash_distribution_yield"] = np.nan
    total_reconstruction_errors: list[float] = []

    for asset in ASSETS:
        asset_rows = panel.index[panel["asset_id"].astype(str) == asset].tolist()
        asset_market = market[market["asset_id"] == asset].set_index("date").sort_index()
        asset_dates = panel.loc[asset_rows, "date"].tolist()
        for row_index, current_date, next_date in zip(
            asset_rows[:-1],
            asset_dates[:-1],
            asset_dates[1:],
            strict=True,
        ):
            current_market_date = _asof_market_date(asset_market.index, current_date, asset=asset)
            next_market_date = _asof_market_date(asset_market.index, next_date, asset=asset)
            total_excess = float(panel.at[row_index, "target_total_excess"])
            if not math.isfinite(total_excess):
                continue
            start_close = float(asset_market.at[current_market_date, "close"])
            end_close = float(asset_market.at[next_market_date, "close"])
            start_adjusted = float(asset_market.at[current_market_date, "adj_close"])
            end_adjusted = float(asset_market.at[next_market_date, "adj_close"])
            adjusted_log_return = math.log(end_adjusted / start_adjusted)
            implied_rf_log = adjusted_log_return - math.log1p(total_excess)
            price_excess = math.expm1(math.log(end_close / start_close) - implied_rf_log)
            distribution_contribution = total_excess - price_excess
            dividends = float(
                asset_market.loc[
                    (asset_market.index > current_market_date) & (asset_market.index <= next_market_date),
                    "dividends",
                ].sum()
            )
            reconstructed_total = price_excess + distribution_contribution
            total_reconstruction_errors.append(abs(reconstructed_total - total_excess))
            panel.at[row_index, "target_price_excess"] = price_excess
            panel.at[row_index, "target_distribution_contribution"] = distribution_contribution
            panel.at[row_index, "cash_distribution_yield"] = dividends / start_close

    panel["lag_total_excess"] = panel.groupby("asset_id", observed=True)["target_total_excess"].shift(1)
    panel["lag_price_excess"] = panel.groupby("asset_id", observed=True)["target_price_excess"].shift(1)
    panel["lag_distribution_contribution"] = panel.groupby("asset_id", observed=True)[
        "target_distribution_contribution"
    ].shift(1)
    panel["asset_id"] = pd.Categorical(panel["asset_id"], categories=ASSETS, ordered=True)
    panel = panel.sort_values(["date", "asset_id"]).reset_index(drop=True)
    audit = {
        "selected_assets": ASSETS,
        "asset_classes": ASSET_CLASS,
        "market_date_min": market["date"].min().date().isoformat(),
        "market_date_max": market["date"].max().date().isoformat(),
        "stock_split_event_count": int((market["stock_splits"].abs() > 0).sum()),
        "dividend_event_count": int((market["dividends"].abs() > 0).sum()),
        "max_total_reconstruction_error": float(max(total_reconstruction_errors, default=0.0)),
        "price_target_definition": (
            "Yahoo unadjusted close return minus the risk-free log return implied by the frozen adjusted-total target"
        ),
        "distribution_target_definition": "frozen adjusted total excess return minus price-only excess return",
    }
    return panel, market, audit


def _asof_market_date(index: pd.DatetimeIndex, label: pd.Timestamp, *, asset: str) -> pd.Timestamp:
    eligible = index[index <= label]
    if eligible.empty:
        raise ValueError(f"Yahoo market data has no observation on or before {label.date()} for {asset}")
    selected = pd.Timestamp(eligible[-1])
    if (label - selected).days > 7:
        raise ValueError(f"Yahoo market data is more than seven days stale at feature date {label.date()} for {asset}")
    return selected


def _load_or_build_panel(args: argparse.Namespace, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.input_panel is not None:
        panel = pd.read_parquet(args.input_panel)
        panel["date"] = pd.to_datetime(panel["date"])
        panel["asset_id"] = pd.Categorical(panel["asset_id"], categories=ASSETS, ordered=True)
        return panel, {
            "mode": "offline_replay",
            "input_panel": str(args.input_panel),
            "input_panel_sha256": _sha256(args.input_panel),
        }

    panel, market, audit = _build_input_panel(args.feature_path)
    market_path = output_dir / "market_data_snapshot.csv"
    market.to_csv(market_path, index=False)
    return panel, {
        "mode": "constructed_from_frozen_features_and_yahoo_actions",
        "feature_path": str(args.feature_path),
        "feature_sha256": _sha256(args.feature_path),
        "market_data_snapshot": str(market_path),
        "market_data_sha256": _sha256(market_path),
        "target_construction_audit": audit,
    }


def _evaluation_dates(panel: pd.DataFrame) -> list[pd.Timestamp]:
    valid_counts = panel.groupby("date", observed=True)["target_total_excess"].apply(
        lambda values: values.notna().sum()
    )
    dates = [
        pd.Timestamp(date)
        for date, count in valid_counts.items()
        if count == len(ASSETS) and EVALUATION_START <= pd.Timestamp(date) <= EVALUATION_END
    ]
    if len(dates) != 24:
        raise ValueError(f"Expected the established 24 evaluation windows, found {len(dates)}: {dates}")
    return dates


def _forecast_config(args: argparse.Namespace) -> MonthlyForecastConfig:
    return MonthlyForecastConfig(
        etf_feature_columns=ETF_FEATURES,
        macro_feature_columns=MACRO_FEATURES,
        rank=5,
        min_inferred_noise_level=args.min_inferred_noise_level,
        seed=args.seed,
        maxiter=args.maxiter,
        posterior_scenario_count=args.posterior_scenarios,
        time_modulation="lengthscale_only",
    )


def _stable_seed(base_seed: int, label: str, window_index: int, offset: int = 0) -> int:
    label_offset = sum((index + 1) * ord(character) for index, character in enumerate(label))
    return int(base_seed + label_offset + 10_000 * window_index + offset)


def _fit_component(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    target_key: str,
    config: MonthlyForecastConfig,
    fit_seed: int,
    sample_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    target_column = TARGET_COLUMNS[target_key]
    lag_column = LAG_COLUMNS[target_key]
    train = train_df.copy()
    evaluate = eval_df.copy()
    train["lag_target"] = train[lag_column]
    evaluate["lag_target"] = evaluate[lag_column]
    needed = [*FEATURE_COLUMNS, target_column]
    train = train.dropna(subset=needed).sort_values(["date", "asset_id"])
    evaluate = evaluate.dropna(subset=FEATURE_COLUMNS).sort_values(["asset_id"])
    if evaluate["asset_id"].astype(str).tolist() != ASSETS:
        raise ValueError(f"Evaluation rows are incomplete for target component {target_key}")

    feature_index = {name: index for index, name in enumerate(FEATURE_COLUMNS)}
    covar_config, mean_config = build_default_monthly_gp_configs(
        forecast_config=config,
        feature_index=feature_index,
    )
    task_map = {asset: index for index, asset in enumerate(ASSETS)}
    train_features = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    train_tasks = train["asset_id"].map(task_map).to_numpy(dtype=float).reshape(-1, 1)
    eval_features = evaluate[FEATURE_COLUMNS].to_numpy(dtype=float)
    eval_tasks = evaluate["asset_id"].map(task_map).to_numpy(dtype=float).reshape(-1, 1)
    train_x = torch.tensor(np.concatenate([train_features, train_tasks], axis=1), dtype=torch.float64)
    train_y = torch.tensor(train[target_column].to_numpy(dtype=float), dtype=torch.float64).unsqueeze(-1)
    eval_x = torch.tensor(np.concatenate([eval_features, eval_tasks], axis=1), dtype=torch.float64)
    task_feature_index = train_x.shape[-1] - 1
    all_task_values = train_x[:, task_feature_index].to(torch.long).unique(sorted=True)
    outcome_transform = StratifiedStandardize(
        stratification_idx=task_feature_index,
        all_task_values=all_task_values,
        batch_shape=train_y.shape[:-2],
    )
    input_transform = Normalize(
        d=train_x.shape[-1],
        indices=list(range(len(FEATURE_COLUMNS))),
    )

    torch.manual_seed(fit_seed)
    started = time.perf_counter()
    model = build_multitask_gp(
        train_X=train_x,
        train_Y=train_y,
        task_feature=-1,
        covar_config=covar_config,
        mean_config=mean_config,
        rank=config.rank,
        min_inferred_noise_level=config.min_inferred_noise_level,
        outcome_transform=outcome_transform,
        input_transform=input_transform,
        task_covar_prior=None,
    )
    data_kernel, task_kernel = model.covar_module.kernels
    model.covar_module = (
        build_time_varying_kernel(
            data_kernel,
            time_feature_index=0,
            target="lengthscale",
        )
        * task_kernel
    )
    model.train()
    likelihood = model.likelihood
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={"options": {"maxiter": config.maxiter}},
        )
    fit_seconds = time.perf_counter() - started

    model.eval()
    likelihood.eval()
    torch.manual_seed(sample_seed)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(eval_x, observation_noise=True)
        mean = posterior.mean.squeeze(-1).detach().cpu().numpy()
        std = posterior.variance.squeeze(-1).clamp_min(0.0).sqrt().detach().cpu().numpy()
        samples = posterior.rsample(torch.Size([config.posterior_scenario_count])).squeeze(-1).detach().cpu().numpy()
    noise = model.likelihood.noise.detach().cpu().reshape(-1).numpy()
    diagnostics = {
        "component": target_key,
        "train_rows": int(len(train)),
        "train_months": int(train["date"].nunique()),
        "train_start": train["date"].min().date().isoformat(),
        "train_end": train["date"].max().date().isoformat(),
        "fit_seconds": fit_seconds,
        "likelihood_noise_mean": float(noise.mean()),
        "likelihood_noise_min": float(noise.min()),
        "likelihood_noise_max": float(noise.max()),
    }
    return mean, std, samples, diagnostics


def _empirical_crps(samples: np.ndarray, observation: float) -> float:
    ordered = np.sort(np.asarray(samples, dtype=float))
    count = ordered.size
    coefficients = 2 * np.arange(1, count + 1) - count - 1
    half_pairwise = float(np.sum(coefficients * ordered) / (count**2))
    return float(np.mean(np.abs(ordered - observation)) - half_pairwise)


def _interval_score(observation: float, lower: float, upper: float, alpha: float) -> float:
    score = upper - lower
    if observation < lower:
        score += 2.0 / alpha * (lower - observation)
    elif observation > upper:
        score += 2.0 / alpha * (observation - upper)
    return float(score)


def _uncertainty_row(
    *,
    setup: str,
    view: str,
    date: pd.Timestamp,
    asset: str,
    observation: float,
    samples: np.ndarray,
) -> dict[str, Any]:
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    residual = observation - mean
    row: dict[str, Any] = {
        "setup": setup,
        "view": view,
        "date": date.date().isoformat(),
        "asset_id": asset,
        "observation": observation,
        "predictive_mean": mean,
        "predictive_std": std,
        "standardized_residual": residual / max(std, 1e-12),
        "nlpd": 0.5 * math.log(2.0 * math.pi * max(std, 1e-12) ** 2) + 0.5 * (residual / max(std, 1e-12)) ** 2,
        "crps": _empirical_crps(samples, observation),
    }
    for level in INTERVAL_LEVELS:
        alpha = 1.0 - level
        lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
        suffix = int(level * 100)
        row[f"lower_{suffix}"] = float(lower)
        row[f"upper_{suffix}"] = float(upper)
        row[f"covered_{suffix}"] = bool(lower <= observation <= upper)
        row[f"width_{suffix}"] = float(upper - lower)
        row[f"interval_score_{suffix}"] = _interval_score(observation, float(lower), float(upper), alpha)
    return row


def _prediction_summary(rows: pd.DataFrame, *, truth_column: str) -> dict[str, Any]:
    truth = rows.pivot(index="date", columns="asset_id", values=truth_column).reindex(columns=ASSETS)
    prediction = rows.pivot(index="date", columns="asset_id", values="prediction").reindex(columns=ASSETS)
    pricing = evaluate_asset_pricing(truth, prediction)
    residual = rows[truth_column] - rows["prediction"]
    by_asset: dict[str, Any] = {}
    for asset, group in rows.groupby("asset_id", observed=True):
        asset_residual = group[truth_column] - group["prediction"]
        by_asset[str(asset)] = {
            "n": int(len(group)),
            "rmse": float(np.sqrt(np.mean(np.square(asset_residual)))),
            "mae": float(np.mean(np.abs(asset_residual))),
            "bias_prediction_minus_actual": float(np.mean(group["prediction"] - group[truth_column])),
            "directional_accuracy": float(np.mean(np.sign(group[truth_column]) == np.sign(group["prediction"]))),
        }
    return {
        "n_observations": int(len(rows)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_prediction_minus_actual": float(np.mean(rows["prediction"] - rows[truth_column])),
        "directional_accuracy": float(np.mean(np.sign(rows[truth_column]) == np.sign(rows["prediction"]))),
        "asset_pricing_metrics": pricing,
        "by_asset": by_asset,
    }


def _uncertainty_summary(rows: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_observations": int(len(rows)),
        "mean_nlpd": float(rows["nlpd"].mean()),
        "mean_crps": float(rows["crps"].mean()),
        "mean_predictive_std": float(rows["predictive_std"].mean()),
        "standardized_residual_mean": float(rows["standardized_residual"].mean()),
        "standardized_residual_std": float(rows["standardized_residual"].std(ddof=1)),
    }
    coverage_errors: list[float] = []
    for level in INTERVAL_LEVELS:
        suffix = int(level * 100)
        coverage = float(rows[f"covered_{suffix}"].mean())
        error = coverage - level
        coverage_errors.append(abs(error))
        result[f"coverage_{suffix}"] = coverage
        result[f"coverage_error_{suffix}"] = error
        result[f"mean_interval_width_{suffix}"] = float(rows[f"width_{suffix}"].mean())
        result[f"mean_interval_score_{suffix}"] = float(rows[f"interval_score_{suffix}"].mean())
    result["mean_absolute_coverage_error"] = float(np.mean(coverage_errors))
    return result


def _bayesian_block_comparison(
    frame: pd.DataFrame,
    *,
    metric_column: str,
    reference: str,
    alternatives: list[str],
    higher_is_better: bool,
    seed: int,
    square_root_metric: bool = False,
    draws: int = 20_000,
) -> dict[str, Any]:
    monthly = frame.groupby(["date", "setup"], observed=True)[metric_column].mean().unstack("setup").sort_index()
    required = [reference, *alternatives]
    monthly = monthly.dropna(subset=required)
    block_ids = np.arange(len(monthly)) // 3
    blocks = monthly.groupby(block_ids).mean()
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(blocks)), size=draws)
    reference_draws = weights @ blocks[reference].to_numpy(dtype=float)
    if square_root_metric:
        reference_draws = np.sqrt(np.clip(reference_draws, 0.0, None))

    comparisons: dict[str, Any] = {}
    for alternative in alternatives:
        alternative_draws = weights @ blocks[alternative].to_numpy(dtype=float)
        if square_root_metric:
            alternative_draws = np.sqrt(np.clip(alternative_draws, 0.0, None))
        advantage = reference_draws - alternative_draws if higher_is_better else alternative_draws - reference_draws
        comparisons[alternative] = {
            "posterior_probability_reference_better": float(np.mean(advantage > 0.0)),
            "posterior_mean_reference_advantage": float(np.mean(advantage)),
            "credible_interval_90": np.quantile(advantage, [0.05, 0.95]).tolist(),
            "credible_interval_95": np.quantile(advantage, [0.025, 0.975]).tolist(),
        }
    return {
        "reference": reference,
        "metric": metric_column,
        "higher_is_better": higher_is_better,
        "square_root_after_weighting": square_root_metric,
        "method": "Bayesian bootstrap over eight non-overlapping three-month blocks",
        "draws": draws,
        "comparisons": comparisons,
    }


def _optimizer_request() -> OptimizeCommand:
    return OptimizeCommand(
        objective="Sharpe",
        risk_measure="CVaR",
        model="Classic",
        rf=0.0,
        alpha=0.5,
        min_weight=0.0,
        max_weight=0.40,
        nea=3,
        hist=True,
    )


def _portfolio_results(
    weight_rows: pd.DataFrame,
    panel: pd.DataFrame,
    evaluation_dates: list[pd.Timestamp],
) -> dict[str, Any]:
    all_results: dict[str, Any] = {}
    next_date = {
        date: pd.Timestamp(next_value)
        for date, next_value in zip(
            sorted(panel["date"].unique())[:-1],
            sorted(panel["date"].unique())[1:],
            strict=True,
        )
    }
    for setup in [*SETUPS, "equal_weight"]:
        periods: list[WalkForwardPeriod] = []
        setup_weights = weight_rows[weight_rows["setup"] == setup]
        for date in evaluation_dates:
            date_weights = setup_weights[setup_weights["date"] == date.date().isoformat()]
            realized = panel[panel["date"] == date].set_index("asset_id")["target_total_excess"]
            weights = dict(zip(date_weights["asset_id"].astype(str), date_weights["weight"], strict=True))
            returns = {asset: float(realized.loc[asset]) for asset in ASSETS}
            periods.append(
                WalkForwardPeriod(
                    strategy_id=setup,
                    source_run_label=f"{setup}_{date.date().isoformat()}",
                    decision_date=date.date(),
                    realization_date=next_date[np.datetime64(date)].date(),
                    weights=weights,
                    realized_returns=returns,
                )
            )
        result = run_walk_forward_backtest(
            WalkForwardBacktestCommand(
                strategy_id=setup,
                periods=periods,
                periods_per_year=12,
                starting_value=10_000.0,
                transaction_cost_bps=0.0,
                charge_initial_trade=False,
            )
        )
        all_results[setup] = result.model_dump(mode="json")
        all_results[setup]["average_effective_asset_count"] = float(
            setup_weights.groupby("date", observed=True)["weight"]
            .apply(lambda values: 1.0 / np.square(values).sum())
            .mean()
        )
        all_results[setup]["average_max_weight"] = float(
            setup_weights.groupby("date", observed=True)["weight"].max().mean()
        )
        exact_equal = setup_weights.groupby("date", observed=True)["weight"].apply(
            lambda values: bool(np.allclose(values.to_numpy(dtype=float), 1.0 / len(ASSETS), rtol=0.0, atol=1e-10))
        )
        all_results[setup]["exact_equal_weight_window_count"] = int(exact_equal.sum())
        all_results[setup]["exact_equal_weight_windows"] = exact_equal[exact_equal].index.astype(str).tolist()
    return all_results


def _model_configuration(args: argparse.Namespace, config: MonthlyForecastConfig) -> dict[str, Any]:
    feature_index = {name: index for index, name in enumerate(FEATURE_COLUMNS)}
    covar_config, mean_config = build_default_monthly_gp_configs(
        forecast_config=config,
        feature_index=feature_index,
    )
    return {
        "schema": "bayesfolio.return_target_comparison.gp_config.v1",
        "accepted_configuration_source": "tests/regression/test_july_2026_full_pipeline_parity.py",
        "forecast_config": config.model_dump(mode="json"),
        "feature_columns": FEATURE_COLUMNS,
        "covariance_config": covar_config.model_dump(mode="json"),
        "mean_config": mean_config.model_dump(mode="json"),
        "input_transform": "BoTorch Normalize over all non-task features",
        "outcome_transform": "StratifiedStandardize by ETF task",
        "task_covariance_prior": None,
        "time_modulation": "lengthscale_only",
        "decomposed_dependence_assumption": (
            "Price and distribution GPs share covariates and configuration but posterior draws are "
            "conditionally independent."
        ),
        "runtime": {
            "seed": args.seed,
            "maxiter": args.maxiter,
            "posterior_scenarios": args.posterior_scenarios,
            "dtype": "float64",
        },
    }


def _summarize_prediction_views(predictions: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for setup in SETUPS:
        setup_rows = predictions[predictions["setup"] == setup]
        result[setup] = {
            "decision_total_return": _prediction_summary(setup_rows, truth_column="actual_total"),
            "native_target": _prediction_summary(setup_rows, truth_column="actual_native"),
        }
    distribution_rows = predictions[predictions["setup"] == "decomposed_distribution_component"]
    result["decomposed_distribution_component"] = {
        "native_target": _prediction_summary(distribution_rows, truth_column="actual_native")
    }
    return result


def _summarize_uncertainty_views(intervals: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (setup, view), group in intervals.groupby(["setup", "view"], observed=True):
        result.setdefault(str(setup), {})[str(view)] = _uncertainty_summary(group)
    return result


def _prediction_comparisons(predictions: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    decision_rows = predictions[predictions["setup"].isin(SETUPS)].copy()
    decision_rows["squared_error"] = np.square(decision_rows["actual_total"] - decision_rows["prediction"])
    decision_rows["absolute_error"] = np.abs(decision_rows["actual_total"] - decision_rows["prediction"])
    return {
        "rmse": _bayesian_block_comparison(
            decision_rows,
            metric_column="squared_error",
            reference="total_return",
            alternatives=["price_only", "decomposed"],
            higher_is_better=False,
            square_root_metric=True,
            seed=seed + 101,
        ),
        "mae": _bayesian_block_comparison(
            decision_rows,
            metric_column="absolute_error",
            reference="total_return",
            alternatives=["price_only", "decomposed"],
            higher_is_better=False,
            seed=seed + 102,
        ),
    }


def _uncertainty_comparisons(intervals: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    decision_rows = intervals[(intervals["view"] == "decision_total_return") & intervals["setup"].isin(SETUPS)].copy()
    comparisons: dict[str, Any] = {}
    for index, metric in enumerate(("nlpd", "crps", "interval_score_80"), start=1):
        comparisons[metric] = _bayesian_block_comparison(
            decision_rows,
            metric_column=metric,
            reference="total_return",
            alternatives=["price_only", "decomposed"],
            higher_is_better=False,
            seed=seed + 200 + index,
        )
    return comparisons


def _portfolio_comparisons(portfolio_payload: dict[str, Any], *, seed: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for setup, result in portfolio_payload.items():
        for period in result["periods"]:
            rows.append(
                {
                    "setup": setup,
                    "date": period["decision_date"],
                    "net_return": period["net_return"],
                }
            )
    return {
        "mean_monthly_return": _bayesian_block_comparison(
            pd.DataFrame(rows),
            metric_column="net_return",
            reference="total_return",
            alternatives=["price_only", "decomposed", "equal_weight"],
            higher_is_better=True,
            seed=seed + 301,
        )
    }


def run(args: argparse.Namespace) -> None:
    """Run the complete three-target walk-forward comparison."""

    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.torch_num_threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    panel, data_provenance = _load_or_build_panel(args, args.output_dir)
    input_panel_path = args.output_dir / "input_panel.parquet"
    panel.to_parquet(input_panel_path, index=False)
    evaluation_dates = _evaluation_dates(panel)
    config = _forecast_config(args)
    gp_config = _model_configuration(args, config)
    gp_config_path = args.output_dir / "gp_config.json"
    _write_json(gp_config_path, gp_config)

    prediction_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    optimizer = _optimizer_request()

    for window_index, window_date in enumerate(evaluation_dates):
        print(f"[{window_index + 1:02d}/{len(evaluation_dates)}] {window_date.date()}", flush=True)
        train_df = panel[panel["date"] < window_date].copy()
        eval_df = panel[panel["date"] == window_date].copy()
        component_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for component in ("total", "price", "distribution"):
            mean, std, samples, diagnostics = _fit_component(
                train_df,
                eval_df,
                target_key=component,
                config=config,
                fit_seed=_stable_seed(args.seed, component, window_index),
                sample_seed=_stable_seed(args.seed, component, window_index, offset=5_000),
            )
            diagnostics["window_date"] = window_date.date().isoformat()
            diagnostic_rows.append(diagnostics)
            component_results[component] = (mean, std, samples)

        total_mean, total_std, total_samples = component_results["total"]
        price_mean, price_std, price_samples = component_results["price"]
        distribution_mean, distribution_std, distribution_samples = component_results["distribution"]
        decomposed_samples = price_samples + distribution_samples
        decomposed_mean = decomposed_samples.mean(axis=0)
        decomposed_std = decomposed_samples.std(axis=0, ddof=1)
        setup_values = {
            "total_return": (total_mean, total_std, total_samples),
            "price_only": (price_mean, price_std, price_samples),
            "decomposed": (decomposed_mean, decomposed_std, decomposed_samples),
        }
        eval_ordered = eval_df.sort_values("asset_id")
        actual_total = eval_ordered["target_total_excess"].to_numpy(dtype=float)
        actual_price = eval_ordered["target_price_excess"].to_numpy(dtype=float)

        for setup, (mean, std, samples) in setup_values.items():
            actual_native = actual_price if setup == "price_only" else actual_total
            for asset_index, asset in enumerate(ASSETS):
                prediction_rows.append(
                    {
                        "setup": setup,
                        "date": window_date.date().isoformat(),
                        "asset_id": asset,
                        "asset_class": ASSET_CLASS[asset],
                        "actual_total": float(actual_total[asset_index]),
                        "actual_native": float(actual_native[asset_index]),
                        "prediction": float(mean[asset_index]),
                        "predictive_std": float(std[asset_index]),
                    }
                )
                uncertainty_rows.append(
                    _uncertainty_row(
                        setup=setup,
                        view="decision_total_return",
                        date=window_date,
                        asset=asset,
                        observation=float(actual_total[asset_index]),
                        samples=samples[:, asset_index],
                    )
                )
                uncertainty_rows.append(
                    _uncertainty_row(
                        setup=setup,
                        view="native_target",
                        date=window_date,
                        asset=asset,
                        observation=float(actual_native[asset_index]),
                        samples=samples[:, asset_index],
                    )
                )

            optimized = optimize_from_historical_returns(
                pd.DataFrame(samples, columns=ASSETS),
                request=optimizer,
            )
            for asset, weight in zip(optimized.asset_order, optimized.weights, strict=True):
                weight_rows.append(
                    {
                        "setup": setup,
                        "date": window_date.date().isoformat(),
                        "asset_id": asset,
                        "weight": float(weight),
                    }
                )

        for asset_index, asset in enumerate(ASSETS):
            actual_distribution = float(eval_ordered.iloc[asset_index]["target_distribution_contribution"])
            prediction_rows.append(
                {
                    "setup": "decomposed_distribution_component",
                    "date": window_date.date().isoformat(),
                    "asset_id": asset,
                    "asset_class": ASSET_CLASS[asset],
                    "actual_total": actual_distribution,
                    "actual_native": actual_distribution,
                    "prediction": float(distribution_mean[asset_index]),
                    "predictive_std": float(distribution_std[asset_index]),
                }
            )
            uncertainty_rows.append(
                _uncertainty_row(
                    setup="decomposed_distribution_component",
                    view="native_target",
                    date=window_date,
                    asset=asset,
                    observation=actual_distribution,
                    samples=distribution_samples[:, asset_index],
                )
            )
            weight_rows.append(
                {
                    "setup": "equal_weight",
                    "date": window_date.date().isoformat(),
                    "asset_id": asset,
                    "weight": 1.0 / len(ASSETS),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    intervals = pd.DataFrame(uncertainty_rows)
    weights = pd.DataFrame(weight_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    predictions.to_csv(args.output_dir / "prediction_details.csv", index=False)
    intervals.to_csv(args.output_dir / "uncertainty_details.csv", index=False)
    weights.to_csv(args.output_dir / "portfolio_weights.csv", index=False)
    diagnostics.to_csv(args.output_dir / "fit_diagnostics.csv", index=False)

    prediction_results = {
        "schema": "bayesfolio.return_target_comparison.prediction_results.v1",
        "primary_evaluation_target": "realized adjusted total excess return",
        "results": _summarize_prediction_views(predictions),
        "paired_bayesian_comparisons": _prediction_comparisons(predictions, seed=args.seed),
    }
    uncertainty_results = {
        "schema": "bayesfolio.return_target_comparison.uncertainty_results.v1",
        "proper_scores": ["negative_log_predictive_density", "continuous_ranked_probability_score"],
        "results": _summarize_uncertainty_views(intervals),
        "paired_bayesian_comparisons": _uncertainty_comparisons(intervals, seed=args.seed),
    }
    portfolio_payload = _portfolio_results(weights, panel, evaluation_dates)
    portfolio_results = {
        "schema": "bayesfolio.return_target_comparison.portfolio_results.v1",
        "realized_return_target": "adjusted total excess return",
        "optimizer": optimizer.model_dump(mode="json"),
        "optimizer_status_note": (
            "The maintained adapter does not expose solver status. Exact equal-weight windows are reported because "
            "they can indicate either the numerical optimum or the adapter's fallback after Riskfolio infeasibility."
        ),
        "results": portfolio_payload,
        "paired_bayesian_comparisons": _portfolio_comparisons(portfolio_payload, seed=args.seed),
    }
    result_paths = {
        "prediction_results": args.output_dir / "prediction_results.json",
        "uncertainty_results": args.output_dir / "uncertainty_results.json",
        "portfolio_results": args.output_dir / "portfolio_results.json",
    }
    _write_json(result_paths["prediction_results"], prediction_results)
    _write_json(result_paths["uncertainty_results"], uncertainty_results)
    _write_json(result_paths["portfolio_results"], portfolio_results)

    manifest = {
        "schema": "bayesfolio.return_target_comparison.manifest.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git_sha": _git_output("rev-parse", "HEAD"),
        "git_status_at_start": _git_output("status", "--short"),
        "objective": (
            "Compare total, price-only, and decomposed monthly multitask-GP targets on prediction, uncertainty, "
            "and downstream total-return portfolio quality."
        ),
        "assets": ASSETS,
        "asset_classes": ASSET_CLASS,
        "evaluation": {
            "design": "expanding-window one-month-ahead walk-forward",
            "comparison_uncertainty": "Bayesian bootstrap over eight non-overlapping three-month blocks",
            "dates": [date.date().isoformat() for date in evaluation_dates],
            "window_count": len(evaluation_dates),
            "start": evaluation_dates[0].date().isoformat(),
            "end": evaluation_dates[-1].date().isoformat(),
            "training_start": panel["date"].min().date().isoformat(),
            "primary_target": "adjusted total excess return",
        },
        "data_provenance": data_provenance,
        "input_panel": {
            "path": str(input_panel_path),
            "sha256": _sha256(input_panel_path),
            "rows": int(len(panel)),
            "columns": panel.columns.tolist(),
        },
        "gp_config": {"path": str(gp_config_path), "sha256": _sha256(gp_config_path)},
        "results": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in result_paths.items()},
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "botorch": botorch.__version__,
            "gpytorch": gpytorch.__version__,
            "riskfolio": riskfolio.__version__,
            "yfinance": yf.__version__,
        },
        "replay_command": (
            "uv run --locked python experiments/2026-10-return-target-comparison/"
            "run_return_target_comparison.py --input-panel "
            f"{input_panel_path} --output-dir /tmp/bayesfolio-return-target-replay"
        ),
        "limitations": [
            "The five-asset universe is diagnostic and does not estimate behavior for the full retirement universe.",
            "Common non-lag covariates are held fixed and were originally engineered from the accepted "
            "feature artifact.",
            "The decomposed setup adds independent price and distribution posterior draws and therefore omits residual "
            "cross-component dependence.",
            "No transaction costs or taxes are applied; the account of interest is a tax-deferred IRA.",
        ],
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    print(f"Wrote experiment evidence to {args.output_dir}", flush=True)


def main() -> int:
    """Run the command-line experiment and return a process status."""

    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
