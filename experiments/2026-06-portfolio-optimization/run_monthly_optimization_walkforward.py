"""Walk-forward monthly portfolio optimization from multitask GP scenarios.

Usage:
    poetry run python experiments/2026-06-portfolio-optimization/run_monthly_optimization_walkforward.py \
        --run-id 20260616_gp_scenario_portfolio --maxiter 50
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
import riskfolio as rp
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import ScaleKernel
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
    "signed_lkj_eta_2",
    "signed_lkj_eta_2_turnover_blend",
    "signed_lkj_eta_2_lengthscale_floor",
}
LENGTHSCALE_FLOOR_EXPERIMENTS = {"lengthscale_floor", "signed_lkj_eta_2_lengthscale_floor"}
TURNOVER_BLEND_EXPERIMENTS = {"turnover_blend", "signed_lkj_eta_2_turnover_blend"}


@dataclass(frozen=True)
class StrategyResult:
    name: str
    weights: pd.DataFrame
    returns: pd.Series


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
    parser.add_argument(
        "--gp-experiment",
        choices=[
            "control",
            "signed_lkj_eta_2",
            "lengthscale_floor",
            "component_outputscale_floor",
            "scenario_mean_scale",
            "turnover_blend",
            "signed_lkj_eta_2_turnover_blend",
            "signed_lkj_eta_2_lengthscale_floor",
        ],
        default="control",
    )
    parser.add_argument("--scenario-mean-scale", type=float, default=1.0)
    parser.add_argument("--turnover-blend", type=float, default=0.50)
    parser.add_argument("--lengthscale-floor", type=float, default=0.02)
    parser.add_argument("--outputscale-floor", type=float, default=0.01)
    parser.add_argument("--outputscale-prior-median", type=float, default=0.05)
    parser.add_argument("--outputscale-prior-sigma", type=float, default=0.75)
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


def build_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    df: pd.DataFrame,
    scored_dates: list[pd.Timestamp],
) -> dict[str, Any]:
    final_universe = [asset for asset in task_exp.ETF_TICKERS if asset not in HELPER_ASSETS]
    is_signed = args.gp_experiment in SIGNED_EXPERIMENTS
    uses_lengthscale_floor = args.gp_experiment in LENGTHSCALE_FLOOR_EXPERIMENTS
    uses_turnover_blend = args.gp_experiment in TURNOVER_BLEND_EXPERIMENTS
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
            "target_col": task_exp.TARGET_COL,
            "training_universe": task_exp.ETF_TICKERS,
            "helper_assets_fit_but_excluded": sorted(HELPER_ASSETS),
            "final_portfolio_universe": final_universe,
        },
        "rebalance_dates": [date.date().isoformat() for date in scored_dates],
        "portfolio": {
            "starting_value": STARTING_VALUE,
            "rebalance_frequency": "monthly",
            "n_rebalances": len(scored_dates),
            "strategies": [
                "gp_scenarios_riskfolio",
                "historical_y_ewma2_riskfolio",
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
                "method_mu": "ewma2",
                "method_cov": "ewma2",
                "hist": True,
                "upperlng": args.upperlng,
                "nea": args.nea,
            },
        },
        "modeling": {
            "experiment": args.gp_experiment,
            "variant": "signed_lkj_eta_2" if is_signed else "positive_beta_prior",
            "task_kernel": "IndexKernel" if is_signed else "PositiveIndexKernel",
            "task_covar_prior": None if is_signed else "BetaPrior(2.5, 1.5)",
            "time_modulation_mode": "lengthscale_only",
            "lengthscale_floor": args.lengthscale_floor if uses_lengthscale_floor else None,
            "outputscale_floor": args.outputscale_floor
            if args.gp_experiment == "component_outputscale_floor"
            else None,
            "outputscale_prior_median": args.outputscale_prior_median
            if args.gp_experiment == "component_outputscale_floor"
            else None,
            "scenario_mean_scale": args.scenario_mean_scale if args.gp_experiment == "scenario_mean_scale" else 1.0,
            "turnover_blend": args.turnover_blend if uses_turnover_blend else None,
            "rank": task_exp.RANK,
            "outcome_transform": "StratifiedStandardize by ETF task",
            "input_transform": "BoTorch Normalize on non-task feature columns",
            "posterior_scenarios_per_rebalance": args.posterior_scenarios,
            "maxiter": args.maxiter,
            "seed": args.seed,
        },
    }


def optimize_riskfolio(
    returns: pd.DataFrame,
    *,
    method_mu: str,
    method_cov: str,
    upperlng: float,
    nea: int,
) -> pd.Series:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any").dropna(axis=0, how="any")
    if clean.shape[1] < 2 or clean.empty:
        return equal_weight(clean.columns.tolist())

    try:
        n_assets = clean.shape[1]
        portfolio = rp.Portfolio(returns=clean, nea=max(1, min(int(nea), n_assets - 1)))
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
    except Exception:
        weights = equal_weight(clean.columns.tolist())

    weights = weights.reindex(clean.columns).fillna(0.0).clip(lower=0.0)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return equal_weight(clean.columns.tolist())
    return weights / total


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


def build_experiment_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    args: argparse.Namespace,
) -> Any:
    is_signed = args.gp_experiment in SIGNED_EXPERIMENTS
    uses_lengthscale_floor = args.gp_experiment in LENGTHSCALE_FLOOR_EXPERIMENTS
    variant_name = "signed_lkj_eta_2" if is_signed else "positive_beta_prior"
    variant = task_exp.VARIANTS[variant_name]
    if not uses_lengthscale_floor:
        return task_exp.build_model(train_x, train_y, variant, time_modulation_mode="lengthscale_only")

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
        return task_exp.build_model(train_x, train_y, variant, time_modulation_mode="lengthscale_only")
    finally:
        task_exp.build_time_varying_kernel = original_builder


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
    train_x, train_y, eval_x, _, _, _ = task_exp.prepare_window_tensors(train_df, eval_df)
    torch.manual_seed(seed)
    model = build_experiment_model(train_x, train_y, args=args)
    outputscale_prior_rows: list[dict[str, Any]] = []
    if args.gp_experiment == "component_outputscale_floor":
        outputscale_prior_rows = apply_outputscale_floor_prior(
            model,
            floor=args.outputscale_floor,
            prior_median=args.outputscale_prior_median,
            prior_sigma=args.outputscale_prior_sigma,
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
    for row in outputscale_prior_rows:
        row["date"] = pd.Timestamp(eval_df["date"].iloc[0]).date().isoformat()
        row["experiment"] = args.gp_experiment
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
    return scenarios, predictions, [*diagnostics, *outputscale_prior_rows], task_diag


def realized_return(weights: pd.Series, eval_returns: pd.Series) -> float:
    aligned = weights.reindex(eval_returns.index).fillna(0.0)
    value = float(np.dot(aligned.to_numpy(dtype=float), eval_returns.to_numpy(dtype=float)))
    return value if np.isfinite(value) else math.nan


def information_coefficient(predictions: pd.DataFrame, final_universe: list[str]) -> float:
    frame = predictions[predictions["asset_id"].isin(final_universe)]
    if frame["y_pred"].nunique() < 2 or frame["y_true"].nunique() < 2:
        return math.nan
    return float(frame["y_pred"].corr(frame["y_true"], method="spearman"))


def performance_stats(returns: pd.Series, weights: pd.DataFrame, *, starting_value: float) -> dict[str, float]:
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
            "mean_monthly_return": math.nan,
            "hit_rate": math.nan,
            "avg_turnover": math.nan,
            "max_weight": math.nan,
        }

    equity = (1.0 + returns).cumprod()
    cumulative_return = float(equity.iloc[-1] - 1.0)
    years = len(returns) / PERIODS_PER_YEAR
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else math.nan
    ann_vol = float(returns.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR))
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
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def run(args: argparse.Namespace) -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = task_exp.load_features(args.feature_path)
    scored_dates, _ = task_exp.scored_and_live_dates(df, args.max_windows)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    final_universe = [asset for asset in task_exp.ETF_TICKERS if asset not in HELPER_ASSETS]

    manifest = build_manifest(args, output_dir, df, scored_dates)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

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
    previous_gp_weights: pd.Series | None = None

    for window_index, window_date in enumerate(scored_dates):
        print(f"rebalance {window_date.date()}", flush=True)
        train_df = df[(df["date"] < window_date) & df[task_exp.TARGET_COL].notna()].copy()
        eval_df = df[df["date"] == window_date].copy()
        eval_returns = (
            eval_df.set_index(eval_df["asset_id"].astype(str))[task_exp.TARGET_COL]
            .reindex(final_universe)
            .astype(float)
        )

        scenarios, predictions, model_diagnostics, task_diag = fit_gp_window(
            train_df,
            eval_df,
            args=args,
            seed=task_exp.stable_seed(args.seed, "positive_beta_prior", window_index),
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

        gp_weights = optimize_riskfolio(
            scenarios.loc[:, final_universe],
            method_mu="hist",
            method_cov="hist",
            upperlng=args.upperlng,
            nea=args.nea,
        )
        if args.gp_experiment in TURNOVER_BLEND_EXPERIMENTS:
            gp_weights = blend_with_previous_weights(
                gp_weights,
                previous_gp_weights,
                blend=args.turnover_blend,
            )
        previous_gp_weights = gp_weights.reindex(final_universe).fillna(0.0)
        hist_panel = train_df.pivot(index="date", columns="asset_id", values=task_exp.TARGET_COL).reindex(
            columns=final_universe
        )
        hist_weights = optimize_riskfolio(
            hist_panel,
            method_mu="ewma2",
            method_cov="ewma2",
            upperlng=args.upperlng,
            nea=args.nea,
        )
        ew_weights = equal_weight(final_universe)

        weights_by_strategy = {
            "gp_scenarios_riskfolio": gp_weights.reindex(final_universe).fillna(0.0),
            "historical_y_ewma2_riskfolio": hist_weights.reindex(final_universe).fillna(0.0),
            "equal_weight": ew_weights.reindex(final_universe).fillna(0.0),
        }
        gp_ic = information_coefficient(predictions, final_universe)
        ic_rows.append({"date": window_date.date().isoformat(), "strategy": "gp_scenarios_riskfolio", "ic": gp_ic})

        for strategy, weights in weights_by_strategy.items():
            weight_frames[strategy].append(pd.Series(weights, name=window_date))
            return_rows.append(
                {
                    "date": window_date.date().isoformat(),
                    "strategy": strategy,
                    "return": realized_return(weights, eval_returns),
                    "gp_ic": gp_ic if strategy == "gp_scenarios_riskfolio" else math.nan,
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
        summary = {"strategy": strategy, **performance_stats(strategy_ret, weights, starting_value=STARTING_VALUE)}
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
            f"Rebalances: `{len(scored_dates)}` monthly windows from "
            f"`{scored_dates[0].date()}` to `{scored_dates[-1].date()}`."
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
            f"- `gp_scenarios_riskfolio` experiment variant: `{args.gp_experiment}`. "
            "Control is positive beta task covariance, lengthscale-only time modulation, rank 5."
        ),
        (
            "- `historical_y_ewma2_riskfolio` ignores GP predictions and optimizes directly on historical "
            "`y_excess_lead` with EWMA2, Sharpe, and CVaR."
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
