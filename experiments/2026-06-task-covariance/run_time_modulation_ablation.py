"""Roll-forward ablation for time-varying GP outputscale/lengthscale wrappers."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import run_task_covariance_rollforward as exp
import torch
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import StratifiedStandardize
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.priors import LogNormalPrior

import bayesfolio.engine.forecast.gp.multitask_builder as multitask_builder
from bayesfolio.engine.forecast.gp.multitask_builder import MeanKind, MeanModuleConfig, build_multitask_gp
from bayesfolio.engine.forecast.gp.time_varying_kernel import build_time_varying_kernel

RUN_ID = "20260613_time_modulation_ablation"
RUN_ID_OUTPUTSCALE_PRIOR = "20260613_time_modulation_ablation_outputscale_prior"
RUN_ID_COMPONENT_BUDGET_PRIOR = "20260613_time_modulation_ablation_component_budget_prior"
BASE_VARIANT = exp.VARIANTS["positive_no_prior"]
MODES = ["both", "outputscale_only", "lengthscale_only", "neither"]
LOGNORMAL_SIGMA = 0.5
SCALE_COMPONENT_BUDGETS = [
    ("time", 0.10),
    ("etf", 0.25),
    ("macro_matern", 0.20),
    ("macro_rq", 0.10),
    ("macro_linear", 0.10),
    ("time_x_etf", 0.07),
    ("time_x_macro", 0.09),
    ("macro_x_etf", 0.09),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=exp.DEFAULT_FEATURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument(
        "--outputscale-prior",
        choices=["none", "lognormal_unit", "lognormal_component_budget"],
        default="none",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        if args.outputscale_prior == "lognormal_component_budget":
            run_id = RUN_ID_COMPONENT_BUDGET_PRIOR
        elif args.outputscale_prior != "none":
            run_id = RUN_ID_OUTPUTSCALE_PRIOR
        else:
            run_id = RUN_ID
        args.output_dir = exp.OUTPUT_ROOT / "runs" / run_id
    return args


def modulation_builder(mode: str) -> tuple[bool, Callable[[Kernel], Kernel]]:
    original = multitask_builder.add_time_varying_os_ls
    if mode == "both":
        return True, original
    if mode == "outputscale_only":
        return True, lambda covar: build_time_varying_kernel(covar, time_feature_index=0, target="outputscale")
    if mode == "lengthscale_only":
        return True, lambda covar: build_time_varying_kernel(covar, time_feature_index=0, target="lengthscale")
    if mode == "neither":
        return False, original
    raise ValueError(f"Unknown mode: {mode}")


def apply_outputscale_prior(model: torch.nn.Module, prior_name: str) -> int:
    if prior_name == "none":
        return 0
    if prior_name not in {"lognormal_unit", "lognormal_component_budget"}:
        raise ValueError(f"Unknown outputscale prior: {prior_name}")

    applied = 0
    scale_modules = [module for module in model.modules() if isinstance(module, ScaleKernel)]
    if prior_name == "lognormal_component_budget" and len(scale_modules) != len(SCALE_COMPONENT_BUDGETS):
        raise ValueError(
            "Component-budget outputscale prior expected "
            f"{len(SCALE_COMPONENT_BUDGETS)} ScaleKernels but found {len(scale_modules)}"
        )
    for index, module in enumerate(scale_modules):
        if prior_name == "lognormal_unit":
            component_name = "unit"
            median = 1.0
        else:
            component_name, median = SCALE_COMPONENT_BUDGETS[index]
        module.register_prior(
            f"{component_name}_lognormal_outputscale_prior",
            LogNormalPrior(loc=float(np.log(median)), scale=LOGNORMAL_SIGMA),
            "outputscale",
        )
        module.initialize(outputscale=median)
        applied += 1
    return applied


def collect_outputscale_diagnostics(
    *,
    model: torch.nn.Module,
    variant: str,
    window_date: pd.Timestamp | None,
    window_index: int,
    outputscale_prior: str,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    scale_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, ScaleKernel)]
    for index, (path, module) in enumerate(scale_modules):
        if index < len(SCALE_COMPONENT_BUDGETS):
            component, budget_median = SCALE_COMPONENT_BUDGETS[index]
        else:
            component, budget_median = path, np.nan
        outputscale = module.outputscale.detach().cpu().reshape(-1)
        diagnostics.append(
            {
                "variant": variant,
                "date": window_date.date().isoformat() if window_date is not None else "live",
                "window_index": window_index,
                "outputscale_prior": outputscale_prior,
                "scale_index": index,
                "component": component,
                "kernel_path": path,
                "kernel_type": module.base_kernel.__class__.__name__,
                "prior_median": float(budget_median),
                "learned_outputscale_mean": float(outputscale.mean()),
                "learned_outputscale_min": float(outputscale.min()),
                "learned_outputscale_max": float(outputscale.max()),
            }
        )
    return diagnostics


def fit_mode(
    *,
    mode: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    window_index: int,
    window_date: pd.Timestamp | None,
    seed: int,
    maxiter: int,
    outputscale_prior: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    add_tv_os_ls, builder = modulation_builder(mode)
    original = multitask_builder.add_time_varying_os_ls
    multitask_builder.add_time_varying_os_ls = builder
    try:
        torch.manual_seed(exp.stable_seed(seed, BASE_VARIANT.name, window_index))
        train_x, train_y, eval_x, _, _, _ = exp.prepare_window_tensors(train_df, eval_df)
        task_idx = train_x.shape[-1] - 1
        outcome_transform = StratifiedStandardize(
            stratification_idx=task_idx,
            all_task_values=train_x[:, task_idx].to(torch.long).unique(sorted=True),
            batch_shape=train_y.shape[:-2],
        )
        input_transform = Normalize(
            d=train_x.shape[-1],
            indices=list(range(len(exp.INPUT_COLUMNS))),
        )
        model = build_multitask_gp(
            train_X=train_x,
            train_Y=train_y,
            task_feature=exp.TASK_FEATURE,
            covar_config=exp.build_covar_config(),
            mean_config=MeanModuleConfig(kind=MeanKind.MULTITASK_CONSTANT),
            rank=exp.RANK,
            min_inferred_noise_level=5e-3,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            task_covar_prior=BASE_VARIANT.task_covar_prior,
            add_tv_os_ls=add_tv_os_ls,
        )
        apply_outputscale_prior(model, outputscale_prior)
        y_pred, y_std = exp.fit_and_predict(model, eval_x, maxiter=maxiter)
        diagnostics = collect_outputscale_diagnostics(
            model=model,
            variant=f"{BASE_VARIANT.name}_{mode}",
            window_date=window_date,
            window_index=window_index,
            outputscale_prior=outputscale_prior,
        )
        return y_pred, y_std, diagnostics
    finally:
        multitask_builder.add_time_varying_os_ls = original


def prediction_records(
    *,
    variant: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    train_means = exp.training_asset_means(train_df, eval_df)
    for row, mean, std, train_mean in zip(eval_df.itertuples(index=False), y_pred, y_std, train_means, strict=True):
        records.append(
            {
                "variant": variant,
                "date": pd.Timestamp(row.date).date().isoformat(),
                "asset_id": str(row.asset_id),
                "train_mean": float(train_mean),
                "y_true": float(row.y_excess_lead),
                "y_pred": float(mean),
                "y_std": float(std),
                "y_true_resid": float(row.y_excess_lead - train_mean),
                "y_pred_resid": float(mean - train_mean),
            }
        )
    return records


def write_asset_plots(output_dir: Path, pred_rows: list[dict[str, Any]]) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(pred_rows)
    df["date"] = pd.to_datetime(df["date"])
    order = ["historical_mean", *[f"{BASE_VARIANT.name}_{mode}" for mode in MODES]]
    colors = {
        "historical_mean": "#7f7f7f",
        f"{BASE_VARIANT.name}_both": "#1f77b4",
        f"{BASE_VARIANT.name}_outputscale_only": "#ff7f0e",
        f"{BASE_VARIANT.name}_lengthscale_only": "#2ca02c",
        f"{BASE_VARIANT.name}_neither": "#9467bd",
    }
    labels = {
        "historical_mean": "historical_mean",
        f"{BASE_VARIANT.name}_both": "both wrappers",
        f"{BASE_VARIANT.name}_outputscale_only": "outputscale only",
        f"{BASE_VARIANT.name}_lengthscale_only": "lengthscale only",
        f"{BASE_VARIANT.name}_neither": "neither wrapper",
    }
    for asset in ["SPY", "VEA"]:
        sub = df[df["asset_id"] == asset]
        actual = sub[["date", "y_true"]].drop_duplicates().sort_values("date")
        fig, ax = plt.subplots(figsize=(11, 6.2))
        ax.axhline(0, color="#dddddd", linewidth=1, zorder=0)
        ax.plot(actual["date"], actual["y_true"], color="black", linewidth=2.8, marker="o", label="actual", zorder=5)
        for variant in order:
            vdf = sub[sub["variant"] == variant].sort_values("date")
            if vdf.empty:
                continue
            linestyle = "--" if variant == "historical_mean" else "-"
            ax.plot(
                vdf["date"],
                vdf["y_pred"],
                color=colors[variant],
                linestyle=linestyle,
                linewidth=1.8,
                marker=".",
                label=labels[variant],
            )
        ax.set_title(f"{asset} actual vs predicted excess return\nTime modulation ablation", fontsize=14, pad=12)
        ax.set_xlabel("Prediction month")
        ax.set_ylabel("Excess return")
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(True, axis="x", alpha=0.12)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{asset.lower()}_time_modulation_ablation.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = exp.load_features(args.feature_path)
    scored_dates, live_date = exp.scored_and_live_dates(df, args.max_windows)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema": "bayesfolio.task_covariance_time_modulation_ablation.manifest.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git_sha": exp.git_sha(short=False),
        "git_dirty_summary_at_start": exp.git_dirty_summary(),
        "feature_path": str(args.feature_path),
        "feature_sha256": exp.sha256_file(args.feature_path),
        "base_variant": BASE_VARIANT.name,
        "modes": MODES,
        "scored_dates": [date.date().isoformat() for date in scored_dates],
        "live_date": live_date.date().isoformat() if live_date is not None else None,
        "maxiter": args.maxiter,
        "seed": args.seed,
        "outputscale_prior": args.outputscale_prior,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    pred_rows: list[dict[str, Any]] = []
    live_rows: list[dict[str, Any]] = []
    outputscale_diag_rows: list[dict[str, Any]] = []
    variants = ["historical_mean", *[f"{BASE_VARIANT.name}_{mode}" for mode in MODES]]

    for window_index, window_date in enumerate(scored_dates):
        print(f"window {window_date.date()}", flush=True)
        train_df = df[(df["date"] < window_date) & df[exp.TARGET_COL].notna()].copy()
        eval_df = df[df["date"] == window_date].copy()
        y_pred, y_std = exp.historical_mean_predict(train_df, eval_df)
        pred_rows.extend(
            prediction_records(
                variant="historical_mean",
                train_df=train_df,
                eval_df=eval_df,
                y_pred=y_pred,
                y_std=y_std,
            )
        )
        for mode in MODES:
            variant_name = f"{BASE_VARIANT.name}_{mode}"
            print(f"  {variant_name}", flush=True)
            y_pred, y_std, diagnostics = fit_mode(
                mode=mode,
                train_df=train_df,
                eval_df=eval_df,
                window_index=window_index,
                window_date=window_date,
                seed=args.seed,
                maxiter=args.maxiter,
                outputscale_prior=args.outputscale_prior,
            )
            outputscale_diag_rows.extend(diagnostics)
            pred_rows.extend(
                prediction_records(
                    variant=variant_name,
                    train_df=train_df,
                    eval_df=eval_df,
                    y_pred=y_pred,
                    y_std=y_std,
                )
            )

    if live_date is not None:
        train_df = df[(df["date"] < live_date) & df[exp.TARGET_COL].notna()].copy()
        eval_df = df[df["date"] == live_date].copy()
        for mode in MODES:
            variant_name = f"{BASE_VARIANT.name}_{mode}"
            y_pred, y_std, diagnostics = fit_mode(
                mode=mode,
                train_df=train_df,
                eval_df=eval_df,
                window_index=len(scored_dates),
                window_date=None,
                seed=args.seed,
                maxiter=args.maxiter,
                outputscale_prior=args.outputscale_prior,
            )
            outputscale_diag_rows.extend(diagnostics)
            live_rows.extend(
                prediction_records(
                    variant=variant_name,
                    train_df=train_df,
                    eval_df=eval_df,
                    y_pred=y_pred,
                    y_std=y_std,
                )
            )

    summary_rows = [
        exp.summarize_variant(variant, [row for row in pred_rows if row["variant"] == variant]) for variant in variants
    ]
    summary = pd.DataFrame(summary_rows)
    pd.DataFrame(pred_rows).to_csv(args.output_dir / "window_predictions.csv", index=False)
    pd.DataFrame(live_rows).to_csv(args.output_dir / "live_predictions.csv", index=False)
    pd.DataFrame(outputscale_diag_rows).to_csv(args.output_dir / "outputscale_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "variant_summary.csv", index=False)
    write_asset_plots(args.output_dir, pred_rows)

    print(summary.to_string(index=False))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    run(parse_args())
