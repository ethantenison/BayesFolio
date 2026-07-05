"""Ablate GP data-kernel outputscale structure for the task-covariance model."""

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

import bayesfolio.engine.forecast.gp.multitask_builder as multitask_builder
from bayesfolio.engine.forecast.gp.multitask_builder import (
    CovarModuleConfig,
    KernelBlockConfig,
    KernelInteractionConfig,
    MeanKind,
    MeanModuleConfig,
    build_multitask_gp,
)
from bayesfolio.engine.forecast.gp.time_varying_kernel import build_time_varying_kernel

RUN_ID = "20260614_outputscale_structure_ablation"
TIME_MODES = ["lengthscale_only", "neither"]
SCALE_STRUCTURES = ["component_scales", "no_component_scales", "global_scale"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=exp.DEFAULT_FEATURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[name for name in exp.VARIANTS if name != "historical_mean"],
        default=["positive_beta_prior"],
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = exp.OUTPUT_ROOT / "runs" / RUN_ID
    return args


def _without_outputscales(config: CovarModuleConfig) -> CovarModuleConfig:
    blocks: list[KernelBlockConfig] = []
    for block in config.blocks:
        components = [component.model_copy(update={"use_outputscale": False}) for component in block.components]
        blocks.append(block.model_copy(update={"components": components, "use_outputscale": False}))
    interactions: list[KernelInteractionConfig] = [
        interaction.model_copy(update={"use_outputscale": False}) for interaction in config.custom_interactions
    ]
    return config.model_copy(update={"blocks": blocks, "custom_interactions": interactions})


def covar_config_for_structure(scale_structure: str) -> CovarModuleConfig:
    config = exp.build_covar_config()
    if scale_structure == "component_scales":
        return config
    if scale_structure in {"no_component_scales", "global_scale"}:
        return _without_outputscales(config)
    raise ValueError(f"Unknown scale structure: {scale_structure}")


def modulation_builder(mode: str) -> tuple[bool, Callable[[Kernel], Kernel]]:
    original = multitask_builder.add_time_varying_os_ls
    if mode == "lengthscale_only":
        return True, lambda covar: build_time_varying_kernel(covar, time_feature_index=0, target="lengthscale")
    if mode == "neither":
        return False, original
    raise ValueError(f"Unknown time mode: {mode}")


def _wrap_global_data_scale(model: torch.nn.Module, train_x: torch.Tensor) -> None:
    data_kernel, task_kernel = model.covar_module.kernels
    model.covar_module = ScaleKernel(data_kernel, batch_shape=train_x.shape[:-2]) * task_kernel


def build_model(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    variant: exp.Variant,
    time_mode: str,
    scale_structure: str,
) -> Any:
    task_idx = train_x.shape[-1] - 1
    outcome_transform = StratifiedStandardize(
        stratification_idx=task_idx,
        all_task_values=train_x[:, task_idx].to(torch.long).unique(sorted=True),
        batch_shape=train_y.shape[:-2],
    )
    input_transform = Normalize(d=train_x.shape[-1], indices=list(range(len(exp.INPUT_COLUMNS))))
    add_tv_os_ls, builder = modulation_builder(time_mode)
    original = multitask_builder.add_time_varying_os_ls
    multitask_builder.add_time_varying_os_ls = builder
    try:
        model = build_multitask_gp(
            train_X=train_x,
            train_Y=train_y,
            task_feature=exp.TASK_FEATURE,
            covar_config=covar_config_for_structure(scale_structure),
            mean_config=MeanModuleConfig(kind=MeanKind.MULTITASK_CONSTANT),
            rank=exp.RANK,
            min_inferred_noise_level=5e-3,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            task_covar_prior=variant.task_covar_prior,
            add_tv_os_ls=add_tv_os_ls,
        )
    finally:
        multitask_builder.add_time_varying_os_ls = original
    if scale_structure == "global_scale":
        _wrap_global_data_scale(model, train_x)
    if variant.task_kernel == "signed":
        exp.replace_with_signed_index_kernel(model, eta=variant.lkj_eta)
    return model


def collect_hyperparameter_diagnostics(
    *,
    model: torch.nn.Module,
    train_x: torch.Tensor,
    eval_x: torch.Tensor,
    variant: str,
    window_date: pd.Timestamp | None,
    window_index: int,
    time_mode: str,
    scale_structure: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    date = window_date.date().isoformat() if window_date is not None else "live"
    base = {
        "variant": variant,
        "date": date,
        "window_index": window_index,
        "time_mode": time_mode,
        "scale_structure": scale_structure,
    }
    for path, module in model.named_modules():
        if isinstance(module, ScaleKernel):
            outputscale = module.outputscale.detach().cpu().reshape(-1)
            rows.append(
                {
                    **base,
                    "diagnostic": "outputscale",
                    "module_path": path,
                    "module_type": module.base_kernel.__class__.__name__,
                    "mean": float(outputscale.mean()),
                    "min": float(outputscale.min()),
                    "max": float(outputscale.max()),
                }
            )
        lengthscale = getattr(module, "lengthscale", None)
        if lengthscale is not None:
            values = lengthscale.detach().cpu().reshape(-1)
            rows.append(
                {
                    **base,
                    "diagnostic": "lengthscale",
                    "module_path": path,
                    "module_type": module.__class__.__name__,
                    "mean": float(values.mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
        if hasattr(module, "_effective_lengthscale"):
            transformed_train = model.transform_inputs(train_x)
            transformed_eval = model.transform_inputs(eval_x)
            for label, x in [("train", transformed_train), ("eval", transformed_eval)]:
                values = module._effective_lengthscale(x).detach().cpu().reshape(-1)
                rows.append(
                    {
                        **base,
                        "diagnostic": f"time_varying_effective_lengthscale_{label}",
                        "module_path": path,
                        "module_type": module.__class__.__name__,
                        "mean": float(values.mean()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
                )
    noise = getattr(getattr(model, "likelihood", None), "noise", None)
    if noise is not None:
        values = noise.detach().cpu().reshape(-1)
        rows.append(
            {
                **base,
                "diagnostic": "likelihood_noise",
                "module_path": "likelihood",
                "module_type": model.likelihood.__class__.__name__,
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    for name, parameter in model.named_parameters():
        if "constant" in name:
            values = parameter.detach().cpu().reshape(-1)
            rows.append(
                {
                    **base,
                    "diagnostic": "mean_constant",
                    "module_path": name,
                    "module_type": "Parameter",
                    "mean": float(values.mean()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return rows


def prediction_records(
    *,
    variant: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> list[dict[str, Any]]:
    train_means = exp.training_asset_means(train_df, eval_df)
    records: list[dict[str, Any]] = []
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
    colors = {
        "component_scales_lengthscale_only": "#1f77b4",
        "component_scales_neither": "#17becf",
        "no_component_scales_lengthscale_only": "#d62728",
        "no_component_scales_neither": "#ff9896",
        "global_scale_lengthscale_only": "#2ca02c",
        "global_scale_neither": "#98df8a",
    }
    for asset in ["SPY", "VEA"]:
        sub = df[df["asset_id"] == asset]
        actual = sub[["date", "y_true"]].drop_duplicates().sort_values("date")
        fig, ax = plt.subplots(figsize=(11, 6.2))
        ax.axhline(0, color="#dddddd", linewidth=1, zorder=0)
        ax.plot(actual["date"], actual["y_true"], color="black", linewidth=2.8, marker="o", label="actual", zorder=5)
        for variant in sorted(sub["variant"].unique()):
            if variant == "historical_mean":
                continue
            vdf = sub[sub["variant"] == variant].sort_values("date")
            suffix = variant.split("__", 1)[1]
            ax.plot(
                vdf["date"],
                vdf["y_pred"],
                color=colors.get(suffix, None),
                linewidth=1.7,
                marker=".",
                label=suffix,
            )
        ax.set_title(f"{asset} actual vs predicted excess return\nOutputscale structure ablation", fontsize=14, pad=12)
        ax.set_xlabel("Prediction month")
        ax.set_ylabel("Excess return")
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(True, axis="x", alpha=0.12)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{asset.lower()}_outputscale_structure_ablation.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = exp.load_features(args.feature_path)
    scored_dates, live_date = exp.scored_and_live_dates(df, args.max_windows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "bayesfolio.task_covariance_outputscale_structure_ablation.manifest.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": " ".join(sys.argv),
        "git_sha": exp.git_sha(short=False),
        "git_dirty_summary_at_start": exp.git_dirty_summary(),
        "feature_path": str(args.feature_path),
        "feature_sha256": exp.sha256_file(args.feature_path),
        "variants": args.variants,
        "time_modes": TIME_MODES,
        "scale_structures": SCALE_STRUCTURES,
        "scored_dates": [date.date().isoformat() for date in scored_dates],
        "live_date": live_date.date().isoformat() if live_date is not None else None,
        "maxiter": args.maxiter,
        "seed": args.seed,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    pred_rows: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    variants = [exp.VARIANTS[name] for name in args.variants]
    output_variant_names: list[str] = []

    for variant in variants:
        for window_index, window_date in enumerate(scored_dates):
            print(f"{variant.name} window {window_date.date()}", flush=True)
            train_df = df[(df["date"] < window_date) & df[exp.TARGET_COL].notna()].copy()
            eval_df = df[df["date"] == window_date].copy()
            for scale_structure in SCALE_STRUCTURES:
                for time_mode in TIME_MODES:
                    name = f"{variant.name}__{scale_structure}_{time_mode}"
                    if name not in output_variant_names:
                        output_variant_names.append(name)
                    print(f"  {name}", flush=True)
                    torch.manual_seed(exp.stable_seed(args.seed, name, window_index))
                    train_x, train_y, eval_x, _, _, _ = exp.prepare_window_tensors(train_df, eval_df)
                    model = build_model(
                        train_x=train_x,
                        train_y=train_y,
                        variant=variant,
                        time_mode=time_mode,
                        scale_structure=scale_structure,
                    )
                    y_pred, y_std = exp.fit_and_predict(model, eval_x, maxiter=args.maxiter)
                    hyper_rows.extend(
                        collect_hyperparameter_diagnostics(
                            model=model,
                            train_x=train_x,
                            eval_x=eval_x,
                            variant=name,
                            window_date=window_date,
                            window_index=window_index,
                            time_mode=time_mode,
                            scale_structure=scale_structure,
                        )
                    )
                    pred_rows.extend(
                        prediction_records(
                            variant=name,
                            train_df=train_df,
                            eval_df=eval_df,
                            y_pred=y_pred,
                            y_std=y_std,
                        )
                    )

    summary_rows = [
        exp.summarize_variant(variant, [row for row in pred_rows if row["variant"] == variant])
        for variant in output_variant_names
    ]
    summary = pd.DataFrame(summary_rows)
    pd.DataFrame(pred_rows).to_csv(args.output_dir / "window_predictions.csv", index=False)
    pd.DataFrame(hyper_rows).to_csv(args.output_dir / "hyperparameter_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "variant_summary.csv", index=False)
    write_asset_plots(args.output_dir, pred_rows)
    print(summary.to_string(index=False))
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    run(parse_args())
