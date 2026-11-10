"""Single-thread independent monthly fits for registered GP tuning components.

Research adapter around the frozen experiment model builder and fitter. Only
registered task-kernel rank, fitting cap, feature mask and causal training window
may vary. The global month-index seed remains unchanged, including rolling fits.
Predictions are decimal discounted excess returns; covariance is decimal squared.
Usage: initialized by campaign.py in a shared bounded spawned-process queue.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STATE: dict[str, Any] = {}


def training_panel(frame: pd.DataFrame, date: str, index: int, tickers: list[str],
                   rolling_months: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a complete expanding or fixed trailing panel strictly before origin.

    Args:
        frame: Registered complete monthly task panel with decimal target returns.
        date: Original forecast-origin ISO date.
        index: Original global index in the 28-month evaluation, never renumbered.
        tickers: Exact fifteen-task order, including IWM rather than IJR.
        rolling_months: None for expanding 84+index dates; otherwise fixed count.

    Returns:
        Complete training and evaluation frames without future targets in training.
    """
    if index not in range(28) or rolling_months not in {None, 48, 60, 84}:
        raise ValueError("Unregistered index or rolling window")
    train = frame[frame.date.lt(pd.Timestamp(date)) & frame.y_excess_lead.notna()].copy()
    evaluation = frame[frame.date.eq(pd.Timestamp(date))].copy()
    if train.date.nunique() != 84 + index:
        raise ValueError("Unexpected full-history chronology")
    count = 84 + index if rolling_months is None else rolling_months
    if rolling_months is not None:
        retained = sorted(train.date.unique())[-count:]
        train = train[train.date.isin(retained)].copy()
    counts = train.groupby("asset_id", observed=True).size()
    if len(train) != count * len(tickers) or set(counts.index) != set(tickers) or not counts.eq(count).all():
        raise ValueError("Incomplete causal training panel")
    if len(evaluation) != len(tickers) or set(evaluation.asset_id) != set(tickers):
        raise ValueError("Incomplete evaluation universe")
    return train, evaluation


def initialize(configs: dict, framework: str) -> None:
    """Initialize one worker with immutable model specs and one numerical thread.

    Args:
        configs: Registered model specifications; no worker creates candidates.
        framework: Read-only existing evaluation directory for dense GP readout.
    """
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    sys.path.extend([framework, str(Path(framework) / "parallel_pipeline_v1")])
    STATE.clear()
    STATE.update(configs=configs, component=None)


def _configure(component: str) -> None:
    config = STATE["configs"][component]
    spec = importlib.util.spec_from_file_location("tuning_frozen_wrapper", config["command"][1])
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the frozen GP wrapper")
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    outer, remaining = wrapper.parse_wrapper_args(config["command"][2:])
    runner = wrapper.load_portfolio_runner()
    tickers = outer.tickers.split(",")
    wrapper.patch_first_round_constants(
        runner, mtgp_rank=outer.mtgp_rank, feature_set=outer.feature_set, tickers=tickers,
        custom_etf_cols=wrapper.parse_optional_column_list(outer.custom_etf_cols),
        custom_macro_cols=wrapper.parse_optional_column_list(outer.custom_macro_cols),
        covar_grouping_proposal=outer.covar_grouping_proposal,
        macro_component_proposal=outer.macro_component_proposal,
    )
    original = sys.argv
    try:
        sys.argv = [str(wrapper.PORTFOLIO_RUNNER_PATH), "--feature-path", str(outer.feature_path),
                    "--output-dir", str(outer.output_root / outer.run_id), "--run-id", outer.run_id,
                    "--periods-per-year", "12", "--rebalance-frequency-label", "monthly_bme", *remaining]
        args = runner.parse_args()
    finally:
        sys.argv = original
    if (outer.mtgp_rank, args.maxiter, args.torch_num_threads, args.scenario_workers) != (
        config["rank"], config["maxiter"], 1, 1
    ):
        raise ValueError("Actual numerical controls differ from registration")
    if args.gp_experiment != "signed_lkj_eta_2" or args.time_modulation_mode != "neither":
        raise ValueError("Unregistered kernel/prior/time-modulation change")
    if tickers != config["tickers"] or runner.task_exp.INPUT_COLUMNS != config["inputs"]:
        raise ValueError("Actual task/feature ordering differs")
    frame = runner.task_exp.load_features(args.feature_path)
    if args.drop_incomplete_feature_dates:
        frame = runner.drop_incomplete_feature_dates(frame)
    frame = runner.apply_min_feature_date(frame, args.min_feature_date)
    STATE.update(component=component, runner=runner, args=vars(args), frame=frame)


def fit_job(job: dict) -> dict:
    """Fit one registered model/month/seed and retain full observed covariance.

    Args:
        job: Component, global index, original seed, date and isolated attempt path.

    Returns:
        Fit paths, training coverage, exact-vs-legacy parity and optimizer telemetry.
        Exhausting maxiter is recorded, not misreported as convergence or failure.
    """
    from botorch.fit import fit_gpytorch_mll
    from botorch.optim.fit import fit_gpytorch_mll_scipy
    from workers import analytic_readout

    started = time.monotonic()
    destination = Path(job["run"])
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "started.json").write_text(json.dumps(job, sort_keys=True) + "\n")
    config = STATE["configs"][job["component"]]
    if job["seed"] not in [27, 127, 227] or config["dates"][job["index"]] != job["date"]:
        raise ValueError("Date/seed/index registration mismatch")
    if STATE["component"] != job["component"]:
        _configure(job["component"])
    runner = STATE["runner"]
    args = argparse.Namespace(**STATE["args"])
    args.seed, args.resume_scenarios, args.output_dir = job["seed"], False, Path(job["run"])
    train, evaluation = training_panel(STATE["frame"], job["date"], job["index"],
                                       config["tickers"], config["rolling_months"])
    fit_seed = runner.task_exp.stable_seed(job["seed"], runner.gp_variant_name(args.gp_experiment), job["index"])
    np.random.seed(job["seed"])
    random.seed(job["seed"])
    context, optimizer = {}, []
    original_builder = runner.build_experiment_model

    def build(*values: Any, **keywords: Any) -> Any:
        model = original_builder(*values, **keywords)
        context["model"] = model
        return model

    def fit(mll: Any, **keywords: Any) -> Any:
        def optimize(model: Any, **options: Any) -> Any:
            if model.model.train_inputs[0].shape != (len(train), len(config["inputs"]) + 1):
                raise ValueError("Actual training tensor dimensions changed")
            if list(model.model.covar_module.kernels[1].covar_factor.shape) != [15, config["rank"]]:
                raise ValueError("Actual covariance rank differs")
            if options.get("options", {}).get("maxiter") != config["maxiter"]:
                raise ValueError("Actual fitting budget differs")
            result = fit_gpytorch_mll_scipy(model, **options)
            optimizer.append({"attempt": len(optimizer) + 1, "iterations": result.step,
                              "status": result.status.name, "message": result.message,
                              "runtime_seconds": result.runtime})
            return result
        return fit_gpytorch_mll(mll, optimizer=optimize, **keywords)

    runner.build_experiment_model, runner.fit_gpytorch_mll = build, fit
    try:
        result = runner.fit_gp_window_artifact({
            "args": vars(args), "window_date": job["date"], "window_index": job["index"],
            "train_df": train, "eval_df": evaluation, "seed": fit_seed, "output_dir": job["run"],
            "final_universe": config["tickers"],
        })
    finally:
        runner.build_experiment_model = original_builder
    _, _, eval_x, _, _ = runner.prepare_window_tensors_for_run(train, evaluation, args=args)
    exported = analytic_readout(context["model"], eval_x)
    if evaluation.asset_id.astype(str).tolist() != config["tickers"]:
        raise ValueError("Dense readout task order changed")
    legacy = pd.read_csv(result["paths"]["predictions"])
    difference = float(np.max(np.abs(legacy[["y_pred", "y_std"]].to_numpy()
                                     - np.column_stack([exported["mean"], exported["std"]]))))
    if difference > 1e-8 or not optimizer:
        raise ValueError("Dense readout parity or optimizer telemetry failed")
    folder = Path(job["run"])
    covariance = folder / "analytic.npz"
    np.savez_compressed(covariance, date=np.array(job["date"]), assets=np.array(config["tickers"]),
                        actual=legacy.y_true.to_numpy(), **exported)
    fitted = {name: value.detach().cpu().numpy().tolist()
              for name, value in context["model"].named_parameters()}
    parameters = folder / "fitted_parameters.json"
    parameters.write_text(json.dumps(fitted, sort_keys=True, allow_nan=False) + "\n")
    result["paths"].update(analytic=str(covariance), fitted_parameters=str(parameters))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.update(component=job["component"], seed=job["seed"], fit_seed=fit_seed,
                  global_window_index=job["index"], rank=config["rank"], maxiter=config["maxiter"],
                  rolling_months=config["rolling_months"], train_months=train.date.nunique(),
                  train_start=str(train.date.min().date()), train_end=str(train.date.max().date()),
                  optimizer=optimizer, dense_legacy_max_abs=difference,
                  peak_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024),
                  seconds=time.monotonic() - started)
    context.clear()
    gc.collect()
    return result
