"""Spawn-safe matched-input family/regional GP workers for ranking research.

Uses the frozen production experiment's model builder and fitting routine, not
a replacement GP. Only the task universe changes. One thread per process; all
month/seed jobs are independent and write isolated, resumable artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STATE: dict[str, Any] = {}


def initialize(configs: dict[str, Any]) -> None:
    """Set process-local numerical controls and registered component specs.

    Args:
        configs: Component configurations from the immutable experiment contract.
    """
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    STATE.clear()
    STATE.update(configs=configs, component=None)


def _configure(component: str) -> None:
    from botorch.fit import fit_gpytorch_mll
    from botorch.optim.fit import fit_gpytorch_mll_scipy

    config = STATE["configs"][component]
    spec = importlib.util.spec_from_file_location("ranking_real_wrapper", config["command"][1])
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen wrapper")
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
        macro_component_proposal=outer.macro_component_proposal)
    original = sys.argv
    try:
        sys.argv = [str(wrapper.PORTFOLIO_RUNNER_PATH), "--feature-path", str(outer.feature_path),
                    "--output-dir", str(outer.output_root / outer.run_id), "--run-id", outer.run_id,
                    "--periods-per-year", "12", "--rebalance-frequency-label", "monthly_bme", *remaining]
        args = runner.parse_args()
    finally:
        sys.argv = original
    if (outer.mtgp_rank, args.maxiter, args.torch_num_threads, args.scenario_workers) != (2, 15, 1, 1):
        raise ValueError("Registered numerical controls changed")
    if args.gp_experiment != "signed_lkj_eta_2" or args.time_modulation_mode != "neither":
        raise ValueError("Kernel/prior policy changed")
    if tickers != config["tickers"] or runner.task_exp.INPUT_COLUMNS != config["inputs"]:
        raise ValueError("Actual model inputs/tasks differ from registration")
    context: dict[str, Any] = {}

    def fit(mll: Any, **kwargs: Any) -> Any:
        if mll.model.train_inputs[0].shape[-1] != len(config["inputs"]) + 1:
            raise ValueError("Actual tensor width differs")
        attempts = 0

        def optimize(model: Any, **options: Any) -> Any:
            nonlocal attempts
            rows = model.model.train_inputs[0].shape[0]
            tasks = len(tickers)
            index = rows // tasks - 84
            if rows % tasks or not 0 <= index < 24:
                raise ValueError("Unexpected training panel")
            if list(model.model.covar_module.kernels[1].covar_factor.shape) != [tasks, 2]:
                raise ValueError("Actual task rank differs")
            if options.get("options", {}).get("maxiter") != 15:
                raise ValueError("Actual optimizer cap differs")
            attempts += 1
            result = fit_gpytorch_mll_scipy(model, **options)
            payload = {"date": config["dates"][index], "seed": context["seed"], "attempt": attempts,
                       "train_rows": rows, "tasks": tasks, "rank": 2, "maxiter": 15,
                       "iterations": result.step, "status": result.status.name, "message": result.message,
                       "runtime_seconds": result.runtime}
            with context["log"].open("a") as handle:
                handle.write(json.dumps(payload) + "\n")
            return result

        return fit_gpytorch_mll(mll, optimizer=optimize, **kwargs)

    runner.fit_gpytorch_mll = fit
    frame = runner.task_exp.load_features(args.feature_path)
    if args.drop_incomplete_feature_dates:
        frame = runner.drop_incomplete_feature_dates(frame)
    frame = runner.apply_min_feature_date(frame, args.min_feature_date)
    STATE.update(component=component, runner=runner, args=vars(args), frame=frame, context=context)


def training_panel(frame: pd.DataFrame, date: str, index: int, tickers: list[str]) -> tuple:
    """Select complete expanding windows strictly before the forecast origin.

    Args:
        frame: Frozen panel, already restricted to registered tasks.
        date: Original monthly forecast-origin date.
        index: Original month index, never renumbered on resume.
        tickers: Exact training/evaluation task universe.

    Returns:
        Training and evaluation data with exactly 84+index dates per task.
    """
    point = pd.Timestamp(date)
    train = frame[frame.date.lt(point) & frame.y_excess_lead.notna()].copy()
    evaluation = frame[frame.date.eq(point)].copy()
    if train.date.nunique() != 84 + index or len(train) != (84 + index) * len(tickers):
        raise ValueError("Incomplete expanding training window")
    counts = train.groupby("asset_id", observed=True).size()
    if set(counts.index) != set(tickers) or not counts.eq(84 + index).all():
        raise ValueError("Training task coverage differs")
    if len(evaluation) != len(tickers) or set(evaluation.asset_id) != set(tickers):
        raise ValueError("Incomplete evaluation task coverage")
    return train, evaluation


def fit_job(job: dict[str, Any]) -> dict[str, Any]:
    """Fit one component/month/seed with the original deterministic window seed.

    Args:
        job: Component identity, original seed/month index and isolated output path.

    Returns:
        Saved artifact paths, numerical controls, training coverage and telemetry.
    """
    started = time.perf_counter()
    if STATE["component"] != job["component"]:
        _configure(job["component"])
    config = STATE["configs"][job["component"]]
    runner = STATE["runner"]
    args = argparse.Namespace(**STATE["args"])
    args.seed, args.resume_scenarios, args.output_dir = job["seed"], False, Path(job["run"])
    train, evaluation = training_panel(STATE["frame"], job["date"], job["index"], config["tickers"])
    seed = runner.task_exp.stable_seed(job["seed"], runner.gp_variant_name(args.gp_experiment), job["index"])
    np.random.seed(job["seed"])
    random.seed(job["seed"])
    log = Path(job["run"]) / "optimizer_logs" / f"{job['date']}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    STATE["context"].update(log=log, seed=job["seed"])
    result = runner.fit_gp_window_artifact({"args": vars(args), "window_date": job["date"],
        "window_index": job["index"], "train_df": train, "eval_df": evaluation, "seed": seed,
        "output_dir": job["run"], "final_universe": config["tickers"]})
    result["paths"]["optimizer_log"] = str(log)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {**result, "component": job["component"], "seed": job["seed"], "window_index": job["index"],
            "fit_seed": seed, "train_dates": train.date.nunique(), "train_rows": len(train),
            "task_count": len(config["tickers"]), "rank": 2, "maxiter": 15, "pid": os.getpid(),
            "peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
            "seconds": time.perf_counter() - started}
