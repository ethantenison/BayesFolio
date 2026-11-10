"""Spawn-safe monthly GP and strict portfolio workers for frozen ETF research.

Each process reconstructs the real wrapper's feature/kernel configuration and
reads the immutable panel once. Jobs are keyed by original seed/month index;
only rows strictly before that date train the GP. All returns remain decimals.
Portfolio failures raise: no previous weights, equal-weight substitutions or
constraint relaxation. Explicit benchmark strategies are not failure fallbacks.
This module never changes the production runner or historical experiment files.
"""

from __future__ import annotations

import argparse
import importlib.util
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


def initialize_worker(config: dict[str, Any]) -> None:
    """Initialize the actual model configuration explicitly in a spawned process.

    Args:
        config: Frozen command, input list, dates, data hash and output directory.
    """
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    spec = importlib.util.spec_from_file_location("month_parallel_real_wrapper", config["command"][1])
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the real feature wrapper")
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
    if outer.mtgp_rank != 5 or args.maxiter != 36 or args.torch_num_threads != 1 or args.scenario_workers != 1:
        raise ValueError("Rank, iteration cap or per-worker thread controls changed")
    if args.gp_experiment != "signed_lkj_eta_2" or args.time_modulation_mode != "neither":
        raise ValueError("Signed prior or time policy changed")
    if runner.task_exp.INPUT_COLUMNS != config["inputs"] or tickers != config["tickers"]:
        raise ValueError("Spawned worker did not reconstruct the exact feature/task specification")
    frame = runner.task_exp.load_features(args.feature_path)
    if args.drop_incomplete_feature_dates:
        frame = runner.drop_incomplete_feature_dates(frame)
    frame = runner.apply_min_feature_date(frame, args.min_feature_date)
    from botorch.fit import fit_gpytorch_mll
    from log_fit_budget36 import observed_optimizer

    context: dict[str, Any] = {}

    def fit(mll: Any, **kwargs: Any) -> Any:
        if mll.model.train_inputs[0].shape[-1] != len(config["inputs"]) + 1:
            raise ValueError("Actual model tensor width differs from the registered inputs plus task index")
        optimizer = observed_optimizer(context["log"], config["dates"], context["seed"])
        return fit_gpytorch_mll(mll, optimizer=optimizer, **kwargs)

    runner.fit_gpytorch_mll = fit
    STATE.clear()
    STATE.update(config=config, runner=runner, args=vars(args), frame=frame, context=context)


def training_panel(frame: pd.DataFrame, date: str, window_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the exact expanding train/evaluation slices without future rows.

    Args:
        frame: Frozen monthly panel with decimal excess-return labels.
        date: Forecast-origin date, at business month-end.
        window_index: Original index within the complete 24-date scored grid.

    Returns:
        Training and evaluation frames with 84+index training dates per task.

    Raises:
        ValueError: Any training count, date cutoff or evaluation task count differs.
    """
    point = pd.Timestamp(date)
    train = frame[frame.date.lt(point) & frame.y_excess_lead.notna()].copy()
    evaluation = frame[frame.date.eq(point)].copy()
    if train.date.nunique() != 84 + window_index or len(train) != (84 + window_index) * 15:
        raise ValueError("Training coverage differs from 84+k complete task-months")
    if len(evaluation) != 15 or evaluation.asset_id.nunique() != 15 or not train.date.lt(point).all():
        raise ValueError("Evaluation coverage or as-of cutoff differs")
    return train, evaluation


def fit_job(job: dict[str, Any]) -> dict[str, Any]:
    """Fit one independent seed/month using its original deterministic window seed.

    Args:
        job: Base seed, original date/index and isolated run directory.

    Returns:
        Timing, actual seed, training counts, artifact paths and worker RSS telemetry.
    """
    started = time.perf_counter()
    runner, frame, config = STATE["runner"], STATE["frame"], STATE["config"]
    args = argparse.Namespace(**STATE["args"])
    args.seed, args.resume_scenarios = job["seed"], False
    args.output_dir = Path(job["run"])
    train, evaluation = training_panel(frame, job["date"], job["index"])
    seed = runner.task_exp.stable_seed(job["seed"], runner.gp_variant_name(args.gp_experiment), job["index"])
    np.random.seed(job["seed"])
    random.seed(job["seed"])
    STATE["context"].update(log=Path(job["run"]) / "optimizer_logs" / f"{job['date']}.jsonl", seed=job["seed"])
    result = runner.fit_gp_window_artifact({"args": vars(args), "window_date": job["date"],
        "window_index": job["index"], "train_df": train, "eval_df": evaluation, "seed": seed,
        "output_dir": job["run"], "final_universe": config["tickers"]})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {**result, "seed": job["seed"], "window_index": job["index"], "fit_seed": seed,
            "train_dates": train.date.nunique(), "train_rows": len(train), "pid": os.getpid(),
            "peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
            "seconds": time.perf_counter() - started, "input_columns": config["inputs"]}


def strict_solve(solver: Any, panel: pd.DataFrame, **kwargs: Any) -> pd.Series:
    """Solve the requested constraints once; failure never produces replacement weights.

    Args:
        solver: The real deterministic Riskfolio solver, injectable for guard tests.
        panel: Decimal return scenarios or trailing historical returns.
        **kwargs: Exact requested objective, risk measure and allocation constraints.

    Returns:
        Valid nonnegative fully invested solution on the original asset columns.

    Raises:
        RuntimeError: Any missing input, solver failure or invalid output weights.
    """
    if not np.isfinite(panel.to_numpy()).all() or panel.shape[1] != 15:
        raise RuntimeError("Invalid portfolio inputs; dropping assets is not permitted")
    try:
        weights = solver(panel, **kwargs)
    except Exception as error:
        raise RuntimeError(f"Strict portfolio solve failed; no fallback used: {error}") from error
    if set(weights.index) != set(panel.columns) or not np.isfinite(weights.to_numpy()).all() or (
        weights.lt(-1e-9).any() or not np.isclose(weights.sum(), 1, atol=1e-7, rtol=0)
    ):
        raise RuntimeError("Invalid portfolio weights; no fallback used")
    return weights.reindex(panel.columns)


def portfolio_job(job: dict[str, Any]) -> dict[str, Any]:
    """Compute independent monthly portfolios without previous-month state.

    Args:
        job: Base seed, original date/index and completed forecast run directory.

    Returns:
        Paths, timing and explicit primary-solver status; raises on solver failure.
    """
    started = time.perf_counter()
    runner, config = STATE["runner"], STATE["config"]
    args = argparse.Namespace(**STATE["args"])
    args.output_dir = Path(job["run"])
    np.random.seed(job["seed"])
    train, evaluation = training_panel(STATE["frame"], job["date"], job["index"])
    paths = runner.scenario_artifact_paths(Path(job["run"]), pd.Timestamp(job["date"]))
    scenarios = pd.read_csv(paths["scenarios"])[config["tickers"]]
    historical = train.pivot(index="date", columns="asset_id", values="y_excess_lead").reindex(
        columns=config["tickers"])
    common = {"upperlng": args.upperlng, "nea": min(args.nea, len(config["tickers"]) - 1)}
    hist = strict_solve(runner.solve_riskfolio, historical, method_mu=args.historical_method_mu,
                        method_cov=args.historical_method_cov, **common)
    gp = strict_solve(runner.solve_riskfolio, scenarios, method_mu="hist", method_cov="hist",
                      rm=args.gp_riskfolio_rm, obj=args.gp_riskfolio_obj, **common)
    portfolios = {"gp_scenarios_riskfolio": gp, runner.historical_strategy_name(args): hist,
                  "equal_weight": runner.equal_weight(config["tickers"]),
                  runner.SCHWAB_MODERATE_AGGRESSIVE_STRATEGY:
                      runner.schwab_moderate_aggressive_weights(config["tickers"])}
    actual = evaluation.set_index("asset_id").y_excess_lead.reindex(config["tickers"])
    weights, returns = [], []
    for strategy, allocation in portfolios.items():
        returns.append({"date": job["date"], "strategy": strategy,
                        "return": runner.realized_return(allocation, actual)})
        for ticker, weight in allocation.reindex([*config["tickers"], runner.CASH_ASSET]).fillna(0).items():
            weights.append({"date": job["date"], "strategy": strategy, "asset_id": ticker, "weight": weight})
    folder = Path(job["run"]) / "portfolio_cache"
    folder.mkdir(parents=True, exist_ok=True)
    weight_path, return_path = folder / f"weights_{job['date']}.csv", folder / f"returns_{job['date']}.csv"
    pd.DataFrame(weights).to_csv(weight_path, index=False)
    pd.DataFrame(returns).to_csv(return_path, index=False)
    return {"date": job["date"], "seed": job["seed"], "status": "solved", "fallback_used": False,
            "constraint_relaxation_used": False, "seconds": time.perf_counter() - started,
            "pid": os.getpid(), "paths": {"weights": str(weight_path), "returns": str(return_path)}}
