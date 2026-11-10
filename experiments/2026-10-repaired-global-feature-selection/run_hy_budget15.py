"""Run the frozen HY-change joint model with a 15-iteration cap, then compare SPY.

Experiment-only adapter: reuses the frozen monthly fit/artifact/strict-portfolio
workers and checkpoint scheduler, without modifying their source or old results.
Only maxiter changes from the completed HY-change/36 run. Returns are decimals;
plots display percentages. Three seeds, 24 dates, rank five, one thread per worker.

Usage:
    MPLCONFIGDIR=/private/tmp/bayesfolio-mpl .venv/bin/python \
        experiments/2026-10-repaired-global-feature-selection/run_hy_budget15.py --execute --workers 14
    Add --report-only to verify/rebuild the comparison without fitting.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import month_parallel_worker as worker
import numpy as np
import pandas as pd
import run_month_parallel as pipeline

ROOT = pipeline.data.HERE / "month_parallel_hy_iter15"
REFERENCE = pipeline.data.HERE / "month_parallel"
SEEDS = pipeline.SEEDS


def budget15_variant(reference: dict[str, Any]) -> dict[str, Any]:
    """Copy the exact HY-change specification, changing only its fitting cap.

    Args:
        reference: Frozen completed month-parallel experiment contract.

    Returns:
        Independent variant with identical inputs and maxiter fifteen.
    """
    variant = copy.deepcopy(reference["variants"]["plus_hy_change"])
    index = variant["command"].index("--maxiter") + 1
    if variant["command"][index] != "36" or "hy_spread_chg_1p" not in variant["inputs"]:
        raise ValueError("Expected the completed HY-change/36 configuration")
    variant["command"][index] = "15"
    return variant


def initialize_worker(config: dict[str, Any]) -> None:
    """Reconstruct the frozen wrapper with the new cap and observational logging.

    Args:
        config: Exact registered command, inputs, task universe and scored dates.
    """
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.optim.fit import fit_gpytorch_mll_scipy

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    spec = importlib.util.spec_from_file_location("hy_budget15_real_wrapper", config["command"][1])
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the frozen real wrapper")
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
    if (outer.mtgp_rank, args.maxiter, args.torch_num_threads, args.scenario_workers) != (5, 15, 1, 1):
        raise ValueError("Unexpected rank, iteration cap or thread settings")
    if args.gp_experiment != "signed_lkj_eta_2" or args.time_modulation_mode != "neither":
        raise ValueError("Kernel/prior configuration changed")
    if runner.task_exp.INPUT_COLUMNS != config["inputs"] or tickers != config["tickers"]:
        raise ValueError("Actual spawned task/input specification differs")
    frame = runner.task_exp.load_features(args.feature_path)
    if args.drop_incomplete_feature_dates:
        frame = runner.drop_incomplete_feature_dates(frame)
    frame = runner.apply_min_feature_date(frame, args.min_feature_date)
    context: dict[str, Any] = {}

    def fit(mll: Any, **kwargs: Any) -> Any:
        if mll.model.train_inputs[0].shape[-1] != len(config["inputs"]) + 1:
            raise ValueError("Actual tensor input width differs")
        attempts = 0

        def optimize(model: Any, **options: Any) -> Any:
            nonlocal attempts
            rows = model.model.train_inputs[0].shape[0]
            index = rows // 15 - 84
            if rows % 15 or not 0 <= index < len(config["dates"]):
                raise ValueError("Unexpected training panel")
            if list(model.model.covar_module.kernels[1].covar_factor.shape) != [15, 5]:
                raise ValueError("Unexpected task rank")
            if options.get("options", {}).get("maxiter") != 15 or options.get("callback") is not None:
                raise ValueError("Actual optimizer cap differs from fifteen")
            attempts += 1
            result = fit_gpytorch_mll_scipy(model, **options)
            payload = {"date": config["dates"][index], "seed": context["seed"], "attempt": attempts,
                       "train_rows": rows, "maxiter": 15, "event": "optimizer_return",
                       "iterations": result.step, "status": result.status.name,
                       "message": result.message, "runtime_seconds": result.runtime}
            log = context["log"]
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as handle:
                handle.write(json.dumps(payload) + "\n")
            return result

        return fit_gpytorch_mll(mll, optimizer=optimize, **kwargs)

    runner.fit_gpytorch_mll = fit
    worker.STATE.clear()
    worker.STATE.update(config=config, runner=runner, args=vars(args), frame=frame, context=context)


def _verify_previous() -> dict[str, Any]:
    contract = json.loads((REFERENCE / "contract.json").read_text())
    pipeline._fresh(contract)
    seal = json.loads((REFERENCE / "RESULTS_VERIFICATION.json").read_text())
    for name, digest in seal["artifact_sha256"].items():
        if pipeline.data.digest(pipeline.data.ROOT / name) != digest:
            raise ValueError(f"Previous publication changed: {name}")
    return contract


def execute(requested_workers: int) -> None:
    """Execute seventy-two fresh monthly fits and strict independent portfolios.

    Args:
        requested_workers: Global pool ceiling, bounded by CPU and observed RSS.
    """
    previous = _verify_previous()
    seal = json.loads((REFERENCE / "RESULTS_VERIFICATION.json").read_text())
    sources = {**previous["source_sha256"], **seal["artifact_sha256"],
               **pipeline._hashes([Path(__file__).resolve(), REFERENCE / "RESULTS_VERIFICATION.json"])}
    contract = {key: previous[key] for key in ["feature_path", "date_path", "feature_sha256", "date_sha256",
                                              "dates", "tickers", "seeds"]}
    contract.update(source_sha256=sources, variants={"hy_iter15": budget15_variant(previous)},
                    rank=5, maxiter=15, previous_weights_fallback=False, constraint_relaxations=False,
                    requested_workers=requested_workers, only_scientific_change="maxiter: 36 -> 15")
    pipeline.immutable_json(ROOT / "contract.json", contract)
    pipeline._fresh(contract)
    if (ROOT / "completed.json").exists():
        report()
        return
    frame = pd.read_parquet(contract["feature_path"])
    if not np.isfinite(frame[contract["variants"]["hy_iter15"]["inputs"]].to_numpy()).all():
        raise ValueError("Incomplete frozen input coverage")
    rss = json.loads((REFERENCE / "plus_hy_change/timing.json").read_text())["worker_peak_rss_bytes"]
    workers = pipeline.worker_limit(requested_workers, os.cpu_count() or 16, rss)
    config = copy.deepcopy(contract["variants"]["hy_iter15"])
    signature = pipeline.data.digest(ROOT / "contract.json")
    config.update(dates=contract["dates"], tickers=contract["tickers"], signature=signature)
    # The imported scheduler writes only beneath this new root; old source stays frozen.
    pipeline.HERE = ROOT
    jobs = pipeline._jobs("hy_iter15", contract["dates"])
    started = time.perf_counter()
    pipeline.data.write_json(ROOT / "status.json", {"state": "running", "stage": "fit", "completed_jobs": 0,
                                                   "required_jobs": 72, "workers": workers})
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize_worker, initargs=(config,)) as pool:
        fits, fresh_fits, fit_wall = pipeline._stage(pool, "hy_iter15", "fit", jobs, signature)
        pipeline._fresh(contract)
        portfolios, fresh_portfolios, portfolio_wall = pipeline._stage(
            pool, "hy_iter15", "portfolio", jobs, signature)
    for seed in SEEDS:
        run = ROOT / "hy_iter15/runs" / f"s{seed}"
        selected = [row for row in fits if row["seed"] == seed]
        pd.concat([pd.read_csv(row["paths"]["predictions"]) for row in selected], ignore_index=True).to_csv(
            run / "gp_window_predictions.csv", index=False)
        for key, name in [("weights", "portfolio_weights.csv"), ("returns", "portfolio_returns.csv")]:
            pd.concat([pd.read_csv(row["paths"][key]) for row in portfolios if row["seed"] == seed],
                      ignore_index=True).to_csv(run / name, index=False)
        pd.DataFrame([{"date": row["date"], "unique_train_dates": row["train_dates"],
                       "train_rows": row["train_rows"]} for row in selected]).to_csv(
            run / "window_training_history.csv", index=False)
        pipeline.study.verify_training(run, contract["tickers"])
        pipeline.data.write_json(run / "manifest.json", {"rank": 5, "maxiter": 15, "seed": seed,
            "input_columns": config["inputs"], "dates": contract["dates"], "tickers": contract["tickers"],
            "source_signature": signature, "previous_weights_fallback": False, "constraint_relaxations": False})
    pipeline._fresh(contract)
    timing = {"workers": workers, "fresh_gp_jobs": fresh_fits, "fresh_portfolio_jobs": fresh_portfolios,
              "fit_wall_seconds": fit_wall, "portfolio_wall_seconds": portfolio_wall,
              "total_wall_seconds": time.perf_counter() - started, "cold_run": fresh_fits == 72,
              "worker_peak_rss_bytes": max(row["peak_rss_bytes"] for row in fits)}
    pipeline.data.write_json(ROOT / "timing.json", timing)
    paths = [ROOT / "timing.json", *ROOT.glob("hy_iter15/checkpoints/*.json"),
             *ROOT.glob("hy_iter15/runs/s*/*.csv"), *ROOT.glob("hy_iter15/runs/s*/manifest.json"),
             *ROOT.glob("hy_iter15/runs/s*/optimizer_logs/*.jsonl")]
    pipeline.data.write_json(ROOT / "completed.json", {"signature": signature,
                                                       "artifact_sha256": pipeline._hashes(paths)})
    report()


def report() -> None:
    """Verify both budgets, reconstruct all objectives, and plot unsmoothed SPY forecasts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    _verify_previous()
    contract = json.loads((ROOT / "contract.json").read_text())
    pipeline._fresh(contract)
    completed = json.loads((ROOT / "completed.json").read_text())
    assert completed["signature"] == pipeline.data.digest(ROOT / "contract.json")
    for name, digest in completed["artifact_sha256"].items():
        assert pipeline.data.digest(pipeline.data.ROOT / name) == digest, name
    for stamp in ROOT.glob("hy_iter15/checkpoints/*.json"):
        for name, digest in json.loads(stamp.read_text())["artifact_sha256"].items():
            assert pipeline.data.digest(pipeline.data.ROOT / name) == digest, name
    objectives, etfs, panels = [], [], {}
    for budget, folder in [(15, ROOT / "hy_iter15/runs"), (36, REFERENCE / "plus_hy_change/runs")]:
        for seed in SEEDS:
            run = folder / f"s{seed}"
            metrics = pipeline.reconstruct(run, contract["tickers"], contract["dates"])
            objectives.append({"maxiter": budget, "seed": seed, **{key: metrics[key] for key in
                ["mean_ic", "mean_gaussian_nll", "sharpe", "cumulative_return_diagnostic_only",
                 "rmse_diagnostic_only"]}})
            frame = pd.read_csv(run / "gp_window_predictions.csv", parse_dates=["date"])
            for ticker, panel in frame.groupby("asset_id"):
                error = panel.y_true - panel.y_pred
                etfs.append({"maxiter": budget, "seed": seed, "ETF": ticker,
                    "nll": (0.5 * np.log(2 * np.pi * panel.y_std**2) + 0.5 * (error / panel.y_std)**2).mean(),
                    "rmse": np.sqrt(np.square(error).mean()), "predicted_time_sd": panel.y_pred.std(ddof=1)})
            panels[budget, seed] = frame[frame.asset_id.eq("SPY")].sort_values("date")
    objective_rows, etf_rows = pd.DataFrame(objectives), pd.DataFrame(etfs)
    objective_rows.to_csv(ROOT / "objectives_by_seed.csv", index=False)
    etf_rows.to_csv(ROOT / "etf_metrics_by_seed.csv", index=False)
    means = objective_rows.drop(columns="seed").groupby("maxiter").mean()
    spy = etf_rows[etf_rows.ETF.eq("SPY")].drop(columns=["seed", "ETF"]).groupby("maxiter").mean()
    delta = objective_rows[objective_rows.maxiter.eq(15)].set_index("seed").drop(columns="maxiter") - (
        objective_rows[objective_rows.maxiter.eq(36)].set_index("seed").drop(columns="maxiter"))
    summary = {"completed": True, "rank": 5, "only_change": "maxiter 36 -> 15; HY change retained",
        "new_gp_windows": 72, "seed_runs": 3, "strict_portfolios_solved": True,
        "timing": json.loads((ROOT / "timing.json").read_text()),
        "previous_timing": json.loads((REFERENCE / "plus_hy_change/timing.json").read_text()),
        "means": means.to_dict("index"), "spy_means": spy.to_dict("index"),
        "paired_deltas_15_minus_36": delta.mean().to_dict(),
        "seed_wins_15": {key: int((delta[key] * sign > 0).sum()) for key, sign in
                         [("mean_ic", 1), ("mean_gaussian_nll", -1), ("sharpe", 1)]},
        "boundary": "Adaptive reused-date research; capped fits, not convergence or holdout confirmation"}
    pipeline.data.write_json(ROOT / "comparison.json", summary)
    values = np.concatenate([p[["y_true", "y_pred"]].to_numpy().ravel() for p in panels.values()])
    low, high = min(0, values.min()), max(0, values.max())
    pad = max((high - low) * 0.12, 0.001)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.4), sharex=True, sharey=True)
    for axis, seed in zip(axes, SEEDS, strict=True):
        current, old = panels[15, seed], panels[36, seed]
        np.testing.assert_array_equal(current[["date", "y_true"]].to_numpy(), old[["date", "y_true"]].to_numpy())
        axis.plot(old.date, old.y_pred, color="#72767c", linestyle="--", linewidth=1.8, label="HY change | 36 iter")
        axis.plot(current.date, current.y_true, color="#1764c0", marker="o", markersize=3.5,
                  linewidth=1.8, label="Actual excess return")
        axis.plot(current.date, current.y_pred, color="#d62728", marker="o", markersize=3.5,
                  linewidth=2, label="HY change | 15 iter")
        axis.axhline(0, color="#808b98", linestyle="--", linewidth=0.8)
        axis.set_ylim(low - pad, high + pad)
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        axis.set_ylabel("Excess return")
        axis.set_title(f"Seed {seed}", loc="left", fontsize=11)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.907),
               frameon=False, ncol=3, fontsize=10)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].set_xlabel("Forecast date (month-end; return target is the following month)")
    fig.suptitle("SPY | Joint model with HY spread change: 15 versus 36 iterations", fontsize=15, y=0.98)
    fig.text(0.5, 0.935, "Rank 5 | Same inputs, data, seeds and portfolio constraints | Only fitting cap changed",
             ha="center", fontsize=10, color="#465160")
    fig.text(0.5, 0.018, "24 forecasts per seed; shared y-axis scale. No smoothing or seed averaging.",
             ha="center", fontsize=9, color="#465160")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92), h_pad=1)
    figure = ROOT / "SPY_hy_iter15_vs36.png"
    fig.savefig(figure, dpi=160, facecolor="white")
    plt.close(fig)
    pipeline.data.write_json(ROOT / "RESULTS_VERIFICATION.json", {"passed": True,
        "artifact_sha256": pipeline._hashes([ROOT / "completed.json", ROOT / "comparison.json", figure,
                                             ROOT / "objectives_by_seed.csv", ROOT / "etf_metrics_by_seed.csv"]),
        "new_fit_windows": 72, "seed_runs": 3, "independent_reconstruction": True})
    pipeline.data.write_json(ROOT / "status.json", {"state": "completed", "comparison": "comparison.json"})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--report-only", action="store_true")
    options = parser.parse_args()
    os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "VECLIB_MAXIMUM_THREADS": "1", "MPLBACKEND": "Agg",
                       "MPLCONFIGDIR": "/private/tmp/bayesfolio-mpl"})
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "dispatcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if options.report_only:
            report()
        else:
            execute(options.workers)
