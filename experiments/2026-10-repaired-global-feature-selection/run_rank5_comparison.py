"""Run a separately authorized rank-only joint sensitivity test on frozen data.

Research orchestration only: changes rank 2 to 5 for the fourteen-input SPY
lead, with three seeds and all other fit/data/portfolio controls unchanged.
Writes isolated rank5_followup outputs, immutable commands and source hashes.
Decimal excess-return arithmetic is independently reconstructed, never traded.
The completed rank-two campaigns and their validators are not modified.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/run_rank5_comparison.py --execute
    Same command resumes only matching signatures. No additional masks or ranks.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import followup_workflow as followup
import numpy as np
import pandas as pd
import repair_dataset as data
import run_study as study
import workflow
from audit_joint_objectives import reconstruct
from summarize_international_experiments import validate_run_manifest

HERE = data.HERE / "rank5_followup"
BASE_ID = "global_e6c41f2020fd"
IDENTIFIER = BASE_ID + "_rank5"
SEEDS = [27, 127, 227]


def rank5_command(command: list[str]) -> list[str]:
    """Change only rank and isolated output identity on a validated baseline command.

    Args:
        command: Original rank-two command from the frozen runner's command builder.

    Returns:
        New command preserving every non-rank scientific argument.

    Raises:
        ValueError: Baseline command has duplicate/missing controls or wrong rank.
    """
    result = list(command)
    for flag in ["--mtgp-rank", "--output-root", "--run-id"]:
        if result.count(flag) != 1:
            raise ValueError(f"Missing or duplicated baseline flag: {flag}")
    if result[result.index("--mtgp-rank") + 1] != "2":
        raise ValueError("Expected the exact rank-two baseline")
    result[result.index("--mtgp-rank") + 1] = "5"
    result[result.index("--output-root") + 1] = str(HERE / "runs")
    run_index = result.index("--run-id") + 1
    old_id = result[run_index]
    if old_id not in [f"{BASE_ID}_s{seed}" for seed in SEEDS]:
        raise ValueError("Unexpected baseline mask or seed")
    result[run_index] = old_id.replace(BASE_ID, IDENTIFIER)
    return result


def _probe(command: list[str]) -> dict[str, Any]:
    """Build the actual first-window model and inspect the task-factor dimension."""
    spec = importlib.util.spec_from_file_location("rank5_actual_wrapper", study.RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load actual walk-forward wrapper")
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    runner = wrapper.load_portfolio_runner()
    outer, remaining = wrapper.parse_wrapper_args(command[2:])
    wrapper.patch_first_round_constants(
        runner, mtgp_rank=outer.mtgp_rank, feature_set=outer.feature_set, tickers=data.TICKERS,
        custom_etf_cols=wrapper.parse_optional_column_list(outer.custom_etf_cols),
        custom_macro_cols=wrapper.parse_optional_column_list(outer.custom_macro_cols),
        covar_grouping_proposal=outer.covar_grouping_proposal, macro_component_proposal=outer.macro_component_proposal)
    original = sys.argv
    try:
        sys.argv = [str(wrapper.PORTFOLIO_RUNNER_PATH), "--feature-path", str(outer.feature_path),
                    "--output-dir", str(outer.output_root / outer.run_id), "--run-id", outer.run_id, *remaining]
        args = runner.parse_args()
    finally:
        sys.argv = original
    frame = pd.read_parquet(outer.feature_path)
    first = pd.Timestamp(data.STUDY["first_score"])
    tx, ty, _, yvar, _ = runner.prepare_window_tensors_for_run(
        frame[frame.date.lt(first)].copy(), frame[frame.date.eq(first)].copy(), args=args)
    model = runner.build_experiment_model(tx, ty, yvar, args=args)
    task = model.covar_module.kernels[1]
    shape = list(task.covar_factor.shape)
    if shape != [15, 5]:
        raise ValueError(f"Actual task factor is not fifteen tasks by rank five: {shape}")
    priors = [p for _, _, p, _, _ in task.named_priors() if type(p).__name__ == "LKJCovariancePrior"]
    if len(priors) != 1 or float(priors[0].correlation_prior.eta) != 2:
        raise ValueError("Actual signed-LKJ covariance prior changed")
    return {"task_factor_shape": shape, "rank": 5, "task_kernel": type(task).__name__, "lkj_eta": 2}


def _run(command: list[str], log: Path) -> int:
    """Execute one deterministic runner with isolated logs and one thread per worker."""
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1", "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(HERE / "cache/matplotlib"), "XDG_CACHE_HOME": str(HERE / "cache")})
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as out, log.with_suffix(".stderr.log").open("w") as err:
        return subprocess.run(command, cwd=data.ROOT, env=env, stdout=out, stderr=err, check=False).returncode


def _validate(run: Path, item: dict[str, Any], dates: list[str], seed: int) -> dict[str, Any]:
    """Validate all inherited controls, explicitly replacing only the rank-two check."""
    manifest = json.loads((run / "manifest.json").read_text())
    failures = validate_run_manifest(
        IDENTIFIER, item, manifest, dates, data.TICKERS, expected_seed=seed)
    if failures != [f"{IDENTIFIER}: rank"]:
        raise ValueError(f"Unexpected inherited manifest violations: {failures}")
    if manifest["modeling"]["rank"] != 5 or manifest["us_equity_first_round_wrapper"]["mtgp_rank"] != 5:
        raise ValueError("Saved rank-five command was not honored")
    study.verify_training(run, data.TICKERS)
    return reconstruct(run, data.TICKERS, dates)


def execute() -> None:
    """Preregister, preflight, run three seeds concurrently and audit rank sensitivity."""
    followup.validate_base()
    baseline_audit = data.HERE / "followup/results/all_joint/independent_audit.json"
    for relative, expected in json.loads(baseline_audit.read_text())["source_sha256"].items():
        if data.digest(data.ROOT / relative) != expected:
            raise ValueError(f"Baseline evidence changed: {relative}")
    plan_path = data.HERE / "followup/plans/10_global_round1.json"
    config = json.loads(plan_path.read_text())
    workflow.validate_plan(config, plan_path, workflow.validate_frozen_data())
    paths = study.previous.validate_config(config, plan_path)
    item = next(copy.deepcopy(x) for x in config["configurations"] if x["id"] == BASE_ID)
    commands = {str(seed): rank5_command(study.previous.build_command(
        config, paths, item, seed=seed, preflight=False)) for seed in SEEDS}
    sources = {**study.code_hashes(), **{str(p.relative_to(data.ROOT)): data.digest(p) for p in
        [Path(__file__).resolve(), baseline_audit, plan_path, data.HERE / "audit_joint_objectives.py"]}}
    contract = {"base_configuration": BASE_ID, "new_configuration": IDENTIFIER, "rank": 5,
                "only_scientific_change": "task covariance rank 2 -> 5", "seeds": SEEDS,
                "expected_inputs": item["expected_input_columns"], "feature_sha256": data.digest(paths["feature"]),
                "dates_sha256": data.digest(paths["dates"]), "commands": commands, "source_sha256": sources}
    followup.immutable_json(HERE / "contract.json", contract)
    signature = data.digest(HERE / "contract.json")

    def fresh() -> None:
        if any(data.digest(data.ROOT / relative) != digest for relative, digest in sources.items()):
            raise ValueError("Execution-critical sources changed")
        if data.digest(paths["feature"]) != contract["feature_sha256"] or (
            data.digest(paths["dates"]) != contract["dates_sha256"] or
            data.digest(HERE / "contract.json") != signature
        ):
            raise ValueError("Frozen feature/date/contract bytes changed")

    env_defaults = {"MPLBACKEND": "Agg", "MPLCONFIGDIR": str(HERE / "cache/matplotlib")}
    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)
    data.write_json(HERE / "actual_model_probe.json", _probe(commands["27"]))
    log = HERE / "logs/preflight.log"
    if _run([*commands["27"], "--preflight-only"], log):
        raise RuntimeError(f"Rank-five preflight failed: {log}")
    dates = study.previous.read_dates(paths["dates"])
    preflight = json.loads(log.read_text())
    if preflight["scored_dates"] != dates or len(preflight["training_history"]) != 24:
        raise ValueError("Preflight evaluation/training coverage changed")
    status: dict[str, Any] = {"state": "running", "signature": signature, "workers": 3, "runs": {}}
    pending = []
    for seed in SEEDS:
        run = HERE / f"runs/{IDENTIFIER}_s{seed}"
        stamp = run / "rank5_fingerprint.json"
        if run.exists() and not stamp.exists():
            raise ValueError(f"Unfingerprinted output already exists: {run}")
        if stamp.exists():
            saved = json.loads(stamp.read_text())
            if saved["signature"] != signature:
                raise ValueError(f"Conflicting rank-five evidence: {run}")
            if saved["state"] == "completed":
                _validate(run, item, dates, seed)
                status["runs"][str(seed)] = "completed_existing"
                continue
        data.write_json(stamp, {"signature": signature, "state": "running"})
        status["runs"][str(seed)] = "running"
        pending.append(seed)
    data.write_json(HERE / "status.json", status)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(_run, commands[str(seed)], HERE / f"logs/seed{seed}.log"): seed for seed in pending}
        for job in as_completed(jobs):
            seed = jobs[job]
            code = job.result()
            fresh()
            if code:
                status["runs"][str(seed)] = "failed"
                data.write_json(HERE / "status.json", status)
                raise RuntimeError(f"Rank-five seed {seed} exited {code}")
            run = HERE / f"runs/{IDENTIFIER}_s{seed}"
            _validate(run, item, dates, seed)
            data.write_json(run / "rank5_fingerprint.json", {"signature": signature, "state": "completed"})
            status["runs"][str(seed)] = "completed"
            data.write_json(HERE / "status.json", status)
            print(f"Validated rank-five seed {seed}", flush=True)
    summaries, per_etf = [], []
    for rank, prefix in [(2, data.HERE / f"followup/runs/{BASE_ID}"), (5, HERE / f"runs/{IDENTIFIER}")]:
        for seed in SEEDS:
            run = Path(str(prefix) + f"_s{seed}")
            metrics = reconstruct(run, data.TICKERS, dates)
            summaries.append({"rank": rank, "seed": seed, **metrics})
            frame = pd.read_csv(run / "gp_window_predictions.csv")
            frame["nll"] = 0.5 * np.log(2 * np.pi * frame.y_std**2) + 0.5 * (
                (frame.y_true - frame.y_pred) / frame.y_std)**2
            for ticker, panel in frame.groupby("asset_id"):
                base = pd.read_csv(data.HERE / f"followup/runs/{BASE_ID}_s{seed}/gp_window_predictions.csv")
                base = base[base.asset_id.eq(ticker)].sort_values("date")
                np.testing.assert_array_equal(panel.sort_values("date").y_true.to_numpy(), base.y_true.to_numpy())
                per_etf.append({"rank": rank, "seed": seed, "ETF": ticker, "nll": panel.nll.mean(),
                    "rmse": float(np.sqrt(np.square(panel.y_true - panel.y_pred).mean())),
                    "mean_predictive_sd": panel.y_std.mean(), "predicted_time_sd": panel.y_pred.std(ddof=1),
                    "coverage95": ((panel.y_true - panel.y_pred).abs() <= 1.959963984540054 * panel.y_std).mean()})
    fresh()
    data.write_json(HERE / "independent_audit.json", {"passed": True, "seed_runs": 6,
        "contract_sha256": signature, "source_sha256": sources, "checked": summaries,
        "note": "Three frozen rank-two controls and three new rank-five runs; reused adaptive research dates."})
    keys = ["rank", "seed", "mean_ic", "mean_gaussian_nll", "sharpe",
            "cumulative_return_diagnostic_only", "rmse_diagnostic_only"]
    pd.DataFrame(summaries)[keys].to_csv(HERE / "objectives_by_seed.csv", index=False)
    pd.DataFrame(per_etf).to_csv(HERE / "etf_metrics_by_seed.csv", index=False)
    status["state"] = "completed"
    data.write_json(HERE / "status.json", status)
    print(pd.DataFrame(summaries)[keys].groupby("rank").mean().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "dispatcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute()
