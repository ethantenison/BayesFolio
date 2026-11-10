"""Increase only rank-five fitting effort from maxiter 15 to 150 on frozen inputs.

Isolated research adapter reuses the completed rank-five command, validation
contracts, actual-model probe and decimal-return arithmetic. Three seeds run in
parallel with per-window observational optimizer diagnostics; old rank-two and
rank-five evidence is read-only. No feature selection, trading or promotion.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/run_rank5_budget150.py --execute
    Repeating this command resumes only matching experiment fingerprints.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import repair_dataset as data
import run_rank5_comparison as prior
import run_study as study
from audit_joint_objectives import reconstruct
from followup_workflow import immutable_json
from summarize_international_experiments import validate_run_manifest

HERE = data.HERE / "rank5_budget150"
IDENTIFIER = prior.IDENTIFIER + "_iter150"
HOOK = data.HERE / "log_fit_budget.py"


def larger_budget_command(command: list[str]) -> list[str]:
    """Change only maxiter and output identity on the completed rank-five command.

    Args:
        command: Frozen rank-five maxiter-fifteen baseline command.

    Returns:
        Uninstrumented rank-five command with maxiter 150 and isolated output paths.

    Raises:
        ValueError: Rank, budget, command identity or flags differ from the baseline.
    """
    result = list(command)
    for flag in ["--mtgp-rank", "--maxiter", "--output-root", "--run-id"]:
        if result.count(flag) != 1:
            raise ValueError(f"Missing or duplicate flag {flag}")
    if result[result.index("--mtgp-rank") + 1] != "5" or result[result.index("--maxiter") + 1] != "15":
        raise ValueError("Expected rank five with maxiter fifteen")
    index = result.index("--run-id") + 1
    if result[index] not in [f"{prior.IDENTIFIER}_s{seed}" for seed in prior.SEEDS]:
        raise ValueError("Unexpected baseline mask or seed")
    result[index] = result[index].replace(prior.IDENTIFIER, IDENTIFIER)
    result[result.index("--maxiter") + 1] = "150"
    result[result.index("--output-root") + 1] = str(HERE / "runs")
    return result


def _validate(run: Path, item: dict[str, Any], dates: list[str], seed: int) -> dict[str, Any]:
    manifest = json.loads((run / "manifest.json").read_text())
    failures = validate_run_manifest(IDENTIFIER, item, manifest, dates, data.TICKERS, expected_seed=seed)
    if failures != [f"{IDENTIFIER}: rank", f"{IDENTIFIER}: maxiter"]:
        raise ValueError(f"Unexpected manifest differences: {failures}")
    if manifest["modeling"]["rank"] != 5 or manifest["modeling"]["maxiter"] != 150 or (
        manifest["us_equity_first_round_wrapper"]["mtgp_rank"] != 5
    ):
        raise ValueError("Actual fitting rank/budget changed")
    study.verify_training(run, data.TICKERS)
    diagnostics = pd.read_json(run / "fitting_diagnostics.jsonl", lines=True)
    returned = diagnostics[diagnostics.event.eq("optimizer_return")]
    if set(returned.date.astype(str)) != set(dates) or not returned.maxiter.eq(150).all():
        raise ValueError("Missing per-window fitting diagnostics")
    return reconstruct(run, data.TICKERS, dates)


def execute() -> None:
    """Run/resume all three rank-five larger-budget seeds and compare frozen controls."""
    baseline = json.loads((prior.HERE / "contract.json").read_text())
    verification = json.loads((prior.HERE / "RESULTS_VERIFICATION.json").read_text())
    for relative, expected in {**baseline["source_sha256"], **verification["artifact_sha256"]}.items():
        if data.digest(data.ROOT / relative) != expected:
            raise ValueError(f"Completed rank-five baseline changed: {relative}")
    sources = {**baseline["source_sha256"], **verification["artifact_sha256"]}
    critical = [Path(__file__).resolve(), HOOK, prior.HERE / "contract.json", prior.HERE / "RESULTS_VERIFICATION.json"]
    for path in critical:
        sources[str(path.relative_to(data.ROOT))] = data.digest(path)
    commands = {str(seed): larger_budget_command(baseline["commands"][str(seed)]) for seed in prior.SEEDS}
    contract = {"rank": 5, "maxiter": 150, "baseline_maxiter": 15, "seeds": prior.SEEDS,
                "only_scientific_change": "maxiter 15 -> 150; same rank five, inputs, prior, dates and scenarios",
                "expected_inputs": baseline["expected_inputs"], "commands": commands, "source_sha256": sources,
                "feature_sha256": baseline["feature_sha256"], "dates_sha256": baseline["dates_sha256"]}
    immutable_json(HERE / "contract.json", contract)
    signature = data.digest(HERE / "contract.json")
    plan = json.loads((data.HERE / "followup/plans/10_global_round1.json").read_text())
    item = next(x for x in plan["configurations"] if x["id"] == prior.BASE_ID)
    dates_path = data.ROOT / plan["scored_date_file"]
    feature_path = data.ROOT / plan["feature_path"]
    dates = dates_path.read_text().splitlines()

    def fresh() -> None:
        if any(data.digest(data.ROOT / rel) != digest for rel, digest in sources.items()) or (
            data.digest(dates_path) != contract["dates_sha256"] or
            data.digest(feature_path) != contract["feature_sha256"] or data.digest(HERE / "contract.json") != signature
        ):
            raise ValueError("Frozen rank-five comparison source/data/contract changed")

    fresh()
    data.write_json(HERE / "actual_model_probe.json", prior._probe(commands["27"]))
    log = HERE / "logs/preflight.log"
    if prior._run([*commands["27"], "--preflight-only"], log):
        raise RuntimeError(f"Larger-budget preflight failed: {log}")
    if json.loads(log.read_text())["scored_dates"] != dates:
        raise ValueError("Preflight scored dates changed")
    status: dict[str, Any] = {"state": "running", "workers": 3, "signature": signature, "runs": {}}
    pending = []
    for seed in prior.SEEDS:
        run = HERE / f"runs/{IDENTIFIER}_s{seed}"
        stamp = run / "budget_fingerprint.json"
        if run.exists() and not stamp.exists():
            raise ValueError(f"Unfingerprinted output exists: {run}")
        if stamp.exists():
            saved = json.loads(stamp.read_text())
            if saved["signature"] != signature:
                raise ValueError("Conflicting budget-run fingerprint")
            if saved["state"] == "completed":
                _validate(run, item, dates, seed)
                status["runs"][str(seed)] = "completed_existing"
                continue
        data.write_json(stamp, {"signature": signature, "state": "running"})
        status["runs"][str(seed)] = "running"
        pending.append(seed)
    data.write_json(HERE / "status.json", status)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(prior._run, [sys.executable, str(HOOK), *commands[str(seed)][1:]],
                            HERE / f"logs/seed{seed}.log"): seed for seed in pending}
        for job in as_completed(jobs):
            seed, code = jobs[job], job.result()
            fresh()
            if code:
                status["runs"][str(seed)] = "failed"
                data.write_json(HERE / "status.json", status)
                raise RuntimeError(f"Larger-budget seed {seed} failed with exit {code}")
            run = HERE / f"runs/{IDENTIFIER}_s{seed}"
            _validate(run, item, dates, seed)
            data.write_json(run / "budget_fingerprint.json", {"signature": signature, "state": "completed"})
            status["runs"][str(seed)] = "completed"
            data.write_json(HERE / "status.json", status)
            print(f"Validated rank-five maxiter150 seed {seed}", flush=True)
    objectives, etfs, diag = [], [], []
    for budget, prefix in [(15, prior.HERE / f"runs/{prior.IDENTIFIER}"), (150, HERE / f"runs/{IDENTIFIER}")]:
        for seed in prior.SEEDS:
            run = Path(str(prefix) + f"_s{seed}")
            metrics = reconstruct(run, data.TICKERS, dates)
            objectives.append({"maxiter": budget, "seed": seed, **metrics})
            frame = pd.read_csv(run / "gp_window_predictions.csv")
            base = pd.read_csv(prior.HERE / f"runs/{prior.IDENTIFIER}_s{seed}/gp_window_predictions.csv")
            np.testing.assert_array_equal(frame[["date", "asset_id", "y_true"]].to_numpy(),
                                          base[["date", "asset_id", "y_true"]].to_numpy())
            for ticker, panel in frame.groupby("asset_id"):
                error = panel.y_true - panel.y_pred
                etfs.append({"maxiter": budget, "seed": seed, "ETF": ticker,
                    "nll": (0.5 * np.log(2 * np.pi * panel.y_std**2) + 0.5 * (error / panel.y_std)**2).mean(),
                    "rmse": float(np.sqrt(np.square(error).mean())), "mean_predictive_sd": panel.y_std.mean(),
                    "predicted_time_sd": panel.y_pred.std(ddof=1)})
            if budget == 150:
                rows = pd.read_json(run / "fitting_diagnostics.jsonl", lines=True)
                rows.insert(0, "seed_run", seed)
                diag.append(rows)
    fresh()
    columns = ["maxiter", "seed", "mean_ic", "mean_gaussian_nll", "sharpe",
               "cumulative_return_diagnostic_only", "rmse_diagnostic_only"]
    pd.DataFrame(objectives)[columns].to_csv(HERE / "objectives_by_seed.csv", index=False)
    pd.DataFrame(etfs).to_csv(HERE / "etf_metrics_by_seed.csv", index=False)
    pd.concat(diag, ignore_index=True).to_csv(HERE / "fitting_diagnostics.csv", index=False)
    data.write_json(HERE / "independent_audit.json", {"passed": True, "checked": objectives,
        "seed_runs": 6, "source_sha256": sources, "contract_sha256": signature,
        "boundary": "Budget sensitivity within rank five; no matched-budget rank-two conclusion."})
    status["state"] = "completed"
    data.write_json(HERE / "status.json", status)
    print(pd.DataFrame(objectives)[columns].groupby("maxiter").mean().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "dispatcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute()
