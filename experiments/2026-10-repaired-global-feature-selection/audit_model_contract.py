"""Inspect actual model construction and training-only transforms without GP fits.

This experiment-only audit loads the exact legacy wrapper and command parser,
constructs unfitted representative models, and checks priors, likelihood and
adversarial evaluation isolation. It never optimizes, samples or writes run files.
Returns/noise are in decimal/standardized-target units as explicitly recorded.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_model_contract.py
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import repair_dataset as data
import run_study as study
import torch
import workflow


def inspect_candidate(config: dict[str, Any], item: dict[str, Any], plan: Path) -> dict[str, Any]:
    """Construct and inspect the exact command-selected model without optimization.

    Args:
        config: Validated immutable study batch configuration.
        item: A registered feature mask in that batch.
        plan: Actual batch path used by the normal command builder.

    Returns:
        Actual module/prior/noise configuration and evaluation-isolation checks.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(data.HERE / "artifacts/matplotlib"))
    spec = importlib.util.spec_from_file_location("repaired_contract_wrapper", study.RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the actual study wrapper")
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    runner = wrapper.load_portfolio_runner()
    paths = study.previous.validate_config(config, plan)
    command = study.previous.build_command(config, paths, item, seed=27, preflight=True)
    outer, remaining = wrapper.parse_wrapper_args(command[2:])
    wrapper.patch_first_round_constants(
        runner, mtgp_rank=outer.mtgp_rank, feature_set=outer.feature_set, tickers=config["tickers"],
        custom_etf_cols=wrapper.parse_optional_column_list(outer.custom_etf_cols),
        custom_macro_cols=wrapper.parse_optional_column_list(outer.custom_macro_cols),
        covar_grouping_proposal=outer.covar_grouping_proposal, macro_component_proposal=outer.macro_component_proposal)
    original_argv = sys.argv
    try:
        sys.argv = [str(wrapper.PORTFOLIO_RUNNER_PATH), "--feature-path", str(outer.feature_path),
                    "--output-dir", str(outer.output_root / outer.run_id), "--run-id", outer.run_id, *remaining]
        args = runner.parse_args()
    finally:
        sys.argv = original_argv
    if args.input_transform_mode != "botorch_normalize" or args.time_modulation_mode != "neither":
        raise ValueError("Actual command does not use the frozen transform/modulation policy")
    frame = pd.read_parquet(paths["feature"])
    first = pd.Timestamp(data.STUDY["first_score"])
    train, evaluation = frame[frame.date.lt(first)].copy(), frame[frame.date.eq(first)].copy()
    tx, ty, ex, yvar, _ = runner.prepare_window_tensors_for_run(train, evaluation, args=args)
    hostile = evaluation.copy()
    hostile[item["expected_input_columns"]] = 1e9
    hostile["y_excess_lead"] = -1e9
    other_tx, other_ty, _, _, _ = runner.prepare_window_tensors_for_run(train, hostile, args=args)
    torch.testing.assert_close(tx, other_tx, rtol=0, atol=0)
    torch.testing.assert_close(ty, other_ty, rtol=0, atol=0)
    torch.manual_seed(27)
    model = runner.build_experiment_model(tx, ty, yvar, args=args)
    transform = model.input_transform
    width = len(item["expected_input_columns"])
    expected_bounds = torch.stack([tx[:, :width].amin(0), tx[:, :width].amax(0)])
    torch.testing.assert_close(transform.bounds, expected_bounds)
    before = transform.bounds.detach().clone()
    target_means = model.outcome_transform.means.detach().clone()
    target_stds = model.outcome_transform.stdvs.detach().clone()
    model.eval()
    distant = ex.clone()
    distant[:, :width] = 1e12
    transform(distant)
    torch.testing.assert_close(transform.bounds, before, rtol=0, atol=0)
    torch.testing.assert_close(model.outcome_transform.means, target_means, rtol=0, atol=0)
    torch.testing.assert_close(model.outcome_transform.stdvs, target_stds, rtol=0, atol=0)
    task_kernel = model.covar_module.kernels[1]
    priors = [prior for _, _, prior, _, _ in task_kernel.named_priors()]
    actual = [prior for prior in priors if type(prior).__name__ == "LKJCovariancePrior"]
    if type(task_kernel).__name__ != "IndexKernel" or len(actual) != 1:
        raise ValueError("Signed IndexKernel with LKJ covariance prior was not actually constructed")
    prior = actual[0]
    assert float(prior.correlation_prior.eta) == 2.0
    assert prior.correlation_prior.n == len(config["tickers"])
    assert float(prior.sd_prior.loc) == 0 and float(prior.sd_prior.scale) == 0.5
    assert task_kernel.covar_factor.shape[-1] == 2
    assert type(model.likelihood).__name__ == "HadamardGaussianLikelihood"
    floor = model.likelihood.noise_covar.raw_noise_constraint.lower_bound
    torch.testing.assert_close(floor, torch.full_like(floor, 0.005))
    saved_run = data.HERE / "runs" / f"{item['id']}_s27"
    saved = pd.read_csv(saved_run / "scenario_cache" / f"model_diagnostics_{first.date()}.csv")
    noise = saved[saved.diagnostic.eq("likelihood_noise")]
    assert len(noise) == 1 and noise.iloc[0].module_type == "HadamardGaussianLikelihood"
    assert noise.iloc[0]["min"] >= 0.005 - 1e-9
    recorded = json.loads((saved_run / "manifest.json").read_text())
    assert recorded["modeling"]["input_transform"] == "BoTorch Normalize on non-task feature columns"
    return {"family": config["family"], "configuration": item["id"], "training_rows": len(train),
            "plan_path": str(plan.relative_to(data.ROOT)), "plan_sha256": data.digest(plan),
            "feature_path": str(paths["feature"].relative_to(data.ROOT)),
            "feature_sha256": data.digest(paths["feature"]),
            "initial_training_months": int(train.date.nunique()), "tasks": model.num_tasks,
            "task_kernel": type(task_kernel).__name__, "task_rank": 2,
            "task_prior": "LKJCovariancePrior(eta=2, sd_prior=LogNormal(0,0.5))",
            "mean_module": type(model.mean_module).__name__,
            "likelihood": type(model.likelihood).__name__,
            "noise_floor_standardized_variance": 0.005,
            "likelihood_noise_prior": "LogNormal(-4,1)",
            "input_transform": type(transform).__name__, "input_bounds_fit_on_training_only": True,
            "outcome_transform": type(model.outcome_transform).__name__,
            "evaluation_features_and_targets_cannot_change_training_tensors": True,
            "evaluation_cannot_update_learned_bounds_or_outcome_scaling": True,
            "completed_seed27_first_window_likelihood_and_transform_match": True,
            "legacy_recorded_prior_label": recorded["modeling"]["task_covar_prior"],
            "covariance_config": runner.task_exp.build_covar_config().model_dump(mode="json"),
            "actual_model_repr": str(model), "optimizer_called": False}


def audit() -> dict[str, Any]:
    """Check each universe's largest completed registered mask without performing a fit.

    Returns:
        A source-hashed construction audit with explicit legacy metadata caveats.
    """
    torch.set_num_threads(1)
    available = workflow.validate_frozen_data()
    hashes = study.code_hashes()
    chosen = {}
    for path in sorted((data.HERE / "plans").glob("*.json")):
        config = json.loads(path.read_text())
        workflow.validate_plan(config, path, available)
        for item in config["configurations"]:
            stamp_path = data.HERE / "runs" / f"{item['id']}_s27/repaired_execution_fingerprint.json"
            if not stamp_path.is_file():
                continue
            stamp = json.loads(stamp_path.read_text())
            if stamp.get("state") != "completed" or stamp.get("signature") != study.signature(
                config, item, 27, hashes
            ):
                continue
            family = config["family"]
            if family not in chosen or len(item["expected_input_columns"]) > len(
                chosen[family][1]["expected_input_columns"]
            ):
                chosen[family] = (config, item, path)
    if set(chosen) != {*data.FAMILIES, "global"}:
        raise ValueError("A completed representative is missing for an approved universe")
    rows = [inspect_candidate(*chosen[family]) for family in sorted(chosen)]
    if hashes != study.code_hashes():
        raise RuntimeError("Execution-critical sources changed during construction audit")
    report = {"passed": True, "constructed_unfitted_models": rows, "optimizer_calls": 0,
              "runtime": {"python": sys.version, "packages": {
                  name: importlib.metadata.version(name) for name in [
                      "torch", "gpytorch", "botorch", "numpy", "pandas", "scipy", "Riskfolio-Lib"]}},
              "source_sha256": {**hashes, str(Path(__file__).resolve().relative_to(data.ROOT)): data.digest(
                  Path(__file__).resolve())},
              "metadata_caveats": [
                  "Legacy manifest modeling.task_covar_prior incorrectly says BetaPrior(2.5,1.5) for signed LKJ. "
                  "Actual construction above uses LKJ eta 2; original manifests and fit sources are preserved.",
                  "Legacy wrapper assigns every task the us_equity label only in its ASSET_GROUPS correlation "
                  "diagnostics. Do not interpret its within/between-group correlations as four-family results. "
                  "Selection reports independently use the approved four-family ticker map."]}
    data.write_json(data.HERE / "results/model_contract_audit.json", report)
    return report


if __name__ == "__main__":
    result = audit()
    print(json.dumps({"passed": result["passed"], "constructed_models": len(result["constructed_unfitted_models"]),
                      "optimizer_calls": result["optimizer_calls"]}))
