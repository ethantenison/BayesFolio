"""Spawn-safe workers for strict portfolio policies and matched-task GP fits.

Usage: initialized only by run_study.py under one numerical thread per process.
All writes target this new study. Existing predictors and producers are read-only;
no portfolio optimizer from the legacy fitting command is invoked. Raw return
units are discounted-excess decimals and covariance is decimal squared.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import study_common as common
from study_common import ROOT, SEEDS, _hash, _save
from study_math import ScoreAllocator, _fixed, _risk, _score, _scores

STATE = {}


def initialize_portfolios(context: dict) -> None:
    """Store shared evaluation context and reusable historical-risk allocators.

    Args:
        context: Chronological targets, historical estimates and variance budgets.
    """
    STATE.clear()
    STATE.update(context=context, historical={})


def portfolio_job(case: dict) -> dict:
    """Evaluate one registered policy across every original seed and month.

    Args:
        case: Model, risk source, count floor, variance multiplier and objective.

    Returns:
        Complete-case aggregate metrics and a hashed detailed portfolio artifact.
    """
    context = STATE["context"]
    path = ROOT / "panels" / f"{case['configuration']}.npz"
    source_hash = _hash(path)
    outcomes = []
    with np.load(path, allow_pickle=False) as values:
        for offset, seed in enumerate(SEEDS):
            mean = values["mean"][offset]
            native = values["covariance"][offset] if "covariance" in values else None
            matrix, factors = _risk(
                case["risk_source"], context["actual"], mean, native, context["historical_covariance"]
            )
            scores, penalty = _scores(mean, context["baseline_mean"], case["objective"])
            budget = context["risk_budget"] * case["budget_factor"]
            if case["risk_source"] == "historical":
                key = (case["minimum"], case["budget_factor"], penalty)
                if key not in STATE["historical"]:
                    STATE["historical"][key] = ScoreAllocator(matrix, budget, case["minimum"], penalty)
                allocator = STATE["historical"][key]
            else:
                allocator = ScoreAllocator(matrix, budget, case["minimum"], penalty)
            scored = _score(context, mean, matrix, allocator.allocate(scores))
            outcomes.append(
                {
                    "seed": seed,
                    **scored,
                    "prequential_variance_scale": factors.tolist(),
                    "fixed_reference_scores": _fixed(
                        context, case["configuration"], case["risk_source"], seed, mean, matrix
                    ),
                }
            )
    if _hash(path) != source_hash:
        raise ValueError("Forecast panel changed while evaluating")
    complete = all(row["solved_months"] == 24 for row in outcomes)
    summary = {
        **case,
        "complete_seeds": sum(row["solved_months"] == 24 for row in outcomes),
        "solved_months": sum(row["solved_months"] for row in outcomes),
        "ETF_position_cap": None,
        "CVaR": False,
        "fallback_weights": False,
    }
    for metric in [
        "conventional_sharpe",
        "compounded_total_return",
        "annualized_excess_vol",
        "max_drawdown_total_nav",
        "mean_effective_count",
        "largest_weight",
    ]:
        numbers = [row["performance"][metric] for row in outcomes] if complete else []
        summary[metric] = float(np.mean(numbers)) if complete else None
        summary[f"seed_sd_{metric}"] = float(np.std(numbers, ddof=1)) if complete else None
    for metric in ["error_rms_to_predicted_sd_rms", "coverage_95", "mean_nll", "mean_crps"]:
        summary[f"adaptive_{metric}"] = (
            float(np.mean([r["adaptive_calibration"][metric] for r in outcomes])) if complete else None
        )
    artifact = ROOT / "portfolio_cases" / f"{case['case_id']}.json"
    _save(
        artifact,
        {
            "case": case,
            "summary": summary,
            "seeds": outcomes,
            "source_sha256": {str(path): source_hash},
            "calibration_role": "adaptive_scores_not_like_for_like_across_different_weights",
        },
        immutable=True,
    )
    return {
        "summary": summary,
        "artifact_sha256": {str(artifact): _hash(artifact)},
        "source_sha256": {str(path): source_hash},
    }


def initialize_fits(configs: dict) -> None:
    """Initialize the existing exact-readout fitter under unchanged model controls.

    Args:
        configs: Existing masks registered to fixed task groups and new outputs.
    """
    import refresh_workers

    refresh_workers.initialize_fit(configs, str(common.LEGACY), str(common.FRAMEWORK / "parallel_pipeline_v1"))


def fit_job(job: dict) -> dict:
    """Fit one independent month, exporting exact covariance and fitted parameters.

    Args:
        job: Registered component, original seed/index/date and isolated run path.

    Returns:
        Existing fitter metadata plus diagnostic parameter and replay evidence.
    """
    import refresh_workers

    generic = refresh_workers.STATE["generic"]
    if generic.STATE["component"] != job["component"]:
        generic._configure(job["component"])
    runner = generic.STATE["runner"]
    original = runner.build_experiment_model
    capture = {}

    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        capture["model"] = model
        return model

    runner.build_experiment_model = build
    try:
        result = refresh_workers.fit_job(job)
    finally:
        runner.build_experiment_model = original
    model = capture["model"]
    parameters = {name: tensor.detach().cpu().numpy().tolist() for name, tensor in model.named_parameters()}
    transformed = {}
    for path, module in model.named_modules():
        for attribute in ["lengthscale", "outputscale", "noise", "means", "stdvs"]:
            value = getattr(module, attribute, None)
            if hasattr(value, "detach"):
                array = value.detach().cpu().numpy()
                if not np.isfinite(array).all():
                    raise ValueError("Nonfinite fitted diagnostic")
                transformed[f"{path}.{attribute}"] = array.tolist()
    diagnostic = Path(job["run"]) / "fitted_parameters" / f"{job['date']}.json"
    _save(
        diagnostic,
        {
            "raw_parameters": parameters,
            "transformed_parameters": transformed,
            "interpretation": "diagnostics_only_not_new_optimizer_steps_or_verified_convergence",
        },
    )
    result["paths"]["fitted_parameters"] = str(diagnostic)
    config = generic.STATE["configs"][job["component"]]
    reference = config.get("reference_component")
    if reference:
        with np.load(ROOT / "components" / f"{reference}.npz", allow_pickle=False) as old:
            with np.load(result["paths"]["analytic_covariance"], allow_pickle=False) as current:
                old_mean = old["mean"][SEEDS.index(job["seed"]), job["index"]]
                difference = float(np.max(np.abs(old_mean - current["mean"])))
                if difference > 1e-8:
                    raise ValueError("Audit replay differs from protected forecast")
                result["audit_replay_mean_max_abs"] = difference
    return result
