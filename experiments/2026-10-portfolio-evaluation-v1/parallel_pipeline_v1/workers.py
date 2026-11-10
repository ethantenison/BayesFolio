"""Independent monthly fit and allocation workers for the isolated GP pipeline.

The frozen rank-two initializer and monthly fit function are reused unchanged.
The added readout uses dense Cholesky Gaussian conditioning with LOVE disabled,
after saving legacy predictions/scenarios. Financial arrays use raw decimal
discounted-excess returns; covariance includes observation noise in decimal^2.
"""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STATE: dict[str, Any] = {}


def initialize(config: dict) -> None:
    """Initialize one spawned process with frozen model controls and shared context.

    Args:
        config: Registered baseline command, raw inputs, dates, assets and causal
            historical covariance; one torch/BLAS thread per spawned process.
    """
    legacy = config["legacy_root"]
    if legacy not in sys.path:
        sys.path.append(legacy)
    import month_parallel_worker
    import run_hy_rank2_iter15

    # Only the validated initializer is reused, never that module's HY variant,
    # execution function, output locations or feature specification.
    run_hy_rank2_iter15.initialize_worker(config)
    runner = month_parallel_worker.STATE["runner"]
    captured: dict[str, Any] = {}
    original = runner.build_experiment_model

    def build(*args: Any, **kwargs: Any) -> Any:
        model = original(*args, **kwargs)
        captured["model"] = model
        return model

    runner.build_experiment_model = build
    STATE.clear()
    STATE.update(config=config, legacy_worker=month_parallel_worker, runner=runner, captured=captured)


def analytic_readout(model: Any, eval_x: Any) -> dict[str, np.ndarray]:
    """Extract complete observed Gaussian covariance without sampling or refitting.

    Args:
        model: Fitted BoTorch model, including trained outcome/input transforms.
        eval_x: Original untransformed test inputs including explicit task indices.

    Returns:
        Raw mean/SD, full observed covariance and raw likelihood-noise diagonal.
        Exact here means dense conditioning at fitted hyperparameters, not
        integration over their posterior or a claim of exact financial truth.
    """
    import gpytorch
    import torch

    # Clear only prediction caches. Do not call train(), change parameters, or
    # apply normalization again: the fitted model owns those transforms.
    model.prediction_strategy = None
    with (torch.no_grad(), gpytorch.settings.fast_pred_var(False),
          gpytorch.settings.max_cholesky_size(100000),
          gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False)):
        observed = model.posterior(eval_x, observation_noise=True)
        mean = observed.mean.squeeze(-1).detach().cpu().numpy().copy()
        covariance = observed.distribution.covariance_matrix.detach().cpu().numpy().copy()
        latent = model.posterior(eval_x, observation_noise=False)
        latent_covariance = latent.distribution.covariance_matrix.detach().cpu().numpy()
        std = observed.variance.squeeze(-1).sqrt().detach().cpu().numpy().copy()
    delta = covariance - latent_covariance
    noise = np.diag(delta).copy()
    if not (noise > 0).all() or not np.allclose(delta, np.diag(noise), rtol=0, atol=1e-11):
        raise ValueError("Observed covariance does not contain the expected diagonal likelihood noise")
    if not np.allclose(np.diag(covariance), std**2, rtol=1e-9, atol=1e-13):
        raise ValueError("Observed covariance diagonal/SD mismatch")
    np.linalg.cholesky(covariance)  # Fail rather than repair the exported density.
    return {"mean": mean, "std": std, "covariance": covariance, "likelihood_noise_diagonal": noise}


def fit_job(job: dict) -> dict:
    """Fit one original seed/month and then export its dense predictive readout.

    Args:
        job: Immutable original seed/index/date and new isolated run directory.

    Returns:
        Original fit artifacts plus analytic predictions/covariance, timing, and
        numerical differences from the legacy fast-variance readout.
    """
    worker, runner = STATE["legacy_worker"], STATE["runner"]
    STATE["captured"].clear()
    result = worker.fit_job(job)
    model = STATE["captured"].pop("model")
    import argparse

    args = argparse.Namespace(**worker.STATE["args"])
    train, evaluation = worker.training_panel(worker.STATE["frame"], job["date"], job["index"])
    _, _, eval_x, _, _ = runner.prepare_window_tensors_for_run(train, evaluation, args=args)
    started = time.perf_counter()
    exported = analytic_readout(model, eval_x)
    dense_seconds = time.perf_counter() - started
    assets = evaluation.asset_id.astype(str).tolist()
    if assets != STATE["config"]["tickers"]:
        raise ValueError("Analytic task ordering differs from registration")
    original = pd.read_csv(result["paths"]["predictions"])
    predictions = original.copy()
    predictions["y_pred"] = exported["mean"]
    predictions["y_std"] = exported["std"]
    predictions["score"] = exported["mean"] / exported["std"]
    folder = Path(job["run"]) / "analytic_months"
    folder.mkdir(parents=True, exist_ok=True)
    prediction_path = folder / f"predictions_{job['date']}.csv"
    covariance_path = folder / f"covariance_{job['date']}.npz"
    predictions.to_csv(prediction_path, index=False)
    np.savez(covariance_path, date=np.array(job["date"]), assets=np.array(assets), **exported)
    result["paths"].update(analytic_predictions=str(prediction_path), analytic_covariance=str(covariance_path))
    log_path = Path(job["run"]) / "optimizer_logs" / f"{job['date']}.jsonl"
    result["paths"]["optimizer_log"] = str(log_path)
    result.update(analytic_readout_seconds=dense_seconds,
        legacy_vs_analytic_mean_max_abs=float(np.max(np.abs(original.y_pred.to_numpy() - exported["mean"]))),
        legacy_vs_analytic_std_max_abs=float(np.max(np.abs(original.y_std.to_numpy() - exported["std"]))),
        dense_cholesky=True, fast_pred_var=False, observation_noise=True,
        likelihood_noise_min=float(exported["likelihood_noise_diagonal"].min()),
        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return result


def solve_weights(mean: np.ndarray, covariance: np.ndarray, policy: str) -> tuple[np.ndarray, dict]:
    """Solve one strict long-only portfolio without past weights or relaxation.

    Args:
        mean: Shape (15,), expected returns or scenario sample means in decimals.
        covariance: Training-only common covariance in decimal squared.
        policy: legacy_scenario_maxret, risk_rank_cap35 or risk_rank_no_cap.

    Returns:
        Raw solver weights and constraint diagnostics. Risk rank policies use a
        research variance budget equal to the common equal-weight variance.
    """
    import cvxpy as cp
    from scipy.stats import rankdata

    if (mean.shape != (15,) or covariance.shape != (15, 15) or not np.isfinite(mean).all()
            or not np.isfinite(covariance).all()):
        raise ValueError("Strict allocator needs the complete finite 15-ETF inputs")
    np.linalg.cholesky(covariance)
    equal = np.ones(15) / 15
    reference = float(equal @ covariance @ equal)
    variable = cp.Variable(15)
    cap = 1.0 if policy == "risk_rank_no_cap" else 0.35
    constraints = [variable >= 0, cp.sum(variable) == 1, cp.sum_squares(variable) <= 1 / 3]
    if cap < 1:
        constraints.append(variable <= cap)
    if policy == "legacy_scenario_maxret":
        objective = cp.Maximize(mean @ variable)
    elif policy in {"risk_rank_cap35", "risk_rank_no_cap"}:
        constraints.append(cp.quad_form(variable, cp.psd_wrap(covariance / reference)) <= 1)
        objective = cp.Maximize((rankdata(mean) / 15) @ variable - 1e-8 * cp.sum_squares(variable))
    else:
        raise ValueError(f"Unregistered allocation policy: {policy}")
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
    if problem.status != cp.OPTIMAL or variable.value is None:
        raise RuntimeError(f"Strict allocation failed: {problem.status}; no fallback")
    weights = np.asarray(variable.value).reshape(-1)
    variance_ratio = float(weights @ covariance @ weights / reference)
    nea = float(1 / (weights @ weights))
    if (not np.isfinite(weights).all() or weights.min() < -1e-8 or weights.max() > cap + 1e-7
            or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-7) or nea < 3 - 1e-6):
        raise RuntimeError("Invalid strict allocation; no clipping or renormalization")
    if policy.startswith("risk_") and variance_ratio > 1 + 1e-6:
        raise RuntimeError("Variance budget violated; no relaxation")
    return weights, {"status": problem.status, "cap": cap, "effective_assets": nea,
                     "variance_ratio_to_equal_weight": variance_ratio, "largest_weight": float(weights.max()),
                     "fallback_used": False, "relaxations_used": False}


def portfolio_job(job: dict) -> dict:
    """Solve independent monthly allocations only after the full fit stage.

    Args:
        job: Same original date/index/seed and run directory as the completed fit.

    Returns:
        Separate saved weight/constraint artifacts for every registered policy.
    """
    started = time.perf_counter()
    config = STATE["config"]
    run = Path(job["run"])
    forecasts = pd.read_csv(run / "analytic_months" / f"predictions_{job['date']}.csv")
    expected = forecasts.set_index("asset_id").y_pred.reindex(config["tickers"]).to_numpy()
    scenarios = pd.read_csv(run / f"gp_scenarios_{job['date']}.csv")[config["tickers"]].to_numpy()
    common = np.asarray(config["historical_covariance"])[job["index"]]
    rows, diagnostics = [], []
    for policy in config["allocation_policies"]:
        objective_mean = scenarios.mean(axis=0) if policy == "legacy_scenario_maxret" else expected
        weights, evidence = solve_weights(objective_mean, common, policy)
        diagnostics.append({"policy": policy, "date": job["date"], **evidence})
        rows.extend({"date": job["date"], "policy": policy, "asset_id": asset, "weight": float(weight)}
                    for asset, weight in zip(config["tickers"], weights, strict=True))
    folder = run / "allocation_months"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"weights_{job['date']}.csv"
    status_path = folder / f"diagnostics_{job['date']}.json"
    pd.DataFrame(rows).to_csv(path, index=False)
    status_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    return {"seed": job["seed"], "date": job["date"], "seconds": time.perf_counter() - started,
            "paths": {"weights": str(path), "diagnostics": str(status_path)},
            "fallback_used": False, "constraint_relaxations": False}


def hash_files(paths: list[Path]) -> dict[str, str]:
    """Fingerprint exact new artifacts for checkpoint validation on resume.

    Args:
        paths: Explicit files produced by one monthly job.

    Returns:
        Absolute artifact paths and SHA-256 digests.
    """
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
