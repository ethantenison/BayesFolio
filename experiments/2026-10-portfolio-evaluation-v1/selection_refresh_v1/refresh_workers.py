"""Spawn-safe cache scoring and generic month-parallel exact GP readouts.

Usage: workers are initialized by refresh.py, with a single shared process
ceiling and one numerical thread. Existing producer/evaluator code is read-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from refresh_math import CommonAllocator, _screen_one

STATE = {}


def initialize_screen(context: dict, anchor: np.ndarray) -> None:
    """Cache the shared portfolio risk problems in each scoring process.

    Args:
        context: Immutable common dates, actuals, RF and historical covariance.
        anchor: Frozen highest-IC family forecasts with axes seed, month, asset.
    """
    STATE.clear()
    STATE.update(context=context, anchor=anchor, allocator=CommonAllocator(context["historical_covariance"]))


def screen_job(candidate: dict) -> dict:
    """Rescore one saved configuration without fitting or changing its forecasts.

    Args:
        candidate: Validated family/joint forecasts for the original three seeds.

    Returns:
        Forecast and common-universe portfolio metrics, with chronological paths.
    """
    return _screen_one(candidate, STATE["context"], STATE["anchor"], STATE["allocator"])


def initialize_fit(configs: dict, legacy_root: str, parallel_root: str) -> None:
    """Load the frozen generic fitter under strict one-thread process controls.

    Args:
        configs: Explicit feature/task/command configurations.
        legacy_root: Read-only original study module directory.
        parallel_root: Validated analytic covariance exporter directory.
    """
    sys.path.extend([legacy_root, parallel_root])
    import ranking_architecture_worker as generic

    generic.initialize(configs)
    STATE.clear()
    STATE.update(generic=generic, configs=configs)


def fit_job(job: dict) -> dict:
    """Fit one independent month and retain exact observed predictive covariance.

    Args:
        job: Original month index, date, seed and isolated component output path.

    Returns:
        Hashed-checkpoint-ready artifact paths, strict controls and parity checks.
    """
    from workers import analytic_readout

    generic = STATE["generic"]
    if generic.STATE["component"] != job["component"]:
        generic._configure(job["component"])
    runner = generic.STATE["runner"]
    captured = {}
    original = runner.build_experiment_model

    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        captured["model"] = model
        return model

    runner.build_experiment_model = build
    try:
        result = generic.fit_job(job)
    finally:
        runner.build_experiment_model = original
    config = STATE["configs"][job["component"]]
    train, evaluation = generic.training_panel(generic.STATE["frame"], job["date"], job["index"], config["tickers"])
    args = argparse.Namespace(**generic.STATE["args"])
    _, _, eval_x, _, _ = runner.prepare_window_tensors_for_run(train, evaluation, args=args)
    exported = analytic_readout(captured["model"], eval_x)
    assets = evaluation.asset_id.astype(str).tolist()
    if assets != config["tickers"]:
        raise ValueError("Exact readout task order differs")
    original_frame = pd.read_csv(result["paths"]["predictions"])
    predictions = original_frame.copy()
    predictions["y_pred"], predictions["y_std"] = exported["mean"], exported["std"]
    predictions["score"] = predictions.y_pred / predictions.y_std
    folder = Path(job["run"]) / "analytic_months"
    folder.mkdir(parents=True, exist_ok=True)
    prediction_path = folder / f"predictions_{job['date']}.csv"
    covariance_path = folder / f"covariance_{job['date']}.npz"
    predictions.to_csv(prediction_path, index=False)
    np.savez(covariance_path, date=np.array(job["date"]), assets=np.array(assets), **exported)
    dense_difference = float(
        np.max(np.abs(predictions[["y_pred", "y_std"]].to_numpy() - original_frame[["y_pred", "y_std"]].to_numpy()))
    )
    if dense_difference > 1e-8:
        raise ValueError("Dense and legacy predictions differ beyond predeclared parity tolerance")
    old_difference = None
    if config.get("old_run_root"):
        old = pd.read_csv(
            Path(config["old_run_root"]) / f"{job['component']}_s{job['seed']}" / "gp_window_predictions.csv"
        )
        old = old[old.date.eq(job["date"])].set_index("asset_id").reindex(assets)
        old_difference = float(
            np.max(
                np.abs(
                    old[["y_true", "y_pred", "y_std"]].to_numpy()
                    - original_frame[["y_true", "y_pred", "y_std"]].to_numpy()
                )
            )
        )
        if not np.isfinite(old_difference) or old_difference > 1e-8:
            raise ValueError(f"Replay forecasts differ from old saved run: {old_difference}")
    result["paths"].update(analytic_predictions=str(prediction_path), analytic_covariance=str(covariance_path))
    return {
        **result,
        "legacy_vs_dense_max_abs": dense_difference,
        "old_replay_max_abs": old_difference,
        "observation_noise": True,
        "dense_cholesky": True,
        "raw_discounted_excess_units": True,
    }
