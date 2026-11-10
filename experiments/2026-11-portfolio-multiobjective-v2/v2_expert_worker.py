"""Single-thread v2 adapter around protected historical GP fitting code.

The root exclusively owns scheduling, fit budgets and retries. This worker fits
one registered component/month/seed, exports all fifteen observed GP moments
(decimal returns, squared-decimal covariance), fitted parameters and telemetry.
It performs no policy scoring or portfolio allocation and modifies no old source.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from v2_common import ASSETS, CHRONO, HERE, SEEDS, TUNING, digest, read
from v2_experts import DATASET, REGISTERED, forecast_dates


def training_panel(frame: pd.DataFrame, date: str, index: int, tickers: list[str],
                   rolling_months: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select complete available labels strictly before one registered origin.

    Args:
        frame: Sealed monthly panel, with unknown November labels left NaN.
        date: Canonical business-month origin May2022 through October2026.
        index: Original May2024-relative index -24 through29.
        tickers: All fifteen exact approved tasks, including MGK.
        rolling_months: None for expanding or48 for the existing broad reference.

    Returns:
        Causal training and evaluation frames with no manufactured live targets.
    """
    dates = forecast_dates()
    if (index not in range(-24, 30) or dates[index + 24] != date or tickers != ASSETS
            or rolling_months not in {None, 48}):
        raise ValueError("Unregistered GP chronology, tasks or training window")
    train = frame[frame.date.lt(pd.Timestamp(date)) & frame.y_excess_lead.notna()].copy()
    evaluation = frame[frame.date.eq(pd.Timestamp(date))].copy()
    if train.date.nunique() != 84 + index:
        raise ValueError("Available-label chronology changed")
    count = rolling_months or 84 + index
    if rolling_months:
        train = train[train.date.isin(sorted(train.date.unique())[-count:])].copy()
    counts = train.groupby("asset_id", observed=True).size()
    if (len(train) != count * 15 or set(counts.index) != set(tickers) or not counts.eq(count).all()
            or not np.isfinite(train.y_excess_lead.to_numpy()).all()
            or len(evaluation) != 15 or set(evaluation.asset_id) != set(tickers)):
        raise ValueError("Incomplete causal fit panel")
    if date == "2026-10-30":
        if not evaluation.y_excess_lead.isna().all():
            raise ValueError("November targets cannot be known in October")
    elif not np.isfinite(evaluation.y_excess_lead.to_numpy()).all():
        raise ValueError("Historical evaluation targets must be complete")
    return train, evaluation


def initialize(configs: dict, framework: str) -> None:
    """Initialize one worker; queue creation remains the root's responsibility.

    Args:
        configs: Frozen references/compact rank1 with original features/windows.
        framework: Existing protected dense-covariance readout directory.
    """
    sys.path.insert(0, str(TUNING))
    import tuning_worker

    dates = forecast_dates()
    originals = read(CHRONO / "CONTRACT.json")["configs"]
    for name, config in configs.items():
        if (name not in REGISTERED or config["maxiter"] != 15
                or config["rank"] != (1 if name == "compact_r1_i15_exp" else 2)
                or config["tickers"] != ASSETS or config["kernel_recipe"] != "original_default"
                or config["task_prior_lkj_eta"] != 2.0
                or config["rolling_months"] != (48 if name == "broad_roll48_r2_i15" else None)):
            raise ValueError("Worker received an unregistered GP modification")
        source = originals[name]
        if config["inputs"] != source["inputs"]:
            raise ValueError("Registered feature mask cannot change")
        for flag in ["--custom-etf-cols", "--custom-macro-cols", "--feature-set", "--covar-grouping-proposal",
                     "--macro-component-proposal", "--time-modulation-mode", "--gp-experiment"]:
            if config["command"][config["command"].index(flag) + 1] != source["command"][
                    source["command"].index(flag) + 1]:
                raise ValueError(f"Unregistered kernel/feature command change: {flag}")
        if Path(config["command"][config["command"].index("--feature-path") + 1]).resolve() != DATASET:
            raise ValueError("Only the sealed v1 October dataset is permitted")
    mapped = {name: {**config, "dates": {dates.index(date) - 24: date for date in config["dates"]}}
              for name, config in configs.items()}
    tuning_worker.initialize(mapped, framework)
    tuning_worker.training_panel = training_panel


def _model_telemetry(model: Any) -> dict:
    """Read actual constructed/fitted kernel properties, without changing them."""
    from gpytorch.kernels import Kernel
    from gpytorch.likelihoods import Likelihood

    nodes, priors = {}, {}
    for name, module in model.named_modules():
        if not name.startswith(("covar_module", "likelihood", "input_transform", "outcome_transform")):
            continue
        item = {"class": type(module).__name__}
        attributes = []
        if isinstance(module, Kernel):
            attributes = ["active_dims", "lengthscale", "outputscale", "variance", "covar_factor"]
        elif isinstance(module, Likelihood):
            attributes = ["noise"]
        for key in attributes:
            value = getattr(module, key, None)
            if value is not None and hasattr(value, "detach"):
                item[key] = value.detach().cpu().numpy().tolist()
        nodes[name] = item
    for name, _module, prior, _closure, _setting in model.named_priors():
        priors[name] = {"class": type(prior).__name__, "state": {
            key: value.detach().cpu().numpy().tolist() for key, value in prior.state_dict().items()}}
    return {"nodes": nodes, "priors": priors}


def fit_job(job: dict) -> dict:
    """Fit one immutable monthly job and add exact prior/kernel provenance.

    Args:
        job: Registered expert/seed/date/index and a unique new v2 attempt path.

    Returns:
        Original analytic moments/parameters and convergence, seed and timing
        telemetry. Maxiter exhaustion remains explicit, without a fallback fit.
    """
    import tuning_worker

    destination = Path(job["run"])
    if not destination.resolve().is_relative_to(HERE) or job["seed"] not in SEEDS:
        raise ValueError("Isolated attempt path and registered seed required")
    if tuning_worker.STATE["component"] != job["component"]:
        tuning_worker._configure(job["component"])
    runner = tuning_worker.STATE["runner"]
    original_builder, captured = runner.build_experiment_model, {}

    def observed_builder(*values: Any, **keywords: Any) -> Any:
        model = original_builder(*values, **keywords)
        captured["model"] = model
        return model

    runner.build_experiment_model = observed_builder
    try:
        result = tuning_worker.fit_job(job)
    finally:
        runner.build_experiment_model = original_builder
    config = tuning_worker.STATE["configs"][job["component"]]
    manifest = destination / "v2_model_spec.json"
    manifest.write_text(json.dumps({
        "component": job["component"], "inputs": config["inputs"], "tasks": ASSETS,
        "rank": config["rank"], "maxiter": 15, "rolling_months": config["rolling_months"],
        "kernel_recipe": config["kernel_recipe"], "task_prior_lkj_eta": 2.0,
        "task_standard_deviation_prior": "LogNormal(0,0.5)",
        "gp_experiment": "signed_lkj_eta_2", "time_modulation_mode": "neither",
        "frozen_worker_sha256": digest(TUNING / "tuning_worker.py"),
        "adapter_sha256": digest(Path(__file__)),
        "original_month_seed_convention_preserved": True,
        "actual_fitted_model": _model_telemetry(captured["model"]),
    }, sort_keys=True, indent=2, allow_nan=False) + "\n")
    result["paths"]["v2_model_spec"] = str(manifest)
    return result
