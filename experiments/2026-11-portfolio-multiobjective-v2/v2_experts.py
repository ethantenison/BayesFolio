"""Registered GP expert evidence and queue jobs for the isolated v2 decision.

This composition adapter reuses protected monthly moment caches and prepares
jobs for the root-owned single fit pool. It never launches workers or scores
policies. Means are decimal excess returns; observed covariance is decimal
squared. All fifteen tasks remain in every expert, including allocation-ineligible
MGK. Only compact expanding rank1 is registered beyond the three references.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from v2_common import ASSETS, CHRONO, COMPONENTS, HERE, SEEDS, TUNING, V1, arrays, digest, read, save, verify_hashes

REGISTERED = [*COMPONENTS, "compact_r1_i15_exp"]
DATASET = V1 / "data/dataset/global.parquet"


def forecast_dates() -> list[str]:
    """Return exact business-month origins May2022 through October2026.

    Returns:
        Fifty-four ISO origins; the final November-return target is unknown.
    """
    return [*read(CHRONO / "CONTRACT.json")["configs"][COMPONENTS[0]]["dates"], "2026-09-30", "2026-10-30"]


def registered_configs(names: list[str] | None = None, dates: list[str] | None = None,
                       output_root: Path | None = None) -> dict:
    """Prepare exact immutable specifications without starting any fit.

    Args:
        names: Registered references and/or compact rank1; defaults to all four.
        dates: Ordered subset of canonical origins; defaults to all54.
        output_root: New attempt directory under this v2 folder only.

    Returns:
        Fixed command specifications compatible with the existing fit adapter.
    """
    selected = REGISTERED if names is None else names
    canonical = forecast_dates()
    dates = canonical if dates is None else dates
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(REGISTERED):
        raise ValueError("Only the registered three references and compact rank1 are allowed")
    if not dates or len(set(dates)) != len(dates) or dates != [date for date in canonical if date in dates]:
        raise ValueError("Forecast dates must be an ordered unique canonical subset")
    output_root = HERE / "outputs/fits" if output_root is None else output_root
    if not output_root.resolve().is_relative_to(HERE):
        raise ValueError("Fit outputs must remain inside the new v2 folder")
    seal = read(V1 / "data/COMPLETED.json")
    if not seal["passed"]:
        raise ValueError("A passed sealed October dataset is required")
    verify_hashes(seal["artifact_sha256"])
    date_file = HERE / "expert_dates.txt"
    save(date_file, ("\n".join(canonical) + "\n").encode())
    source = read(CHRONO / "CONTRACT.json")["configs"]
    result = {}
    for name in selected:
        config = copy.deepcopy(source[name])
        config.update(dates=dates, reuse=name, kernel_recipe="original_default", task_prior_lkj_eta=2.0)
        command = config["command"]
        replacements = {"--feature-path": str(DATASET), "--output-root": str(output_root),
                        "--run-id": "v2_" + name, "--scored-date-file": str(date_file),
                        "--gp-riskfolio-rm": "MV", "--gp-riskfolio-obj": "Sharpe", "--nea": "7"}
        for flag, value in replacements.items():
            command[command.index(flag) + 1] = value
        if (config["maxiter"] != 15 or config["tickers"] != ASSETS or "t_index" not in config["inputs"]
                or set(config["inputs"]) & {"hy_spread", "hy_spread_chg_1p"}
                or config["rank"] != (1 if name == "compact_r1_i15_exp" else 2)
                or config["rolling_months"] != (48 if name == "broad_roll48_r2_i15" else None)):
            raise ValueError("Registered feature/task/rank/fitting controls changed")
        result[name] = config
    return result


def _checked_cache(path: Path) -> dict:
    values = arrays(path)
    dates = values["dates"].tolist()
    if (values["assets"].tolist() != ASSETS or values["seeds"].tolist() != SEEDS
            or len(dates) != len(set(dates)) or set(dates) - set(forecast_dates())
            or values["mean"].shape != (3, len(dates), 15)
            or values["covariance"].shape != (3, len(dates), 15, 15)
            or not np.isfinite(values["mean"]).all() or not np.isfinite(values["covariance"]).all()):
        raise ValueError(f"Incompatible cached GP moments: {path}")
    np.testing.assert_allclose(values["covariance"], values["covariance"].swapaxes(-1, -2), rtol=0, atol=1e-12)
    np.linalg.cholesky(values["covariance"])
    return values


def checkpoint_path(job: dict) -> Path:
    """Return the v2-only completion-record path for one model/seed/origin.

    Args:
        job: Exact component, seed and origin keys.

    Returns:
        Root-owned immutable JSON checkpoint destination.
    """
    return HERE / "outputs/fit_checkpoints" / f"{job['component']}_s{job['seed']}_{job['date']}.json"


def load_cached_panel(name: str, dates: list[str] | None = None) -> dict:
    """Assemble protected historical/recent moments and validated v2 completions.

    Args:
        name: One registered expert ID.
        dates: Requested canonical origins, default all54.

    Returns:
        Explicit mean/covariance/date/task/seed axes, availability mask and source
        hashes. Missing fits alone contain NaNs and cannot be scored as complete.
    """
    if name not in REGISTERED:
        raise ValueError("Unregistered expert")
    canonical = forecast_dates()
    dates = canonical if dates is None else dates
    if not dates or len(set(dates)) != len(dates) or dates != [date for date in canonical if date in dates]:
        raise ValueError("Invalid canonical date subset")
    mean = np.full((3, len(dates), 15), np.nan)
    covariance = np.full((3, len(dates), 15, 15), np.nan)
    available = np.zeros((3, len(dates)), dtype=bool)
    paths = [CHRONO / "components" / f"{name}.npz"]
    if name in COMPONENTS:
        paths.append(V1 / "outputs/recent_components" / f"{name}.npz")
    hashes = {}
    for path in paths:
        values = _checked_cache(path)
        hashes[str(path)] = digest(path)
        for i, date in enumerate(values["dates"].tolist()):
            if date not in dates:
                continue
            j = dates.index(date)
            if available[:, j].any():
                raise ValueError("Overlapping cached origin sources")
            mean[:, j], covariance[:, j], available[:, j] = values["mean"][:, i], values["covariance"][:, i], True
    config = None
    for j, date in enumerate(dates):
        for s, seed in enumerate(SEEDS):
            job = {"component": name, "date": date, "seed": seed, "index": canonical.index(date) - 24}
            path = checkpoint_path(job)
            if not path.exists():
                continue
            record = read(path)
            if config is None:
                config = registered_configs([name])[name]
            validate_fit(record, job, config)
            exported = arrays(Path(record["result"]["paths"]["analytic"]))
            if available[s, j]:
                np.testing.assert_allclose(mean[s, j], exported["mean"], rtol=0, atol=1e-10)
                np.testing.assert_allclose(covariance[s, j], exported["covariance"], rtol=0, atol=1e-10)
            mean[s, j], covariance[s, j], available[s, j] = exported["mean"], exported["covariance"], True
            hashes[str(path)] = digest(path)
            hashes.update(record["artifact_sha256"])
    return {"mean": mean, "covariance": covariance, "available": available, "dates": np.array(dates),
            "assets": np.array(ASSETS), "seeds": np.array(SEEDS), "source_sha256": hashes}


def cache_inventory(names: list[str] | None = None) -> dict:
    """Audit cache coverage, SPD covariance and prior28 exact cache compatibility.

    Args:
        names: Selected registered experts, default all four.

    Returns:
        Coverage/hash evidence without scoring or inspecting portfolio outcomes.
    """
    result = {}
    for name in REGISTERED if names is None else names:
        panel = load_cached_panel(name)
        old_path = TUNING / "outputs/components" / f"{name}.npz"
        old = _checked_cache(old_path)
        for key in ["mean", "covariance"]:
            np.testing.assert_array_equal(panel[key][:, 24:52], old[key])
        available = panel["available"]
        result[name] = {"cached_month_seed_cells": int(available.sum()),
                        "cached_complete_origins": int(available.all(axis=0).sum()),
                        "required_month_seed_cells": 3 * 54, "missing_fits": int((~available).sum()),
                        "prior28_bitwise_compatible": True, "all_cached_covariances_spd": True,
                        "source_sha256": {**panel["source_sha256"], str(old_path): digest(old_path)}}
    return result


def missing_jobs(configs: dict) -> list[dict]:
    """Return deterministic missing jobs for admission to the shared root pool.

    Args:
        configs: Output of registered_configs; no retries or submission occur here.

    Returns:
        Component/seed/date/original-index records lacking validated moments.
    """
    canonical, jobs = forecast_dates(), []
    for name, config in configs.items():
        panel = load_cached_panel(name, config["dates"])
        for i, date in enumerate(config["dates"]):
            for s, seed in enumerate(SEEDS):
                if not panel["available"][s, i]:
                    jobs.append({"component": name, "seed": seed, "date": date,
                                 "index": canonical.index(date) - 24, "audit": False})
    return jobs


def validate_fit(record: dict, job: dict, config: dict) -> None:
    """Reject invalid completed monthly outputs before they enter a policy panel.

    Args:
        record: Root checkpoint containing result and artifact SHA256 mapping.
        job: Registered component, seed, original month index and origin.
        config: Exact expert specification used by the shared worker.
    """
    verify_hashes(record["artifact_sha256"])
    result = record["result"]
    if any(result[key] != job[key] for key in ["component", "seed"]):
        raise ValueError("Checkpoint expert/seed mismatch")
    if (result["rank"], result["maxiter"], result["train_months"], result["global_window_index"]) != (
        config["rank"], 15, config["rolling_months"] or 84 + job["index"], job["index"]
    ):
        raise ValueError("Fit controls differ from registration")
    values = arrays(Path(result["paths"]["analytic"]))
    if (str(values["date"]) != job["date"] or values["assets"].tolist() != ASSETS
            or values["mean"].shape != (15,) or values["covariance"].shape != (15, 15)
            or not np.isfinite(values["mean"]).all() or not np.isfinite(values["covariance"]).all()):
        raise ValueError("Invalid all-task analytic moments")
    np.testing.assert_allclose(values["covariance"], values["covariance"].T, rtol=0, atol=1e-12)
    np.linalg.cholesky(values["covariance"])
    live = job["date"] == "2026-10-30"
    if (live and not np.isnan(values["actual"]).all()) or (not live and not np.isfinite(values["actual"]).all()):
        raise ValueError("Actual target availability changed")
