"""Read-only GP artifact adapters and complete paired portfolio recordings.

Usage: spawned by run_comparison.py only after the full 499-study writer exits.
Writes this policy study only, never upstream fits/evaluations. Forecast targets
are decimal discounted excess; covariance/budgets are in matching squared units.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from policy_math import PolicyAllocator, _policies, _score_allocations, _summary

from artifacts import Reader

STATE = {}


def initialize(context: dict, anchor: np.ndarray) -> None:
    """Precompile common historical-risk problems for all three count settings.

    Args:
        context: Frozen full-study targets, dates, RF, historical covariance/budget.
        anchor: Frozen 3-seed by 24-month by 15-task other-family forecasts.
    """
    STATE.clear()
    STATE.update(
        context=context,
        anchor=anchor,
        allocators={
            minimum: PolicyAllocator(context["historical_covariance"], context["risk_budget"], minimum)
            for minimum in [None, 3, 10]
        },
    )


def evaluate(job: dict) -> dict:
    """Record both variants for every seed/month of a registered configuration.

    Args:
        job: Candidate, immutable upstream artifact root and separate output root.

    Returns:
        Summaries and hashes for seed/month weights, returns and risk diagnostics.
    """
    reader = Reader()
    candidate, context = job["candidate"], STATE["context"]
    output, upstream = Path(job["output"]), Path(job["upstream"])
    name, assets = candidate["configuration"], candidate["tickers"]
    is_joint = candidate["family"] == "joint"
    if is_joint and assets != context["assets"]:
        raise ValueError("Joint policy requires the exact complete 15-ETF universe")
    results, weight_rows = [], []
    for offset, seed in enumerate([27, 127, 227]):
        source = upstream / "analytic" / name / f"s{seed}.npz"
        values = reader.read(source)
        if reader.hashes[str(source.resolve())] != job["expected_analytic_sha"][str(source.resolve())]:
            raise ValueError("Predictive arrays differ from the completed upstream evidence seal")
        if (
            values["dates"].tolist() != context["dates"]
            or values["assets"].tolist() != assets
            or not bool(values["observation_noise"].item())
            or values["units"].item() != "raw_discounted_excess"
        ):
            raise ValueError("Forecast date/task/noise/unit metadata mismatch")
        mean, std, covariance = values["mean"], values["std"], values["covariance"]
        if (
            mean.shape != (24, len(assets))
            or std.shape != mean.shape
            or covariance.shape != (24, len(assets), len(assets))
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or not (std > 0).all()
            or not (values["likelihood_noise_diagonal"] > 0).all()
            or not np.allclose(np.diagonal(covariance, axis1=-2, axis2=-1), std**2, rtol=1e-8, atol=1e-12)
        ):
            raise ValueError("Incomplete raw predictive arrays or observed covariance")
        np.linalg.cholesky(covariance)
        full_mean = STATE["anchor"][offset].copy()
        indices = [context["assets"].index(asset) for asset in assets]
        full_mean[:, indices] = mean
        calibration_covariance = {"historical_control_not_GP": context["historical_covariance"]}
        if is_joint:
            calibration_covariance["exact_observed_GP"] = covariance
        portfolios = {}
        for variant, minimum in _policies(candidate["family"]).items():
            allocators = {"historical": STATE["allocators"][minimum]}
            if is_joint:
                allocators["GP"] = PolicyAllocator(covariance, context["risk_budget"], minimum)
            for source, allocator in allocators.items():
                key = f"{source}__{variant}"
                records = allocator.allocate(full_mean)
                portfolios[key] = _score_allocations(context, full_mean, records, calibration_covariance)
                for record in records:
                    for index, asset in enumerate(context["assets"]):
                        weight_rows.append(
                            {
                                "configuration": name,
                                "seed": seed,
                                "date": context["dates"][record["index"]],
                                "asset": asset,
                                "covariance_policy": source,
                                "variant": variant,
                                "minimum_effective": minimum,
                                "ETF_position_cap": None,
                                "status": record["status"],
                                "weight": record["weights"][index] if record["weights"] is not None else None,
                                "variance_budget": record["variance_budget"],
                                "fallback_used": False,
                            }
                        )
        results.append(
            {
                "seed": seed,
                "portfolios": portfolios,
                "GP_covariance_status": "full_joint_available"
                if is_joint
                else "no_cross_family_GP_covariance_not_fabricated",
            }
        )
    reader.verify()
    summaries = _summary(candidate, results)
    folder = output / "portfolios"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(
        json.dumps(
            {
                "candidate": candidate,
                "summaries": summaries,
                "seeds": results,
                "upstream_sha256": reader.hashes,
                "risk_budget_shared_between_variants": True,
                "CVaR": False,
                "ETF_position_cap": None,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    weights = output / "monthly_weights" / f"{name}.csv"
    weights.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(weight_rows).to_csv(weights, index=False)
    return {
        "configuration": name,
        "summaries": summaries,
        "upstream_sha256": reader.hashes,
        "artifact_sha256": {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in [path, weights]},
    }
