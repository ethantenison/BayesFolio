"""Deterministic fixed-policy evaluation with a locked confirmation boundary.

Experiment composition layer reads isolated explicit moment inputs, allocates
one pooled portfolio per month, and saves raw evidence. No fitting or trading.
Decimal excess/total returns, squared-decimal risk matrices, fractional weights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from v2_common import ASSETS, FAMILIES, HERE, arrays, digest, read, save, snapshot
from v2_metrics import score_portfolio
from v2_policy import allocate

STATE: dict = {}


def policy(label: str, fractions: np.ndarray, *, nea: int = 7, risk: float | None = 1.,
           covariance: list[float] | None = None, equity_cap: float | None = None,
           prediction_experts: list[str] | None = None, covariance_experts: list[str] | None = None,
           control: bool = False) -> dict:
    """Register one content-addressed complete fixed policy configuration.

    Args:
        label: Human-readable hypothesis label, not a ranking criterion.
        fractions: Four by three fixed family prediction fractions.
        nea: Registered eligible-asset minimum effective count, 3..14.
        risk: Historical eligible-EW risk multiplier, or no ceiling.
        covariance: Four nonnegative fractions: broad/compact/rolling/historical.
        equity_cap: Optional static equity ceiling.
        prediction_experts: Exact three registered prediction expert slots.
        covariance_experts: Exact three registered GP covariance slots.
        control: Whether this is an unchanged mandatory reference.

    Returns:
        Exact complete policy definition and reproducible ID.
    """
    from v2_common import COMPONENTS

    fractions = np.asarray(fractions, dtype=float)
    mixing = [.5, .5, 0., 0.] if covariance is None else covariance
    if (fractions.shape != (4, 3) or not np.isfinite(fractions).all() or (fractions < 0).any()
            or not np.allclose(fractions.sum(1), 1, rtol=0, atol=1e-12)
            or len(mixing) != 4 or min(mixing) < 0 or abs(sum(mixing) - 1) > 1e-12
            or nea not in range(3, 15) or (risk is not None and (not np.isfinite(risk) or risk <= 0))
            or (equity_cap is not None and not 0 <= equity_cap <= 1)):
        raise ValueError("Invalid complete static policy")
    body = {"prediction_experts": list(COMPONENTS if prediction_experts is None else prediction_experts),
            "covariance_experts": list(COMPONENTS if covariance_experts is None else covariance_experts),
            "prediction_fractions": fractions.tolist(), "covariance_fractions": list(mixing),
            "minimum_effective_assets": nea, "risk_multiplier": risk, "equity_cap": equity_cap,
            "allocator": "hard_rank", "MGK_weight": 0, "seed_pooling": "equal_moments_before_one_solve"}
    if len(body["prediction_experts"]) != 3 or len(body["covariance_experts"]) != 3:
        raise ValueError("Three expert component slots required")
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"id": "policy_" + hashlib.sha256(encoded).hexdigest()[:20], "label": label,
            "control": control, **body}


def initialize(input_path: str, stage: str) -> None:
    """Initialize one root-owned evaluation worker from explicit frozen inputs.

    Args:
        input_path: Development-only or separately unlocked all53 source NPZ.
        stage: Development or confirmation.
    """
    if stage not in {"development", "confirmation"}:
        raise ValueError("Unknown evaluation stage")
    source = arrays(Path(input_path))
    required = 41 if stage == "development" else 53
    if (len(source["dates"]) != required
            or source["dates"].tolist() != read(HERE / "SPLITS.json")["origins"][:required]):
        raise ValueError("Evaluation input crosses the frozen temporal boundary")
    if stage == "confirmation" and not (HERE / "SHORTLIST_FROZEN.json").exists():
        raise ValueError("Latest twelve candidate outcomes are still locked")
    np.testing.assert_array_equal(source["assets"], ASSETS)
    np.testing.assert_array_equal(source["seeds"], [27, 127, 227])
    if (source["expert_mean"].shape != (len(source["experts"]), 3, required, 15)
            or source["expert_covariance"].shape != (len(source["experts"]), 3, required, 15, 15)):
        raise ValueError("Incorrect expert/seed/month/task axes")
    STATE.clear()
    STATE.update(source=source, stage=stage, input_sha256=digest(Path(input_path)))


def pooled(case: dict, source: dict) -> tuple[np.ndarray, np.ndarray]:
    """Blend expert moments and pool seeds before any eligible-axis allocation.

    Args:
        case: Complete static policy tuple.
        source: Explicit expert/seed/month/task moment panels and historical risk.

    Returns:
        Monthly fifteen-task pooled means and positive-definite observed risk.
    """
    names = source["experts"].tolist()
    mean = source["expert_mean"][[names.index(name) for name in case["prediction_experts"]]]
    cov = source["expert_covariance"][[names.index(name) for name in case["covariance_experts"]]]
    family = np.repeat(np.arange(4), [4, 4, 4, 3])
    fractions = np.asarray(case["prediction_fractions"])
    mixed_mean = np.einsum("ksta,ak->sta", mean, fractions[family]).mean(0)
    fractions_cov = np.asarray(case["covariance_fractions"])
    mixed_cov = np.einsum("k,kstab->tab", fractions_cov[:3] / 3, cov)
    mixed_cov += fractions_cov[3] * source["historical_covariance"]
    if not np.isfinite(mixed_mean).all() or not np.isfinite(mixed_cov).all():
        raise ValueError("Missing forecasts cannot be filled or dropped")
    np.linalg.cholesky(mixed_cov)
    return mixed_mean, mixed_cov


def _flat_paths(paths: dict) -> dict:
    result = {key: value for key, value in paths.items() if isinstance(value, np.ndarray)}
    for key in ["family_weights", "family_arithmetic_return_contributions", "family_ic"]:
        if key in paths:
            for name, value in paths[key].items():
                result[key + "__" + name] = np.asarray(value)
    for bps, values in paths["cost_cases"].items():
        result["cost_" + bps + "_total_returns"] = values["total_returns"]
        result["cost_" + bps + "_notional"] = values["traded_notional_fraction"]
    return result


def evaluate(case: dict) -> dict:
    """Evaluate one registered complete policy without omitting any failed month.

    Args:
        case: Immutable complete policy tuple from a registered round.

    Returns:
        Complete metrics/artifact identity or explicit infeasibility; numerical
        failures raise rather than scoring a partial portfolio path.
    """
    stage, source = STATE["stage"], STATE["source"]
    if stage == "confirmation":
        shortlist = read(HERE / "SHORTLIST_FROZEN.json")
        if (digest(HERE / "SHORTLIST_FROZEN.json") != (HERE / "SHORTLIST_FROZEN.sha256").read_text().strip()
                or len(shortlist["ids"]) > 10 or case["id"] not in shortlist["ids"]):
            raise ValueError("Policy is outside the intact frozen final shortlist")
    output = HERE / "outputs" / stage / f"{case['id']}.json"
    if output.exists():
        existing = read(output)
        if existing["input_sha256"] != STATE["input_sha256"] or existing["policy"] != case:
            raise ValueError("Policy checkpoint input changed")
        if existing["complete"] and digest(output.with_suffix(".npz")) != existing["arrays_sha256"]:
            raise ValueError("Policy numerical checkpoint changed")
        return existing
    mean, cov = pooled(case, source)
    numerical, weights = [], []
    for index, date in enumerate(source["dates"]):
        budget = (None if case["risk_multiplier"] is None else
                  float(case["risk_multiplier"] * source["historical_EW_variance"][index]))
        solved = allocate(mean[index], cov[index], min_effective_assets=case["minimum_effective_assets"],
                          variance_ceiling=budget, equity_cap=case["equity_cap"])
        numerical.append({"origin": str(date), **solved})
        if not solved["passed"]:
            record = {"id": case["id"], "policy": case, "complete": False, "infeasible": True,
                      "failed_origin": str(date), "no_partial_objectives": True,
                      "numerical": numerical, "input_sha256": STATE["input_sha256"]}
            save(output, record)
            return record
        weights.append(solved["weights"])
    weights = np.asarray(weights)
    summary, paths = score_portfolio(source["actual"], source["rf"], mean, weights,
                                     assets=ASSETS, families=FAMILIES)
    blocks = []
    for start, end in [(5, 17), (17, 29), (29, 41)]:
        score, _ = score_portfolio(source["actual"][start:end], source["rf"][start:end], mean[start:end],
                                   weights[start:end], assets=ASSETS, families=FAMILIES, cost_bps=(0,))
        blocks.append(score)
    extra = {}
    if stage == "confirmation":
        extra["latest12"], _ = score_portfolio(source["actual"][41:], source["rf"][41:], mean[41:],
            weights[41:], assets=ASSETS, families=FAMILIES)
    snapshot(output.with_suffix(".npz"), {**_flat_paths(paths), "mean": mean, "covariance": cov,
        "weights": weights, "actual": source["actual"], "rf": source["rf"], "dates": source["dates"],
        "assets": source["assets"], "historical_EW_variance": source["historical_EW_variance"]})
    record = {"id": case["id"], "policy": case, "complete": True, "summary": summary,
              "annual_development_blocks": blocks, "numerical": numerical,
              "risk_binding_fraction": float(np.mean([r["raw_checks"]["variance_binding"] for r in numerical])),
              "equity_binding_fraction": float(np.mean([r["raw_checks"]["equity_binding"] for r in numerical])),
              "analytic_fraction": float(np.mean([r["solver_status"] == "analytic_kkt" for r in numerical])),
              "arrays_sha256": digest(output.with_suffix(".npz")), "artifact_path": str(output.with_suffix(".npz")),
              "input_sha256": STATE["input_sha256"], **extra}
    save(output, record)
    return record
