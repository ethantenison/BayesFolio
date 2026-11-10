"""Reproduce and verify the bounded November portfolio decision end to end.

Isolated research composition/I/O only; no account access or trading. Uses the
frozen preregistration, existing cached forecasts and18 new monthly GP fits.
Returns are decimal discounted excess, covariance squared decimals, NAV dollars,
and delivered weights decimal fractions. All old evidence stays read-only.

Usage:
    uv run --locked --no-sync python <this_file> --execute --workers 14
    uv run --locked --no-sync python <this_file> --verify
    uv run --locked --no-sync python <this_file> --status
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from common import (
    CHRONO,
    COMPONENTS,
    FAMILIES,
    HERE,
    REPO,
    SEEDS,
    arrays,
    digest,
    read,
    save,
    snapshot,
    status,
    verify_hashes,
)
from decision_math import metrics, pool
from historical import evaluate
from independent_validation import validate_portfolio, validate_training_availability, validate_weights_csv
from policy import allocate, allocate_path
from prepare_decision import prepare
from sklearn.covariance import LedoitWolf


def _recent_context() -> dict:
    frame = pd.read_parquet(HERE / "data/dataset/global.parquet")
    old = arrays(CHRONO / "context52.npz")
    assets = old["assets"].tolist()
    labels = frame.pivot(index="date", columns="asset_id", values="y_excess_lead").reindex(columns=assets)
    budgets = []
    for origin in ["2026-09-30", "2026-10-30"]:
        train = labels.loc[labels.index < pd.Timestamp(origin)].to_numpy()
        if not np.isfinite(train).all():
            raise ValueError("Only available historical targets may enter the risk ceiling")
        equal = np.full(15, 1 / 15)
        budgets.append(float(equal @ LedoitWolf().fit(train).covariance_ @ equal))
    audit = read(HERE / "data/coverage_audit.json")
    return {"frame": frame, "assets": assets,
            "dates": np.concatenate([old["dates"], np.array(["2026-09-30"])]),
            "actual": np.concatenate([old["actual"], labels.loc[[pd.Timestamp("2026-09-30")]].to_numpy()]),
            "risk_free": np.r_[old["risk_free"], audit["build_audit"]["october_risk_free_return"]],
            "risk_budget": np.r_[old["risk_budget"], budgets[0]], "live_budget": budgets[1]}


def _distribution_scores(actual: np.ndarray, mean: np.ndarray, covariance: np.ndarray,
                         weights: np.ndarray) -> dict:
    errors = actual - mean
    factors = np.linalg.cholesky(covariance)
    z = np.linalg.solve(factors, errors[..., None])[..., 0]
    logdet = 2 * np.log(np.diagonal(factors, axis1=-2, axis2=-1)).sum(-1)
    joint_nll = .5 * (15 * np.log(2 * np.pi) + logdet + np.sum(z**2, axis=-1))
    marginal_variance = np.diagonal(covariance, axis1=-2, axis2=-1)
    marginal_nll = .5 * (np.log(2 * np.pi * marginal_variance) + errors**2 / marginal_variance)
    portfolio_variance = np.einsum("ta,tab,tb->t", weights, covariance, weights)
    portfolio_error = np.sum(weights * errors, axis=-1)
    equal_error = errors.mean(-1)
    equal_variance = covariance.sum(axis=(-2, -1)) / 225
    return {"joint_nll_per_asset": float(joint_nll.mean() / 15),
            "marginal_nll": float(marginal_nll.mean()), "rmse": float(np.sqrt(np.mean(errors**2))),
            "portfolio_mean_squared_standardized_error": float(np.mean(portfolio_error**2 / portfolio_variance)),
            "portfolio_95pct_coverage": float(np.mean(abs(portfolio_error) <= 1.95996398454
                                                       * np.sqrt(portfolio_variance))),
            "equal_weight_reference_nll": float(np.mean(.5 * (np.log(2 * np.pi * equal_variance)
                                                               + equal_error**2 / equal_variance))),
            "calibration_parameters_fitted": False}


def _benchmark_metrics(total: np.ndarray, rf: np.ndarray) -> dict:
    excess = total - rf
    wealth = 10000 * np.r_[1., np.cumprod(1 + total)]
    return {"sharpe": float(np.sqrt(12) * excess.mean() / excess.std(ddof=1)),
            "total_return": float(wealth[-1] / 10000 - 1),
            "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
            "annualized_excess_vol": float(excess.std(ddof=1) * np.sqrt(12))}


def _compare(selected: dict, context: dict) -> dict:
    from benchmarks import build_benchmarks

    historical = read(HERE / "outputs/historical/ALL_RESULTS.json")
    registry = {row["id"]: row for row in historical}
    recent = [arrays(HERE / "outputs/recent_components" / f"{name}.npz") for name in COMPONENTS]
    means = np.stack([item["mean"] for item in recent])
    covariances = np.stack([item["covariance"] for item in recent])
    records = []
    for identity in selected["shortlist"]:
        row = registry[identity]
        old = arrays(HERE / "outputs/historical" / f"{identity}.npz")
        recent_mean, recent_cov = pool(means, covariances, np.array(row["recipe"]["fractions"]))
        extra_weights, numerical = allocate_path(recent_mean[:1], recent_cov[:1], context["risk_budget"][-1:])
        mean = np.concatenate([old["mean"], recent_mean[:1]])
        covariance = np.concatenate([old["covariance"], recent_cov[:1]])
        weights = np.concatenate([old["weights"], extra_weights])
        all53, path = metrics(context["actual"], context["risk_free"], mean, weights)
        later29, _ = metrics(context["actual"][24:], context["risk_free"][24:], mean[24:], weights[24:])
        distribution = _distribution_scores(context["actual"], mean, covariance, weights)
        live = allocate(recent_mean[1], recent_cov[1], context["live_budget"])
        artifact = HERE / "outputs/finalists" / f"{identity}.npz"
        snapshot(artifact, {**path, "dates": context["dates"], "assets": np.array(context["assets"]),
                           "actual": context["actual"], "rf": context["risk_free"], "mean": mean,
                           "covariance": covariance, "weights": weights, "risk_budget": context["risk_budget"],
                           "live_mean": recent_mean[1], "live_covariance": recent_cov[1],
                           "live_weights": np.array(live["weights"]), "live_budget": np.array(context["live_budget"])})
        result = {"id": identity, "label": row["recipe"]["label"], "all52": row["all52"],
                  "earlier24": row["earlier24"], "later28": row["later28"], "all53": all53,
                  "later29": later29, "october_return": float(path["total_returns"][-1]),
                  "promotion": row["promotion"], "artifact_path": str(artifact),
                  "distribution_diagnostics": distribution,
                  "October_is_diagnostic_not_selection": True, "November_allocation": live,
                  "October_allocation": numerical[0]}
        save(artifact.with_suffix(".json"), result)
        records.append(result)
    benchmarks = build_benchmarks(context["dates"].tolist(), context["actual"], context["risk_free"],
                                  context["assets"])
    benchmark_rows = {}
    for name in ["schwab_aggressive", "momentum_top7"]:
        total = benchmarks[name + "_total_returns"]
        path = HERE / "outputs/benchmarks" / f"{name}.npz"
        snapshot(path, {"dates": context["dates"], "weights": benchmarks[name + "_weights"],
                         "total_returns": total, "wealth": 10000 * np.r_[1., np.cumprod(1 + total)]})
        benchmark_rows[name] = {"all53": _benchmark_metrics(total, context["risk_free"]),
                                "later29": _benchmark_metrics(total[24:], context["risk_free"][24:]),
                                "october_return": float(total[-1]), "artifact_path": str(path)}
    save(HERE / "outputs/benchmarks/AUDIT.json", benchmarks["audit"])
    return {"records": records, "benchmark": benchmark_rows, "dates": context["dates"].tolist(),
            "selected_id": selected["winner"]["id"],
            "incumbent_id": next(row["id"] for row in historical if row["recipe"]["label"] == "incumbent"),
            "standing_benchmarks_only": True, "historical_selection_months": 52,
            "latest_check_months": 1, "benchmark_audit": benchmarks["audit"]}


def finalize() -> dict:
    """Deliver final configuration/weights only after every numerical acceptance gate.

    Returns:
        Passed final validation object. No trades or account changes occur.
    """
    from report_decision import render

    plan = prepare()
    selected = read(HERE / "HISTORICAL_DECISION.json")
    context = _recent_context()
    comparison = _compare(selected, context)
    winner = next(row for row in comparison["records"] if row["id"] == selected["winner"]["id"])
    live = arrays(Path(winner["artifact_path"]))
    weights = live["live_weights"]
    model_contract = read(HERE / "FIT_CONTRACT.json")
    columns = sorted({column for item in model_contract["configs"].values() for column in item["inputs"]})
    availability = validate_training_availability(context["frame"], context["assets"], columns)
    portfolio = validate_portfolio(live["live_mean"], live["live_covariance"], context["live_budget"],
                                   weights, context["assets"])
    frame = pd.DataFrame({"asset_id": context["assets"], "weight": weights,
                          "as_of_date": "2026-10-31", "market_origin": "2026-10-30",
                          "forecast_month": "2026-11"})
    stream = io.StringIO()
    frame.to_csv(stream, index=False, float_format="%.17g")
    save(HERE / "NOVEMBER_WEIGHTS.csv", stream.getvalue().encode())
    serialized = validate_weights_csv(HERE / "NOVEMBER_WEIGHTS.csv", live["live_mean"],
        live["live_covariance"], context["live_budget"], weights, context["assets"],
        expected_metadata={"as_of_date": "2026-10-31", "market_origin": "2026-10-30", "forecast_month": "2026-11"})
    # Recompute pooling and allocation from the underlying per-seed expert panels,
    # not from a preassembled live portfolio or an average of seed portfolios.
    parts = [arrays(HERE / "outputs/recent_components" / f"{name}.npz") for name in COMPONENTS]
    mean2, cov2 = pool(np.stack([p["mean"] for p in parts]), np.stack([p["covariance"] for p in parts]),
                      np.array(selected["winner"]["fractions"]))
    repeated = allocate(mean2[1], cov2[1], context["live_budget"])
    np.testing.assert_array_equal(mean2[1], live["live_mean"])
    np.testing.assert_array_equal(cov2[1], live["live_covariance"])
    np.testing.assert_allclose(repeated["weights"], weights, rtol=0, atol=1e-12)
    seed_diagnostic = []
    seed_blends = np.einsum("ksta,ak->sta", np.stack([p["mean"] for p in parts]),
                            np.array(selected["winner"]["fractions"])[np.repeat(np.arange(4), [4, 4, 4, 3])])
    for index, seed in enumerate(SEEDS):
        seed_diagnostic.append({"seed": seed, "November_blended_mean": seed_blends[index, 1].tolist()})
    snapshot(HERE / "outputs/NOVEMBER_MOMENTS.npz", {
        "mean": live["live_mean"], "covariance": live["live_covariance"], "weights": weights,
        "assets": np.array(context["assets"]), "variance_budget": np.array(context["live_budget"]),
        "source_expert_means": np.stack([p["mean"][:, 1] for p in parts]),
        "source_expert_covariances": np.stack([p["covariance"][:, 1] for p in parts]),
        "seeds": np.array(SEEDS)})
    fits = read(HERE / "FITS_COMPLETED.json")
    config = {
        "schema": "bayesfolio.november_recommended_configuration.v1", "as_of_date": "2026-10-31",
        "market_origin": "2026-10-30", "forecast_month": "2026-11", "assets": context["assets"],
        "family_order": FAMILIES, "component_order": COMPONENTS,
        "selected_recipe": selected["winner"], "incumbent_retained": selected["incumbent_retained"],
        "GP_settings": {"rank": 2, "maxiter": 15, "seeds": SEEDS, "t_index_retained": True,
                        "kernel_variant": "signed_lkj_eta_2", "time_modulation_mode": "neither",
                        "data_kernel": "T + E + M + T*E + T*M + E*M",
                        "T_and_E": "Matern nu0.5 ARD", "M": "Matern nu0.5 ARD + RQ ARD + Linear",
                        "task_covariance": "signed task kernel with LKJ eta2 prior; rank2 factor and task diagonal",
                        "mean": "multitask_constant", "input_transform": "BoTorch Normalize non-task inputs",
                        "outcome_transform": "StratifiedStandardize by ETF task",
                        "min_inferred_noise_level": .005,
                        "fitting_objective": "marginal log likelihood with existing priors, not portfolio Sharpe",
                        "all_15_tasks_jointly_trained": True},
        "models": {name: {key: value for key, value in item.items() if key in
                   ["inputs", "rank", "maxiter", "rolling_months", "tickers", "command"]}
                   for name, item in model_contract["configs"].items()},
        "live_training": {"expanding_months": 113, "rolling_months": 48,
                          "first_available_origin": "2017-05-31", "last_training_origin": "2026-09-30",
                          "last_training_outcome_month": "2026-10", "unknown_live_target": "2026-11"},
        "seed_pooling": "equal mean of three seed predictive means and declared covariance, before one allocation",
        "covariance": "mean_seeds(0.5 broad observed covariance + 0.5 compact observed covariance)",
        "covariance_disagreement_or_independence_term_added": False,
        "allocator": {"method": "hard_rank", "objective": "rank(mean)/15 @ w - 1e-8 * sum(w^2)",
                      "long_only": True, "fully_invested": True, "minimum_effective_assets": 7,
                      "position_cap": None, "continuous_risk_penalty": 0,
                      "variance_ceiling": "expanding pre-origin LedoitWolf historical equal-weight15 variance",
                      "live_variance_budget": context["live_budget"], "CVaR": False, "fallback": False,
                      "monthly_blend_tuning": False, "postsolve_repair": False},
        "fit_counts": {key: fits[key] for key in ["completed_fits", "fit_attempts_used", "fit_attempts_remaining"]},
        "seed_sensitivity": {"interpretation": "initialization sensitivity, not independent market histories",
                             "per_seed": seed_diagnostic,
                             "prediction_sd_across_seeds": seed_blends[:, 1].std(0, ddof=1).tolist()},
        "execution_note": "Oct31 logical as-of uses Oct30 closing observations; possible Nov2 execution "
                          "cannot obtain an Oct30 closing fill after observing that close. Target weights, not orders.",
        "data_limitations": ["Inspected retrospective histories; no untouched validation claim.",
                             "Frozen latest-vintage macro history and existing extra predictor lag remain.",
                             "MGK September dividend adjustment revision0.08247% verified; old history not rewritten.",
                             "Arithmetic covariance pooling is a declared risk recipe, not a calibrated mixture.",
                             "Returns gross of explicit costs/taxes; no execution/future-return guarantee."],
        "reproduction_command": "uv run --locked --no-sync python "
                                "experiments/2026-11-portfolio-decision-v1/run_decision.py --verify",
        "preregistration_sha256": digest(HERE / "PREREGISTRATION.json"),
        "historical_decision_sha256": digest(HERE / "HISTORICAL_DECISION.json"),
        "data_sha256": digest(HERE / "data/dataset/global.parquet")}
    verify_hashes(plan["protected_sha256"])
    verify_hashes(model_contract["protected_sha256"])
    validation = {"passed": True, "data_freshness_and_target_availability": availability,
                  "data_extension_passed": read(HERE / "data/COMPLETED.json")["passed"],
                  "portfolio": portfolio, "serialized_weights": serialized,
                  "deterministic_pool_then_allocate_reproduction": {"passed": True,
                      "maximum_absolute_weight_difference": float(np.max(
                          abs(np.array(repeated["weights"]) - weights)))},
                  "protected_files_unchanged": True, "fit_attempts_used": fits["fit_attempts_used"],
                  "fit_attempts_cap": 128, "finalist_count": len(comparison["records"]),
                  "batches_used": 1, "new_blend_configurations": plan["batch1_new_recipes"],
                  "unresolved_fit_or_allocation_failures": [], "all_November_labels_unknown": True,
                  "Pylance": "unavailable; no completed editor diagnostics gate claimed",
                  "independent_historical_audit": read(HERE / "outputs/INDEPENDENT_HISTORICAL_AUDIT.json"),
                  "historical_decision_uses_October_outcomes": False,
                  "no_trades_or_account_changes": True}
    save(HERE / "CONFIG.json", config)
    save(HERE / "COMPARISON.json", comparison)
    save(HERE / "VALIDATION.json", validation)
    render(config, validation, comparison, frame)
    _checkpoint_integrity()
    reproduce()
    artifacts = [item for item in HERE.rglob("*") if item.is_file()
                 and item.name not in {"COMPLETED.json", "STATUS.json"}
                 and item.suffix not in {".log", ".lock", ".pyc"}
                 and "__pycache__" not in item.parts and ".pytest_cache" not in item.parts]
    save(HERE / "COMPLETED.json", {"passed": True, "completed_at_utc": datetime.now(UTC).isoformat(),
        "fit_attempts_used": fits["fit_attempts_used"], "new_configurations": plan["batch1_new_recipes"],
        "finalists": len(comparison["records"]), "selected_recipe": selected["winner"]["label"],
        "incumbent_retained": selected["incumbent_retained"],
        "independent_final_audit": str(HERE / "outputs/INDEPENDENT_FINAL_AUDIT.json"),
        "environment_sha256": {str(item): digest(item) for item in [REPO / "uv.lock", REPO / "pyproject.toml"]},
        "artifact_sha256": {str(item.resolve()): digest(item) for item in artifacts}})
    status("complete", "Delivered validated November target weights; no trades executed",
           selected_recipe=selected["winner"]["label"], fit_attempts_used=fits["fit_attempts_used"],
           fit_attempts_remaining=128 - fits["fit_attempts_used"], batches_used=1,
           new_configurations=plan["batch1_new_recipes"], finalists=len(comparison["records"]),
           unresolved_failures=[], decision=str(HERE / "DECISION.md"), weights=str(HERE / "NOVEMBER_WEIGHTS.csv"))
    return validation


def _checkpoint_integrity() -> None:
    plan = prepare()
    fit_contract = read(HERE / "FIT_CONTRACT.json")
    expected = {(name, seed, index, date) for name in COMPONENTS for seed in SEEDS
                for index, date in enumerate(["2026-09-30", "2026-10-30"], 28)}
    observed = set()
    for path in (HERE / "outputs/fit_checkpoints").glob("*.json"):
        row = read(path)
        job = row["job"]
        identity = (job["component"], job["seed"], job["index"], job["date"])
        if (identity not in expected or identity in observed
                or row["contract_sha256"] != digest(HERE / "FIT_CONTRACT.json")):
            raise ValueError("Fit checkpoint contract/job identity mismatch")
        observed.add(identity)
        verify_hashes(row["artifact_sha256"])
        result = row["result"]
        if (result["seed"], result["component"], result["global_window_index"], result["rank"], result["maxiter"]) != (
            job["seed"], job["component"], job["index"], 2, 15
        ):
            raise ValueError("Actual fit settings/seed differ from registered job")
    if observed != expected:
        raise ValueError("Missing or duplicate forecast jobs")
    menu = {item["id"]: item for item in plan["candidate_menu"]}
    for path in (HERE / "outputs/historical").glob("recipe_*.json"):
        row = read(path)
        if (row["recipe"] != menu[row["id"]]
                or row["preregistration_sha256"] != digest(HERE / "PREREGISTRATION.json")
                or digest(path.with_suffix(".npz")) != row["arrays_sha256"]):
            raise ValueError("Historical checkpoint preregistration mismatch")
    verify_hashes(plan["protected_sha256"])
    verify_hashes(fit_contract["protected_sha256"])
    data = read(HERE / "data/COMPLETED.json")
    verify_hashes(data["input_sha256"])
    verify_hashes(data["artifact_sha256"])
    historical = read(HERE / "HISTORICAL_DECISION.json")
    verify_hashes(historical["artifact_sha256"])
    fits = read(HERE / "FITS_COMPLETED.json")
    verify_hashes(fits["artifact_sha256"])
    independent = read(HERE / "outputs/INDEPENDENT_FINAL_AUDIT.json")
    if not independent["passed"]:
        raise ValueError("Independent final numerical audit did not pass")
    verify_hashes(independent["source_hashes"])
    np.testing.assert_allclose(
        _recent_context()["live_budget"], independent["source_derived_live_variance_budget"],
        rtol=0, atol=1e-15,
    )


def reproduce() -> dict:
    """Recompute delivered weights from original expert moments and re-audit CSV.

    Returns:
        Independent original-conic and serialized-weight validation evidence.
        No GP fitting, network access or output mutation occurs.
    """
    config = read(HERE / "CONFIG.json")
    parts = [arrays(HERE / "outputs/recent_components" / f"{name}.npz") for name in COMPONENTS]
    mean, covariance = pool(np.stack([p["mean"] for p in parts]),
                            np.stack([p["covariance"] for p in parts]),
                            np.array(config["selected_recipe"]["fractions"]))
    context = _recent_context()
    result = allocate(mean[1], covariance[1], context["live_budget"])
    expected = arrays(HERE / "outputs/NOVEMBER_MOMENTS.npz")
    np.testing.assert_array_equal(mean[1], expected["mean"])
    np.testing.assert_array_equal(covariance[1], expected["covariance"])
    np.testing.assert_allclose(result["weights"], expected["weights"], rtol=0, atol=1e-12)
    return validate_weights_csv(HERE / "NOVEMBER_WEIGHTS.csv", mean[1], covariance[1], context["live_budget"],
                                expected["weights"], context["assets"], expected_metadata={
                                    "as_of_date": "2026-10-31", "market_origin": "2026-10-30",
                                    "forecast_month": "2026-11"})


def verify() -> dict:
    """Verify the completed decision, protected inputs, checkpoints and reproduction.

    Returns:
        Evidence summary; raises rather than accepting missing/stale outputs.
    """
    seal = read(HERE / "COMPLETED.json")
    if not seal["passed"]:
        raise ValueError("November decision is not complete")
    verify_hashes(seal["artifact_sha256"])
    verify_hashes(seal["environment_sha256"])
    _checkpoint_integrity()
    actual = reproduce()
    return {"passed": actual["passed"], "fit_attempts": seal["fit_attempts_used"],
            "selected_recipe": seal["selected_recipe"], "target_weights_reproduced": True,
            "protected_files_unchanged": True, "trades_executed": False}


def main() -> None:
    """Execute the finite goal or perform read-only status/reproduction checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ["execute", "verify", "status"]:
        mode.add_argument("--" + name, action="store_true")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read(HERE / "STATUS.json"), indent=2))
        return
    if args.verify:
        print(json.dumps(verify(), indent=2))
        return
    if (HERE / "COMPLETED.json").exists():
        print(json.dumps(verify(), indent=2))
        return
    from data_extension import prepare_data
    from live_fits import fit
    from reversal_analysis import analyze

    prepare()
    analyze(HERE / "outputs", HERE / "PREREGISTRATION.json")
    evaluate()
    prepare_data(HERE / "data", fetch=True)
    fit(args.workers)
    finalize()
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
