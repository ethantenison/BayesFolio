"""Finite v2 campaign, read-only reproduction and independent-audited sealing.

Research composition only: no trades, brokerage, git mutations or production
changes. Cached decimal forecasts/covariance recreate raw portfolio fractions;
the verifier never needs new GP fits or new market data.
Usage:
    uv run --locked --no-sync python <this_file> --execute
    uv run --locked --no-sync python <this_file> --render
    uv run --locked --no-sync python <this_file> --seal
    uv run --locked --no-sync python <this_file> --verify
"""

from __future__ import annotations

import os

for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-v2-mpl")

import argparse  # noqa: E402
import json  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
from v2_common import (  # noqa: E402
    ASSETS,
    ELIGIBLE_INDEX,
    FAMILIES,
    HERE,
    REPO,
    V1,
    arrays,
    digest,
    read,
    save,
    status,
    verify_hashes,
)
from v2_evaluate import pooled  # noqa: E402
from v2_experts import REGISTERED, load_cached_panel  # noqa: E402
from v2_finish import _solve_live, _validate_inputs, live_source, prepare_final  # noqa: E402
from v2_independent_validation import validate_weights_csv  # noqa: E402
from v2_metrics import score_portfolio  # noqa: E402
from v2_policy import allocate  # noqa: E402


def reproduce() -> dict:
    """Repool original expert caches, recompute risk and recreate every finalist.

    Returns:
        Passed exact live/CSV and historical-policy numerical reproduction.
        No output, fit, data-download or allocation-policy mutation occurs.
    """
    shortlist = _validate_inputs()
    config, selected = read(HERE / "CONFIG.json"), read(HERE / "FINAL_SELECTION.json")
    if config["selected_policy"] != selected["selected_policy"]:
        raise ValueError("Delivered configuration differs from frozen final choice")
    case = config["selected_policy"]
    source = live_source()
    actual = _solve_live(case, source)
    expected = arrays(HERE / "outputs/NOVEMBER_MOMENTS.npz")
    np.testing.assert_array_equal(actual["mean"], expected["mean"])
    np.testing.assert_array_equal(actual["covariance"], expected["covariance"])
    np.testing.assert_allclose(actual["result"]["weights"], expected["weights"], rtol=0, atol=1e-12)
    panel = pd.read_parquet(V1 / "data/dataset/global.parquet")
    labels = panel.pivot(index="date", columns="asset_id", values="y_excess_lead").reindex(columns=ASSETS)
    training = labels.loc[labels.index < pd.Timestamp("2026-10-30")].to_numpy()[:, ELIGIBLE_INDEX]
    if training.shape != (113, 14) or not np.isfinite(training).all():
        raise ValueError("Live historical risk used unavailable targets or changed universe")
    risk = float(LedoitWolf().fit(training).covariance_.sum() / 196)
    np.testing.assert_allclose(risk, source["historical_EW_variance"][0], rtol=0, atol=1e-15)
    serialized = validate_weights_csv(HERE / "NOVEMBER_WEIGHTS.csv", actual["mean"], actual["covariance"],
        expected["weights"], ASSETS, min_effective_assets=case["minimum_effective_assets"],
        variance_ceiling=actual["budget"], equity_cap=case["equity_cap"],
        solver_status=actual["result"]["solver_status"], solver_attempts=actual["result"]["numerical_attempts"],
        analytic_certificate=actual["result"]["certificate"])
    panels = [load_cached_panel(name) for name in REGISTERED]
    history = arrays(HERE / "outputs/confirmation_inputs.npz")
    source_history = {**history,
        "expert_mean": np.stack([p["mean"][:, :53] for p in panels]),
        "expert_covariance": np.stack([p["covariance"][:, :53] for p in panels])}
    np.testing.assert_array_equal(source_history["expert_mean"], history["expert_mean"])
    np.testing.assert_array_equal(source_history["expert_covariance"], history["expert_covariance"])
    maximum_weight_difference = 0.
    checked = []
    for item in shortlist["cases"]:
        mean, covariance = pooled(item, source_history)
        saved = arrays(HERE / "outputs/confirmation" / f"{item['id']}.npz")
        np.testing.assert_array_equal(mean, saved["mean"])
        np.testing.assert_array_equal(covariance, saved["covariance"])
        recreated = []
        for index in range(53):
            budget = None if item["risk_multiplier"] is None else float(
                item["risk_multiplier"] * history["historical_EW_variance"][index])
            result = allocate(mean[index], covariance[index], min_effective_assets=item["minimum_effective_assets"],
                              variance_ceiling=budget, equity_cap=item["equity_cap"])
            if not result["passed"]:
                raise ValueError("A finalist failed its identical reproduction")
            recreated.append(result["weights"])
        recreated = np.asarray(recreated)
        difference = float(np.max(abs(recreated - saved["weights"])))
        maximum_weight_difference = max(maximum_weight_difference, difference)
        np.testing.assert_allclose(recreated, saved["weights"], rtol=0, atol=1e-12)
        metrics, path = score_portfolio(history["actual"], history["rf"], mean, recreated,
                                        assets=ASSETS, families=FAMILIES)
        old = read(HERE / "outputs/confirmation" / f"{item['id']}.json")
        for key in ["cagr", "total_return", "max_drawdown", "sharpe", "rank_ic"]:
            np.testing.assert_allclose(metrics[key], old["summary"][key], rtol=0, atol=1e-12)
        np.testing.assert_allclose(path["wealth"], saved["wealth"], rtol=0, atol=1e-8)
        checked.append({"id": item["id"], "months": 53, "maximum_weight_difference": difference})
    return {"passed": True, "live_maximum_weight_difference": float(np.max(abs(
        np.asarray(actual["result"]["weights"]) - expected["weights"]))),
        "historical_maximum_weight_difference": maximum_weight_difference, "finalists": checked,
        "CSV": serialized, "fit_attempts_launched": 0, "new_policies_scored": 0,
        "protected_inputs_unchanged": True, "trades_executed": False}


def render_delivery() -> dict:
    """Combine passed root/external audit evidence and render final reports.

    Returns:
        Final validation object; completion seal is a separate explicit step.
    """
    from v2_report import render

    _validate_inputs()
    independent = read(HERE / "outputs/INDEPENDENT_FINAL_AUDIT.json")
    if not independent["passed"]:
        raise ValueError("External final audit must pass before delivery")
    if "source_hashes" in independent:
        verify_hashes(independent["source_hashes"])
    repeat = reproduce()
    root = read(HERE / "ROOT_VALIDATION.json")
    validation = {**root, "root_checks_only_external_final_audit_required_before_completion": False,
        "external_final_audit": independent, "deterministic_reproduction": repeat,
        "minimum_raw_weight_tolerance": -1e-8, "weight_sum_tolerance": 1e-7,
        "independent_conic_weight_parity_tolerance": 5e-6,
        "exact_delivered_reproduction_tolerance": 1e-12,
        "code_checks": read(HERE / "CODE_CHECKS.json"), "passed": True}
    save(HERE / "VALIDATION.json", validation)
    render(read(HERE / "CONFIG.json"), validation, read(HERE / "COMPARISON.json"),
           pd.read_csv(HERE / "NOVEMBER_WEIGHTS.csv"))
    return validation


def seal() -> dict:
    """Seal only a reproduced delivery with independent and visual acceptance.

    Returns:
        Passed completion manifest and all reproducible artifact fingerprints.
    """
    if (HERE / "COMPLETED.json").exists():
        return verify()
    _validate_inputs()
    visual = read(HERE / "VISUAL_QA.json")
    if not visual["passed"]:
        raise ValueError("Plots must be inspected before first completion")
    verify_hashes(visual["image_sha256"])
    validation = read(HERE / "VALIDATION.json")
    if not validation["passed"] or not validation["external_final_audit"]["passed"]:
        raise ValueError("Incomplete final validation")
    repeated = reproduce()
    if not repeated["passed"]:
        raise ValueError("Reproduction failed")
    artifacts = [path for path in HERE.rglob("*") if path.is_file()
        and path.name not in {"COMPLETED.json", "STATUS.json"}
        and path.suffix not in {".log", ".lock", ".pyc"}
        and "__pycache__" not in path.parts and ".pytest_cache" not in path.parts]
    value = {"passed": True, "completed_at_utc": datetime.now(UTC).isoformat(),
        "selected_id": read(HERE / "FINAL_SELECTION.json")["selected_id"],
        "new_policies": 874, "controls": 2, "rounds": 4, "new_fit_attempts": 6,
        "artifact_sha256": {str(path.resolve()): digest(path) for path in artifacts},
        "environment_sha256": {str(path): digest(path) for path in [REPO / "uv.lock", REPO / "pyproject.toml"]},
        "no_trades_git_or_production_changes": True}
    save(HERE / "COMPLETED.json", value)
    status("complete", "Delivered validated November target weights and reproducible decision; stop research",
           selected_id=value["selected_id"], fit_attempts_used=6, fit_attempts_remaining=1194,
           new_policies=874, overall_policy_budget_remaining=150, rounds_used=4, finalists=10,
           unresolved_failures=[], retained_design_coverage_limitations=16,
           decision=str(HERE / "DECISION.md"), weights=str(HERE / "NOVEMBER_WEIGHTS.csv"))
    return value


def verify() -> dict:
    """Read-only complete-artifact and numerical decision reproduction.

    Returns:
        Compact evidence summary, without fits, mutations or new market data.
    """
    completed = read(HERE / "COMPLETED.json")
    if not completed["passed"]:
        raise ValueError("Campaign is not complete")
    verify_hashes(completed["artifact_sha256"])
    verify_hashes(completed["environment_sha256"])
    repeated = reproduce()
    return {"passed": repeated["passed"], "selected_id": completed["selected_id"],
        "MGK_weight": 0, "new_policy_configurations": 874, "new_fit_attempts": 6,
        "all_ten_finalists_reproduced": True, "maximum_weight_difference":
            max(repeated["live_maximum_weight_difference"], repeated["historical_maximum_weight_difference"]),
        "protected_inputs_unchanged": True, "trades_executed": False}


def execute(workers: int) -> None:
    """Resume the finite registered campaign without repeating sealed work.

    Args:
        workers: One shared queue of at most14 single-thread workers.
    """
    from v2_diagnostics import produce_diagnostics
    from v2_driver import initialize_controls, round1, run_cases
    from v2_fit_queue import run_missing
    from v2_protocol import prepare
    from v2_research import review, round2, round3, round4
    from v2_select import freeze

    if (HERE / "COMPLETED.json").exists():
        print(json.dumps(verify(), indent=2))
        return
    prepare()
    if not (HERE / "CONTROLS.json").exists():
        initialize_controls(workers)
    for number, factory in [(1, round1), (2, round2), (3, round3), (4, round4)]:
        if not (HERE / "rounds" / f"REVIEW{number:02d}.json").exists():
            run_cases(factory(), workers=workers)
            review(number)
    run_missing(workers)
    shortlist = freeze()
    run_cases(shortlist["cases"], stage="confirmation", workers=workers)
    prepare_final()
    produce_diagnostics()
    # CODE_CHECKS, external audit and first visual QA are honest independent
    # gates, not synthesized by declaring this function successful.
    render_delivery()
    seal()
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for option in ["execute", "render", "seal", "verify", "reproduce", "status"]:
        group.add_argument("--" + option, action="store_true")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    if args.execute:
        execute(args.workers)
    elif args.render:
        render_delivery()
    elif args.seal:
        print(json.dumps(seal(), indent=2))
    elif args.verify:
        print(json.dumps(verify(), indent=2))
    elif args.reproduce:
        result = reproduce()
        print(json.dumps({key: result[key] for key in ["passed", "live_maximum_weight_difference",
            "historical_maximum_weight_difference", "fit_attempts_launched", "new_policies_scored"]}, indent=2))
    elif args.status:
        print(json.dumps(read(HERE / "STATUS.json"), indent=2))
