"""Freeze v2 temporal/decision protocol and assemble sealed causal cached axes.

Research composition/I/O only. Forecasts/actuals are decimal discounted excess;
historical risk matrices are squared decimals. No candidate scoring or GP fits
occur here. Latest twelve return outcomes remain locked for later confirmation.
Usage: uv run --locked --no-sync python <this_file>
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from v2_common import (
    ASSETS,
    CHRONO,
    COMPONENTS,
    ELIGIBLE_INDEX,
    FRAMEWORK,
    HERE,
    REPO,
    SEEDS,
    TUNING,
    V1,
    arrays,
    digest,
    read,
    save,
    snapshot,
    status,
    verify_hashes,
)


def prepare() -> dict:
    """Freeze the exact temporal protocol and protected reference evidence.

    Returns:
        Passed immutable preregistration with source/data hashes and limits.
    """
    target = HERE / "PREREGISTRATION.json"
    if target.exists():
        registered = read(target)
        verify_hashes(registered["protected_sha256"])
        if (digest(HERE / "SPLITS.json") != registered["splits_sha256"]
                or digest(HERE / "SELECTION_RULE.json") != registered["selection_rule_sha256"]):
            raise ValueError("Frozen split or selection rule changed")
        return registered
    v1 = read(V1 / "COMPLETED.json")
    if not v1["passed"]:
        raise ValueError("Passed v1 evidence required")
    verify_hashes(v1["artifact_sha256"])
    verify_hashes(v1["environment_sha256"])
    files = [HERE / "GOAL.md", V1 / "COMPLETED.json", V1 / "CONFIG.json", V1 / "VALIDATION.json",
             CHRONO / "CONTRACT.json", CHRONO / "RESULTS.md", CHRONO / "context52.npz",
             TUNING / "outputs/RESULTS.md", TUNING / "three_way_joint_v1/outputs/RESULTS.md",
             FRAMEWORK / "CURRENT_BASELINES.json", FRAMEWORK / "CURRENT_BENCHMARKS.json",
             REPO / "uv.lock", REPO / "pyproject.toml"]
    files.extend(CHRONO / "components" / f"{name}.npz" for name in [*COMPONENTS, "compact_r1_i15_exp"])
    files.extend(V1 / "outputs/recent_components" / f"{name}.npz" for name in COMPONENTS)
    context = arrays(CHRONO / "context52.npz")
    dates = np.concatenate([context["dates"], np.array(["2026-09-30"])])
    realized = (pd.to_datetime(dates).to_period("M") + 1).astype(str).tolist()
    splits = {"origins": dates.tolist(), "realized_return_months": realized,
        "partial_diagnostic": {"indices": [0, 5], "months": realized[:5]},
        "development_blocks": [{"indices": [a, b], "months": realized[a:b]}
                               for a, b in [(5, 17), (17, 29), (29, 41)]],
        "development_all_indices": [0, 41], "confirmation_indices": [41, 53],
        "confirmation_months": realized[41:], "training_window_is_not_evaluation_window": True,
        "confirmation_is_retrospective_previously_inspected": True,
        "selection_before_confirmation": True, "unknown_November_target_excluded": True,
        "label_rule": "Only target realizations actually completed before each forecast cutoff enter training",
        "macro_rule": "Preserve sealed extra feature lag and frozen vintage/provenance conventions"}
    if realized[41:] != pd.period_range("2025-11", "2026-10", freq="M").astype(str).tolist():
        raise ValueError("Incorrect latest twelve-month confirmation axis")
    rule = {"schema": "bayesfolio.multiobjective.selection.v2",
        "primary": ["gross_CAGR", "drawdown_loss_magnitude"], "secondary": ["Sharpe", "eligible14_rank_IC"],
        "normalization": {"source": "MGK-free fixed v1 three-model reference development41 ONLY",
            "aspiration_CAGR": "reference_CAGR + 0.02",
            "aspiration_drawdown_loss": "max(0, reference_drawdown_loss - 0.02)",
            "CAGR_scale": .02, "drawdown_scale": .02,
            "distance": "hypot(max(0,(aspiration_CAGR-CAGR)/0.02), "
                        "max(0,(drawdown_loss-aspiration_drawdown_loss)/0.02))",
            "anchors_saved_once_before_new_policy_scoring": True,
            "interpretation": "Balanced operational aspiration, not a personal loss limit or promised return"},
        "development_eligibility": {"complete_raw_feasible_path": True,
            "distance_gain_over_best_control_min": .05,
            "annual_blocks_no_worse_by_distance_0p25_min": 2,
            "paired_3month_bootstrap_distance_gain_nonnegative_fraction_min": .60,
            "leave_one_month_distance_gain_nonnegative_fraction_min": .80,
            "ten_bps_cost_distance_gain_min": -.10,
            "IC_noninferiority_veto": False},
        "shortlist": {"maximum": 10, "mandatory_controls": 2,
            "remaining": "Development eligible Pareto distance/stability plus return/defensive leaders "
                         "and matched compact-rank pair when supported; deterministic ID ties"},
        "confirmation_acceptance": {"complete_all53_and_November_raw_feasibility": True,
            "all53_distance_gain_over_best_control_min": .05,
            "confirmation12_distance_no_worse_than_best_control_plus": .50,
            "confirmation12_drawdown_loss_no_worse_than_best_control_plus": .03,
            "no_retuning_after_unlock": True},
        "final_selection": "Eligible full53 Pareto minimum fixed-anchor distance; ties within1e-8 by "
                           "development annual-block stability, Sharpe, eligible14IC, recipe complexity, ID. "
                           "If no accepted challenger, best matched control by same full53 fixed-anchor distance.",
        "robustness": {"bootstrap_draws": 512, "block_lengths": [3, 6], "seed": 20261101,
            "cost_bps": [0, 5, 10, 20], "cost_objective": False, "cost_gate_is_sensitivity_only": True,
            "seed_sensitivity_is_not_market_evidence": True,
            "stress_drawdown_thresholds": [.10, .15, .20], "crash_probabilities_calibrated": False},
        "frontier_stopping": {"minimum_tracks": ["cached policy investigation", "allowed GP evidence assessment"],
            "negligible_round": "HV relative gain<1% AND best distance gain<0.05",
            "consecutive_negligible_rounds": 2, "HV_reference": "reference_CAGR-0.10, reference_DDloss+0.10",
            "budgets_are_ceilings_not_targets": True}}
    save(HERE / "SPLITS.json", splits)
    save(HERE / "SELECTION_RULE.json", rule)
    registered = {"schema": "bayesfolio.multiobjective.preregistration.v2", "passed": True,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "deadline": "2026-11-01T18:00:00-06:00", "exploration_stop": "2026-11-01T16:30:00-06:00",
        "caps": {"rounds": 4, "policies_per_round": 256, "new_policies": 1024, "new_expert_specs": 6,
                 "fit_attempts": 1200, "reserved_fit_attempts": 96, "workers": 14, "finalists": 10},
        "new_GP_scope": ["compact rank1", "registered minimal kernel/prior proposal with development evidence"],
        "fixed_GP": {"tasks": ASSETS, "seeds": SEEDS, "maxiter": 15,
                     "broad_rank": 2, "compact_ranks": [1, 2], "feature_window_search": False},
        "splits_sha256": digest(HERE / "SPLITS.json"), "selection_rule_sha256": digest(HERE / "SELECTION_RULE.json"),
        "protected_sha256": {str(path): digest(path) for path in files},
        "v1_seal_sha256": digest(V1 / "COMPLETED.json")}
    save(target, registered)
    status("preregistered", "Assemble causal14-EW context and score unchanged matched controls; freeze anchors",
           new_policy_count=0, fit_attempts_used=0, fit_attempts_remaining=1200, unresolved_failures=[])
    return registered


def context() -> dict:
    """Assemble all saved source axes; caller controls which outcomes are scored.

    Returns:
        Complete53-month actual/source panels and separate unknown November
        target. Risk matrices use only labels available strictly pre-origin.
    """
    prepare()
    old = arrays(CHRONO / "context52.npz")
    frame = pd.read_parquet(V1 / "data/dataset/global.parquet")
    labels = frame.pivot(index="date", columns="asset_id", values="y_excess_lead").reindex(columns=ASSETS)
    dates = np.array(read(HERE / "SPLITS.json")["origins"])
    actual = labels.loc[pd.to_datetime(dates)].to_numpy()
    np.testing.assert_allclose(actual[:52], old["actual"], rtol=0, atol=1e-15)
    rf = np.r_[old["risk_free"], read(V1 / "data/coverage_audit.json")["build_audit"]["october_risk_free_return"]]
    means, covariance = [], []
    for component in COMPONENTS:
        first = arrays(CHRONO / "components" / f"{component}.npz")
        tail = arrays(V1 / "outputs/recent_components" / f"{component}.npz")
        for axis, expected in [("assets", ASSETS), ("seeds", SEEDS)]:
            np.testing.assert_array_equal(first[axis], expected)
            np.testing.assert_array_equal(tail[axis], expected)
        means.append(np.concatenate([first["mean"], tail["mean"]], axis=1))
        covariance.append(np.concatenate([first["covariance"], tail["covariance"]], axis=1))
    historical = []
    for origin in [*dates, "2026-10-30"]:
        available = labels.loc[labels.index < pd.Timestamp(str(origin))].to_numpy()[:, ELIGIBLE_INDEX]
        if not np.isfinite(available).all():
            raise ValueError("Unavailable target entered causal risk reference")
        historical.append(LedoitWolf().fit(available).covariance_)
    cov14 = np.stack(historical)
    cov15 = np.zeros((54, 15, 15))
    cov15[:, ELIGIBLE_INDEX[:, None], ELIGIBLE_INDEX] = cov14
    # MGK is a noninvestable task, not a conditional future observation.
    # This diagonal only completes the historical15 storage axis; never used
    # by the eligible principal-risk matrix or as a GP predictive covariance.
    cov15[:, 1, 1] = cov14.diagonal(axis1=-2, axis2=-1).mean(-1)
    live_target = labels.loc[pd.Timestamp("2026-10-30")].to_numpy()
    if not np.isnan(live_target).all():
        raise ValueError("November labels must remain unknown")
    value = {"dates": dates, "assets": np.array(ASSETS), "seeds": np.array(SEEDS), "actual": actual,
             "rf": rf, "mean": np.stack(means), "covariance": np.stack(covariance),
             "historical_covariance": cov15, "historical_EW_variance": cov14.sum((-2, -1)) / 196,
             "live_target": live_target}
    snapshot(HERE / "outputs/context.npz", value)
    return value


if __name__ == "__main__":
    prepare()
    context()
