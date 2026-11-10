"""Freeze November candidates and promotion gates before new scores/downloads.

Experiment-only I/O; prior artifacts and production are read-only. This stage
does not allocate portfolios or access fresh October market outcomes.
Usage: uv run --locked --no-sync python <this_file>
"""

from __future__ import annotations

from datetime import UTC, datetime

from common import CHRONO, COMPONENTS, FRAMEWORK, HERE, REPO, TUNING, digest, read, save, status, verify_hashes
from decision_math import BLOCKS, candidates


def prepare() -> dict:
    """Freeze exact candidate identities, temporal blocks, operational gates and inputs.

    Returns:
        Immutable preregistration; no new scoring has occurred in this stage.
    """
    path = HERE / "PREREGISTRATION.json"
    if path.exists():
        value = read(path)
        verify_hashes(value["protected_sha256"])
        return value
    menu = candidates()
    files = [HERE / "PLAN.md", CHRONO / "CONTRACT.json", CHRONO / "COMPLETED.json",
             CHRONO / "context52.npz", CHRONO / "RESULTS.md",
             FRAMEWORK / "CURRENT_BASELINES.json", FRAMEWORK / "CURRENT_BENCHMARKS.json",
             TUNING / "three_way_joint_v1/outputs/RESULTS.md",
             TUNING / "scenario_numerics_v2/outputs/RESULTS.md",
             TUNING / "scenario_ne7_uncapped_v1/outputs/RESULTS.md",
             REPO / "experiments/2026-10-monthly-data-extension-v1/outputs/dataset/global.parquet",
             REPO / "experiments/2026-10-monthly-data-extension-v1/outputs/dataset/unscored_september_origin.parquet"]
    files.extend(CHRONO / "components" / f"{name}.npz" for name in COMPONENTS)
    protected = {str(item.resolve()): digest(item) for item in files}
    value = {
        "schema": "bayesfolio.november_decision.preregistration.v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "as_of_date": "2026-10-31", "market_origin": "2026-10-30", "forecast_month": "2026-11",
        "deadline": "2026-11-01T18:00:00-06:00", "exploration_stop": "2026-11-01T16:30:00-06:00",
        "caps": {"batches": 2, "new_recipes_per_batch": 48, "monthly_fit_attempts": 128,
                 "fit_workers": 14, "finalists": 15},
        "candidate_menu": menu,
        "batch1_new_recipes": sum(not item["control"] for item in menu),
        "mandatory_controls": [item["id"] for item in menu if item["control"]],
        "batch2": "not required; only a documented unresolved mechanism, never absence of a win",
        "components": COMPONENTS, "seeds": [27, 127, 227],
        "live_policy": "static family blend then equal seed mean/covariance pooling then one allocation",
        "covariance": "0.5 broad + 0.5 compact observed covariance; no third/seed disagreement term",
        "historical_origins": "2022-05-31 through 2026-08-31",
        "historical_realized_months": "June2022 through September2026",
        "blocks": BLOCKS,
        "promotion_gates": {
            "all52_sharpe_delta_min": .05, "all52_IC_delta_min": .01,
            "all52_return_delta_min": -1e-10, "later28_sharpe_delta_min": -.05,
            "later28_IC_delta_min": -.01, "later28_return_delta_min": -.01,
            "nonnegative_excess_advantage_blocks_min": 4,
            "all_leave_one_month_out_sharpe_deltas_positive": True,
            "max_drawdown_delta_min": -.01, "any_family_IC_delta_min": -.05,
            "interpretation": "conservative operational gates, not significance tests"},
        "finalist_selection": "five controls plus up to10 ranked by worst normalized gate slack, Sharpe/IC/return",
        "winner_selection": "eligible Pareto set then minimum coefficient distance from incumbent; "
                            "ties higher Sharpe, IC, return, then ID; otherwise incumbent",
        "fresh_outcomes": "October is a separately reported diagnostic after historical selection freezes",
        "protected_sha256": protected,
        "provenance_limitations": ["All52 historical outcomes and candidate ideas were already inspected.",
                                   "Macro sources preserve latest-vintage limitations and existing extra feature lag.",
                                   "Returns are gross; Friday closing assumptions differ from Monday execution."]}
    save(path, value)
    save(HERE / "PREREGISTRATION.sha256", (digest(path) + "\n").encode())
    status("preregistered", "Analyze cached reversal and replay pooled candidates; refresh October prices separately",
           fit_attempts_used=0, fit_attempts_remaining=128, batches_used=0,
           preregistered_new_recipes=value["batch1_new_recipes"], failures=[])
    return value


if __name__ == "__main__":
    result = prepare()
    print(f"Frozen {len(result['candidate_menu'])} controls/recipes before new scoring or October downloads.")
