"""Test preregistered scenario case enumeration without allocating or fitting."""

from __future__ import annotations

import run_track as track


def test_manifest_grid_contains_36_cases_and_one_frozen_hard_control() -> None:
    """The rank/budget Cartesian grid contains only the approved fixed recipes."""
    cases = track.build_cases()
    assert len(cases) == len({case["id"] for case in cases}) == 37
    scenarios = [case for case in cases if case["allocator"] == "scenarios"]
    hard = [case for case in cases if case["allocator"] == "hard_rank"]
    assert len(scenarios) == 36 and len(hard) == 1
    assert hard[0]["alpha"] == [0.5, 0, 0.375, 0.25]
    assert hard[0]["broad_id"] == "broad_r2_i15_exp"
    assert hard[0]["compact_id"] == "compact_r2_i15_exp"
    assert hard[0]["cov_compact_fraction"] == 0.5
    for rank in [1, 2, 5]:
        for maxiter in [5, 15, 36, 75]:
            ids = {case["id"] for case in scenarios}
            assert {f"t1_r{rank}_i{maxiter}_{source}" for source in ["broad", "compact", "blend"]} <= ids
    assert all(case["third_id"] is None and case["third_fraction"] == 0 for case in cases)


def test_source_endpoints_use_consistent_means_and_covariance() -> None:
    """Broad/compact-only recipes cannot blend the other model's covariance."""
    for case in track.build_cases():
        if case["id"].endswith("_broad"):
            assert case["alpha"] == [0, 0, 0, 0] and case["cov_compact_fraction"] == 0
        elif case["id"].endswith("_compact"):
            assert case["alpha"] == [1, 1, 1, 1] and case["cov_compact_fraction"] == 1
        else:
            assert case["alpha"] == [0.5, 0, 0.375, 0.25] and case["cov_compact_fraction"] == 0.5


def test_selection_rows_exclude_incomplete_and_hard_rank_controls() -> None:
    """Failed cases are explicit elsewhere and cannot choose the scenario winner."""
    metric = {"sharpe": 1, "total_return": 0.2, "rank_ic": 0.3}
    rows = [
        {"id": "valid", "status": "complete", "successful_allocations": 84,
         "case": {"allocator": "scenarios"}, "periods": {"tuning_first16": metric}},
        {"id": "failed", "status": "failed", "successful_allocations": 83,
         "case": {"allocator": "scenarios"}, "periods": {}},
        {"id": "hard", "status": "complete", "successful_allocations": 84,
         "case": {"allocator": "hard_rank"}, "periods": {"tuning_first16": metric}},
    ]
    selected = track._candidate_rows(rows, "tuning_first16")
    assert selected == [{"id": "valid", "period": "tuning_first16", **metric}]


def test_controlled_contrasts_and_report_keep_all_periods_separate() -> None:
    """Every rank/budget contrast is paired by representation, seed and period."""
    records = []
    for case in track.build_cases():
        metric = {"sharpe": 1.2, "total_return": 0.3, "rank_ic": 0.2,
                  "max_drawdown": -0.06, "ending_value": 13000,
                  "effective_assets": 10.5, "equity_weight": 0.6}
        metric["per_seed"] = [{"seed": seed, **{key: metric[key] for key in track._OBJECTIVES}}
                              for seed in [27, 127, 227]]
        records.append({"id": case["id"], "case": case, "status": "complete", "successful_allocations": 84,
                        "periods": {period: metric for period in track._PERIODS}})
    contrasts = track._contrasts(records)
    assert len(contrasts) == (3 * 4 * 2 + 3 * 3 * 3) * 3 == 153
    assert all(row["status"] == "complete" for row in contrasts)
    assert all(row["deltas"] == {key: 0 for key in track._OBJECTIVES} for row in contrasts)
    summary = {"complete_scenario_cases": 36, "failed_scenario_cases": 0,
               "selected_case_ids": ["t1_r2_i15_blend"], "benchmarks": {},
               "paired_contrasts": contrasts, "failures": []}
    report = track._report(summary, records).decode()
    assert "not a pure allocator experiment" in report
    assert "tuning_first16" in report and "validation_last12" in report and "all28" in report
    assert "t1_reference_hard_rank" in report
    table = track._table(records, summary["selected_case_ids"], ["t1_r2_i15_blend"]).decode()
    assert len(table.splitlines()) == 38
