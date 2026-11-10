"""Fast mock execution tests; no real fits or portfolio allocations are launched."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import recipe_math as recipes
import run_track2 as runner


def _prepare_root(root: Path) -> None:
    component_root = root / "components"
    component_root.mkdir(parents=True)
    (root / "FITS_COMPLETED.json").write_text(json.dumps({"passed": True}))
    for family in ("broad", "compact"):
        for rank in recipes.RANKS:
            for maxiter in recipes.MAXITERS:
                (component_root / f"{family}_r{rank}_i{maxiter}_exp.npz").write_bytes(b"synthetic-component")
    np.savez(
        root / "context.npz", seeds=np.array([27, 127, 227]), assets=np.array(recipes.ASSETS),
        dates=np.array([f"origin_{index:02d}" for index in range(28)]), risk_free=np.full(28, 0.003),
        benchmark_schwab_total=np.tile([0.01, -0.005, 0.02, 0.005], 7),
        benchmark_momentum_total=np.tile([0.02, -0.01, 0.03, 0.005], 7),
    )


def _fake_batch(
    cases: list[dict], *, validation_factor: float = 1.0, incomplete_id: str | None = None
) -> list[dict]:
    records = []
    for case in cases:
        alpha = np.array(case["alpha"])
        broad_rank = int(case["broad_id"].split("_")[1][1:])
        compact_rank = int(case["compact_id"].split("_")[1][1:])
        fraction = case["cov_compact_fraction"]
        sharpe = 1.0 + broad_rank * 0.04 - compact_rank * 0.01 + np.dot(alpha, [0.1, -0.03, 0.05, 0.01])
        sharpe -= abs(fraction - 0.25) * 0.1
        total_return = 0.2 - broad_rank * 0.005 + compact_rank * 0.003 + np.mean(alpha) * 0.1
        rank_ic = 0.2 + np.dot(alpha, [-0.01, 0.03, -0.02, 0.02]) + broad_rank * 0.002
        periods = {}
        for period in runner.PERIODS:
            factor = 1.0 if period == recipes.SELECTION_PERIOD else validation_factor
            periods[period] = {
                "sharpe": float(sharpe * factor), "total_return": float(total_return * factor),
                "rank_ic": float(rank_ic * factor), "max_drawdown": -0.1,
                "ending_value": float(10000 * (1 + total_return * factor)), "per_seed": [],
            }
        records.append({
            "id": case["id"], "case": case, "status": "complete",
            "successful_allocations": 83 if case["id"] == incomplete_id else 84,
            "expected_allocations": 84, "periods": periods,
        })
    return records


def test_mock_track_runs_stages_deduplicates_and_writes_report(tmp_path: Path) -> None:
    """All static stages execute against a fake evaluator within their budget."""
    _prepare_root(tmp_path)
    called = []

    def fake(cases: list[dict]) -> list[dict]:
        called.extend(cases)
        return _fake_batch(cases)

    summary = runner._run_track(tmp_path, fake)
    assert 144 <= len(called) <= 468
    assert len(called) == len({runner._key(case) for case in called})
    assert summary["unique_case_count"] == len(called)
    assert summary["complete_case_count"] == len(called)
    assert summary["additional_gp_fits"] == 0
    assert len(summary["selected_best_case_ids"]) <= 3
    assert len(summary["selected_component_pair_case_ids"]) <= 3
    assert summary["selected_component_pair_case_ids"][0] == summary["baseline_case_id"]
    assert all(case["allocator"] == "hard_rank" and len(case["alpha"]) == 4 for case in called)
    assert (tmp_path / "track2/RESULTS.md").is_file()
    assert (tmp_path / "track2/tables.csv").is_file()
    assert json.loads((tmp_path / "track2/COMPLETED.json").read_text())["all_cases_successful"]
    assert "Schwab aggressive proxy" in summary["benchmarks"]
    for pair in summary["pair_results"]:
        for period in runner.PERIODS:
            for key in recipes.OBJECTIVES:
                component = pair["fixed_blend_minus_original_baseline"][period][key]
                mean = pair["mean_tuning_minus_fixed_blend"][period][key]
                covariance = pair["covariance_tuning_minus_selected_mean"][period][key]
                overall = pair["retuned_minus_original_baseline"][period][key]
                assert overall == pytest.approx(component + mean + covariance, rel=1e-12, abs=1e-14)
    repeated = runner._run_track(tmp_path, fake)
    assert repeated == summary


def test_changed_validation_or_all28_metrics_cannot_change_selection(tmp_path: Path) -> None:
    """Counterfactually reversed later results leave every selection unchanged."""
    positive_root, negative_root = tmp_path / "positive", tmp_path / "negative"
    _prepare_root(positive_root)
    _prepare_root(negative_root)
    first = runner._run_track(positive_root, lambda cases: _fake_batch(cases, validation_factor=1.0))
    second = runner._run_track(negative_root, lambda cases: _fake_batch(cases, validation_factor=-2.0))
    assert first["selected_best_case_ids"] == second["selected_best_case_ids"]
    assert first["selected_component_pair_case_ids"] == second["selected_component_pair_case_ids"]
    for path in sorted((positive_root / "track2/stages").glob("*selection.json")):
        other = negative_root / "track2/stages" / path.name
        assert json.loads(path.read_text()) == json.loads(other.read_text())
    assert first["selected_best_metrics"] != second["selected_best_metrics"]


def test_partially_completed_candidate_is_not_selected_or_partially_averaged(tmp_path: Path) -> None:
    """A status-complete record with only 83 allocations remains ineligible."""
    _prepare_root(tmp_path)
    bad_id = recipes.screen_cases([path.stem for path in (tmp_path / "components").glob("*.npz")])[0]["id"]
    summary = runner._run_track(tmp_path, lambda cases: _fake_batch(cases, incomplete_id=bad_id))
    assert bad_id in summary["ineligible_ids"]
    assert bad_id not in summary["selected_best_case_ids"]
    assert all(bad_id not in identifiers for identifiers in summary["pareto_ids_by_period"].values())
    seal = json.loads((tmp_path / "track2/COMPLETED.json").read_text())
    assert not seal["all_cases_successful"]


def test_incomplete_original_baseline_blocks_interpretation(tmp_path: Path) -> None:
    """A missing baseline cannot be silently replaced by another strategy."""
    _prepare_root(tmp_path)

    def fake(cases: list[dict]) -> list[dict]:
        records = _fake_batch(cases)
        for record in records:
            case = record["case"]
            if (case["broad_id"], case["compact_id"]) == runner.BASELINE_COMPONENTS:
                record["successful_allocations"] = 83
        return records

    with pytest.raises(ValueError, match="Original baseline is incomplete"):
        runner._run_track(tmp_path, fake)


def test_changed_immutable_selection_evidence_is_rejected(tmp_path: Path) -> None:
    """Resumption must not overwrite selected IDs or their tuning evidence."""
    path = tmp_path / "selection.json"
    runner._save(tmp_path, path, {"selected": ["old"]})
    with pytest.raises(ValueError, match="Conflicting immutable"):
        runner._save(tmp_path, path, {"selected": ["new"]})
    assert json.loads(path.read_text()) == {"selected": ["old"]}


def test_worker_limits_reject_oversubscription_before_starting_pool() -> None:
    """No accidental independent pool larger than the approved cap can start."""
    with pytest.raises(ValueError, match="at most fourteen"):
        runner.run_track("/unused", workers=15)
    with pytest.raises(ValueError, match="at most fourteen"):
        runner.run_track("/unused", workers=True)


def test_incomplete_fit_queue_blocks_allocation_before_any_case(tmp_path: Path) -> None:
    """The track cannot race the shared fitting queue or score missing panels."""
    _prepare_root(tmp_path)
    (tmp_path / "FITS_COMPLETED.json").write_text(json.dumps({"passed": False}))
    with pytest.raises(ValueError, match="fit queue must complete"):
        runner._run_track(tmp_path, _fake_batch)
