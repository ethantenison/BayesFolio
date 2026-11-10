"""Pure construction and chronological-selection tests; no fits or allocations."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import run_track3 as runner
from run_track3 import build_cases, select_representatives


def _record(identifier: str, scores: tuple[float, float, float], *, complete: bool = True) -> dict:
    values = dict(zip(("sharpe", "total_return", "rank_ic"), scores, strict=True))
    return {"id": identifier, "status": "complete" if complete else "failed",
            "periods": {"tuning_first16": values, "validation_last12": values.copy(), "all28": values.copy()}}


def test_case_grid_is_bounded_static_and_deduplicated() -> None:
    """54 grid cells contain48 unique allocations plusone mandatory control."""
    cases, aliases = build_cases([f"component{index}" for index in range(6)])
    assert len(cases) == 49
    assert len(aliases) == 6
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["allocator"] == "hard_rank" for case in cases)
    assert all(isinstance(case["third_fraction"], float) for case in cases)
    assert all(case["alpha"] == [0.5, 0, 0.375, 0.25] for case in cases)
    assert cases[0]["third_id"] is None
    assert all(alias["third_fraction"] == alias["third_cov_fraction"] == 0.5 for alias in aliases)
    assert max(len(case["id"]) for case in cases) <= 160


def test_secondary_grid_preserves_selected_base_and_separates_ids() -> None:
    """Secondary recipes retain static Track2 settings without altering primary."""
    components = [f"component{index}" for index in range(6)]
    primary, _ = build_cases(components)
    reference = copy.deepcopy(primary[0])
    reference.update(broad_id="broad_r5_i36_exp", alpha=[0.25, 0, 0.5, 0.125], cov_compact_fraction=0.25)
    secondary, _ = build_cases(components, base_case=reference, stage="secondary_track2")
    assert all(case["broad_id"] == "broad_r5_i36_exp" for case in secondary)
    assert all(case["alpha"] == [0.25, 0, 0.5, 0.125] for case in secondary)
    assert all(case["cov_compact_fraction"] == 0.25 for case in secondary)
    assert {case["id"] for case in primary}.isdisjoint(case["id"] for case in secondary)


def test_selection_never_reads_later12_or_all28_values() -> None:
    """Even absurd future scores cannot alterfirst16 selected recipes."""
    records = [_record("control", (1, 0.2, 0.2)), _record("a", (2, 0.3, 0.3)),
               _record("b", (1.5, 0.4, 0.4))]
    expected = select_representatives(records, "control")
    mutated = copy.deepcopy(records)
    for record in mutated:
        record["periods"]["validation_last12"] = {"sharpe": float("nan")}
        record["periods"]["all28"] = {"sharpe": -1e99}
    assert select_representatives(mutated, "control") == expected
    assert expected["status"] == "complete"
    assert set(expected["selected_ids"]) == {"a", "b"}


def test_failed_case_prevents_winner_selection() -> None:
    """No partial campaign is mislabeledas a completedwinner."""
    result = select_representatives([_record("control", (1, 0.2, 0.2)),
                                     _record("missing", (9, 9, 9), complete=False)], "control")
    assert result["status"] == "incomplete"
    assert result["selected_ids"] == []
    assert result["failed_ids"] == ["missing"]


def test_dominated_complements_are_not_presented_as_winners() -> None:
    """The control mayremain best; no complementpromotion is forced."""
    result = select_representatives([_record("control", (2, 0.4, 0.4)),
                                     _record("worse", (1, 0.2, 0.2))], "control")
    assert result["status"] == "complete"
    assert result["selected_ids"] == []
    assert result["first16_pareto_ids"] == ["control"]


def test_nested_ensemble_base_is_rejected() -> None:
    """A thirdmodel cannot quietlybecome a hiddenfourth component."""
    cases, _ = build_cases([f"component{index}" for index in range(6)])
    with pytest.raises(ValueError, match="two-model"):
        build_cases([f"component{index}" for index in range(6)], base_case=cases[1], stage="secondary_track2")


def test_report_pipeline_with_mocked_case_evaluator_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise report wiring with synthetic panels and no real allocation calls."""
    generator = np.random.default_rng(37)
    registry = json.loads((Path(runner.__file__).parent / "components.json").read_text())
    complements = [item["modelname"] for item in registry["components"]]
    axes = {"seeds": np.array([27, 127, 227]), "assets": np.array([f"asset{index}" for index in range(15)]),
            "dates": np.arange(28)}
    context = {**axes, "actual": generator.normal(0, 0.03, (28, 15)), "risk_free": np.zeros(28),
               "benchmark_schwab_total": generator.normal(0.01, 0.02, 28),
               "benchmark_momentum_total": generator.normal(0.01, 0.02, 28)}
    np.savez_compressed(tmp_path / "context.npz", **context)
    (tmp_path / "components").mkdir()
    means = {}
    for identifier in ["broad_r2_i15_exp", "compact_r2_i15_exp", *complements]:
        means[identifier] = generator.normal(0, 0.02, (3, 28, 15))
        np.savez_compressed(tmp_path / "components" / f"{identifier}.npz", **axes, mean=means[identifier])

    def fake_cases(cases: list[dict], root: Path, workers: int) -> list[dict]:
        assert workers == 1
        records = []
        (root / "cases").mkdir(exist_ok=True)
        for case in cases:
            alpha = np.repeat(case["alpha"], [4, 4, 4, 3])
            mean = (1 - alpha) * means[case["broad_id"]] + alpha * means[case["compact_id"]]
            if case["third_id"] is not None:
                fraction = case["third_fraction"]
                mean = (1 - fraction) * mean + fraction * means[case["third_id"]]
            np.savez_compressed(root / "cases" / f"{case['id']}.npz", mean=mean)
            score = (1 + case["third_fraction"], 0.2 + 0.1 * case["third_fraction"],
                     0.2 + 0.1 * case["third_fraction"])
            record = _record(case["id"], score)
            record["case"] = case
            for period in record["periods"].values():
                period.update(max_drawdown=-0.03, ending_value=10000 * (1 + period["total_return"]),
                              per_seed=[{"seed": int(seed), **dict(zip(runner._OBJECTIVES, score, strict=True))}
                                        for seed in axes["seeds"]])
            records.append(record)
        return records

    def forbid_real_evaluator() -> None:
        pytest.fail("Real evaluator must never run in this report-wiring test")

    monkeypatch.setattr(runner, "_evaluate_cases", fake_cases)
    monkeypatch.setattr(runner, "_evaluator", forbid_real_evaluator)
    summary = runner.run_track(str(tmp_path), workers=1)
    assert summary["status"] == "complete"
    assert summary["stages"]["primary_approved"]["case_count"] == 49
    assert len(summary["stages"]["primary_approved"]["selection"]["selected_ids"]) <= 3
    assert (tmp_path / "track3/RESULTS.md").exists()
    assert (tmp_path / "track3/tables.csv").exists()
    assert (tmp_path / "track3/diversity.json").exists()
