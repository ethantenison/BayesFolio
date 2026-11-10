"""Synthetic smoke test of final report wiring, without GP fits or allocations."""

from __future__ import annotations

import json
from pathlib import Path

import campaign
import numpy as np
import pandas as pd
import run_study


def test_global_report_preserves_selection_period_and_builds_artifacts(tmp_path: Path, monkeypatch) -> None:
    """Final integration handles controls, fixed recipes, benchmarks and plots."""
    monkeypatch.setattr(campaign, "OUTPUT", tmp_path)
    monkeypatch.setattr(run_study, "OUTPUT", tmp_path)
    cases = tmp_path / "cases"
    cases.mkdir()
    periods = ["tuning_first16", "validation_last12", "all28"]
    origins = pd.date_range("2024-05-31", periods=28, freq="BME").strftime("%Y-%m-%d").to_numpy(dtype=str)
    rf = np.full(28, .002)
    schwab = .007 + .02 * np.sin(np.arange(28))
    momentum = .006 + .02 * np.cos(np.arange(28))
    np.savez(tmp_path / "context.npz", dates=origins, risk_free=rf,
             benchmark_schwab_total=schwab, benchmark_momentum_total=momentum)
    for identity, allocator, level in [("control_hard_rank", "hard_rank", 1.),
                                       ("control_scenarios", "scenarios", .8),
                                       ("candidate", "hard_rank", 1.1)]:
        case = run_study._baseline_case(allocator, identity)
        case["track"] = "control" if identity.startswith("control_") else "track2"
        record = {"id": identity, "case": case, "status": "complete", "periods": {
            period: {"sharpe": level, "total_return": .2 * level, "rank_ic": .2 * level,
                     "max_drawdown": -.05, "ending_value": 10000 * (1 + .2 * level)}
            for period in periods}}
        (cases / f"{identity}.json").write_text(json.dumps(record))
        wealth = np.broadcast_to(np.linspace(10000, 12000 * level, 29), (3, 29))
        np.savez(cases / f"{identity}.npz", wealth=wealth)
    run_study._combine({"track1": {}, "track2": {}, "track3": {}})
    summary = json.loads((tmp_path / "GLOBAL_SUMMARY.json").read_text())
    assert summary["first16_selected_case_ids"] == ["candidate"]
    assert summary["paired_findings"][0]["later12_pareto_dominates_reference"] is True
    assert summary["automatic_promotion"] is False
    assert summary["monthly_blend_adaptation"] is False
    assert (tmp_path / "GLOBAL_COMPARISON.png").stat().st_size > 10000
    assert (tmp_path / "RESULTS.md").exists()
