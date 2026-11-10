"""Synthetic I/O, reusable-comparator and static-chart smoke checks.

Usage: pytest -q <this_directory>. Uses temporary synthetic data only; no fitting,
optimizer calls, data refresh or modification of any existing study artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import run_benchmarks as runner
from plot_suite import render_growth

from artifacts import Reader


def test_custom_comparator_preserves_exact_date_and_asset_axes(tmp_path: Path) -> None:
    """A one-path custom comparator becomes a one-run array without repairs."""
    dates, assets = np.array(["2024-05-31", "2024-06-28"]), np.array(["A", "B"])
    path = tmp_path / "weights.npz"
    weights = np.array([[0.4, 0.6], [0.7, 0.3]])
    np.savez(path, weights=weights, dates=dates, assets=assets)
    options = argparse.Namespace(comparison_npz=str(path), run_id="other", label="Other")
    loaded, label = runner._comparison(Reader(), options, {"dates": dates, "assets": assets})
    np.testing.assert_array_equal(loaded, weights[None, ...])
    assert label == "Other"
    with pytest.raises(AssertionError):
        runner._comparison(Reader(), options, {"dates": dates, "assets": assets[::-1]})


def test_custom_comparator_cannot_replace_current_leader(tmp_path: Path) -> None:
    """A distinct run ID is mandatory before custom input is read."""
    options = argparse.Namespace(
        comparison_npz=str(tmp_path / "unused.npz"), run_id="current_sharpe_leader", label="Other"
    )
    with pytest.raises(ValueError, match="distinct"):
        runner._comparison(Reader(), options, {})


def test_output_writer_is_confined_and_immutable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the isolated output subtree can be written, and results cannot change."""
    monkeypatch.setattr(runner, "HERE", tmp_path)
    target = tmp_path / "outputs/run/result.json"
    runner._write(target, {"a": 1})
    runner._write(target, {"a": 1})
    assert json.loads(target.read_text()) == {"a": 1}
    with pytest.raises(ValueError, match="immutable"):
        runner._write(target, {"a": 2})
    with pytest.raises(ValueError, match="writable"):
        runner._write(tmp_path / "unrelated.json", {})


def test_monthly_history_is_registered_complete_and_cut_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read real parquet bytes but never include prices after the last decision."""
    monkeypatch.setattr(runner, "REPAIRED", tmp_path)
    raw = tmp_path / "artifacts/v1/raw"
    raw.mkdir(parents=True)
    index = pd.bdate_range("2021-01-01", "2024-12-31")
    metadata = {}
    for ticker, multiplier in [("A", 1), ("B", 2)]:
        frame = pd.DataFrame({"Adj Close": multiplier * np.linspace(90, 140, len(index))}, index=index)
        path = raw / f"{ticker}.parquet"
        frame.to_parquet(path)
        metadata[ticker] = {"sha256": runner.previous.base._hash(path)}
    (raw / "sources.json").write_text(json.dumps(metadata))
    dates, prices = runner._history(Reader(), ["A", "B"], ["2024-01-31", "2024-02-29"], 36)
    assert prices.shape == (38, 2)
    np.testing.assert_array_equal(dates[[0, -1]], np.array(["2021-01-29", "2024-02-29"], dtype="datetime64[D]"))
    np.testing.assert_allclose(prices[:, 1], 2 * prices[:, 0], rtol=0, atol=1e-12)
    metadata["A"]["sha256"] = "wrong"
    (raw / "sources.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="registered"):
        runner._history(Reader(), ["A", "B"], ["2024-01-31", "2024-02-29"], 36)


def test_chart_render_does_not_mutate_paths() -> None:
    """All six strategies render to a PNG without changing numerical evidence."""
    origins = np.arange(np.datetime64("2024-05", "M"), np.datetime64("2026-05", "M"))
    dates = ((origins + 1).astype("datetime64[D]") - 1).astype(str)
    arrays = {"dates": dates, "comparison_wealth": np.tile(np.linspace(10000, 16000, 25), (3, 1))}
    for index, name in enumerate(runner.LABELS):
        arrays[f"{name}_wealth"] = np.linspace(10000, 13500 + 300 * index, 25)
    before = {name: value.copy() for name, value in arrays.items()}
    png = render_growth(arrays, "Synthetic comparator", runner.LABELS)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    for key in arrays:
        np.testing.assert_array_equal(before[key], arrays[key])


def test_completed_run_cannot_silently_reuse_a_different_comparator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume must match comparator identity, not silently return another study."""
    monkeypatch.setattr(runner, "HERE", tmp_path)
    output = tmp_path / "outputs/current_sharpe_leader"
    output.mkdir(parents=True)
    (output / "COMPLETED.json").write_text("{}")
    (output / "contract.json").write_text(
        json.dumps({"comparison_source": "sealed_Sharpe_leader", "comparison_label": "Sharpe leader — hard rank"})
    )
    options = argparse.Namespace(
        run_id="current_sharpe_leader", comparison_npz=None, label="Different", verify=False, execute=True
    )
    with pytest.raises(ValueError, match="different"):
        runner._run(options)


def test_completed_run_same_identity_verifies_without_recalculation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completed reruns perform read-only verification instead of recomputation."""
    monkeypatch.setattr(runner, "HERE", tmp_path)
    output = tmp_path / "outputs/current_sharpe_leader"
    output.mkdir(parents=True)
    (output / "COMPLETED.json").write_text("{}")
    (output / "contract.json").write_text(
        json.dumps({"comparison_source": "sealed_Sharpe_leader", "comparison_label": "Sharpe leader — hard rank"})
    )
    calls = []
    monkeypatch.setattr(runner, "_verify", lambda path: calls.append(path))
    options = argparse.Namespace(
        run_id="current_sharpe_leader", comparison_npz=None, label=None, verify=False, execute=True
    )
    runner._run(options)
    assert calls == [output]
