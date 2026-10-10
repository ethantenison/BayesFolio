"""Tests for the concrete monthly portfolio runtime adapter."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest

from bayesfolio.contracts.base import Meta
from bayesfolio.io.fingerprints import sha256_digest
from bayesfolio.io.monthly_portfolio_runtime import LocalMlflowMonthlyPortfolioRunIO


def test_start_run_configures_mlflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []
    context = nullcontext(object())
    monkeypatch.setattr(
        "bayesfolio.io.monthly_portfolio_runtime.mlflow.set_tracking_uri",
        lambda value: calls.append(("tracking_uri", value)),
    )
    monkeypatch.setattr(
        "bayesfolio.io.monthly_portfolio_runtime.mlflow.set_experiment",
        lambda value: calls.append(("experiment", value)),
    )
    monkeypatch.setattr(
        "bayesfolio.io.monthly_portfolio_runtime.mlflow.start_run",
        lambda *, run_name: calls.append(("run_name", run_name)) or context,
    )
    adapter = LocalMlflowMonthlyPortfolioRunIO(
        artifact_root=tmp_path,
        mlflow_tracking_uri="sqlite:///tracking.db",
        mlflow_experiment_name="monthly-test",
    )

    assert adapter.start_run(run_name="july") is context
    assert calls == [
        ("tracking_uri", "sqlite:///tracking.db"),
        ("experiment", "monthly-test"),
        ("run_name", "july"),
    ]


def test_adapter_persists_exact_csv_and_json_bytes(tmp_path: Path) -> None:
    adapter = LocalMlflowMonthlyPortfolioRunIO(artifact_root=tmp_path)
    frame = pd.DataFrame({"asset": ["SPY", "VTV"], "weight": [0.55, 0.45]})

    csv_pointer = adapter.save_csv(frame, "run/weights.csv")
    json_pointer = adapter.save_json_contract(Meta(producer="test"), "run/summary.json")

    csv_path = tmp_path / "run/weights.csv"
    json_path = tmp_path / "run/summary.json"
    csv_bytes = csv_path.read_bytes()
    json_bytes = json_path.read_bytes()
    assert csv_bytes == frame.to_csv(index=False).encode("utf-8")
    assert b'  "producer": "test"' in json_bytes
    assert csv_pointer.path == str(csv_path)
    assert csv_pointer.digest == sha256_digest(csv_bytes)
    assert csv_pointer.byte_size == len(csv_bytes)
    assert json_pointer.path == str(json_path)
    assert json_pointer.digest == sha256_digest(json_bytes)
    assert json_pointer.byte_size == len(json_bytes)


def test_adapter_loads_parquet(tmp_path: Path) -> None:
    path = tmp_path / "features.parquet"
    expected = pd.DataFrame({"asset_id": ["SPY"], "y_excess_lead": [0.02]})
    expected.to_parquet(path, index=False)
    adapter = LocalMlflowMonthlyPortfolioRunIO(artifact_root=tmp_path)

    actual = adapter.load_parquet(str(path))

    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("revision", ["abc123", None])
def test_log_source_revision_is_fail_soft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    revision: str | None,
) -> None:
    logged: list[tuple[str, str]] = []
    monkeypatch.setattr("bayesfolio.io.monthly_portfolio_runtime._get_git_sha", lambda: revision)
    monkeypatch.setattr(
        "bayesfolio.io.monthly_portfolio_runtime.mlflow.log_param",
        lambda name, value: logged.append((name, value)),
    )
    adapter = LocalMlflowMonthlyPortfolioRunIO(artifact_root=tmp_path)

    adapter.log_source_revision()

    assert logged == ([] if revision is None else [("git_sha", revision)])
