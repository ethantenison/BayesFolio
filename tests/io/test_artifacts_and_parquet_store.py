from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bayesfolio.contracts.base import Meta
from bayesfolio.contracts.results.report import ArtifactPointer, ReportResult
from bayesfolio.io.artifacts import save_dataframe_csv, save_json_contract, save_plotly_figure_html
from bayesfolio.io.parquet_store import write_parquet_with_metadata
from bayesfolio.io.report_artifacts import persist_report_diagnostic_figures


class FakeBackend:
    """In-memory backend for IO artifact tests."""

    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}

    def put_bytes(self, key: str, payload: bytes) -> str:
        self.payloads[key] = payload
        return f"memory://{key}"

    def exists(self, key: str) -> bool:
        return key in self.payloads


class FakeFigure:
    def __init__(self, text: str) -> None:
        self._text = text

    def to_html(self, *, full_html: bool, include_plotlyjs: str) -> str:
        return (
            f"<html><head><title>{include_plotlyjs}</title></head>"
            f"<body data-full='{full_html}'>{self._text}</body></html>"
        )


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_id": ["SPY", "VTV"],
            "ret": [0.01, 0.02],
        }
    )


def test_save_json_contract_uses_backend_factory() -> None:
    payload = Meta(producer="unit-test")
    backend = FakeBackend()

    pointer = save_json_contract(payload, "contracts/meta.json", backend=backend)

    assert isinstance(pointer, ArtifactPointer)
    assert pointer.path == "memory://contracts/meta.json"
    assert pointer.artifact_format == "json"
    assert pointer.byte_size > 0

    persisted = json.loads(backend.payloads["contracts/meta.json"].decode("utf-8"))
    assert persisted["producer"] == "unit-test"


def test_save_dataframe_csv_uses_backend_factory() -> None:
    backend = FakeBackend()
    pointer = save_dataframe_csv(_sample_frame(), "exports/sample.csv", backend=backend)

    assert pointer.path == "memory://exports/sample.csv"
    assert pointer.artifact_format == "csv"
    assert pointer.byte_size > 0

    csv_text = backend.payloads["exports/sample.csv"].decode("utf-8")
    assert "asset_id" in csv_text
    assert "SPY" in csv_text


def test_write_parquet_with_metadata_uses_backend_factory() -> None:
    backend = FakeBackend()
    pointer = write_parquet_with_metadata(
        frame=_sample_frame(),
        output_path="exports/sample.parquet",
        metadata=Meta(producer="parquet-test"),
        backend=backend,
    )

    assert pointer.path == "memory://exports/sample.parquet"
    assert pointer.artifact_format == "parquet"
    assert pointer.byte_size > 0

    metadata_key = "exports/sample.parquet.meta.json"
    assert metadata_key in backend.payloads
    metadata_payload = json.loads(backend.payloads[metadata_key].decode("utf-8"))
    assert metadata_payload["producer"] == "parquet-test"


def test_write_parquet_with_metadata_absolute_path_is_backward_compatible(tmp_path: Path) -> None:
    output_path = tmp_path / "table.parquet"
    pointer = write_parquet_with_metadata(_sample_frame(), output_path, Meta(producer="local"))

    assert output_path.exists()
    assert pointer.path == output_path.as_uri()


def test_save_plotly_figure_html_uses_backend_factory() -> None:
    backend = FakeBackend()

    pointer = save_plotly_figure_html(
        figure=FakeFigure("diagnostic"),
        output_path="reports/diagnostic.html",
        backend=backend,
    )

    assert pointer.path == "memory://reports/diagnostic.html"
    assert pointer.artifact_format == "html"
    assert pointer.byte_size > 0
    assert "reports/diagnostic.html" in backend.payloads


def test_persist_report_diagnostic_figures_appends_artifacts() -> None:
    backend = FakeBackend()
    base_report = {
        "headline_metrics": {"sharpe_ratio": 0.5},
        "market_structure": None,
        "diagnostic_figures": [],
    }

    report_result = ReportResult(**base_report)
    updated = persist_report_diagnostic_figures(
        report_result=report_result,
        figures={
            "feature_target_correlation_heatmap": FakeFigure("heatmap"),
            "target_histogram": FakeFigure("hist"),
        },
        output_dir="reports/diagnostics",
        backend=backend,
    )

    assert len(updated.artifacts) == 2
    paths = [artifact.path for artifact in updated.artifacts]
    assert "memory://reports/diagnostics/feature_target_correlation_heatmap.html" in paths
    assert "memory://reports/diagnostics/target_histogram.html" in paths
