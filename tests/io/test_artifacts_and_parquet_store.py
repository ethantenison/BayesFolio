from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bayesfolio.contracts.base import Meta
from bayesfolio.contracts.results.report import ArtifactPointer
from bayesfolio.io.artifacts import save_dataframe_csv, save_json_contract
from bayesfolio.io.parquet_store import write_parquet_with_metadata


class FakeBackend:
    """In-memory backend for IO artifact tests."""

    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}

    def put_bytes(self, key: str, payload: bytes) -> str:
        self.payloads[key] = payload
        return f"memory://{key}"

    def exists(self, key: str) -> bool:
        return key in self.payloads


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
