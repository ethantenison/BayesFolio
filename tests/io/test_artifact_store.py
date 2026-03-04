from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bayesfolio.io.artifact_store import ParquetArtifactStore


class FakeBackend:
    """In-memory backend for artifact store tests."""

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
            "date": ["2026-01-31", "2026-02-28"],
            "asset_id": ["SPY", "SPY"],
            "y_excess_lead": [0.01, 0.02],
        }
    )


def test_parquet_artifact_store_supports_pluggable_backend() -> None:
    backend = FakeBackend()
    store = ParquetArtifactStore(base_dir="features", backend=backend)

    pointer = store.save_parquet(
        frame=_sample_frame(),
        artifact_name="features_pluggable",
        metadata={"return_unit": "decimal"},
    )

    assert pointer.uri == "memory://features/features_pluggable.parquet"
    assert pointer.format == "parquet"
    assert pointer.row_count == 2
    assert pointer.column_count == 3
    assert pointer.fingerprint

    metadata_key = "features/features_pluggable.parquet.meta.json"
    assert metadata_key in backend.payloads
    metadata_payload = json.loads(backend.payloads[metadata_key].decode("utf-8"))
    assert metadata_payload["metadata"]["return_unit"] == "decimal"


def test_parquet_artifact_store_absolute_base_dir_is_backward_compatible(tmp_path: Path) -> None:
    store = ParquetArtifactStore(base_dir=tmp_path)

    pointer = store.save_parquet(
        frame=_sample_frame(),
        artifact_name="features_absolute",
        metadata={"run": "local"},
    )

    expected_path = tmp_path / "features_absolute.parquet"
    assert expected_path.exists()
    assert pointer.uri == expected_path.as_uri()


def test_parquet_artifact_store_uses_root_uri_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BAYESFOLIO_ARTIFACT_ROOT_URI", tmp_path.as_uri())
    store = ParquetArtifactStore(base_dir="artifacts/features")

    pointer = store.save_parquet(
        frame=_sample_frame(),
        artifact_name="features_env",
        metadata={"source": "env"},
    )

    expected_path = tmp_path / "artifacts" / "features" / "features_env.parquet"
    assert expected_path.exists()
    assert pointer.uri == expected_path.as_uri()
