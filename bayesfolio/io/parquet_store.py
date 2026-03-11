from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd

from bayesfolio.contracts.base import Meta
from bayesfolio.contracts.results.report import ArtifactPointer
from bayesfolio.io.backends import ArtifactBackend, resolve_backend_and_key
from bayesfolio.io.fingerprints import sha256_digest


def write_parquet_with_metadata(
    frame: pd.DataFrame,
    output_path: str | Path,
    metadata: Meta,
    *,
    root_uri: str | Path | None = None,
    backend: ArtifactBackend | None = None,
) -> ArtifactPointer:
    """Persist DataFrame parquet plus JSON metadata sidecar.

    Args:
        frame: DataFrame payload to persist.
        output_path: Target parquet path or URI. If relative, it is resolved
            against ``root_uri`` or the default artifact root.
        metadata: Cross-boundary metadata to save as sidecar JSON.
        root_uri: Optional root URI/path for relative outputs.
        backend: Optional explicit backend instance.

    Returns:
        ArtifactPointer: Artifact pointer with integrity fingerprint.
    """

    resolved_backend, key = resolve_backend_and_key(output_path, root_uri=root_uri, backend=backend)
    parquet_buffer = BytesIO()
    frame.to_parquet(parquet_buffer, index=True)
    parquet_bytes = parquet_buffer.getvalue()
    artifact_uri = resolved_backend.put_bytes(key, parquet_bytes)

    metadata_key = f"{key}.meta.json"
    metadata_bytes = json.dumps(metadata.model_dump(mode="json"), indent=2).encode("utf-8")
    resolved_backend.put_bytes(metadata_key, metadata_bytes)

    digest = sha256_digest(parquet_bytes)
    return ArtifactPointer(path=artifact_uri, artifact_format="parquet", digest=digest, byte_size=len(parquet_bytes))
