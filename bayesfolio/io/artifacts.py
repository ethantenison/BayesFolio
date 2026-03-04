from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from bayesfolio.contracts.results.report import ArtifactPointer
from bayesfolio.io.backends import ArtifactBackend, resolve_backend_and_key


def save_json_contract(
    payload: BaseModel,
    output_path: str | Path,
    *,
    root_uri: str | Path | None = None,
    backend: ArtifactBackend | None = None,
) -> ArtifactPointer:
    """Serialize a Pydantic contract to JSON with deterministic fingerprint.

    Args:
        payload: Pydantic payload to serialize.
        output_path: Output path or URI. If relative, it is resolved against
            ``root_uri`` or the default artifact root.
        root_uri: Optional root URI/path for relative outputs.
        backend: Optional explicit backend instance.

    Returns:
        Artifact pointer containing storage location and content digest.
    """

    resolved_backend, key = resolve_backend_and_key(output_path, root_uri=root_uri, backend=backend)
    content = json.dumps(payload.model_dump(mode="json"), indent=2).encode("utf-8")
    artifact_uri = resolved_backend.put_bytes(key, content)
    digest = hashlib.sha256(content).hexdigest()
    return ArtifactPointer(path=artifact_uri, artifact_format="json", digest=digest, byte_size=len(content))


def save_dataframe_csv(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    root_uri: str | Path | None = None,
    backend: ArtifactBackend | None = None,
) -> ArtifactPointer:
    """Serialize DataFrame to CSV with deterministic fingerprint.

    Args:
        frame: Tabular payload to serialize.
        output_path: Output path or URI. If relative, it is resolved against
            ``root_uri`` or the default artifact root.
        root_uri: Optional root URI/path for relative outputs.
        backend: Optional explicit backend instance.

    Returns:
        Artifact pointer containing storage location and content digest.
    """

    resolved_backend, key = resolve_backend_and_key(output_path, root_uri=root_uri, backend=backend)
    csv_buffer = StringIO()
    frame.to_csv(csv_buffer, index=True)
    content = csv_buffer.getvalue().encode("utf-8")
    artifact_uri = resolved_backend.put_bytes(key, content)
    digest = hashlib.sha256(content).hexdigest()
    return ArtifactPointer(path=artifact_uri, artifact_format="csv", digest=digest, byte_size=len(content))
