from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path
from typing import Any

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


def save_plotly_figure_html(
    figure: Any,
    output_path: str | Path,
    *,
    root_uri: str | Path | None = None,
    backend: ArtifactBackend | None = None,
) -> ArtifactPointer:
    """Serialize a Plotly-compatible figure to HTML with deterministic digest.

    Args:
        figure: Plotly figure-like object exposing ``to_html``.
        output_path: Output path/URI. If relative, resolved against ``root_uri``.
        root_uri: Optional root URI/path for relative outputs.
        backend: Optional explicit backend instance.

    Returns:
        Artifact pointer containing URI and SHA-256 digest metadata.
    """

    resolved_backend, key = resolve_backend_and_key(output_path, root_uri=root_uri, backend=backend)
    html = figure.to_html(full_html=True, include_plotlyjs="cdn")
    content = html.encode("utf-8")
    artifact_uri = resolved_backend.put_bytes(key, content)
    digest = hashlib.sha256(content).hexdigest()
    return ArtifactPointer(path=artifact_uri, artifact_format="html", digest=digest, byte_size=len(content))


def save_plotly_figures_html(
    figures: dict[str, Any],
    output_dir: str | Path,
    *,
    root_uri: str | Path | None = None,
    backend: ArtifactBackend | None = None,
) -> list[ArtifactPointer]:
    """Persist multiple Plotly-compatible figures to HTML files.

    Args:
        figures: Mapping of figure name to Plotly-compatible figure objects.
        output_dir: Directory under which ``<name>.html`` files are written.
        root_uri: Optional root URI/path for relative outputs.
        backend: Optional explicit backend instance.

    Returns:
        Artifact pointers in insertion order of ``figures``.
    """

    pointers: list[ArtifactPointer] = []
    for name, figure in figures.items():
        file_path = Path(output_dir) / f"{name}.html"
        pointers.append(
            save_plotly_figure_html(
                figure=figure,
                output_path=file_path,
                root_uri=root_uri,
                backend=backend,
            )
        )
    return pointers
