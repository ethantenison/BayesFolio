from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bayesfolio.io.fingerprints import sha256_fingerprint
from bayesfolio.schemas.common import ArtifactRef, SchemaMetadata


def write_parquet_with_metadata(
    frame: pd.DataFrame,
    output_path: str | Path,
    metadata: SchemaMetadata,
) -> ArtifactRef:
    """Persist DataFrame parquet plus JSON metadata sidecar.

    Args:
        frame: DataFrame payload to persist.
        output_path: Target parquet path.
        metadata: Cross-boundary metadata to save as sidecar JSON.

    Returns:
        ArtifactRef: Artifact pointer with integrity fingerprint.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=True)

    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata.model_dump(mode="json"), indent=2), encoding="utf-8")

    return ArtifactRef(path=str(path), format="parquet", fingerprint=sha256_fingerprint(path))
