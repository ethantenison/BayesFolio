from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from bayesfolio.io.fingerprints import sha256_fingerprint
from bayesfolio.schemas.common import ArtifactRef


def save_json_contract(payload: BaseModel, output_path: str | Path) -> ArtifactRef:
    """Serialize a Pydantic contract to JSON with deterministic fingerprint."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload.model_dump(mode="json"), indent=2), encoding="utf-8")
    return ArtifactRef(path=str(path), format="json", fingerprint=sha256_fingerprint(path))


def save_dataframe_csv(frame: pd.DataFrame, output_path: str | Path) -> ArtifactRef:
    """Serialize DataFrame to CSV with deterministic fingerprint."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True)
    return ArtifactRef(path=str(path), format="csv", fingerprint=sha256_fingerprint(path))
