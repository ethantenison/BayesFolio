from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from bayesfolio.contracts.results.report import ArtifactPointer
from bayesfolio.io.fingerprints import sha256_fingerprint


def save_json_contract(payload: BaseModel, output_path: str | Path) -> ArtifactPointer:
    """Serialize a Pydantic contract to JSON with deterministic fingerprint."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload.model_dump(mode="json"), indent=2), encoding="utf-8")
    digest, byte_size = sha256_fingerprint(path)
    return ArtifactPointer(path=str(path), artifact_format="json", digest=digest, byte_size=byte_size)


def save_dataframe_csv(frame: pd.DataFrame, output_path: str | Path) -> ArtifactPointer:
    """Serialize DataFrame to CSV with deterministic fingerprint."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True)
    digest, byte_size = sha256_fingerprint(path)
    return ArtifactPointer(path=str(path), artifact_format="csv", digest=digest, byte_size=byte_size)
