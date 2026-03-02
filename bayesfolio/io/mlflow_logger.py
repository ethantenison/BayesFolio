from __future__ import annotations

import mlflow
from pydantic import BaseModel

from bayesfolio.schemas.common import ArtifactRef


def log_contract(name: str, payload: BaseModel) -> None:
    """Log a Pydantic payload as MLflow JSON artifact."""

    mlflow.log_dict(payload.model_dump(mode="json"), f"contracts/{name}.json")


def log_artifact_ref(name: str, artifact: ArtifactRef) -> None:
    """Log persisted artifact metadata into MLflow params."""

    mlflow.log_param(f"artifact.{name}.path", artifact.path)
    mlflow.log_param(f"artifact.{name}.format", artifact.format)
    mlflow.log_param(f"artifact.{name}.hash_algo", artifact.fingerprint.algorithm)
    mlflow.log_param(f"artifact.{name}.hash", artifact.fingerprint.digest)
    mlflow.log_param(f"artifact.{name}.bytes", artifact.fingerprint.byte_size)
