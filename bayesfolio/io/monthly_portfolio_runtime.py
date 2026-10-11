"""Concrete local and MLflow runtime adapter for monthly portfolio runs.

This I/O-layer adapter configures MLflow, discovers the Git revision, loads
parquet feature tables, and persists index-free CSV or indented JSON artifacts.
It preserves configured local path strings and exact serialized bytes while
leaving GP, forecast, and portfolio decisions to the engine.
"""

from __future__ import annotations

import json
import subprocess
from contextlib import AbstractContextManager
from pathlib import Path

import mlflow
import pandas as pd
from pydantic import BaseModel

from bayesfolio.contracts.commands.monthly_portfolio import MonthlyPortfolioArtifactsConfig
from bayesfolio.contracts.results.report import ArtifactPointer
from bayesfolio.io.fingerprints import sha256_digest
from bayesfolio.io.mlflow_logger import log_contract


class LocalMlflowMonthlyPortfolioRunIO:
    """Provide local artifact persistence and MLflow tracking for monthly runs.

    Args:
        artifact_root: Local root beneath which run-relative artifacts are stored.
        mlflow_tracking_uri: Optional MLflow tracking URI configured before a run.
        mlflow_experiment_name: Optional MLflow experiment selected before a run.
    """

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment_name: str | None = None,
    ) -> None:
        """Initialize the adapter from artifact and MLflow configuration.

        Args:
            artifact_root: Local root beneath which run-relative artifacts are stored.
            mlflow_tracking_uri: Optional MLflow tracking URI configured before a run.
            mlflow_experiment_name: Optional MLflow experiment selected before a run.
        """

        self._artifact_root = Path(artifact_root)
        self._mlflow_tracking_uri = mlflow_tracking_uri
        self._mlflow_experiment_name = mlflow_experiment_name

    def start_run(self, *, run_name: str) -> AbstractContextManager[object]:
        """Configure MLflow and start a named run context.

        Args:
            run_name: Stable label for the monthly run.

        Returns:
            MLflow run context manager.
        """

        if self._mlflow_tracking_uri:
            mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        if self._mlflow_experiment_name:
            mlflow.set_experiment(self._mlflow_experiment_name)
        return mlflow.start_run(run_name=run_name)

    def log_contract(self, name: str, payload: BaseModel) -> None:
        """Log a contract beneath MLflow's ``contracts/`` artifact path.

        Args:
            name: Stable artifact name without a file extension.
            payload: Validated Pydantic contract to serialize.
        """

        log_contract(name, payload)

    def log_source_revision(self) -> None:
        """Log the current Git SHA when discovery succeeds."""

        git_sha = _get_git_sha()
        if git_sha is not None:
            mlflow.log_param("git_sha", git_sha)

    def load_parquet(self, uri: str) -> pd.DataFrame:
        """Load a parquet feature table from a path or URI.

        Args:
            uri: Parquet location returned by the feature artifact store.

        Returns:
            Loaded feature table.
        """

        return pd.read_parquet(uri)

    def save_csv(self, frame: pd.DataFrame, relative_path: str) -> ArtifactPointer:
        """Persist an index-free CSV and return exact byte metadata.

        Args:
            frame: Table to serialize without its pandas index.
            relative_path: Relative output path beneath the artifact root.

        Returns:
            Pointer containing the configured path string and content metadata.
        """

        content = frame.to_csv(index=False).encode("utf-8")
        return self._save_bytes(relative_path=relative_path, content=content, artifact_format="csv")

    def save_json_contract(self, payload: BaseModel, relative_path: str) -> ArtifactPointer:
        """Persist a contract as two-space-indented JSON.

        Args:
            payload: Validated Pydantic contract to serialize.
            relative_path: Relative output path beneath the artifact root.

        Returns:
            Pointer containing the configured path string and content metadata.
        """

        content = json.dumps(payload.model_dump(mode="json"), indent=2).encode("utf-8")
        return self._save_bytes(relative_path=relative_path, content=content, artifact_format="json")

    def _save_bytes(self, *, relative_path: str, content: bytes, artifact_format: str) -> ArtifactPointer:
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError("Artifact output path must be relative to artifact_root.")

        output_path = self._artifact_root / relative
        resolved_root = self._artifact_root.resolve()
        resolved_output = output_path.resolve()
        if not resolved_output.is_relative_to(resolved_root):
            raise ValueError("Artifact output path must remain beneath artifact_root.")

        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_bytes(content)
        return ArtifactPointer(
            path=str(output_path),
            artifact_format=artifact_format,
            digest=sha256_digest(content),
            byte_size=len(content),
        )


def build_monthly_portfolio_run_io(
    config: MonthlyPortfolioArtifactsConfig,
) -> LocalMlflowMonthlyPortfolioRunIO:
    """Build the concrete monthly runtime adapter from its contract config.

    Args:
        config: Monthly artifact and MLflow configuration.

    Returns:
        Configured local/MLflow runtime adapter.
    """

    return LocalMlflowMonthlyPortfolioRunIO(
        artifact_root=config.artifact_root,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        mlflow_experiment_name=config.mlflow_experiment_name,
    )


def _get_git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
