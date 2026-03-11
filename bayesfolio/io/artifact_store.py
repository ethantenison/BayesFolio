from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd

from bayesfolio.contracts.results.features import ArtifactPointer
from bayesfolio.io.backends import ArtifactBackend, make_artifact_backend
from bayesfolio.io.fingerprints import sha256_digest

logger = logging.getLogger(__name__)


class ParquetArtifactStore:
    """Persist feature datasets as parquet with deterministic fingerprints."""

    def __init__(
        self,
        base_dir: str | Path = "artifacts/features",
        *,
        root_uri: str | Path | None = None,
        backend: ArtifactBackend | None = None,
    ) -> None:
        """Initialize artifact store.

        Args:
            base_dir: Relative artifact key prefix or absolute local output
                directory.
            root_uri: Optional local/remote backend root URI. If ``None``,
                defaults to ``BAYESFOLIO_ARTIFACT_ROOT_URI`` or
                ``~/.bayesfolio``.
            backend: Optional custom backend implementation.
        """

        if backend is not None and root_uri is not None:
            msg = "Pass either backend or root_uri, not both."
            raise ValueError(msg)

        base_dir_path = Path(base_dir)
        if backend is not None:
            self._backend = backend
            self._key_prefix = base_dir_path.as_posix().strip("/")
            return

        if base_dir_path.is_absolute() and root_uri is None:
            self._backend = make_artifact_backend(base_dir_path)
            self._key_prefix = ""
            return

        self._backend = make_artifact_backend(root_uri)
        self._key_prefix = base_dir_path.as_posix().strip("/")

    def save_parquet(
        self,
        frame: pd.DataFrame,
        artifact_name: str,
        metadata: dict[str, object],
    ) -> ArtifactPointer:
        """Save dataset and metadata, then return an artifact pointer.

        Args:
            frame: Long-format features dataset to persist.
            artifact_name: Artifact filename. ``.parquet`` is appended when
                missing.
            metadata: Serializable metadata persisted as JSON sidecar.

        Returns:
            ArtifactPointer with parquet URI/path, SHA-256 fingerprint, and
            table shape metadata.
        """

        output_name = artifact_name if artifact_name.endswith(".parquet") else f"{artifact_name}.parquet"
        output_key = self._join_key(self._key_prefix, output_name)

        buffer = BytesIO()
        frame.to_parquet(buffer, index=False)
        parquet_bytes = buffer.getvalue()
        artifact_uri = self._backend.put_bytes(output_key, parquet_bytes)

        logger.info("Persisting features dataset to %s", artifact_uri)

        metadata_key = f"{output_key}.meta.json"
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
        }
        metadata_bytes = json.dumps(payload, indent=2).encode("utf-8")
        self._backend.put_bytes(metadata_key, metadata_bytes)

        fingerprint = sha256_digest(parquet_bytes)

        return ArtifactPointer(
            uri=artifact_uri,
            fingerprint=fingerprint,
            row_count=int(frame.shape[0]),
            column_count=int(frame.shape[1]),
        )

    @staticmethod
    def _join_key(prefix: str, name: str) -> str:
        normalized_name = name.lstrip("/")
        if not prefix:
            return normalized_name
        return f"{prefix.rstrip('/')}/{normalized_name}"
