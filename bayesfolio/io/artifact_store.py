from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bayesfolio.contracts.results.features import ArtifactPointer
from bayesfolio.io.fingerprints import sha256_fingerprint

logger = logging.getLogger(__name__)


class ParquetArtifactStore:
    """Persist feature datasets as parquet with deterministic fingerprints."""

    def __init__(self, base_dir: str | Path = "artifacts/features") -> None:
        """Initialize artifact store.

        Args:
            base_dir: Base directory used for parquet and metadata sidecars.
        """

        self._base_dir = Path(base_dir)

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
        output_path = self._base_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Persisting features dataset to %s", output_path)
        frame.to_parquet(output_path, index=False)

        metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        fingerprint, _ = sha256_fingerprint(output_path)
        return ArtifactPointer(
            uri=str(output_path),
            fingerprint=fingerprint,
            row_count=int(frame.shape[0]),
            column_count=int(frame.shape[1]),
        )
