from __future__ import annotations

import hashlib
from pathlib import Path

from bayesfolio.schemas.common import ArtifactFingerprint


def sha256_fingerprint(file_path: str | Path) -> ArtifactFingerprint:
    """Compute SHA-256 fingerprint for a persisted artifact.

    Args:
        file_path: Absolute or relative path to an existing file.

    Returns:
        ArtifactFingerprint: Hash algorithm, digest, and byte size.
    """

    path = Path(file_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ArtifactFingerprint(algorithm="sha256", digest=digest, byte_size=path.stat().st_size)
