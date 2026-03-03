from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_fingerprint(file_path: str | Path) -> tuple[str, int]:
    """Compute SHA-256 digest and byte size for a persisted artifact.

    Args:
        file_path: Absolute or relative path to an existing file.

    Returns:
        Tuple of (hex digest, byte size).
    """

    path = Path(file_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest, path.stat().st_size
