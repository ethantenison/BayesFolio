from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_digest(content: bytes) -> str:
    """Compute SHA-256 digest for in-memory bytes.

    Args:
        content: Byte payload to hash.

    Returns:
        Lowercase SHA-256 hexadecimal digest.
    """

    return hashlib.sha256(content).hexdigest()


def sha256_fingerprint(file_path: str | Path) -> tuple[str, int]:
    """Compute SHA-256 digest and byte size for a persisted artifact.

    Args:
        file_path: Absolute or relative path to an existing file.

    Returns:
        Tuple of (hex digest, byte size).
    """

    path = Path(file_path)
    digest = sha256_digest(path.read_bytes())
    return digest, path.stat().st_size
