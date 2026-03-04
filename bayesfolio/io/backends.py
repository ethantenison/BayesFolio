from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse


class ArtifactBackend(Protocol):
    """Storage backend protocol for artifact payloads.

    Implementations store bytes under relative keys and return a stable URI.
    """

    def put_bytes(self, key: str, payload: bytes) -> str:
        """Persist bytes and return a location URI.

        Args:
            key: Relative key under backend root.
            payload: Raw payload bytes.

        Returns:
            Fully qualified storage URI.
        """

        ...

    def exists(self, key: str) -> bool:
        """Return whether a key exists in the backend."""

        ...


class LocalArtifactBackend:
    """Local filesystem artifact backend."""

    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir).expanduser().resolve()

    def put_bytes(self, key: str, payload: bytes) -> str:
        normalized_key = key.lstrip("/")
        path = self._root_dir / normalized_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path.as_uri()

    def exists(self, key: str) -> bool:
        normalized_key = key.lstrip("/")
        path = self._root_dir / normalized_key
        return path.exists()


class FsspecArtifactBackend:
    """Remote/object-store backend via ``fsspec``."""

    def __init__(self, root_uri: str) -> None:
        parsed = urlparse(root_uri)
        if not parsed.scheme or parsed.scheme == "file":
            msg = "FsspecArtifactBackend requires a non-file URI scheme."
            raise ValueError(msg)

        try:
            import fsspec
        except ImportError as exc:
            msg = "fsspec is required for remote artifact storage backends. Install it with `poetry add fsspec`."
            raise ImportError(msg) from exc

        self._root_uri = root_uri.rstrip("/")
        self._fs = fsspec.filesystem(parsed.scheme)

        root_path = f"{parsed.netloc}{parsed.path}".strip("/")
        self._root_path = root_path

    def _full_path(self, key: str) -> str:
        normalized_key = key.lstrip("/")
        if not self._root_path:
            return normalized_key
        return f"{self._root_path}/{normalized_key}"

    def put_bytes(self, key: str, payload: bytes) -> str:
        full_path = self._full_path(key)
        parent = str(Path(full_path).parent)
        self._fs.makedirs(parent, exist_ok=True)
        with self._fs.open(full_path, "wb") as stream:
            stream.write(payload)
        return f"{self._root_uri}/{key.lstrip('/')}"

    def exists(self, key: str) -> bool:
        full_path = self._full_path(key)
        return bool(self._fs.exists(full_path))


def default_artifact_root_uri() -> str:
    """Return default artifact root URI.

    Priority:
    1) ``BAYESFOLIO_ARTIFACT_ROOT_URI`` environment variable.
    2) Local user cache under ``~/.bayesfolio``.

    Returns:
        Root URI used for artifact persistence.
    """

    configured = os.getenv("BAYESFOLIO_ARTIFACT_ROOT_URI")
    if configured:
        return configured

    return (Path.home() / ".bayesfolio").as_uri()


def make_artifact_backend(root_uri: str | Path | None = None) -> ArtifactBackend:
    """Create an artifact backend for local or remote storage.

    Args:
        root_uri: Root URI/path for artifacts. If ``None``, use
            :func:`default_artifact_root_uri`.

    Returns:
        Configured artifact backend.
    """

    resolved_root: str | Path = root_uri if root_uri is not None else default_artifact_root_uri()

    if isinstance(resolved_root, Path):
        return LocalArtifactBackend(resolved_root)

    if "://" not in resolved_root:
        return LocalArtifactBackend(Path(resolved_root))

    parsed = urlparse(resolved_root)
    if parsed.scheme == "file":
        return LocalArtifactBackend(Path(parsed.path))

    return FsspecArtifactBackend(resolved_root)


def resolve_backend_and_key(
    output_target: str | Path,
    *,
    root_uri: str | Path | None = None,
    backend: ArtifactBackend | None = None,
) -> tuple[ArtifactBackend, str]:
    """Resolve backend and relative key for an artifact output target.

    Args:
        output_target: Output file target. Supports local paths and URIs.
        root_uri: Optional root URI/path used when ``output_target`` is
            relative.
        backend: Optional explicit backend.

    Returns:
        Tuple of ``(backend, key)`` where key is relative within backend root.
    """

    if backend is not None and root_uri is not None:
        msg = "Pass either backend or root_uri, not both."
        raise ValueError(msg)

    target_str = str(output_target)
    key = target_str.lstrip("/")

    if backend is not None:
        return backend, key

    if isinstance(output_target, Path):
        if output_target.is_absolute():
            return make_artifact_backend(output_target.parent), output_target.name
        resolved_backend = make_artifact_backend(root_uri)
        return resolved_backend, output_target.as_posix().lstrip("/")

    if "://" in target_str:
        parsed = urlparse(target_str)
        if parsed.scheme == "file":
            file_path = Path(parsed.path)
            return make_artifact_backend(file_path.parent), file_path.name

        full_path = f"{parsed.netloc}{parsed.path}".strip("/")
        if "/" in full_path:
            parent, name = full_path.rsplit("/", 1)
            resolved_backend = make_artifact_backend(f"{parsed.scheme}://{parent}")
            return resolved_backend, name

        resolved_backend = make_artifact_backend(f"{parsed.scheme}://{parsed.netloc}")
        return resolved_backend, full_path

    path_target = Path(target_str)
    if path_target.is_absolute():
        return make_artifact_backend(path_target.parent), path_target.name

    resolved_backend = make_artifact_backend(root_uri)
    return resolved_backend, path_target.as_posix().lstrip("/")
