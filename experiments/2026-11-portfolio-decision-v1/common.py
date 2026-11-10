"""Isolated November-decision persistence and reproducibility helpers.

Research composition/I/O only. Writes are restricted to this experiment;
protected evidence is read-only. Financial arrays use decimal returns,
squared-decimal covariance and fractional portfolio weights.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHRONO = REPO / "experiments/2026-10-gp-chronological-selection-v1/outputs"
TUNING = REPO / "experiments/2026-10-gp-tuning-v1"
FRAMEWORK = REPO / "experiments/2026-10-portfolio-evaluation-v1"
SEEDS = [27, 127, 227]
COMPONENTS = ["broad_r2_i15_exp", "compact_r2_i15_exp", "broad_roll48_r2_i15"]
FAMILIES = ["us_equity", "international_equity", "us_fixed_income", "international_fixed_income"]


def digest(path: Path) -> str:
    """Return SHA256 of an existing exact file path.

    Args:
        path: Local artifact or source file.

    Returns:
        Lowercase hexadecimal fingerprint.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    """Read a JSON evidence object without changing it.

    Args:
        path: Existing JSON artifact.

    Returns:
        Parsed object.
    """
    return json.loads(path.read_text())


def arrays(path: Path) -> dict:
    """Read NPZ arrays without executable pickle.

    Args:
        path: Existing compressed array artifact.

    Returns:
        Independent named arrays.
    """
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Nonfinite JSON evidence must be explicitly handled")
    return value


def save(path: Path, value, *, mutable: bool = False) -> None:
    """Atomically save isolated evidence, refusing silent overwrites.

    Args:
        path: Destination within this versioned experiment.
        value: JSON-compatible object or exact bytes.
        mutable: Allow replacement only of explicitly mutable progress ledgers.
    """
    if not path.resolve().is_relative_to(HERE):
        raise ValueError(f"Out-of-scope write: {path}")
    content = value if isinstance(value, bytes) else (
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists() and not mutable:
        if path.read_bytes() != content:
            raise ValueError(f"Conflicting immutable evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def snapshot(path: Path, values: dict) -> None:
    """Save numerical arrays with explicit axes and no pickle.

    Args:
        path: New NPZ artifact path.
        values: Named arrays.
    """
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    save(path, stream.getvalue())


def verify_hashes(values: dict) -> None:
    """Reject any changed protected evidence.

    Args:
        values: Exact absolute file paths mapped to SHA256.
    """
    for name, expected in values.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Protected evidence changed: {name}")


def status(phase: str, action: str, **fields) -> None:
    """Persist current work and the exact next action.

    Args:
        phase: Current finite-workflow stage.
        action: Exact next action, not a vague progress claim.
        **fields: Counters, remaining budgets and resolved/unresolved failures.
    """
    value = {"updated_at": datetime.now(UTC).isoformat(), "phase": phase,
             "next_action": action, **fields}
    save(HERE / "STATUS.json", value, mutable=True)
    print(json.dumps(_jsonable(value)), flush=True)
