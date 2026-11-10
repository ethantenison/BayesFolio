"""Isolated v2 research evidence I/O and protected-source discovery.

This experiment composition layer writes only its versioned folder. Financial
returns/weights are decimals, covariance is squared decimals, and NAV is dollars.
The module does not fit, allocate, trade, or change any production default.
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
V1 = REPO / "experiments/2026-11-portfolio-decision-v1"
CHRONO = REPO / "experiments/2026-10-gp-chronological-selection-v1/outputs"
TUNING = REPO / "experiments/2026-10-gp-tuning-v1"
FRAMEWORK = REPO / "experiments/2026-10-portfolio-evaluation-v1"
ASSETS = ["SPY", "MGK", "VTV", "IWM", "VEA", "VWO", "VSS", "EWX", "BND", "IEF", "LQD", "HYG",
          "BNDX", "VWOB", "HYEM"]
ELIGIBLE = [name for name in ASSETS if name != "MGK"]
ELIGIBLE_INDEX = np.array([ASSETS.index(name) for name in ELIGIBLE])
FAMILIES = {"us_equity": ASSETS[:4], "international_equity": ASSETS[4:8],
            "us_fixed_income": ASSETS[8:12], "international_fixed_income": ASSETS[12:]}
SEEDS = [27, 127, 227]
COMPONENTS = ["broad_r2_i15_exp", "compact_r2_i15_exp", "broad_roll48_r2_i15"]


def digest(path: Path) -> str:
    """Fingerprint an exact existing file.

    Args:
        path: Read-only source or experiment artifact.

    Returns:
        SHA256 hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    """Read JSON evidence without mutations.

    Args:
        path: Existing JSON file.

    Returns:
        Parsed JSON value.
    """
    return json.loads(path.read_text())


def arrays(path: Path) -> dict[str, np.ndarray]:
    """Read explicit numerical axes without executable pickle.

    Args:
        path: Existing NPZ file.

    Returns:
        Independent named numerical arrays.
    """
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


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
        raise ValueError("Nonfinite JSON evidence must be explicitly represented")
    return value


def save(path: Path, value, *, mutable: bool = False) -> None:
    """Atomically write isolated evidence, refusing silent immutable overwrites.

    Args:
        path: Destination inside this experiment only.
        value: Exact bytes or JSON-compatible data.
        mutable: Replacement allowed for explicit status/ledger snapshots only.
    """
    if not path.resolve().is_relative_to(HERE):
        raise ValueError(f"Out-of-scope write: {path}")
    content = value if isinstance(value, bytes) else (
        json.dumps(_jsonable(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and not mutable:
        if path.read_bytes() != content:
            raise ValueError(f"Conflicting immutable evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def snapshot(path: Path, value: dict[str, np.ndarray]) -> None:
    """Write compressed explicit array evidence.

    Args:
        path: New isolated NPZ path.
        value: Named arrays, including explicit date/asset/seed axes.
    """
    stream = io.BytesIO()
    np.savez_compressed(stream, **value)
    save(path, stream.getvalue())


def verify_hashes(mapping: dict[str, str]) -> None:
    """Reject altered protected files.

    Args:
        mapping: Exact absolute path to SHA256 mapping.
    """
    for name, expected in mapping.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Protected source changed: {name}")


def status(phase: str, next_action: str, **fields) -> None:
    """Persist current progress and exact next action.

    Args:
        phase: Current workflow phase.
        next_action: Concrete next operation.
        **fields: Completed work, budgets, failure and artifact evidence.
    """
    save(HERE / "STATUS.json", {"phase": phase, "next_action": next_action,
        "updated_at_utc": datetime.now(UTC).isoformat(), **fields}, mutable=True)
