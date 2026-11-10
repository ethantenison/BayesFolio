"""Freeze the last scored target's daily price/RF endpoints for independent audit.

These April–May 2026 validation sources are not feature inputs and never alter
the frozen v1/v2 datasets or preserved target labels. They close the final-month
source-reconstruction limitation. Existing snapshots are reused, never refreshed.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/freeze_target_tail.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import repair_dataset as data

OUTPUT = data.HERE / "artifacts/target_audit_v1"


def snapshot_path(symbol: str) -> Path:
    """Map a target-validation symbol to an isolated source artifact."""
    return OUTPUT / (symbol.replace("^", "INDEX_") + ".parquet")


def main() -> None:
    """Download only absent endpoint snapshots and persist their provenance hashes."""
    import yfinance as yf

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "sources.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    for symbol in [*data.TICKERS, "^IRX"]:
        target = snapshot_path(symbol)
        if target.exists():
            if symbol not in metadata or metadata[symbol]["sha256"] != data.digest(target):
                raise ValueError("Target-validation source is changed or unregistered")
            continue
        frame = yf.download(symbol, start="2026-04-01", end="2026-06-01", auto_adjust=False,
                            progress=False, threads=False, timeout=30)
        if isinstance(frame.columns, pd.MultiIndex):
            frame = frame.xs(symbol, axis=1, level=1)
        if frame.empty or "Adj Close" not in frame:
            raise ValueError(f"Missing endpoint validation source: {symbol}")
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        if pd.Timestamp("2026-05-29") not in frame.index:
            raise ValueError(f"Final forward endpoint absent: {symbol}")
        frame.to_parquet(target)
        metadata[symbol] = {"symbol": symbol, "provider": "Yahoo Finance via yfinance",
                            "start": "2026-04-01", "end_exclusive": "2026-06-01",
                            "retrieved_at": datetime.now(UTC).isoformat(), "sha256": data.digest(target),
                            "purpose": "Forward-label source validation only; not a model input"}
        data.write_json(path, metadata)
        print(f"Frozen target endpoint: {symbol}", flush=True)
    print("All target-endpoint validation sources frozen; feature datasets unchanged.", flush=True)


if __name__ == "__main__":
    main()
