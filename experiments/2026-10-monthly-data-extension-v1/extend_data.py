"""Extend repaired monthly experiment data without changing frozen prior evidence.

Research I/O orchestration: freeze new vendor snapshots, preserve adjusted-price
vintage continuity, append completed ETF targets/features and retain the prior
108-row-per-ETF panel exactly. No GP fits, allocator changes, production cache
writes or October partial-month targets. Returns are decimal discounted excess.

Usage from repository root:
    uv run --locked --no-sync python <this_file> --fetch
    uv run --locked --no-sync python <this_file> --build
    uv run --locked --no-sync python <this_file> --verify
The first command requires network; subsequent reconstruction is offline.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
import warnings
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-monthly-extension-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-monthly-extension-cache")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / "experiments/2026-10-repaired-global-feature-selection"
OUTPUT = HERE / "outputs"
FRESH = OUTPUT / "raw/fresh"
COMBINED = OUTPUT / "raw/combined"
AS_OF = "2026-10-24"
sys.path.extend([str(ROOT), str(PREVIOUS)])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import repair_dataset as prior  # noqa: E402
from extend_feature_pool import EXTRA_ETF, extra_macro  # noqa: E402
from extension_math import append_macro, monthly_cutoffs, splice_market  # noqa: E402

TICKERS = prior.TICKERS
SYMBOLS = list(dict.fromkeys([*TICKERS, *prior.MACRO_SYMBOLS]))
CUTOFFS = monthly_cutoffs(AS_OF)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(root: Path, symbol: str) -> Path:
    return root / (symbol.replace("^", "INDEX_").replace("=", "_") + ".parquet")


def _write_json(path: Path, value: dict) -> None:
    if not path.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError("Only new extension outputs are writable")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_fred(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    return (
        pd.Series(pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(frame.iloc[:, 0]))
        .dropna()
        .sort_index()
    )


def _fetch() -> None:
    import yfinance as yf

    if (OUTPUT / "COMPLETED.json").exists():
        _verify()
        return
    yf.set_tz_cache_location("/private/tmp/bayesfolio-monthly-extension-yf")
    FRESH.mkdir(parents=True, exist_ok=True)
    metadata_path = FRESH / "sources.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    pending = []
    for symbol in SYMBOLS:
        path = _source_path(FRESH, symbol)
        if path.exists():
            if symbol not in metadata or _hash(path) != metadata[symbol]["sha256"]:
                raise ValueError(f"Changed or unregistered fresh source: {symbol}")
        else:
            pending.append(symbol)
    if pending:
        print(f"Fetching {len(pending)} ETF/macro sources through {CUTOFFS['completed_month']}", flush=True)
        batch = yf.download(
            pending,
            start="2026-04-01",
            end=CUTOFFS["source_end_exclusive"],
            auto_adjust=False,
            threads=4,
            progress=False,
            timeout=30,
        )
        for symbol in pending:
            if isinstance(batch.columns, pd.MultiIndex):
                level = 1 if symbol in batch.columns.get_level_values(1) else 0
                frame = batch.xs(symbol, axis=1, level=level).dropna(how="all")
            else:
                frame = batch.dropna(how="all")
            if frame.empty or "Adj Close" not in frame:
                raise ValueError(f"Missing adjusted source: {symbol}")
            frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
            frame = frame.sort_index()
            if frame.index[-1] != pd.Timestamp(CUTOFFS["completed_month"]):
                raise ValueError(f"Incomplete source endpoint: {symbol}, {frame.index[-1]}")
            path = _source_path(FRESH, symbol)
            frame.to_parquet(path)
            metadata[symbol] = {
                "provider": "Yahoo Finance via yfinance",
                "requested_start": "2026-04-01",
                "end_exclusive": CUTOFFS["source_end_exclusive"],
                "rows": len(frame),
                "first_observation": str(frame.index[0].date()),
                "last_observation": str(frame.index[-1].date()),
                "sha256": _hash(path),
                "retrieved_at": datetime.now(UTC).isoformat(),
            }
            _write_json(metadata_path, metadata)
    for name in ["cpi_fred", "hy_fred_live"]:
        path = FRESH / f"{name}.csv"
        if path.exists():
            if name not in metadata or _hash(path) != metadata[name]["sha256"]:
                raise ValueError("Changed fresh macro snapshot")
            continue
        payload = urllib.request.urlopen(prior.URLS[name], timeout=45).read()
        pd.read_csv(io.BytesIO(payload))
        path.write_bytes(payload)
        metadata[name] = {
            "url": prior.URLS[name],
            "sha256": _hash(path),
            "retrieved_at": datetime.now(UTC).isoformat(),
            "internal_research_only": name.startswith("hy"),
        }
        _write_json(metadata_path, metadata)
    print("Fresh source snapshots frozen; existing datasets and caches unchanged", flush=True)


def _protected() -> dict:
    path = ROOT / "experiments/2026-10-portfolio-evaluation-v1/scenario_sharpe_v3/outputs/COMPLETED.json"
    seal = json.loads(path.read_text())
    if not seal["passed"]:
        raise ValueError("Prior experiment is not complete")
    protected = {**seal["input_sha256"], **seal["artifact_sha256"], str(path.resolve()): _hash(path)}
    _verify_hashes(protected)
    return protected


def _verify_hashes(values: dict) -> None:
    for name, digest in values.items():
        if _hash(Path(name)) != digest:
            raise ValueError(f"Protected evidence changed: {name}")


def _combine() -> tuple[dict, dict]:
    metadata = json.loads((FRESH / "sources.json").read_text())
    old_metadata = json.loads((prior.RAW / "sources.json").read_text())
    COMBINED.mkdir(parents=True, exist_ok=True)
    audit, hashes = {}, {}
    for symbol in SYMBOLS:
        old_path, fresh_path = prior.source_path(symbol), _source_path(FRESH, symbol)
        if _hash(old_path) != old_metadata[symbol]["sha256"] or _hash(fresh_path) != metadata[symbol]["sha256"]:
            raise ValueError(f"Source fingerprint mismatch: {symbol}")
        old, fresh = pd.read_parquet(old_path), pd.read_parquet(fresh_path)
        frame, evidence = splice_market(old, fresh, symbol=symbol, completed=CUTOFFS["completed_month"])
        destination = _source_path(COMBINED, symbol)
        frame.to_parquet(destination)
        audit[symbol] = {
            **evidence,
            "first_date": str(frame.index[0].date()),
            "last_date": str(frame.index[-1].date()),
            "fresh_sha256": _hash(fresh_path),
            "original_sha256": _hash(old_path),
            "combined_sha256": _hash(destination),
        }
        hashes.update({str(old_path.resolve()): _hash(old_path), str(fresh_path.resolve()): _hash(fresh_path)})
    old_hy, _ = prior.verified_hy()
    for name, old in [("cpi_fred", prior.read_series("cpi_fred")), ("hy_fred_live", old_hy)]:
        fresh_path = FRESH / f"{name}.csv"
        if _hash(fresh_path) != metadata[name]["sha256"]:
            raise ValueError("Fresh macro fingerprint changed")
        series, evidence = append_macro(old, _read_fred(fresh_path), completed=CUTOFFS["completed_month"])
        destination = COMBINED / f"{name}.csv"
        series.rename(name).rename_axis("date").to_csv(destination)
        audit[name] = {
            **evidence,
            "last_observation": str(series.index.max().date()),
            "combined_sha256": _hash(destination),
            "fresh_sha256": _hash(fresh_path),
        }
        hashes[str(fresh_path.resolve())] = _hash(fresh_path)
    return audit, hashes


def _build_panel() -> tuple[pd.DataFrame, dict]:
    from bayesfolio.core.settings import Horizon
    from bayesfolio.engine.features.asset_prices import (
        add_cross_sectional_momentum_rank,
        compute_excess_future_return_calendar,
        fetch_etf_features,
    )

    dates = pd.date_range(prior.STUDY["training_start"], CUTOFFS["completed_month"], freq="BME")
    settings = {**prior.STUDY, "last_score": CUTOFFS["completed_month"]}
    with patch.multiple(prior, RAW=COMBINED, STUDY=settings, DATES=dates, SOURCE_END=CUTOFFS["source_end_exclusive"]):
        sources = prior.market_sources()
        daily = pd.concat({ticker: sources[ticker] for ticker in TICKERS}, axis=1)
        print("Rebuilding unchanged ETF/macro formulas with extended trailing history", flush=True)
        with (
            patch("bayesfolio.engine.features.asset_prices.yf.download", return_value=daily),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            etf = fetch_etf_features(TICKERS, prior.SOURCE_START, CUTOFFS["source_end_exclusive"], Horizon.MONTHLY)
        etf = add_cross_sectional_momentum_rank(etf)[["date", "asset_id", *prior.ETF_COLUMNS, *EXTRA_ETF]]
        hy = prior.read_series("hy_fred_live")
        macro = prior.rebuild_macro(sources, hy).merge(extra_macro(sources), on="date", validate="one_to_one")
        rf = sources["^IRX"]["Adj Close"] / 100 / 252
        rf = rf.reindex(pd.bdate_range(rf.index.min(), rf.index.max())).ffill()
        original = pd.read_parquet(PREVIOUS / "artifacts/v2/global.parquet")
        original_last = pd.to_datetime(original.date).max()
        # Preserve every existing target; incomplete future buckets in the cache are not used.
        frozen_labels = pd.read_parquet(PREVIOUS / "artifacts/v1/raw/labels.parquet")
        frozen_labels = frozen_labels[
            frozen_labels.asset_id.isin(TICKERS) & pd.to_datetime(frozen_labels.date).le(original_last)
        ].copy()
        rows, target_audit = [], []
        for ticker in TICKERS:
            rebuilt = compute_excess_future_return_calendar(
                sources[ticker]["Adj Close"], rf, Horizon.MONTHLY, include_unlabeled_tail=True
            )
            expected = original[original.asset_id.eq(ticker)].set_index("date").y_excess_lead
            delta = (rebuilt.loc[expected.index] - expected).abs()
            if len(delta) != 108 or delta.max() >= 2e-6:
                raise ValueError(
                    f"Historical target reconstruction changed beyond prior tolerance: {ticker}, {delta.max()}"
                )
            added = rebuilt.loc[rebuilt.index > original_last].rename_axis("date").reset_index()
            added["asset_id"] = ticker
            rows.append(added[["date", "asset_id", "y_excess_lead"]])
            target_audit.append(
                {
                    "asset": ticker,
                    "old_labels_checked": len(delta),
                    "maximum_old_label_reconstruction_difference": float(delta.max()),
                }
            )
        labels = pd.concat([frozen_labels, *rows], ignore_index=True)
        panel = prior.align_panel(etf, macro, labels)
        panel = panel[list(original.columns)]
        replay = panel[panel.date.le(original_last)].reset_index(drop=True)
        pd.testing.assert_frame_equal(replay, original, check_exact=False, rtol=1e-11, atol=1e-12)
        max_difference = float(
            np.max(np.abs(replay.iloc[:, 2:].to_numpy(float) - original.iloc[:, 2:].to_numpy(float)))
        )
        # Freeze the exact historical cells, not their numerically reproduced copies.
        panel = pd.concat([original, panel[panel.date.gt(original_last)]], ignore_index=True)
        pd.testing.assert_frame_equal(panel.iloc[: len(original)], original, check_exact=True)
        inflation_audit = prior.inflation_signals(prior.read_series("cpi_fred"))[1]
    return panel, {
        "historical_replay_maximum_absolute_difference": max_difference,
        "historical_prefix_bitwise_unchanged": True,
        "target_reconstruction": target_audit,
        "inflation_missingness_policy": inflation_audit,
    }


def _build() -> None:
    if (OUTPUT / "COMPLETED.json").exists():
        _verify()
        return
    protected = _protected()
    source_audit, sources = _combine()
    panel, build_audit = _build_panel()
    dates = pd.date_range(prior.STUDY["training_start"], CUTOFFS["completed_month"], freq="BME")
    expected = pd.MultiIndex.from_product([dates, sorted(TICKERS)], names=["date", "asset_id"])
    if not pd.MultiIndex.from_frame(panel[["date", "asset_id"]]).equals(expected):
        raise ValueError("Complete common ETF-month grid required")
    live_mask = panel.date.eq(pd.Timestamp(CUTOFFS["completed_month"]))
    predictors = panel.drop(columns=["date", "asset_id", "y_excess_lead"])
    if not np.isfinite(predictors.to_numpy(float)).all():
        raise ValueError("Missing/nonfinite predictors; no zero filling is allowed")
    if not panel.loc[live_mask, "y_excess_lead"].isna().all():
        raise ValueError("The incomplete October forward target must remain unresolved")
    complete = panel.loc[~live_mask].reset_index(drop=True)
    if not np.isfinite(complete.drop(columns=["date", "asset_id"]).to_numpy(float)).all():
        raise ValueError("Incomplete scored return targets")
    artifacts = OUTPUT / "dataset"
    artifacts.mkdir(parents=True, exist_ok=True)
    for family, tickers in {**prior.FAMILIES, "global": TICKERS}.items():
        complete[complete.asset_id.isin(tickers)].reset_index(drop=True).to_parquet(
            artifacts / f"{family}.parquet", index=False
        )
    panel.loc[live_mask].to_parquet(artifacts / "unscored_september_origin.parquet", index=False)
    scored = pd.date_range(prior.STUDY["first_score"], CUTOFFS["last_scored_origin"], freq="BME")
    added = scored[scored > pd.Timestamp(prior.STUDY["last_score"])]
    for name, values in [("scored_dates_28.txt", scored), ("new_scored_dates_4.txt", added)]:
        (artifacts / name).write_text("\n".join(date.date().isoformat() for date in values) + "\n")
    coverage = {
        "passed": True,
        "as_of": AS_OF,
        "cutoffs": CUTOFFS,
        "GP_refits": 0,
        "ETFs": TICKERS,
        "columns": panel.columns.tolist(),
        "initial_training_months": 84,
        "complete_months_per_ETF": len(complete) // len(TICKERS),
        "complete_rows": len(complete),
        "scored_origins": len(scored),
        "new_scored_origins": [str(date.date()) for date in added],
        "new_return_months": [str((date + pd.offsets.BMonthEnd(1)).date()) for date in added],
        "old_outcomes_through": "2026-05-29",
        "new_outcomes_through": CUTOFFS["completed_month"],
        "unresolved_next_month_rows": int(live_mask.sum()),
        "missing_scored_cells": 0,
        "source_audit": source_audit,
        "build_audit": build_audit,
        "limitations": [
            "Macro series retain the frozen vintage where available; overlap revisions reported, not applied.",
            "CPI is not a point-in-time vintage backtest; the original one-month feature lag is unchanged.",
            "New adjusted price paths are anchored to the old overlap scale; raw snapshots are retained.",
            "No GP forecasts or portfolio results for the new months have been computed yet.",
        ],
    }
    _write_json(OUTPUT / "coverage_audit.json", coverage)
    for path in [
        *HERE.glob("*.py"),
        PREVIOUS / "repair_dataset.py",
        PREVIOUS / "extend_feature_pool.py",
        ROOT / "bayesfolio/engine/features/asset_prices.py",
    ]:
        sources[str(path.resolve())] = _hash(path)
    protected.update(sources)
    _verify_hashes(protected)
    outputs = {str(path.resolve()): _hash(path) for path in OUTPUT.rglob("*") if path.is_file()}
    _write_json(
        OUTPUT / "COMPLETED.json",
        {
            "passed": True,
            "schema": "bayesfolio.monthly_data_extension.v1",
            "as_of": AS_OF,
            "input_sha256": protected,
            "artifact_sha256": outputs,
        },
    )
    _verify()
    print(
        json.dumps(
            {
                key: value
                for key, value in coverage.items()
                if key not in {"source_audit", "build_audit", "columns", "limitations"}
            },
            indent=2,
        ),
        flush=True,
    )


def _verify() -> None:
    seal = json.loads((OUTPUT / "COMPLETED.json").read_text())
    if not seal["passed"]:
        raise ValueError("Dataset extension is not complete")
    _verify_hashes(seal["input_sha256"])
    _verify_hashes(seal["artifact_sha256"])
    old = pd.read_parquet(PREVIOUS / "artifacts/v2/global.parquet")
    new = pd.read_parquet(OUTPUT / "dataset/global.parquet")
    pd.testing.assert_frame_equal(new.iloc[: len(old)], old, check_exact=True)
    print("Verified extended data, exact historical prefix, all snapshots and protected prior evidence", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch", action="store_true")
    group.add_argument("--build", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        _fetch()
    elif args.build:
        _build()
    else:
        _verify()
