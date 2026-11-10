"""Freeze sources and rebuild experiment-only monthly predictors (decimal returns).

This research I/O entrypoint reuses the production ETF engineering function on
frozen daily inputs; it never changes a provider, production cache, or original
experiment. All predictors are lagged once; cached excess-return labels remain
unchanged. Downloads require --fetch and are thereafter reused offline.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/repair_dataset.py --fetch
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import warnings
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STUDY = json.loads((HERE / "study.json").read_text())
FAMILIES = STUDY["families"]
TICKERS = [ticker for family in FAMILIES.values() for ticker in family]
ARTIFACTS = HERE / "artifacts" / STUDY["dataset_version"]
RAW = ARTIFACTS / "raw"
SOURCE_START = "2013-01-01"
SOURCE_END = "2026-05-01"
DATES = pd.date_range(STUDY["training_start"], STUDY["last_score"], freq="BME")
SCORES = pd.date_range(STUDY["first_score"], STUDY["last_score"], freq="BME")
MACRO_SYMBOLS = ["RSP", "TLT", "SCHP", "CEW", "^VIX", "^VIX3M", "^TNX", "^IRX", "DX-Y.NYB",
                 "CL=F", "HG=F", "GC=F"]
ALIASES = {"hy_spread_chg_1m": "hy_spread_chg_1p", "hy_spread_z_12m": "hy_spread_z_12p",
           "vix_ts_z_12m": "vix_ts_z_12p", "spy_flow_z_12m": "spy_flow_z_12p",
           "cpi_yoy": "cpi_chg_12p", "cpi_mom": "cpi_chg_1p", "rsp_spy_roc_1m": "rsp_spy_roc_1p"}
URLS = {
    "hy_archive_long": "https://raw.githubusercontent.com/maaurocp/Trading_Protocol/"
    "bf64e83fa4c2a6e72c37d3883476dc81bd9d2e31/data/raw/fred_BAMLH0A0HYM2.csv",
    "hy_archive_independent": "https://raw.githubusercontent.com/csaladenes/eco-archive/main/BAMLH0A0HYM2.csv",
    "hy_fred_live": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2",
    "cpi_fred": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
}
ETF_COLUMNS = ["baspread", "ret_kurt", "chmom", "mom6m", "mom12m", "mom12m_skip1m", "mom36m",
               "cs_mom_rank", "max_dd_3m", "max_dd_6m", "ma_signal", "ret_autocorr", "vol_z", "trend_slope"]
LOCAL_LABELS = ROOT / "artifacts/cache/returns/returns_bme.parquet"
LOCAL_FEATURES = ROOT / "artifacts/cache/etf_features/etf_features_bme.parquet"
LOCAL_MACRO = ROOT / "artifacts/cache/macro_enhanced/macro_bme.parquet"


def digest(path: Path) -> str:
    """Return a file's SHA-256 fingerprint."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically persist a JSON report without nonfinite numeric values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_path(symbol: str) -> Path:
    """Map a market symbol to its frozen daily parquet path."""
    return RAW / (symbol.replace("^", "INDEX_").replace("=", "_") + ".parquet")


def acquire_sources() -> None:
    """Download each absent raw source once, preserving metadata and bytes."""
    import yfinance as yf

    RAW.mkdir(parents=True, exist_ok=True)
    meta_path = RAW / "sources.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for symbol in dict.fromkeys([*TICKERS, *MACRO_SYMBOLS]):
        target = source_path(symbol)
        if target.exists():
            if symbol not in metadata or metadata[symbol]["sha256"] != digest(target):
                raise ValueError(f"Unregistered or changed raw source: {symbol}")
            continue
        frame = yf.download(symbol, start=SOURCE_START, end=SOURCE_END, auto_adjust=False,
                            progress=False, threads=False, timeout=30)
        if isinstance(frame.columns, pd.MultiIndex):
            frame = frame.xs(symbol, axis=1, level=1)
        if frame.empty or "Adj Close" not in frame:
            raise ValueError(f"Missing adjusted daily source for {symbol}")
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame.to_parquet(target)
        metadata[symbol] = {"source": "Yahoo Finance via yfinance", "symbol": symbol,
                            "requested_start": SOURCE_START, "requested_end_exclusive": SOURCE_END,
                            "first_observation": str(frame.index.min().date()),
                            "last_observation": str(frame.index.max().date()), "rows": len(frame),
                            "adjustment": "Adj Close; adjusted High/Low derived with Adj Close / Close",
                            "retrieved_at": datetime.now(UTC).isoformat(), "sha256": digest(target)}
        write_json(meta_path, metadata)
        print(f"Frozen {symbol}: {len(frame)} daily rows", flush=True)
    for name, url in URLS.items():
        target = RAW / (name + ".csv")
        if target.exists():
            if name not in metadata or metadata[name]["sha256"] != digest(target):
                raise ValueError(f"Unregistered or changed raw source: {name}")
            continue
        content = urllib.request.urlopen(url, timeout=45).read()
        pd.read_csv(io.BytesIO(content))  # Fail before freezing a non-CSV response.
        target.write_bytes(content)
        metadata[name] = {"url": url, "retrieved_at": datetime.now(UTC).isoformat(),
                          "sha256": digest(target), "internal_research_only": name.startswith("hy_")}
        write_json(meta_path, metadata)
        print(f"Frozen {name}", flush=True)
    for name, original in {"labels": LOCAL_LABELS, "old_etf_features": LOCAL_FEATURES,
                           "old_macro": LOCAL_MACRO}.items():
        target = RAW / (name + ".parquet")
        if not target.exists():
            target.write_bytes(original.read_bytes())
            metadata[name] = {"original_path": str(original), "sha256": digest(target)}
            write_json(meta_path, metadata)


def read_series(name: str) -> pd.Series:
    """Read a FRED-shaped frozen CSV as a numeric daily or monthly series."""
    frame = pd.read_csv(RAW / (name + ".csv"))
    return pd.Series(pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(),
                     index=pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0]))).dropna().sort_index()


def verified_hy() -> tuple[pd.Series, dict[str, object]]:
    """Join exact-index snapshots after checking independent and official overlap."""
    archive, independent, live = (read_series(name) for name in
                                  ["hy_archive_long", "hy_archive_independent", "hy_fred_live"])
    checks = {}
    for name, other in [("independent", independent), ("live_fred", live)]:
        overlap = archive.index.intersection(other.index)
        delta = float((archive.loc[overlap] - other.loc[overlap]).abs().max())
        if len(overlap) < 100 or delta > 1e-10:
            raise ValueError(f"HY snapshot mismatch: {name}, {len(overlap)} observations, difference {delta}")
        checks[name] = {"overlap_observations": len(overlap), "max_absolute_difference": delta}
    combined = pd.concat([archive, live])
    return combined[~combined.index.duplicated(keep="last")].sort_index(), checks


def monthly_hy(series: pd.Series, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute decimal OAS, one-month difference, and trailing sample-SD z-score."""
    level = series.reindex(grid, method="ffill") / 100.0
    return pd.DataFrame({"hy_spread": level, "hy_spread_chg_1p": level.diff(),
                         "hy_spread_z_12p": (level - level.rolling(12).mean()) / level.rolling(12).std()})


def inflation_signals(series: pd.Series) -> tuple[pd.DataFrame, dict[str, object]]:
    """Hold last valid signals for at most two periods, never invent CPI levels.

    October 2025 has no official CPI observation. Calculate changes with missing
    propagation first, then carry the last known *signal*, not a fabricated CPI
    level or a mislabelled two-month return. Output staleness is explicit evidence.
    """
    cpi = series.resample("BME").last()
    observed = pd.DataFrame({"cpi_chg_12p": cpi.pct_change(12, fill_method=None),
                             "cpi_chg_1p": cpi.pct_change(fill_method=None)})
    held = observed.ffill(limit=2)
    relevant = held.index.to_series().between("2017-04-01", STUDY["last_score"])
    return held, {"policy": "Last valid transformed signal, maximum two months; no level imputation",
                  "missing_source_months": [str(date.date()) for date in cpi.index[cpi.isna() & relevant]],
                  "carried_signal_months": {
                      column: [str(date.date()) for date in observed.index[
                          observed[column].isna() & held[column].notna() & relevant]]
                      for column in observed}}


def market_sources() -> dict[str, pd.DataFrame]:
    """Load immutable adjusted daily OHLCV inputs without network access."""
    result = {}
    for symbol in dict.fromkeys([*TICKERS, *MACRO_SYMBOLS]):
        frame = pd.read_parquet(source_path(symbol)).sort_index()
        factor = frame["Adj Close"] / frame["Close"]
        for column in ["Open", "High", "Low"]:
            frame[column] = frame[column] * factor
        frame["Close"] = frame["Adj Close"]
        result[symbol] = frame
    return result


def rebuild_etf(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reuse existing ETF formulas on adjusted snapshots with full trailing warm-up."""
    from bayesfolio.core.settings import Horizon
    from bayesfolio.engine.features.asset_prices import add_cross_sectional_momentum_rank, fetch_etf_features

    daily = pd.concat({symbol: sources[symbol] for symbol in TICKERS}, axis=1)
    with patch("bayesfolio.engine.features.asset_prices.yf.download", return_value=daily):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            frame = fetch_etf_features(TICKERS, SOURCE_START, SOURCE_END, Horizon.MONTHLY)
    frame = add_cross_sectional_momentum_rank(frame)
    if set(frame.asset_id.unique()) != set(TICKERS):
        raise ValueError("ETF engineering omitted a task")
    return frame[["date", "asset_id", *ETF_COLUMNS]]


def rebuild_macro(sources: dict[str, pd.DataFrame], hy: pd.Series) -> pd.DataFrame:
    """Rebuild existing macro proxies with explicit business-month-end aggregation."""
    prices = {name: frame["Adj Close"].dropna() for name, frame in sources.items()}
    monthly = {name: price.resample("BME").last() for name, price in prices.items()}
    grid = pd.date_range("2013-01-31", STUDY["last_score"], freq="BME")
    frame = monthly_hy(hy, grid)
    frame["vix"] = monthly["^VIX"]
    frame["vix_slope"] = np.log(monthly["^VIX3M"]) - np.log(monthly["^VIX"])
    slope = frame["vix_slope"]
    frame["vix_ts_z_12p"] = (slope - slope.rolling(12).mean()) / slope.rolling(12).std()
    for column, symbol in [("tnote10y", "^TNX"), ("tbill3m", "^IRX"), ("y10_nominal", "^TNX")]:
        frame[column] = monthly[symbol] / 100.0
    frame["term_spread"] = frame["tnote10y"] - frame["tbill3m"]
    frame["credit_spread"] = np.log(monthly["HYG"]) - np.log(monthly["LQD"])
    frame["credit_spread_chg_1p"] = frame["credit_spread"].diff()
    frame["dxy"] = monthly["DX-Y.NYB"]
    frame["spy_ret"] = np.log(prices["SPY"] / prices["SPY"].shift()).resample("BME").last()
    frame["erp"] = frame["spy_ret"] - frame["tbill3m"]
    frame["move_proxy"] = np.log(prices["TLT"] / prices["TLT"].shift()).rolling(21).std().resample(
        "BME").last() * np.sqrt(12)
    ratio = monthly["RSP"] / monthly["SPY"]
    frame["rsp_spy_roc_1p"] = ratio.pct_change(fill_method=None)
    frame["pct_above_50dma"] = (prices["SPY"] / prices["SPY"].rolling(50).mean() * 100).resample("BME").last()
    flow = (np.sign(prices["SPY"].diff()) * prices["SPY"] * sources["SPY"]["Volume"]).resample("BME").sum()
    frame["spy_flow_z_12p"] = (flow - flow.rolling(12).mean()) / flow.rolling(12).std()
    frame["schp_ret"] = np.log(monthly["SCHP"] / monthly["SCHP"].shift())
    frame["em_fx_ret"] = np.log(monthly["CEW"] / monthly["CEW"].shift())
    frame["oil_ret"] = monthly["CL=F"].pct_change(fill_method=None)
    frame["copper_ret"] = monthly["HG=F"].pct_change(fill_method=None)
    frame["gold_crude_ratio"] = monthly["GC=F"] / monthly["CL=F"]
    frame["y10_real_proxy"] = frame["schp_ret"].rolling(3).mean()
    frame["breakeven_proxy"] = frame["y10_nominal"].diff() - frame["y10_real_proxy"].diff()
    inflation, _ = inflation_signals(read_series("cpi_fred"))
    for column in inflation:
        frame[column] = inflation[column]
    return frame.rename_axis("date").reset_index()


def align_panel(etf: pd.DataFrame, macro: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Apply exactly one trailing-month feature lag and preserve label values."""
    labels = labels.loc[labels.asset_id.isin(TICKERS)].copy()
    labels["date"] = pd.to_datetime(labels["date"])
    frame = labels.merge(etf, on=["date", "asset_id"], how="left", validate="one_to_one").merge(
        macro, on="date", how="left", validate="many_to_one").sort_values(["asset_id", "date"])
    features = [column for column in frame if column not in {"date", "asset_id", "y_excess_lead"}]
    frame[features] = frame.groupby("asset_id")[features].shift(1)
    frame["lag_y_excess_lead"] = frame.groupby("asset_id").y_excess_lead.shift(1)
    frame["lag2_y_excess_lead"] = frame.groupby("asset_id").y_excess_lead.shift(2)
    frame = frame[frame.date.isin(DATES)].copy()
    for alias, original in ALIASES.items():
        frame[alias] = frame[original]
    frame["t_index"] = frame.date.map({date: index for index, date in enumerate(DATES)})
    return frame.sort_values(["date", "asset_id"]).reset_index(drop=True)


def validate_panel(frame: pd.DataFrame) -> dict[str, object]:
    """Fail on missing dates, nonfinite cells, duplicate tasks, or empty predictors."""
    if frame.duplicated(["date", "asset_id"]).any():
        raise ValueError("Duplicate ETF-month observations")
    expected = pd.MultiIndex.from_product([DATES, sorted(TICKERS)], names=["date", "asset_id"])
    actual = pd.MultiIndex.from_frame(frame[["date", "asset_id"]]).sort_values()
    if not actual.equals(expected):
        raise ValueError("Panel does not contain all 108 dates for all 15 tasks")
    numeric = frame.drop(columns=["date", "asset_id"])
    bad = numeric.columns[~np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)].tolist()
    if bad:
        raise ValueError(f"Nonfinite observations: {bad}")
    constants = [column for column in numeric if numeric[column].nunique() < 2]
    if constants:
        raise ValueError(f"Uninformative constant columns: {constants}")
    initial = frame[frame.date < SCORES[0]].groupby("asset_id").size()
    if not initial.eq(84).all():
        raise ValueError("Initial training panel is not exactly 84 dates per task")
    return {"passed": True, "rows": len(frame), "dates_per_etf": len(DATES),
            "initial_training_months_per_etf": initial.to_dict(), "scored_months": len(SCORES),
            "missing_cells": 0, "nonfinite_cells": 0, "constant_columns": constants,
            "feature_audit": {column: {"zero_count": int(numeric[column].eq(0).sum()),
                                        "unique_values": int(numeric[column].nunique()),
                                        "min": float(numeric[column].min()),
                                        "max": float(numeric[column].max())} for column in numeric}}


def main() -> None:
    """Acquire optional sources, freeze validated family panels, and write evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        acquire_sources()
    sources_meta = json.loads((RAW / "sources.json").read_text())
    for name, metadata in sources_meta.items():
        path = source_path(name) if name in [*TICKERS, *MACRO_SYMBOLS] else RAW / (
            name + (".csv" if name in URLS else ".parquet"))
        if digest(path) != metadata["sha256"]:
            raise ValueError(f"Frozen source changed: {name}")
    sources = market_sources()
    hy, checks = verified_hy()
    panel = align_panel(rebuild_etf(sources), rebuild_macro(sources, hy), pd.read_parquet(RAW / "labels.parquet"))
    audit = validate_panel(panel)
    manifest_path = ARTIFACTS / "dataset.manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        frozen = pd.read_parquet(ARTIFACTS / "global.parquet")
        pd.testing.assert_frame_equal(panel, frozen, check_exact=False, rtol=1e-12, atol=1e-14)
        if any(digest(ROOT / item["path"]) != item["sha256"] for item in existing["outputs"]):
            raise ValueError("Frozen dataset output changed")
        print("Existing frozen dataset reproduced and verified; no outputs overwritten.", flush=True)
        return
    outputs = []
    score_file = ARTIFACTS / "scored_dates_24.txt"
    score_file.write_text("\n".join(date.date().isoformat() for date in SCORES) + "\n")
    outputs.append({"path": str(score_file.relative_to(ROOT)), "sha256": digest(score_file)})
    for family, tickers in {**FAMILIES, "global": TICKERS}.items():
        subset = panel[panel.asset_id.isin(tickers)].reset_index(drop=True)
        target = ARTIFACTS / f"{family}.parquet"
        subset.to_parquet(target, index=False)
        item = {"path": str(target.relative_to(ROOT)), "sha256": digest(target)}
        outputs.append(item)
        write_json(ARTIFACTS / f"{family}.manifest.json", {
            "artifact_path": str(target), "artifact_sha256": item["sha256"], "tickers": tickers,
            "approved_scored_dates": [date.date().isoformat() for date in SCORES],
            "required_columns": [column for column in subset if column not in {"date", "asset_id"}],
            "row_count": len(subset), "dataset_version": STUDY["dataset_version"]})
    audit["hy_source_overlap"] = checks
    audit["inflation_missingness_policy"] = inflation_signals(read_series("cpi_fred"))[1]
    audit["quarantined_unrequested_columns"] = ["yc_pc1", "yc_pc2", "yc_pc3", "de10y", "jp10y", "uk10y", "cn10y"]
    audit["limitations"] = ["Latest-vintage CPI; not a point-in-time macro-vintage study.",
                            "Original excess-return labels preserved; source revisions not mixed into targets.",
                            "Cross-sectional momentum rank uses the fixed 15-task universe.",
                            "Proxy semantics preserved; daily MOVE and breadth aggregated to business month end."]
    write_json(ARTIFACTS / "coverage_audit.json", audit)
    write_json(manifest_path, {"schema": "bayesfolio.repaired_dataset.v1", "created_at": datetime.now(UTC).isoformat(),
                              "study_sha256": digest(HERE / "study.json"), "builder_sha256": digest(Path(__file__)),
                              "etf_engineering_sha256": digest(ROOT / "bayesfolio/engine/features/asset_prices.py"),
                              "raw_sources": sources_meta, "outputs": outputs,
                              "predictor_lag_periods": 1, "selection_objectives": STUDY["objectives"]})
    print(json.dumps({"dataset": str(manifest_path), "rows": len(panel), "passed": True}), flush=True)


if __name__ == "__main__":
    main()
