"""Freeze an October-close data extension for the November portfolio decision.

Experiment-only I/O adapter. Reuses the existing feature/target definitions on
append-only raw snapshots, preserves every old112-month observation, and keeps
November targets unavailable. Returns are decimal discounted excess; price/yield
units retain source conventions. Only this new experiment's data subtree writes.

Usage:
    uv run --locked --no-sync python <this_file> --fetch-and-build
    uv run --locked --no-sync python <this_file> --verify
Network fetch must only run after the parent has frozen the research plan.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = HERE.parent / "2026-10-monthly-data-extension-v1/outputs"
REPAIRED = HERE.parent / "2026-10-repaired-global-feature-selection"
AS_OF = "2026-10-31"
MARKET_ORIGIN = "2026-10-30"
LAST_SCORED = "2026-09-30"
SOURCE_END_EXCLUSIVE = "2026-10-31"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-november-data-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-november-data-cache")
sys.path.extend([str(REPO), str(REPAIRED)])

import repair_dataset as prior  # noqa: E402
from extend_feature_pool import EXTRA_ETF, extra_macro  # noqa: E402


def _load_math():
    path = OLD.parent / "extension_math.py"
    spec = importlib.util.spec_from_file_location("_november_frozen_extension_math", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the frozen append-only source mathematics")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    """Return a local file's exact SHA-256 fingerprint.

    Args:
        path: Existing snapshot or evidence path.

    Returns:
        Lowercase hexadecimal fingerprint.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cutoffs() -> dict[str, str]:
    """Return explicit October-completed cutoffs, distinguishing Saturday as-of.

    Returns:
        Calendar as-of, market forecast origin, last labeled origin and November
        target end. No Saturday price bar or November target is invented.
    """
    return {"as_of_date": AS_OF, "forecast_origin": MARKET_ORIGIN,
            "last_labeled_origin": LAST_SCORED, "target_month_end": "2026-11-30",
            "source_end_exclusive": SOURCE_END_EXCLUSIVE}


def _check_output(output: Path) -> Path:
    output = output.resolve()
    if not output.is_relative_to(HERE.resolve()) or output == HERE.resolve():
        raise ValueError("Data outputs must be a dedicated subtree of the new November experiment")
    return output


def _save(output: Path, path: Path, value: dict | bytes) -> None:
    if not path.resolve().is_relative_to(output):
        raise ValueError("Data adapter cannot write outside its registered output subtree")
    payload = value if isinstance(value, bytes) else (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Conflicting immutable data artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _frame(output: Path, path: Path, frame: pd.DataFrame, *, index: bool = False) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=index)
    _save(output, path, buffer.getvalue())


def _source_path(folder: Path, symbol: str) -> Path:
    return folder / (symbol.replace("^", "INDEX_").replace("=", "_") + ".parquet")


def _read_series(path: Path) -> pd.Series:
    data = pd.read_csv(path)
    result = pd.Series(pd.to_numeric(data.iloc[:, 1], errors="coerce").to_numpy(),
                       index=pd.to_datetime(data.iloc[:, 0])).dropna().sort_index()
    if result.index.has_duplicates or not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"Invalid macro source: {path}")
    return result


def _verify_hashes(values: dict) -> None:
    for name, expected in values.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Changed protected data input: {name}")


def _protected() -> dict:
    seal_path = OLD / "COMPLETED.json"
    seal = json.loads(seal_path.read_text())
    if not seal["passed"]:
        raise ValueError("Prior data extension must be complete")
    sources = [OLD / "dataset/global.parquet", OLD / "dataset/unscored_september_origin.parquet",
               OLD / "coverage_audit.json", *sorted((OLD / "raw/combined").glob("*"))]
    expected = {**seal["input_sha256"], **seal["artifact_sha256"]}
    protected = {}
    for source in sources:
        name = str(source.resolve())
        if name not in expected or digest(source) != expected[name]:
            raise ValueError(f"Unregistered or changed prior source: {source}")
        protected[name] = expected[name]
    labels = REPAIRED / "artifacts/v1/raw/labels.parquet"
    label_manifest = REPAIRED / "artifacts/v1/raw/sources.json"
    expected_label_hash = json.loads(label_manifest.read_text())["labels"]["sha256"]
    if digest(labels) != expected_label_hash:
        raise ValueError("Frozen pre-study warm-up labels no longer match their source manifest")
    protected[str(labels.resolve())] = expected_label_hash
    protected[str(label_manifest.resolve())] = digest(label_manifest)
    protected[str(seal_path.resolve())] = digest(seal_path)
    for source in [Path(__file__), OLD.parent / "extension_math.py", REPAIRED / "repair_dataset.py",
                   REPAIRED / "extend_feature_pool.py", REPO / "bayesfolio/engine/features/asset_prices.py"]:
        protected[str(source.resolve())] = digest(source)
    return protected


def _fetch(output: Path) -> dict:
    import yfinance as yf

    yf.set_tz_cache_location("/private/tmp/bayesfolio-november-data-yf")
    folder = output / "raw/fresh"
    names = list(dict.fromkeys([*prior.TICKERS, *prior.MACRO_SYMBOLS]))
    metadata = {}
    for symbol in names:
        destination = _source_path(folder, symbol)
        record_path = destination.with_suffix(".json")
        if destination.exists():
            if not record_path.exists():
                raise ValueError(f"Unregistered partial raw snapshot: {symbol}")
            record = json.loads(record_path.read_text())
            if digest(destination) != record["sha256"]:
                raise ValueError(f"Changed fresh snapshot: {symbol}")
            metadata[symbol] = record
            continue
        frame = yf.download(symbol, start="2026-08-01", end=SOURCE_END_EXCLUSIVE, auto_adjust=False,
                            threads=False, progress=False, timeout=30)
        if isinstance(frame.columns, pd.MultiIndex):
            frame = frame.xs(symbol, axis=1, level=1)
        frame = frame.dropna(how="all").copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
        frame = frame.sort_index()
        if frame.empty or "Adj Close" not in frame or frame.index[-1] != pd.Timestamp(MARKET_ORIGIN):
            raise ValueError(f"Missing October30 regular daily endpoint: {symbol}")
        if frame.index.has_duplicates or not np.isfinite(frame["Adj Close"].to_numpy()).all():
            raise ValueError(f"Invalid fresh adjusted-price series: {symbol}")
        _frame(output, destination, frame, index=True)
        record = {"symbol": symbol, "provider": "Yahoo Finance via yfinance",
                  "retrieved_at": datetime.now(UTC).isoformat(),
                  "requested_start": "2026-08-01", "end_exclusive": SOURCE_END_EXCLUSIVE,
                  "first_observation": str(frame.index[0].date()), "last_observation": MARKET_ORIGIN,
                  "rows": len(frame), "sha256": digest(destination), "regular_session_daily_bars": True}
        _save(output, record_path, record)
        metadata[symbol] = record
    _save(output, folder / "sources.json", metadata)
    # Obtain corporate-action evidence only when the append-only overlap guard
    # finds a real adjusted-history revision. Never overwrite the raw snapshot.
    for symbol in names:
        if symbol in {"^IRX", "^TNX"}:
            continue
        old = pd.read_parquet(_source_path(OLD / "raw/combined", symbol))
        fresh = pd.read_parquet(_source_path(folder, symbol))
        shared = old.index.intersection(fresh.index)
        factor = old.iloc[-1]["Adj Close"] / fresh.loc[old.index[-1], "Adj Close"]
        relative = fresh.loc[shared, "Adj Close"] * factor / old.loc[shared, "Adj Close"]
        if float((relative - 1).abs().max()) <= 1e-4:
            continue
        actions_path = _source_path(output / "raw/revision_evidence", symbol)
        if actions_path.exists():
            continue
        evidence = yf.download(symbol, start="2026-08-01", end=SOURCE_END_EXCLUSIVE, auto_adjust=False,
                               actions=True, threads=False, progress=False, timeout=30)
        if isinstance(evidence.columns, pd.MultiIndex):
            evidence = evidence.xs(symbol, axis=1, level=1)
        evidence = evidence.dropna(how="all").copy()
        evidence.index = pd.to_datetime(evidence.index).tz_localize(None).normalize()
        if "Dividends" not in evidence or "Stock Splits" not in evidence:
            raise ValueError(f"Missing corporate-action evidence for revised adjusted history: {symbol}")
        pd.testing.assert_series_equal(evidence["Close"].reindex(fresh.index), fresh["Close"],
                                       check_names=False, check_dtype=False, check_exact=True)
        _frame(output, actions_path, evidence, index=True)
        _save(output, actions_path.with_suffix(".json"), {
            "symbol": symbol, "provider": "Yahoo Finance corporate actions", "sha256": digest(actions_path),
            "retrieved_at": datetime.now(UTC).isoformat(), "raw_snapshots_overwritten": False})
    return metadata


def _splice_dividend_revision(old: pd.DataFrame, fresh: pd.DataFrame, evidence: pd.DataFrame) -> tuple:
    # A known dividend correction before the unchanged overlap anchor must not
    # silently rewrite frozen history. Admit only price-identical, action-verified
    # piecewise-constant adjustments; arbitrary vendor revisions remain blocked.
    shared = old.index.intersection(fresh.index)
    anchor = old.index[-1]
    if len(shared) < 10 or fresh.index[-1] != pd.Timestamp(MARKET_ORIGIN):
        raise ValueError("Incomplete dividend-revision overlap")
    for name in ["Open", "High", "Low", "Close", "Volume"]:
        np.testing.assert_array_equal(old.loc[shared, name], fresh.loc[shared, name])
    factor = float(old.loc[anchor, "Adj Close"] / fresh.loc[anchor, "Adj Close"])
    ratio = fresh.loc[shared, "Adj Close"] * factor / old.loc[shared, "Adj Close"]
    jumps = np.log(ratio).diff()
    dates = jumps.index[jumps.abs() > 1e-6]
    if len(dates) == 0 or (evidence["Stock Splits"] != 0).any():
        raise ValueError("Only explicit dividend-only historical adjustment corrections are admissible")
    checks = []
    for date in dates:
        dividend = float(evidence.loc[date, "Dividends"])
        previous = evidence.loc[evidence.index < date, "Close"].iloc[-1]
        expected = -float(np.log1p(-dividend / previous))
        if dividend <= 0 or abs(float(jumps.loc[date]) - expected) > 1e-6:
            raise ValueError("Adjusted-history jump does not match recorded dividend arithmetic")
        checks.append({"ex_date": str(date.date()), "dividend": dividend,
                       "prior_close": float(previous), "log_adjustment_jump": float(jumps.loc[date]),
                       "expected_log_jump": expected})
    terminal = ratio.loc[ratio.index >= dates[-1]]
    if len(terminal) < 3 or float((terminal - 1).abs().max()) > 1e-6:
        raise ValueError("The post-correction anchor neighborhood is not stable")
    tail = fresh.loc[fresh.index > anchor, old.columns].copy()
    tail["Adj Close"] *= factor
    for name, dtype in old.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            values = tail[name].to_numpy(float)
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                raise ValueError("Invalid integer source field")
        tail[name] = tail[name].astype(dtype)
    combined = pd.concat([old, tail])
    pd.testing.assert_frame_equal(combined.loc[old.index], old, check_exact=True)
    return combined, {"anchor_date": str(anchor.date()), "adjusted_price_scale": factor,
                      "max_overlap_relative_difference": float((ratio - 1).abs().max()),
                      "appended_daily_rows": len(tail), "old_rows_unchanged": True,
                      "historical_dividend_revision_recognized_not_applied": True,
                      "corporate_action_checks": checks,
                      "known_limitation": "Frozen historical MGK inputs/labels retain their original vendor vintage; "
                                          "a newly populated pre-anchor dividend correction is not back-applied."}


def _combine(output: Path) -> dict:
    math = _load_math()
    metadata = json.loads((output / "raw/fresh/sources.json").read_text())
    folder = output / "raw/combined"
    audit = {}
    for symbol in dict.fromkeys([*prior.TICKERS, *prior.MACRO_SYMBOLS]):
        original = _source_path(OLD / "raw/combined", symbol)
        fresh = _source_path(output / "raw/fresh", symbol)
        if digest(fresh) != metadata[symbol]["sha256"]:
            raise ValueError(f"Changed fresh source: {symbol}")
        try:
            combined, evidence = math.splice_market(pd.read_parquet(original), pd.read_parquet(fresh),
                                                    symbol=symbol, completed=MARKET_ORIGIN)
        except ValueError as error:
            if "Nonconstant historical price revisions" not in str(error):
                raise
            evidence_path = _source_path(output / "raw/revision_evidence", symbol)
            record = json.loads(evidence_path.with_suffix(".json").read_text())
            if digest(evidence_path) != record["sha256"]:
                raise ValueError("Changed dividend correction evidence") from error
            combined, evidence = _splice_dividend_revision(pd.read_parquet(original), pd.read_parquet(fresh),
                                                            pd.read_parquet(evidence_path))
            evidence["corporate_action_source_sha256"] = digest(evidence_path)
        _frame(output, _source_path(folder, symbol), combined, index=True)
        audit[symbol] = {**evidence, "old_sha256": digest(original), "fresh_sha256": digest(fresh),
                         "combined_sha256": digest(_source_path(folder, symbol))}
    # October-origin predictors are lagged once and need only September macro
    # observations. These snapshots were already available by the Oct24 freeze,
    # so avoid importing post-cutoff CPI/HY revisions from a newer download.
    for name in ["cpi_fred", "hy_fred_live"]:
        source = OLD / "raw/combined" / f"{name}.csv"
        series = _read_series(source)
        minimum = pd.Timestamp("2026-09-01" if name == "cpi_fred" else "2026-09-30")
        if series.index.max() < minimum:
            raise ValueError(f"Frozen publication-available macro does not cover required September inputs: {name}")
        _save(output, folder / f"{name}.csv", source.read_bytes())
        audit[name] = {"source": str(source), "sha256": digest(source),
                       "last_observation": str(series.index.max().date()), "new_download": False,
                       "available_no_later_than": "2026-10-24", "required_by_lagged_live_origin": str(minimum.date()),
                       "post_cutoff_revisions_applied": False}
    return audit


def _build_panel(output: Path) -> tuple[pd.DataFrame, dict]:
    from bayesfolio.core.settings import Horizon
    from bayesfolio.engine.features.asset_prices import (
        add_cross_sectional_momentum_rank,
        compute_excess_future_return_calendar,
        fetch_etf_features,
    )

    old = pd.read_parquet(OLD / "dataset/global.parquet")
    previous_live = pd.read_parquet(OLD / "dataset/unscored_september_origin.parquet")
    dates = pd.date_range(prior.STUDY["training_start"], MARKET_ORIGIN, freq="BME")
    settings = {**prior.STUDY, "last_score": MARKET_ORIGIN}
    combined = output / "raw/combined"
    with patch.multiple(prior, RAW=combined, STUDY=settings, DATES=dates, SOURCE_END=SOURCE_END_EXCLUSIVE):
        sources = prior.market_sources()
        daily = pd.concat({ticker: sources[ticker] for ticker in prior.TICKERS}, axis=1)
        with (patch("bayesfolio.engine.features.asset_prices.yf.download", return_value=daily),
              warnings.catch_warnings()):
            warnings.simplefilter("ignore", FutureWarning)
            engineered = fetch_etf_features(prior.TICKERS, prior.SOURCE_START, SOURCE_END_EXCLUSIVE, Horizon.MONTHLY)
        etf = add_cross_sectional_momentum_rank(engineered)[["date", "asset_id", *prior.ETF_COLUMNS, *EXTRA_ETF]]
        macro = prior.rebuild_macro(sources, prior.read_series("hy_fred_live")).merge(
            extra_macro(sources), on="date", validate="one_to_one")
        rf = (sources["^IRX"]["Adj Close"] / 100 / 252).reindex(
            pd.bdate_range(sources["^IRX"].index.min(), MARKET_ORIGIN)).ffill()
        october_rf = rf.loc[(rf.index > LAST_SCORED) & (rf.index <= MARKET_ORIGIN)]
        if october_rf.empty or october_rf.isna().any():
            raise ValueError("Complete October risk-free outcome required")
        raw_labels = pd.read_parquet(REPAIRED / "artifacts/v1/raw/labels.parquet")
        labels = raw_labels[raw_labels.asset_id.isin(prior.TICKERS) & raw_labels.date.lt(old.date.min())][
            ["date", "asset_id", "y_excess_lead"]].copy()
        label_parts = [labels, old[["date", "asset_id", "y_excess_lead"]]]
        target_audit = []
        for ticker in prior.TICKERS:
            rebuilt = compute_excess_future_return_calendar(sources[ticker]["Adj Close"], rf,
                Horizon.MONTHLY, include_unlabeled_tail=True)
            expected = old[old.asset_id.eq(ticker)].set_index("date").y_excess_lead
            difference = (rebuilt.reindex(expected.index) - expected).abs()
            if difference.isna().any() or float(difference.max()) >= 2e-6:
                raise ValueError(f"Historical labels fail the existing reconstruction tolerance: {ticker}")
            added = rebuilt.loc[rebuilt.index > old.date.max()].rename_axis("date").reset_index()
            added["asset_id"] = ticker
            label_parts.append(added[["date", "asset_id", "y_excess_lead"]])
            target_audit.append({"asset": ticker, "old_labels": len(expected),
                                 "max_reconstruction_difference": float(difference.max())})
        labels = pd.concat(label_parts, ignore_index=True)
        panel = prior.align_panel(etf, macro, labels)[list(old.columns)]
        replay = panel[panel.date.le(old.date.max())].reset_index(drop=True)
        pd.testing.assert_frame_equal(replay, old, check_exact=False, rtol=1e-11, atol=1e-12)
        previous = panel[panel.date.eq(pd.Timestamp(LAST_SCORED))].reset_index(drop=True)
        predictors = [column for column in old if column != "y_excess_lead"]
        pd.testing.assert_frame_equal(previous[predictors], previous_live[predictors].reset_index(drop=True),
                                      check_exact=False, rtol=1e-11, atol=1e-12)
        # Retain exact cached September predictors; only the formerly missing
        # October realized-return target is newly populated.
        september = previous_live.copy()
        september["y_excess_lead"] = previous.y_excess_lead.to_numpy()
        panel = pd.concat([old, september, panel[panel.date.eq(pd.Timestamp(MARKET_ORIGIN))]], ignore_index=True)
        pd.testing.assert_frame_equal(panel.iloc[:len(old)], old, check_exact=True)
        inflation = prior.inflation_signals(prior.read_series("cpi_fred"))[1]
    return panel, {"old112_month_prefix_bitwise_unchanged": True,
                   "september_predictors_bitwise_unchanged": True,
                   "old_target_reconstruction": target_audit,
                   "october_risk_free_return": float(np.expm1(october_rf.sum())),
                   "inflation_missingness_policy": inflation}


def validate_panel(panel: pd.DataFrame, tickers: list[str]) -> dict:
    """Validate complete October-origin features and causal November-label exclusion.

    Args:
        panel: 114 monthly origins per ETF, decimal discounted-excess target.
        tickers: Exact15 approved tickers.

    Returns:
        Coverage and target-availability evidence; raises on any missing feature.
    """
    expected_dates = pd.date_range("2017-05-31", MARKET_ORIGIN, freq="BME")
    expected = pd.MultiIndex.from_product([expected_dates, sorted(tickers)], names=["date", "asset_id"])
    actual = pd.MultiIndex.from_frame(panel[["date", "asset_id"]])
    if len(tickers) != 15 or not actual.equals(expected) or panel.duplicated(["date", "asset_id"]).any():
        raise ValueError("A complete unique114-date/15-ETF grid is required")
    predictors = panel.drop(columns=["date", "asset_id", "y_excess_lead"])
    if not np.isfinite(predictors.to_numpy(float)).all():
        raise ValueError("Missing/nonfinite model features; no zero-filling allowed")
    live = panel.date.eq(pd.Timestamp(MARKET_ORIGIN))
    if not panel.loc[live, "y_excess_lead"].isna().all():
        raise ValueError("November outcomes must remain unavailable at the October origin")
    if not np.isfinite(panel.loc[~live, "y_excess_lead"].to_numpy()).all():
        raise ValueError("Missing completed historical target")
    return {"passed": True, **cutoffs(), "origins_per_etf": 114, "labeled_origins_per_etf": 113,
            "rows": len(panel), "unavailable_november_targets": int(live.sum()),
            "maximum_training_target_origin": LAST_SCORED, "training_target_outcomes_available_through": MARKET_ORIGIN,
            "missing_predictor_cells": 0, "feature_lag_periods": 1,
            "raw_hy_level_change_in_dataset_but_excluded_from_registered_models": True}


def prepare_data(output: Path, fetch: bool = False) -> dict:
    """Fetch once or reproduce the isolated October-close dataset extension.

    Args:
        output: Dedicated data subtree within the new November experiment.
        fetch: Explicit permission to obtain absent fresh market snapshots.
            The parent must freeze the experiment plan before setting this true.

    Returns:
        Passed dataset manifest with artifact paths, cutoffs, hashes and coverage.
        Never computes forecasts, portfolio allocations or October performance.
    """
    output = _check_output(output)
    completed = output / "COMPLETED.json"
    if completed.exists():
        seal = json.loads(completed.read_text())
        _verify_hashes(seal["input_sha256"])
        _verify_hashes(seal["artifact_sha256"])
        if not seal["passed"]:
            raise ValueError("Data extension seal is not passed")
        return seal
    protected = _protected()
    if fetch:
        _fetch(output)
    elif not (output / "raw/fresh/sources.json").exists():
        raise ValueError("Fresh market snapshots absent; wait for frozen-plan fetch authorization")
    source_audit = _combine(output)
    panel, build = _build_panel(output)
    coverage = validate_panel(panel, prior.TICKERS)
    dataset = output / "dataset/global.parquet"
    _frame(output, dataset, panel)
    _frame(output, output / "dataset/scored_global.parquet",
           panel[panel.date.lt(pd.Timestamp(MARKET_ORIGIN))].reset_index(drop=True))
    _frame(output, output / "dataset/november_forecast_origin.parquet",
           panel[panel.date.eq(pd.Timestamp(MARKET_ORIGIN))].reset_index(drop=True))
    _save(output, output / "coverage_audit.json", {**coverage, "source_audit": source_audit, "build_audit": build,
          "limitations": ["Historical CPI uses frozen vintage, not a point-in-time-vintage backtest.",
                          "October live macro features use already-public September observations, lagged once.",
                          "No November target or future market observation is supplied.",
                          "New adjusted prices are audited onto the prior snapshot scale."]})
    _verify_hashes(protected)
    artifacts = {str(path.resolve()): digest(path) for path in output.rglob("*") if path.is_file()}
    seal = {"schema": "bayesfolio.november_data.v1", "passed": True, **coverage,
            "dataset": str(dataset), "coverage_audit": str(output / "coverage_audit.json"),
            "input_sha256": protected, "artifact_sha256": artifacts}
    _save(output, completed, seal)
    return seal


def main() -> None:
    """Build only after explicit fetch authorization, or verify frozen data offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "data")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--fetch-and-build", action="store_true")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify and not (args.output / "COMPLETED.json").exists():
        raise FileNotFoundError("No completed data seal exists; verification never builds or fetches")
    try:
        result = prepare_data(args.output, fetch=args.fetch_and_build)
    except Exception:
        if args.fetch_and_build:
            output = _check_output(args.output)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            _save(output, output / "failures" / f"data_attempt_{stamp}.json",
                  {"timestamp_utc": datetime.now(UTC).isoformat(), "traceback": traceback.format_exc(),
                   "settings_unchanged": True, "no_GP_fits": True})
        raise
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"input_sha256", "artifact_sha256"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
