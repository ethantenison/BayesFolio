"""Check preserved target units, forward horizons, and label availability.

The original decimal excess-return labels stay unchanged. Independently rebuild
labels from frozen daily prices/RF, optionally using separately frozen forward
endpoints to cover the final month. This is an offline audit, with no GP fits.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/audit_target_alignment.py
"""

from __future__ import annotations

import json

import pandas as pd
import repair_dataset as data
from freeze_target_tail import OUTPUT, snapshot_path


def audit_price_series(sources: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.Series], bool, dict[str, float]]:
    """Extend audit-only prices through the last return endpoint without changing feature sources."""
    prices = {symbol: frame["Adj Close"].dropna() for symbol, frame in sources.items()}
    manifest = OUTPUT / "sources.json"
    if not manifest.is_file():
        return prices, False, {}
    metadata = json.loads(manifest.read_text())
    scales = {}
    for symbol in [*data.TICKERS, "^IRX"]:
        path = snapshot_path(symbol)
        if symbol not in metadata or not path.is_file() or metadata[symbol]["sha256"] != data.digest(path):
            raise ValueError("Forward endpoint source set is incomplete or changed")
        tail = pd.read_parquet(path)["Adj Close"].dropna()
        previous = prices[symbol]
        # Dividend-adjustment vintages may rescale the entire old price path.
        # Match the overlap endpoint; append only future values. RF yields are
        # annual percentages, not adjusted stock prices, so never rescale them.
        scale = 1.0 if symbol == "^IRX" else float(previous.iloc[-1] / tail.loc[previous.index[-1]])
        scales[symbol] = scale
        prices[symbol] = pd.concat([previous, (tail * scale).loc[tail.index > previous.index[-1]]])
    return prices, True, scales


def main() -> None:
    """Rebuild supported forward labels per ETF and persist the exact horizon convention."""
    from bayesfolio.core.settings import Horizon
    from bayesfolio.engine.features.asset_prices import compute_excess_future_return_calendar

    sources = data.market_sources()
    prices_for_audit, complete_tail, scales = audit_price_series(sources)
    rf = prices_for_audit["^IRX"] / 100.0 / 252.0
    rf = rf.reindex(pd.bdate_range(rf.index.min(), rf.index.max())).ffill()
    labels = pd.read_parquet(data.RAW / "labels.parquet")
    cached_prices = pd.read_parquet(data.RAW / "old_etf_features.parquet")
    rows = []
    for ticker in data.TICKERS:
        rebuilt = compute_excess_future_return_calendar(prices_for_audit[ticker], rf, Horizon.MONTHLY)
        old = labels[labels.asset_id.eq(ticker)].set_index("date").y_excess_lead
        shared = rebuilt.index.intersection(old.index).intersection(data.DATES)
        delta = (rebuilt.loc[shared] - old.loc[shared]).abs()
        if len(delta) != (108 if complete_tail else 107) or delta.max() >= 2e-6:
            raise ValueError(f"Preserved target source discrepancy requires investigation: {ticker}")
        final_target = float(old.loc[data.DATES[-1]])
        prices = cached_prices[cached_prices.asset_id.eq(ticker)].set_index("date").price
        next_date = data.DATES[-1] + pd.offsets.BMonthEnd(1)
        if prices.loc[[data.DATES[-1], next_date]].isna().any() or not prices.loc[
            [data.DATES[-1], next_date]].gt(0).all():
            raise ValueError(f"Final target forward-price availability is missing: {ticker}")
        rows.append({"asset_id": ticker, "independently_rebuilt_observations": len(delta),
                     "max_abs_decimal_difference": float(delta.max()),
                     "max_abs_difference_basis_points": float(delta.max() * 10000),
                     "median_abs_decimal_difference": float(delta.median()),
                     "final_score_observation_date": str(data.DATES[-1].date()),
                     "final_forward_endpoint": str(next_date.date()), "final_preserved_target": final_target,
                     "final_target_prices_available_in_frozen_monthly_cache": True,
                     "final_target_independently_rebuilt_from_daily_sources": complete_tail})
    report = {"schema": "bayesfolio.target_alignment_audit.v1", "passed": True,
              "source_definition": "exp(log(P[t+1]/P[t]) - integrated daily risk-free continuous rate) - 1",
              "source_code": "bayesfolio/engine/features/asset_prices.py:compute_excess_future_return_calendar",
              "returns_unit": "decimal excess returns; no second risk-free subtraction",
              "labels_unchanged": True, "source_tolerance_decimal": 2e-6,
              "initial_training_observation_dates": ["2017-05-31", "2024-04-30"],
              "initial_training_forward_outcome_endpoints": ["2017-06-30", "2024-05-31"],
              "score_observation_dates": ["2024-05-31", "2026-04-30"],
              "scored_forward_outcome_endpoints": ["2024-06-28", "2026-05-29"],
              "decision_time_policy": "A window t trains on rows dated before t; its last training outcome ends at t. "
              "The forecast target is the return from t to the next month end; predictors are lagged one month.",
              "independent_reconstruction_total": sum(row["independently_rebuilt_observations"] for row in rows),
              "remaining_cache_only_forward_targets": 0 if complete_tail else len(data.TICKERS), "by_asset": rows,
              "forward_tail_manifest": str((OUTPUT / "sources.json").relative_to(data.ROOT)) if complete_tail else None,
              "audit_tail_price_anchoring_scales": scales,
              "limitation": "Latest adjusted-price vintage, not a point-in-time vendor-vintage replication. "
              "All preserved labels were independently checked." if complete_tail else
              "Frozen daily inputs end April 2026; final 15 forward targets have cache-availability checks only."}
    data.write_json(data.HERE / "results/target_alignment_audit.json", report)
    summary = pd.DataFrame(rows)[["asset_id", "independently_rebuilt_observations",
                                 "max_abs_difference_basis_points"]]
    lines = ["# Target alignment and availability audit", "",
             "All original decimal excess-return targets remain unchanged.", "",
             "Observation/decision dates are not the same as the future return endpoints:", "",
             "| Panel | Observation dates | Forward outcome endpoints |", "| --- | --- | --- |",
             "| Initial 84 training rows | May 2017–April 2024 | June 2017–May 2024 |",
             "| 24 scored rows | May 2024–April 2026 | June 2024–May 2026 |", "",
             "At decision month-end t, the last training row is dated t-1 and its forward outcome",
             "has finished at t. The prediction concerns t to the next month-end. Inputs are lagged",
             "once. This preserves the runner's existing causal timing; no extra label shift is applied.", "",
             "## Independent daily-source reconstruction", "", summary.to_markdown(index=False), "",
             f"All {report['independent_reconstruction_total']:,} reconstructed labels differ by less than "
             "0.02 basis points from the preserved",
             "cache, consistent with tiny adjusted-price source-vintage differences. No target was changed.", "",
             "## Source-vintage and endpoint policy", "",
             report["limitation"], "", "Full machine-readable evidence: `target_alignment_audit.json`.", ""]
    (data.HERE / "results/TARGET_ALIGNMENT_AUDIT.md").write_text("\n".join(lines))
    print(json.dumps({"passed": True, "independently_rebuilt_labels": report["independent_reconstruction_total"],
                      "cache_only_last_month_labels": report["remaining_cache_only_forward_targets"]}))


if __name__ == "__main__":
    main()
