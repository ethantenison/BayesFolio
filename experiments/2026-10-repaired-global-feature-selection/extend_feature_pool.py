"""Freeze additional historical predictors without changing any v1 observation.

The v2 feature-pool extension is built exclusively from frozen v1 raw sources.
All shared columns, labels, dates, ranks, and task rows must equal v1 exactly.
Existing v1 runs remain v1 runs, not re-labelled v2 resumes; new masks use their
own v2 artifact hash. This is a matched feature-pool extension, not a new history.

Usage:
    .venv/bin/python experiments/2026-10-repaired-global-feature-selection/extend_feature_pool.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import repair_dataset as data

OUTPUT = data.HERE / "artifacts/v2"
EXTRA_ETF = ["vol_1m", "vol_accel", "vol_ratio_1m_3m"]
EXTRA_MACRO = ["vix3m", "vix_ts_chg_1p", "dealer_gamma_proxy", "oil", "copper", "gold", "em_fx", "rsp_spy"]


def extra_macro(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reconstruct existing level/change/proxy definitions using trailing observations."""
    price = {name: frame["Adj Close"].dropna() for name, frame in sources.items()}
    monthly = {name: series.resample("BME").last() for name, series in price.items()}
    frame = pd.DataFrame(index=monthly["SPY"].index)
    frame["vix3m"] = monthly["^VIX3M"]
    frame["vix_ts_chg_1p"] = (np.log(monthly["^VIX3M"]) - np.log(monthly["^VIX"])).diff()
    realized = np.log(price["SPY"]).diff().rolling(21).std() * np.sqrt(252)
    frame["dealer_gamma_proxy"] = (realized - price["^VIX"] / 100).resample("BME").last()
    for name, symbol in [("oil", "CL=F"), ("copper", "HG=F"), ("gold", "GC=F"), ("em_fx", "CEW")]:
        frame[name] = monthly[symbol]
    frame["rsp_spy"] = monthly["RSP"] / monthly["SPY"]
    return frame.rename_axis("date").reset_index()


def extra_etf(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Reuse mean-daily monthly volatility/volume transforms from the existing builder."""
    from bayesfolio.core.settings import Horizon
    from bayesfolio.engine.features.asset_prices import fetch_etf_features

    daily = pd.concat({ticker: sources[ticker] for ticker in data.TICKERS}, axis=1)
    with patch("bayesfolio.engine.features.asset_prices.yf.download", return_value=daily):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            frame = fetch_etf_features(data.TICKERS, data.SOURCE_START, data.SOURCE_END, Horizon.MONTHLY)
    return frame[["date", "asset_id", *EXTRA_ETF]]


def build_extension() -> pd.DataFrame:
    """Join lagged additional predictors while proving all original cells unchanged."""
    original = pd.read_parquet(data.ARTIFACTS / "global.parquet")
    sources = data.market_sources()
    extra = extra_etf(sources).merge(extra_macro(sources), on="date", how="left", validate="many_to_one")
    extra = extra.sort_values(["asset_id", "date"])
    extra[[*EXTRA_ETF, *EXTRA_MACRO]] = extra.groupby("asset_id")[[*EXTRA_ETF, *EXTRA_MACRO]].shift(1)
    extra = extra[extra.date.isin(data.DATES)]
    result = original.merge(extra[["date", "asset_id", *EXTRA_ETF, *EXTRA_MACRO]],
                            on=["date", "asset_id"], how="left", validate="one_to_one")
    pd.testing.assert_frame_equal(result[original.columns], original, check_exact=True)
    data.validate_panel(result)
    return result


def main() -> None:
    """Create or reproduce v2 and its explicit cross-version comparison evidence."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = build_extension()
    original = pd.read_parquet(data.ARTIFACTS / "global.parquet")
    manifest_path = OUTPUT / "dataset.manifest.json"
    if manifest_path.exists():
        existing = pd.read_parquet(OUTPUT / "global.parquet")
        pd.testing.assert_frame_equal(panel, existing, check_exact=True)
        print("Existing v2 extension reproduced exactly; no artifacts overwritten.")
        return
    outputs = []
    scores = OUTPUT / "scored_dates_24.txt"
    scores.write_bytes((data.ARTIFACTS / "scored_dates_24.txt").read_bytes())
    outputs.append({"path": str(scores.relative_to(data.ROOT)), "sha256": data.digest(scores)})
    for family, tickers in {**data.FAMILIES, "global": data.TICKERS}.items():
        frame = panel[panel.asset_id.isin(tickers)].reset_index(drop=True)
        path = OUTPUT / f"{family}.parquet"
        frame.to_parquet(path, index=False)
        item = {"path": str(path.relative_to(data.ROOT)), "sha256": data.digest(path)}
        outputs.append(item)
        data.write_json(OUTPUT / f"{family}.manifest.json", {
            "dataset_version": "v2", "artifact_path": str(path), "artifact_sha256": item["sha256"],
            "tickers": tickers, "required_columns": [column for column in frame if column not in {"date", "asset_id"}],
            "approved_scored_dates": [date.date().isoformat() for date in data.SCORES], "row_count": len(frame)})
    audit = {"schema": "bayesfolio.feature_pool_extension.v1", "dataset_version": "v2",
             "base_dataset": "v1", "base_global_sha256": data.digest(data.ARTIFACTS / "global.parquet"),
             "all_shared_values_bitwise_equal": True, "shared_columns": list(original.columns),
             "extra_etf_columns": EXTRA_ETF, "extra_macro_columns": EXTRA_MACRO,
             "coverage": data.validate_panel(panel), "builder_sha256": data.digest(Path(__file__)),
             "raw_source_manifest_sha256": data.digest(data.RAW / "sources.json"), "outputs": outputs,
             "cross_version_policy": "V1 runs keep V1 hashes; V2 runs have V2 hashes. Only added input columns differ.",
             "remaining_quarantine": "Turnover-containing historical liquidity blocks remain outside selection policy; "
             "ill_log/dolvol_log require a training-only clipping implementation before becoming eligible."}
    data.write_json(manifest_path, audit)
    data.write_json(data.HERE / "results/feature_pool_extension_audit.json", audit)
    print(json.dumps({"passed": True, "version": "v2", "shared_cells_unchanged": True,
                      "additional_predictors": [*EXTRA_ETF, *EXTRA_MACRO]}))


if __name__ == "__main__":
    main()
