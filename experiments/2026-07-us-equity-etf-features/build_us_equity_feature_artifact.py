"""Build and audit the first-round U.S. equity ETF feature artifact.

This script refreshes ETF-local and macro predictors for the U.S. equity family
so newly added provider columns and horizon-aligned macro transformations are
available, then selects the pruned first-round feature set. Return labels use
the local cache. Outputs are written under this experiment folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from bayesfolio.contracts.commands.features import BuildFeaturesDatasetCommand
from bayesfolio.core.settings import Horizon, Interval
from bayesfolio.engine.features.asset_prices import build_long_panel, fetch_etf_features
from bayesfolio.engine.features.dataset_builder import FeatureProviders, build_features_dataset
from bayesfolio.engine.features.market_fundamentals import fetch_enhanced_macro_features
from bayesfolio.io import EtfFeaturesProvider, MacroProvider, ParquetArtifactStore, ReturnsProvider

EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
US_EQUITY_TICKERS = ["SPY", "MGK", "VTV", "IWM"]
DEFERRED_TICKERS = ["IJR"]

LOOKBACK_DATE = date(2019, 3, 1)
START_DATE = date(2021, 3, 1)
END_DATE = date(2026, 7, 2)

ETF_CANDIDATES = [
    "mom6m",
    "mom12m_skip1m",
    "cs_mom_rank",
    "trend_slope",
    "vol_1m",
    "vol_z",
    "vol_accel",
    "vol_ratio_1m_3m",
    "max_dd_3m",
    "ill_log",
    "dolvol_log",
    "turnover",
]

ENGINEERED_CANDIDATES = [
    "lag_y_excess_lead",
    "lag2_y_excess_lead",
]

MACRO_CANDIDATES = [
    "vix",
    "vix_slope",
    "spy_ret",
    "pct_above_50dma",
    "hy_spread_chg_1p",
    "hy_spread_z_12p",
    "cpi_chg_12p",
    "term_spread",
]

INTERNATIONAL_EQUITY_MACRO_CANDIDATES = [
    # Global risk / volatility
    "vix",
    "vix3m",
    "vix_slope",
    "vix_ts_level",
    "vix_ts_chg_1p",
    "vix_ts_z_12p",
    "skew_proxy",
    "move_proxy",
    "dealer_gamma_proxy",
    # Dollar / FX
    "dxy",
    "em_fx",
    "em_fx_ret",
    # Regional/global equity risk appetite and breadth
    "spy_ret",
    "erp",
    "rsp_spy",
    "rsp_spy_roc_1p",
    "pct_above_50dma",
    "spy_flow_z_12p",
    # Global rates / real-rate proxies
    "tnote10y",
    "tbill3m",
    "term_spread",
    "y10_nominal",
    "schp_ret",
    "y10_real_proxy",
    "breakeven_proxy",
    "de10y",
    "jp10y",
    "uk10y",
    "cn10y",
    # Credit / EM risk
    "credit_spread",
    "credit_spread_chg_1p",
    "hy_spread",
    "hy_spread_chg_1p",
    "hy_spread_z_12p",
    # Inflation
    "cpi_chg_12p",
    "cpi_chg_1p",
    # Commodity sensitivity
    "oil",
    "copper",
    "gold",
    "oil_ret",
    "copper_ret",
    "gold_crude_ratio",
]

MACRO_PRESETS = {
    "us_equity": MACRO_CANDIDATES,
    "international_equity": INTERNATIONAL_EQUITY_MACRO_CANDIDATES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-ijr",
        action="store_true",
        help="Include deferred IJR in the U.S. equity feature artifact.",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated ETF family override. When set, --include-ijr is ignored.",
    )
    parser.add_argument(
        "--family-name",
        default="us_equity",
        help="Artifact/report family name used in output filenames and manifests.",
    )
    parser.add_argument(
        "--macro-preset",
        choices=sorted(MACRO_PRESETS),
        default=None,
        help="Candidate macro feature preset. Defaults to --family-name when available, else us_equity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tickers:
        tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        if not tickers:
            raise ValueError("--tickers must contain at least one ETF ticker")
        deferred_tickers = []
    else:
        tickers = [*US_EQUITY_TICKERS, *DEFERRED_TICKERS] if args.include_ijr else list(US_EQUITY_TICKERS)
        deferred_tickers = [] if args.include_ijr else list(DEFERRED_TICKERS)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    family_name = str(args.family_name).strip().lower().replace(" ", "_")
    macro_preset = args.macro_preset or (family_name if family_name in MACRO_PRESETS else "us_equity")
    macro_candidates = list(MACRO_PRESETS[macro_preset])
    suffix = "custom" if args.tickers else ("with_ijr" if args.include_ijr else "first_round")
    artifact_name = f"{family_name}_{suffix}_feature_candidates_{run_id}.parquet"

    command = BuildFeaturesDatasetCommand(
        tickers=tickers,
        drop_assets=[],
        lookback_date=LOOKBACK_DATE,
        start_date=START_DATE,
        end_date=END_DATE,
        interval=Interval.DAILY,
        horizon=Horizon.MONTHLY,
        etf_cols=ETF_CANDIDATES,
        macro_cols=macro_candidates,
        drop_etf_cols=[],
        drop_macro_cols=[],
        clip_quantile=0.99,
        seed=27,
        artifact_name=artifact_name,
        include_unlabeled_tail=True,
    )

    providers = FeatureProviders(
        returns_provider=ReturnsProvider(
            fetcher=build_long_panel,
            cache_dir="artifacts/cache/returns",
        ),
        macro_provider=MacroProvider(
            fetcher=fetch_enhanced_macro_features,
            cache_dir=None,
            max_retries=1,
            retry_backoff_seconds=0.1,
        ),
        etf_features_provider=EtfFeaturesProvider(
            fetcher=fetch_etf_features,
            cache_dir=None,
        ),
    )
    store = ParquetArtifactStore(base_dir=ARTIFACT_DIR)
    result = build_features_dataset(command=command, providers=providers, artifact_store=store)
    artifact_path = resolve_uri(str(result.artifact.uri))
    audit = audit_correlations(artifact_path, macro_candidates=macro_candidates)

    manifest = {
        "schema": "bayesfolio.etf_family_feature_artifact.v1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family_name": family_name,
        "artifact_path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "row_count": result.artifact.row_count,
        "column_count": result.artifact.column_count,
        "tickers": tickers,
        "deferred_tickers": deferred_tickers,
        "lookback_date": LOOKBACK_DATE.isoformat(),
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "horizon": Horizon.MONTHLY.value,
        "engineered_candidates": ENGINEERED_CANDIDATES,
        "etf_candidates": ETF_CANDIDATES,
        "macro_preset": macro_preset,
        "macro_candidates": macro_candidates,
        "diagnostics": result.diagnostics,
        "correlation_audit": audit,
    }
    manifest_path = ARTIFACT_DIR / f"manifest_{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_name = f"correlation_audit_{family_name}_{suffix}_{run_id}.md"
    write_correlation_report(manifest, tickers=tickers, report_path=EXPERIMENT_DIR / report_name)
    print(json.dumps({"artifact": str(artifact_path), "manifest": str(manifest_path)}, indent=2))


def audit_correlations(artifact_path: Path, *, macro_candidates: list[str]) -> dict[str, object]:
    frame = pd.read_parquet(artifact_path)
    requested_cols = [*ENGINEERED_CANDIDATES, *ETF_CANDIDATES, *macro_candidates]
    candidate_cols = [column for column in requested_cols if column in frame.columns]
    missing_cols = [column for column in requested_cols if column not in frame.columns]
    numeric = frame[candidate_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(min_periods=12)
    perfect = []
    high = []
    for index, left in enumerate(corr.columns):
        for right in corr.columns[index + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            pair = numeric[[left, right]].dropna()
            if len(pair) < 12:
                continue
            row = {
                "feature_a": left,
                "feature_b": right,
                "corr": float(value),
                "n": int(len(pair)),
            }
            if abs(value) >= 0.999999:
                perfect.append(row)
            elif abs(value) >= 0.98:
                high.append(row)

    constant_features = [column for column in candidate_cols if numeric[column].nunique(dropna=True) <= 1]
    return {
        "available_feature_count": len(candidate_cols),
        "missing_features": missing_cols,
        "perfect_correlation_threshold": 0.999999,
        "perfect_correlations": sorted(perfect, key=lambda item: abs(item["corr"]), reverse=True),
        "high_correlation_threshold": 0.98,
        "high_correlations": sorted(high, key=lambda item: abs(item["corr"]), reverse=True),
        "constant_features": constant_features,
    }


def write_correlation_report(manifest: dict[str, object], *, tickers: list[str], report_path: Path) -> None:
    audit = manifest["correlation_audit"]
    assert isinstance(audit, dict)
    lines = [
        "# ETF Family Candidate Feature Correlation Audit\n",
        f"Artifact: `{manifest['artifact_path']}`\n",
        (
            f"Family: `{manifest.get('family_name', 'unknown')}` / `{', '.join(tickers)}`. "
            f"Horizon: `{manifest['horizon']}`.\n"
        ),
        "ETF-local and macro predictors were rebuilt live for this family so newly added columns "
        "and horizon-aligned macro transformations are included. Return labels used local cache.\n",
        "\n## Availability\n",
        f"- Available audited features: `{audit['available_feature_count']}`\n",
        f"- Missing from artifact: `{audit['missing_features']}`\n",
        "\n## Perfect Correlations\n",
    ]
    perfect = audit["perfect_correlations"]
    assert isinstance(perfect, list)
    if perfect:
        for row in perfect:
            lines.append(f"- `{row['feature_a']}` vs `{row['feature_b']}`: corr `{row['corr']:.6f}`, n `{row['n']}`\n")
    else:
        lines.append("- No pair had absolute Pearson correlation >= `0.999999` among available candidates.\n")

    lines.append("\n## High Correlations Worth Watching\n")
    high = audit["high_correlations"]
    assert isinstance(high, list)
    if high:
        for row in high[:30]:
            lines.append(f"- `{row['feature_a']}` vs `{row['feature_b']}`: corr `{row['corr']:.4f}`, n `{row['n']}`\n")
    else:
        lines.append("- No pair had absolute Pearson correlation >= `0.98` below the perfect-correlation threshold.\n")

    lines.append("\n## Constant Features\n")
    constant = audit["constant_features"]
    assert isinstance(constant, list)
    if constant:
        for feature in constant:
            lines.append(f"- `{feature}`\n")
    else:
        lines.append("- No available audited candidate was constant.\n")

    report_path.write_text("".join(lines), encoding="utf-8")


def resolve_uri(uri: str) -> Path:
    if uri.startswith("file://"):
        return Path(urlparse(uri).path)
    return Path(uri)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
