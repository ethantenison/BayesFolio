"""Freeze cached U.S. and international fixed-income experiment panels.

The builder uses the existing long-history 18-ETF feature artifact only. It
does not fetch or refresh external data. Each family receives the same final 24
complete monthly score dates while retaining the full cached training history.

Usage:
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/build_cached_fixed_income_artifacts.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
SOURCE = Path(
    "/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_18etf_common_history_201306_202605.parquet"
)
FAMILIES = {
    "us_fixed_income": ["BND", "IEF", "LQD", "HYG"],
    "international_fixed_income": ["BNDX", "VWOB", "HYEM"],
}
FAMILY_PREFIXES = {
    "us_fixed_income": "us_fixed_income",
    "international_fixed_income": "international_fixed_income",
}
IDENTIFIER_COLUMNS = ["date", "asset_id"]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_correlation_table(frame: pd.DataFrame) -> str:
    """Return a compact Markdown correlation table."""
    return frame.round(3).to_markdown()


def write_family_artifact(source: pd.DataFrame, family: str, tickers: list[str]) -> dict[str, object]:
    """Write one frozen family panel, date file, manifest, and audit payload."""
    prefix = FAMILY_PREFIXES[family]
    artifact_path = ARTIFACT_DIR / f"{prefix}_cached_inputs_20261016.parquet"
    manifest_path = ARTIFACT_DIR / f"{prefix}_cached_inputs_20261016.manifest.json"
    dates_path = ARTIFACT_DIR / f"{prefix}_scored_dates_24.txt"
    output = source.loc[source["asset_id"].isin(tickers)].sort_values(IDENTIFIER_COLUMNS).reset_index(drop=True)
    if sorted(output["asset_id"].unique().tolist()) != sorted(tickers):
        raise ValueError(f"Cached source does not contain the exact {family} universe.")

    value_columns = [column for column in output.columns if column not in IDENTIFIER_COLUMNS]
    complete_by_date = output.groupby("date", observed=True)[value_columns].apply(
        lambda frame: len(frame) == len(tickers) and bool(frame.notna().all().all())
    )
    complete_dates = pd.DatetimeIndex(complete_by_date[complete_by_date].index)
    if len(complete_dates) < 24:
        raise ValueError(f"Expected at least 24 complete dates for {family}; found {len(complete_dates)}.")
    approved_dates = complete_dates[-24:]
    output = output.loc[output["date"].isin(complete_dates)].reset_index(drop=True)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(artifact_path, index=False)
    dates_path.write_text(
        f"# Frozen {family} score panel: 24 business month-end dates.\n"
        + "\n".join(date.date().isoformat() for date in approved_dates)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "bayesfolio.etf_family_cached_confirmation_artifact.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "family": family,
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "source_artifacts": [{"path": str(SOURCE), "sha256": sha256_file(SOURCE)}],
        "tickers": tickers,
        "time_index_included": True,
        "row_count": int(len(output)),
        "column_count": int(len(output.columns)),
        "required_columns": value_columns,
        "complete_date_count": int(len(complete_dates)),
        "complete_date_min": complete_dates.min().date().isoformat(),
        "complete_date_max": complete_dates.max().date().isoformat(),
        "approved_scored_date_min": approved_dates.min().date().isoformat(),
        "approved_scored_date_max": approved_dates.max().date().isoformat(),
        "approved_scored_dates": [date.date().isoformat() for date in approved_dates],
        "scored_dates_path": str(dates_path),
        "cached_data_only": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    target_panel = output.pivot(index="date", columns="asset_id", values="y_excess_lead")
    coverage = output.groupby("asset_id", observed=True).agg(
        rows=("date", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        target_count=("y_excess_lead", "count"),
    )
    return {
        "family": family,
        "tickers": tickers,
        "coverage": coverage,
        "target_correlation": target_panel.corr(),
        "manifest": manifest,
    }


def write_audit(audits: list[dict[str, object]]) -> None:
    """Write the human-readable fixed-income data audit."""
    lines = [
        "# Fixed-Income Cached Data Audit",
        "",
        "The source is the cached 18-ETF common-history artifact; no external data was refreshed.",
        "Both families use the same final 24 complete monthly score dates and retain all prior rows for training.",
    ]
    titles = {"us_fixed_income": "U.S. Fixed Income", "international_fixed_income": "International Fixed Income"}
    for audit in audits:
        manifest = audit["manifest"]
        lines.extend(
            [
                "",
                f"## {titles[str(audit['family'])]}",
                "",
                f"Universe: `{', '.join(audit['tickers'])}`.",
                f"Complete history: `{manifest['complete_date_min']}` through `{manifest['complete_date_max']}` "
                f"(`{manifest['complete_date_count']}` months).",
                f"Scored panel: `{manifest['approved_scored_date_min']}` through "
                f"`{manifest['approved_scored_date_max']}` (24 months).",
                "",
                "### Coverage",
                "",
                audit["coverage"].to_markdown(),
                "",
                "### Target correlation",
                "",
                format_correlation_table(audit["target_correlation"]),
            ]
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "- Keep `BND, IEF, LQD, HYG` together despite BND redundancy; task-universe sensitivity is deferred.",
            "- Keep `BNDX, VWOB, HYEM` as the international family; its three-task IC is intrinsically coarse.",
            "- Keep `t_index` in every configuration.",
            "- Exclude REIT and commodity ETF families from this program. Commodity variables remain eligible inputs.",
        ]
    )
    (EXPERIMENT_DIR / "FIXED_INCOME_DATA_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Build both family artifacts and their shared audit report."""
    if not SOURCE.exists():
        raise FileNotFoundError(f"Cached source artifact is missing: {SOURCE}")
    source = pd.read_parquet(SOURCE)
    audits = [write_family_artifact(source, family, tickers) for family, tickers in FAMILIES.items()]
    write_audit(audits)
    print(json.dumps({audit["family"]: audit["manifest"] for audit in audits}, indent=2, default=str))


if __name__ == "__main__":
    main()
