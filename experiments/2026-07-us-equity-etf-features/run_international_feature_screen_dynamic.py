"""Run a dynamic international-equity feature screen.

This replaces the invalid 2026-07-07 international feature screen that used
`split_groups`, because that covariance path hardcoded the original macro
columns and ignored many custom international features.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
WRAPPER = SCRIPT_DIR / "run_first_round_walkforward.py"
RUNS_DIR = SCRIPT_DIR / "runs"
FEATURE_PATH = SCRIPT_DIR / "artifacts/international_equity_custom_feature_candidates_20260707T182133Z.parquet"
TICKERS = "VEA,VWO,VSS,EWX"

ETF_BASE = ["mom6m", "mom12m_skip1m", "cs_mom_rank", "trend_slope"]

MACRO_GROUPS: dict[str, list[str]] = {
    "market": ["vix", "vix_slope", "spy_ret", "pct_above_50dma"],
    "dollar_fx": ["dxy", "em_fx_ret"],
    "rates": ["tnote10y", "term_spread", "schp_ret", "y10_real_proxy", "breakeven_proxy"],
    "commodities": ["oil_ret", "copper_ret", "gold_crude_ratio"],
    "risk_vol": ["vix", "vix_slope", "vix_ts_chg_1p", "vix_ts_z_12p", "move_proxy", "dealer_gamma_proxy"],
    "breadth_flow": ["spy_ret", "erp", "rsp_spy_roc_1p", "pct_above_50dma", "spy_flow_z_12p"],
    "credit_infl": [
        "credit_spread_chg_1p",
        "hy_spread_chg_1p",
        "hy_spread_z_12p",
        "cpi_chg_12p",
        "cpi_chg_1p",
    ],
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    role: str
    etf_name: str
    etf_cols: list[str]
    macro_names: list[str]
    macro_cols: list[str]
    note: str


def dedupe(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for column in columns:
        if column in seen:
            continue
        deduped.append(column)
        seen.add(column)
    return deduped


def candidate_id(etf_name: str, macro_names: list[str]) -> str:
    raw = json.dumps({"etf": etf_name, "macro": macro_names}, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def build_candidates(limit: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen_feature_sets: set[tuple[str, ...]] = set()

    def append(role: str, macro_names: list[str], note: str) -> None:
        macro_cols = dedupe([column for name in macro_names for column in MACRO_GROUPS[name]])
        key = tuple([*ETF_BASE, *macro_cols])
        if key in seen_feature_sets:
            return
        seen_feature_sets.add(key)
        etf_name = "mom"
        candidates.append(
            Candidate(
                candidate_id=f"d{len(candidates):03d}_{candidate_id(etf_name, macro_names)}",
                role=role,
                etf_name=etf_name,
                etf_cols=list(ETF_BASE),
                macro_names=list(macro_names),
                macro_cols=macro_cols,
                note=note,
            )
        )

    append("control", ["market"], "dynamic-kernel transfer control: ETF momentum/trend plus market macro")

    for name in MACRO_GROUPS:
        if name == "market":
            continue
        append("single_family", [name], f"ETF momentum/trend plus {name}")
        if len(candidates) >= limit:
            return candidates[:limit]

    names = list(MACRO_GROUPS)
    for size in range(2, len(names) + 1):
        for combo in itertools.combinations(names, size):
            role = "group_combo"
            append(role, list(combo), f"ETF momentum/trend plus {'+'.join(combo)}")
            if len(candidates) >= limit:
                return candidates
    return candidates[:limit]


def candidate_command(
    candidate: Candidate,
    *,
    run_root: Path,
    max_windows: int,
    maxiter: int,
    posterior_scenarios: int,
    preflight_only: bool,
) -> list[str]:
    run_id = f"full_{candidate.candidate_id}"
    command = [
        sys.executable,
        str(WRAPPER),
        "--output-root",
        str(run_root / "full_runs"),
        "--run-id",
        run_id,
        "--feature-path",
        str(FEATURE_PATH),
        "--tickers",
        TICKERS,
        "--max-windows",
        str(max_windows),
        "--maxiter",
        str(maxiter),
        "--posterior-scenarios",
        str(posterior_scenarios),
        "--resume-scenarios",
        "--gp-experiment",
        "signed_lkj_eta_2",
        "--time-modulation-mode",
        "neither",
        "--upperlng",
        "0.35",
        "--nea",
        "3",
        "--gp-riskfolio-rm",
        "CVaR",
        "--gp-riskfolio-obj",
        "MaxRet",
        "--feature-set",
        "momentum_plus_macro_market",
        "--custom-etf-cols",
        ",".join(candidate.etf_cols),
        "--custom-macro-cols",
        ",".join(candidate.macro_cols),
        "--covar-grouping-proposal",
        "default",
        "--macro-component-proposal",
        "default",
    ]
    if preflight_only:
        command.append("--preflight-only")
    return command


def read_run_metrics(run_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {}
    portfolio_path = run_dir / "portfolio_summary.csv"
    if portfolio_path.exists():
        portfolio = pd.read_csv(portfolio_path)
        gp_row = portfolio.loc[portfolio["strategy"] == "gp_scenarios_riskfolio"].iloc[0]
        ewma_row = portfolio.loc[portfolio["strategy"] == "historical_y_ewma2_gerber1_riskfolio"].iloc[0]
        equal_row = portfolio.loc[portfolio["strategy"] == "equal_weight"].iloc[0]
        row.update(
            {
                "cumulative_return": float(gp_row["cumulative_return"]),
                "sharpe": float(gp_row["sharpe"]),
                "max_drawdown": float(gp_row["max_drawdown"]),
                "avg_turnover": float(gp_row["avg_turnover"]),
                "equal_weight_return": float(equal_row["cumulative_return"]),
                "ewma_return": float(ewma_row["cumulative_return"]),
            }
        )
    forecast_path = run_dir / "forecast_metrics_overall.csv"
    if forecast_path.exists():
        forecast = pd.read_csv(forecast_path).iloc[0]
        row.update(
            {
                "rmse": float(forecast["rmse"]),
                "mean_pred_std": float(forecast["mean_pred_std"]),
                "mean_gaussian_nll": float(forecast["mean_gaussian_nll"]),
                "coverage_80": float(forecast["coverage_80"]),
            }
        )
    ic_path = run_dir / "gp_window_ic.csv"
    if ic_path.exists():
        ic = pd.read_csv(ic_path)
        ic_column = "spearman_ic" if "spearman_ic" in ic.columns else "ic"
        row["mean_ic"] = float(ic[ic_column].mean())
        row["median_ic"] = float(ic[ic_column].median())
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        wrapper = manifest.get("us_equity_first_round_wrapper", {})
        row["manifest_input_columns"] = ",".join(wrapper.get("input_columns", []))
        row["active_feature_policy"] = "default_dynamic_blocks_include_custom_etf_and_macro_columns"
    return row


def write_manifest(run_root: Path, candidates: list[Candidate], args: argparse.Namespace) -> None:
    manifest = {
        "schema": "bayesfolio.international_dynamic_feature_screen.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "feature_path": str(FEATURE_PATH),
        "tickers": TICKERS.split(","),
        "run_root": str(run_root),
        "summary_csv": str(run_root / "summary_dynamic_feature_screen.csv"),
        "search_policy": (
            "100-candidate deterministic screen; ETF momentum/trend block is present in every candidate; "
            "macro features vary by international macro group combinations; covariance grouping is default "
            "so all selected custom ETF and macro columns are active in GP kernel blocks"
        ),
        "max_windows": args.max_windows,
        "maxiter": args.maxiter,
        "posterior_scenarios": args.posterior_scenarios,
        "covar_grouping_proposal": "default",
        "macro_component_proposal": "default",
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    (run_root / "candidate_plan.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run_screen(args: argparse.Namespace) -> Path:
    run_id = args.run_id or f"international_equity_dynamic_feature_screen_{datetime.now(UTC).strftime('%Y%m%dT%H%MZ')}"
    run_root = RUNS_DIR / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    candidates = build_candidates(args.limit)
    write_manifest(run_root, candidates, args)

    rows: list[dict[str, Any]] = []
    summary_path = run_root / "summary_dynamic_feature_screen.csv"
    for index, candidate in enumerate(candidates, start=1):
        command = candidate_command(
            candidate,
            run_root=run_root,
            max_windows=args.max_windows,
            maxiter=args.maxiter,
            posterior_scenarios=args.posterior_scenarios,
            preflight_only=args.preflight_only,
        )
        run_dir = run_root / "full_runs" / f"full_{candidate.candidate_id}"
        print(f"[{index}/{len(candidates)}] {candidate.candidate_id} {'+'.join(candidate.macro_names)}", flush=True)
        result = subprocess.run(command, check=False, text=True, capture_output=True)
        row: dict[str, Any] = {
            **asdict(candidate),
            "run_id": f"full_{candidate.candidate_id}",
            "run_dir": str(run_dir),
            "status": "complete" if result.returncode == 0 else "failed",
            "returncode": int(result.returncode),
            "command": " ".join(command),
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        if result.returncode == 0 and not args.preflight_only:
            row.update(read_run_metrics(run_dir))
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)
    write_summary(run_root, pd.DataFrame(rows))
    return run_root


def write_summary(run_root: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# International Equity Dynamic Feature Screen",
        "",
        f"Run root: `{run_root}`",
        "",
        (
            "Design: ETF momentum/trend is included in every candidate. "
            "Macro columns vary across international macro groups."
        ),
        "Covariance grouping: `default`, so custom ETF and macro columns are active in fitted GP kernel blocks.",
        "",
        (
            f"Status: {(summary['status'] == 'complete').sum()}/{len(summary)} complete, "
            f"{(summary['status'] == 'failed').sum()} failed."
        ),
        "",
    ]
    if "cumulative_return" in summary.columns and summary["cumulative_return"].notna().any():
        top = summary.sort_values(["cumulative_return", "mean_ic"], ascending=[False, False]).head(15)
        display_cols = [
            "candidate_id",
            "role",
            "macro_names",
            "cumulative_return",
            "sharpe",
            "mean_ic",
            "median_ic",
            "rmse",
            "coverage_80",
            "avg_turnover",
        ]
        lines.extend(
            [
                "## Top Candidates",
                "",
                "```text",
                top.loc[:, display_cols].to_string(index=False),
                "```",
                "",
            ]
        )
        control = summary.iloc[0]
        best = summary.sort_values(["cumulative_return", "mean_ic"], ascending=[False, False]).iloc[0]
        lines.extend(
            [
                "## Best Versus Dynamic Control",
                "",
                (
                    f"Control: `{control['candidate_id']}` return {control['cumulative_return']:.2%}, "
                    f"IC {control['mean_ic']:.3f}."
                ),
                f"Best: `{best['candidate_id']}` return {best['cumulative_return']:.2%}, IC {best['mean_ic']:.3f}.",
                f"Return delta: {best['cumulative_return'] - control['cumulative_return']:.2%}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            "- Candidate plan: `candidate_plan.json`",
            "- Summary CSV: `summary_dynamic_feature_screen.csv`",
        ]
    )
    (run_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=15)
    parser.add_argument("--posterior-scenarios", type=int, default=256)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = run_screen(args)
    print(f"Wrote dynamic screen to {run_root}", flush=True)


if __name__ == "__main__":
    main()
