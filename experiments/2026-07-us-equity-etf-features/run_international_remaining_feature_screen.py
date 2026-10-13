"""Screen remaining BayesFolio international-equity feature candidates.

This follow-up keeps U.S.-equity feature evidence out of the international
screen: features already tested in the U.S. equity campaign and dropped from
the current candidate are not re-tested here. It only varies international
macro columns that exist in the expanded BayesFolio artifact but were omitted
from the corrected dynamic international screen.
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
TRANSFER_CONTROL_MACRO = ["vix", "vix_slope", "spy_ret", "pct_above_50dma"]

REMAINING_MACRO_GROUPS: dict[str, list[str]] = {
    "vol_term_level": ["vix3m"],
    "fx_level": ["em_fx"],
    "breadth_level": ["rsp_spy"],
    "credit_level": ["credit_spread"],
    "commodity_levels": ["oil", "copper", "gold"],
}

EXCLUDED_FEATURES: dict[str, str] = {
    "tbill3m": "tested in the U.S. equity artifact and dropped from the current U.S. candidate",
    "hy_spread": "tested in the U.S. equity artifact; current candidate keeps transformed HY spread columns instead",
    "vix_ts_level": "perfect duplicate of tested vix_slope in the expanded international artifact",
    "skew_proxy": "near duplicate of tested vix_slope/vix_ts_level in the expanded international artifact",
    "y10_nominal": "near duplicate of tested tnote10y in the expanded international artifact",
    "de10y": "constant in the expanded international artifact",
    "jp10y": "constant in the expanded international artifact",
    "uk10y": "constant in the expanded international artifact",
    "cn10y": "constant in the expanded international artifact",
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    role: str
    etf_cols: list[str]
    macro_names: list[str]
    macro_cols: list[str]
    note: str


def dedupe(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for column in columns:
        if column in seen:
            continue
        out.append(column)
        seen.add(column)
    return out


def stable_id(role: str, macro_names: list[str], macro_cols: list[str]) -> str:
    raw = json.dumps({"role": role, "macro_names": macro_names, "macro_cols": macro_cols}, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]


def build_candidates(limit: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, ...]] = set()

    def append(role: str, macro_names: list[str], macro_cols: list[str], note: str) -> None:
        cols = dedupe(macro_cols)
        key = tuple([*ETF_BASE, *cols])
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            Candidate(
                candidate_id=f"r{len(candidates):03d}_{stable_id(role, macro_names, cols)}",
                role=role,
                etf_cols=list(ETF_BASE),
                macro_names=list(macro_names),
                macro_cols=cols,
                note=note,
            )
        )

    append(
        "control",
        ["transfer_market_control"],
        list(TRANSFER_CONTROL_MACRO),
        "ETF momentum/trend plus the prior dynamic market control",
    )
    for name, cols in REMAINING_MACRO_GROUPS.items():
        append("single_remaining_family", [name], cols, f"ETF momentum/trend plus omitted package group {name}")
        if len(candidates) >= limit:
            return candidates[:limit]

    names = list(REMAINING_MACRO_GROUPS)
    for size in range(2, len(names) + 1):
        for combo in itertools.combinations(names, size):
            cols = [column for name in combo for column in REMAINING_MACRO_GROUPS[name]]
            append("remaining_group_combo", list(combo), cols, f"ETF momentum/trend plus {'+'.join(combo)}")
            if len(candidates) >= limit:
                return candidates[:limit]
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
    command = [
        sys.executable,
        str(WRAPPER),
        "--output-root",
        str(run_root / "full_runs"),
        "--run-id",
        f"full_{candidate.candidate_id}",
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
    return row


def write_manifest(run_root: Path, candidates: list[Candidate], args: argparse.Namespace) -> None:
    manifest = {
        "schema": "bayesfolio.international_remaining_feature_screen.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "feature_path": str(FEATURE_PATH),
        "tickers": TICKERS.split(","),
        "run_root": str(run_root),
        "summary_csv": str(run_root / "summary_remaining_feature_screen.csv"),
        "search_policy": (
            "Screen only expanded BayesFolio international macro columns omitted from the corrected "
            "dynamic screen, after excluding U.S.-tested dropped features, duplicate proxies, and constants. "
            "ETF momentum/trend is present in every candidate and default dynamic covariance blocks keep "
            "custom columns active."
        ),
        "included_remaining_macro_groups": REMAINING_MACRO_GROUPS,
        "excluded_features": EXCLUDED_FEATURES,
        "transfer_control_macro": TRANSFER_CONTROL_MACRO,
        "max_windows": args.max_windows,
        "maxiter": args.maxiter,
        "posterior_scenarios": args.posterior_scenarios,
        "covar_grouping_proposal": "default",
        "macro_component_proposal": "default",
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    (run_root / "candidate_plan.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_summary(run_root: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# International Equity Remaining Feature Screen",
        "",
        f"Run root: `{run_root}`",
        "",
        "Design: ETF momentum/trend is included in every candidate. Macro columns vary only across BayesFolio",
        "international artifact columns omitted from the corrected dynamic screen after U.S.-dropped, duplicate,",
        "and constant features are excluded.",
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
        lines.extend(["## Top Candidates", "", "```text", top.loc[:, display_cols].to_string(index=False), "```", ""])
        control = summary.iloc[0]
        best = summary.sort_values(["cumulative_return", "mean_ic"], ascending=[False, False]).iloc[0]
        lines.extend(
            [
                "## Best Versus Transfer Control",
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
            "- Summary CSV: `summary_remaining_feature_screen.csv`",
        ]
    )
    (run_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def run_screen(args: argparse.Namespace) -> Path:
    run_id = (
        args.run_id or f"international_equity_remaining_feature_screen_{datetime.now(UTC).strftime('%Y%m%dT%H%MZ')}"
    )
    run_root = RUNS_DIR / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    candidates = build_candidates(args.limit)
    write_manifest(run_root, candidates, args)

    rows: list[dict[str, Any]] = []
    summary_path = run_root / "summary_remaining_feature_screen.csv"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--maxiter", type=int, default=15)
    parser.add_argument("--posterior-scenarios", type=int, default=256)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = run_screen(args)
    print(f"Wrote remaining-feature screen to {run_root}", flush=True)


if __name__ == "__main__":
    main()
