"""Run a broad U.S. equity ETF kernel/configuration search.

This experiment script intentionally searches a wide candidate universe rather
than hand-curated kernel variants. It first runs a cheap scout pass, then
promotes the strongest candidates to the standard 12-window comparison.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
WRAPPER = SCRIPT_DIR / "run_first_round_walkforward.py"
RUNS_DIR = SCRIPT_DIR / "runs"
INCUMBENT_RUN = RUNS_DIR / "20260706T0018Z_macro_matern_linear_no_rq_12w_gpmaxret"
OUTBOX = Path("/Users/et/.openclaw/agents/bayes-ml-engineer/workspace/outbox")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    role: str
    feature_set: str
    covar_grouping_proposal: str
    macro_component_proposal: str
    gp_experiment: str
    time_modulation_mode: str
    maxiter: int
    include_three_way_interaction: bool = False
    exclude_pairwise_interactions: bool = False
    interactions_only: bool = False
    drop_etf_macro_interaction: bool = False
    exclude_time_index: bool = False
    outputscale_floor: float | None = None


def candidate_command(
    candidate: Candidate,
    *,
    output_root: Path,
    run_id: str,
    max_windows: int,
    posterior_scenarios: int,
) -> list[str]:
    command = [
        sys.executable,
        str(WRAPPER),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--max-windows",
        str(max_windows),
        "--maxiter",
        str(candidate.maxiter),
        "--posterior-scenarios",
        str(posterior_scenarios),
        "--gp-experiment",
        candidate.gp_experiment,
        "--time-modulation-mode",
        candidate.time_modulation_mode,
        "--upperlng",
        "0.35",
        "--nea",
        "3",
        "--gp-riskfolio-rm",
        "CVaR",
        "--gp-riskfolio-obj",
        "MaxRet",
        "--feature-set",
        candidate.feature_set,
        "--covar-grouping-proposal",
        candidate.covar_grouping_proposal,
        "--macro-component-proposal",
        candidate.macro_component_proposal,
    ]
    if candidate.include_three_way_interaction:
        command.append("--include-three-way-interaction")
    if candidate.exclude_pairwise_interactions:
        command.append("--exclude-pairwise-interactions")
    if candidate.interactions_only:
        command.append("--interactions-only")
    if candidate.drop_etf_macro_interaction:
        command.append("--drop-etf-macro-interaction")
    if candidate.exclude_time_index:
        command.append("--exclude-time-index")
    if candidate.outputscale_floor is not None:
        command.extend(
            [
                "--outputscale-floor",
                f"{candidate.outputscale_floor:g}",
                "--outputscale-prior-median",
                "0.1",
                "--outputscale-prior-sigma",
                "1.0",
            ]
        )
    return command


def stable_candidates(
    limit: int,
    *,
    fixed_feature_set: str | None = None,
    etf_unsplit_macro_only: bool = False,
) -> list[Candidate]:
    feature_sets = [
        "momentum_plus_macro",
        "momentum_plus_macro_market",
        "momentum_plus_macro_credit_inflation",
        "target_plus_macro",
        "macro_risk",
        "momentum_plus_macro_plus_volatility",
        "momentum_plus_macro_plus_liquidity",
        "momentum_plus_macro_plus_liquidity_volatility",
        "compact_pruned",
        "target_plus_momentum",
        "momentum_plus_volatility",
    ]
    if fixed_feature_set is not None:
        feature_sets = [fixed_feature_set]
    covar_groupings = [
        "default",
        "split_groups",
        "single_matern",
        "time_plus_matern_interaction",
        "split_macro_structured_pairwise",
        "split_macro_market_nonlinear_credit_linear",
        "split_macro_credit_shrunk",
        "split_macro_no_etf_credit",
        "split_macro_cross_family_gate",
        "macro_split_market_rq_credit_linear",
        "macro_split_market_rq_linear_credit_matern15",
        "macro_split_market_matern_rq_credit_linear",
        "macro_split_market_matern_credit_rbf_linear",
        "macro_split_market_rbf_credit_matern25_linear",
        "macro_split_market_linear_credit_matern25",
        "macro_split_market_rq_credit_rbf",
        "macro_split_market_matern_credit_rq_linear",
    ]
    if etf_unsplit_macro_only:
        covar_groupings = [
            grouping
            for grouping in covar_groupings
            if grouping == "default" or grouping.startswith(("split_macro_", "macro_split_"))
        ]
    macro_components = ["default", "matern_linear", "matern_only", "linear_only"]
    gp_experiments = ["signed_lkj_eta_2", "signed_no_prior", "positive_no_prior"]
    time_modes = ["neither", "lengthscale_only", "outputscale_only"]
    maxiters = [8, 15, 25]
    flag_patterns = [
        {},
        {"include_three_way_interaction": True},
        {"exclude_pairwise_interactions": True},
        {"drop_etf_macro_interaction": True},
        {"interactions_only": True},
        {"exclude_time_index": True},
        {"include_three_way_interaction": True, "drop_etf_macro_interaction": True},
        {"exclude_pairwise_interactions": True, "include_three_way_interaction": True},
    ]

    candidates: list[Candidate] = []
    seen_configs: set[tuple[Any, ...]] = set()

    def config_key(candidate: Candidate) -> tuple[Any, ...]:
        values = asdict(candidate)
        values.pop("candidate_id")
        values.pop("role")
        if values["covar_grouping_proposal"] != "default":
            values["macro_component_proposal"] = "__ignored__"
            values["drop_etf_macro_interaction"] = False
            values["include_three_way_interaction"] = False
        return tuple(values.items())

    def append_candidate(candidate: Candidate) -> None:
        key = config_key(candidate)
        if key in seen_configs:
            return
        candidates.append(candidate)
        seen_configs.add(key)

    # Exploit around the current incumbent, but with broad perturbations.
    exploit_feature_sets = (
        feature_sets
        if fixed_feature_set is not None
        else ["momentum_plus_macro", "momentum_plus_macro_plus_volatility", "momentum_plus_macro_plus_liquidity"]
    )
    exploit_specs = itertools.product(
        exploit_feature_sets,
        covar_groupings,
        ["matern_linear", "default"],
        ["signed_lkj_eta_2"],
        ["neither"],
        [8, 15, 25],
        flag_patterns[:5],
    )
    for i, (feature_set, grouping, macro, gp_exp, time_mode, maxiter, flags) in enumerate(exploit_specs):
        append_candidate(
            Candidate(
                candidate_id=f"c{i:03d}",
                role="exploit",
                feature_set=feature_set,
                covar_grouping_proposal=grouping,
                macro_component_proposal=macro,
                gp_experiment=gp_exp,
                time_modulation_mode=time_mode,
                maxiter=maxiter,
                **flags,
            )
        )
        if len(candidates) >= 35:
            break

    # Space-filling grid over the wider knobs.
    all_specs = list(itertools.product(feature_sets, covar_groupings, macro_components, gp_experiments, time_modes))
    step = max(1, len(all_specs) // 45)
    for j, spec_index in enumerate(range(0, len(all_specs), step)):
        if len(candidates) >= 80:
            break
        feature_set, grouping, macro, gp_exp, time_mode = all_specs[spec_index]
        flags = flag_patterns[(j * 3 + 1) % len(flag_patterns)]
        maxiter = maxiters[(j * 5 + 2) % len(maxiters)]
        append_candidate(
            Candidate(
                candidate_id=f"c{len(candidates):03d}",
                role="space_filling",
                feature_set=feature_set,
                covar_grouping_proposal=grouping,
                macro_component_proposal=macro,
                gp_experiment=gp_exp,
                time_modulation_mode=time_mode,
                maxiter=maxiter,
                **flags,
            )
        )

    # Deliberately novel and sensitivity candidates.
    novelty_specs = [
        (
            "compact_pruned",
            "split_groups",
            "default",
            "signed_lkj_eta_2_outputscale_floor",
            "neither",
            15,
            {"outputscale_floor": 1e-4},
        ),
        (
            "compact_pruned",
            "single_matern",
            "default",
            "signed_lkj_eta_2_outputscale_floor",
            "neither",
            15,
            {"outputscale_floor": 1e-3},
        ),
        (
            "momentum_plus_macro_plus_liquidity_volatility",
            "split_groups",
            "matern_linear",
            "signed_lkj_eta_2",
            "lengthscale_only",
            25,
            {},
        ),
        (
            "momentum_plus_macro_plus_liquidity_volatility",
            "time_plus_matern_interaction",
            "matern_linear",
            "signed_lkj_eta_2",
            "outputscale_only",
            25,
            {},
        ),
        (
            "target_plus_macro",
            "default",
            "matern_linear",
            "signed_lkj_eta_2",
            "neither",
            15,
            {"include_three_way_interaction": True},
        ),
        ("macro_risk", "default", "matern_linear", "signed_lkj_eta_2", "neither", 15, {}),
        ("momentum_plus_macro_credit_inflation", "default", "default", "signed_lkj_eta_2", "lengthscale_only", 15, {}),
        (
            "momentum_plus_macro_market",
            "default",
            "matern_linear",
            "positive_no_prior",
            "neither",
            25,
            {"include_three_way_interaction": True},
        ),
        (
            "momentum_plus_macro_plus_volatility",
            "split_macro_cross_family_gate",
            "default",
            "signed_no_prior",
            "neither",
            15,
            {},
        ),
        (
            "momentum_plus_macro_plus_liquidity",
            "split_macro_structured_pairwise",
            "linear_only",
            "positive_no_prior",
            "outputscale_only",
            8,
            {},
        ),
    ]
    if fixed_feature_set is not None:
        novelty_specs = [
            (fixed_feature_set, grouping, macro, gp_exp, time_mode, maxiter, flags)
            for _, grouping, macro, gp_exp, time_mode, maxiter, flags in novelty_specs
        ]
    if etf_unsplit_macro_only:
        novelty_specs = [
            spec
            for spec in novelty_specs
            if spec[1] == "default" or spec[1].startswith(("split_macro_", "macro_split_"))
        ]
    for feature_set, grouping, macro, gp_exp, time_mode, maxiter, flags in novelty_specs:
        append_candidate(
            Candidate(
                candidate_id=f"c{len(candidates):03d}",
                role="novel_or_sensitivity",
                feature_set=feature_set,
                covar_grouping_proposal=grouping,
                macro_component_proposal=macro,
                gp_experiment=gp_exp,
                time_modulation_mode=time_mode,
                maxiter=maxiter,
                **flags,
            )
        )

    # Fill any remaining slots with deterministic pseudo-random coverage.
    counter = 0
    while len(candidates) < limit:
        feature_set = feature_sets[(counter * 7 + 3) % len(feature_sets)]
        grouping = covar_groupings[(counter * 5 + 4) % len(covar_groupings)]
        macro = macro_components[(counter * 11 + 1) % len(macro_components)]
        gp_exp = gp_experiments[(counter * 13 + 2) % len(gp_experiments)]
        time_mode = time_modes[(counter * 17 + 1) % len(time_modes)]
        flags = flag_patterns[(counter * 19 + 2) % len(flag_patterns)]
        candidate = Candidate(
            candidate_id=f"c{len(candidates):03d}",
            role="space_filling",
            feature_set=feature_set,
            covar_grouping_proposal=grouping,
            macro_component_proposal=macro,
            gp_experiment=gp_exp,
            time_modulation_mode=time_mode,
            maxiter=maxiters[(counter * 23 + 1) % len(maxiters)],
            **flags,
        )
        if candidate.covar_grouping_proposal == "single_matern" and candidate.interactions_only:
            counter += 1
            continue
        append_candidate(candidate)
        counter += 1

    return candidates[:limit]


def read_metrics(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "portfolio_summary.csv"
    forecast_path = run_dir / "forecast_metrics_overall.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)
    gp = summary.loc[summary["strategy"].eq("gp_scenarios_riskfolio")].iloc[0]
    ew = summary.loc[summary["strategy"].eq("equal_weight")].iloc[0]
    row = {
        "cumulative_return": float(gp["cumulative_return"]),
        "return_vs_equal_weight": float(gp["cumulative_return"] - ew["cumulative_return"]),
        "sharpe": float(gp["sharpe"]) if not pd.isna(gp["sharpe"]) else math.nan,
        "max_drawdown": float(gp["max_drawdown"]),
        "avg_turnover": float(gp["avg_turnover"]),
        "mean_ic": float(gp["mean_ic"]) if not pd.isna(gp["mean_ic"]) else math.nan,
        "median_ic": float(gp["median_ic"]) if not pd.isna(gp["median_ic"]) else math.nan,
        "equal_weight_return": float(ew["cumulative_return"]),
    }
    if forecast_path.exists():
        forecast = pd.read_csv(forecast_path).iloc[0]
        row.update(
            {
                "rmse": float(forecast["rmse"]),
                "mean_gaussian_nll": float(forecast["mean_gaussian_nll"]),
                "mean_pred_std": float(forecast["mean_pred_std"]),
                "coverage_80": float(forecast["coverage_80"]),
            }
        )
    return row


def run_candidate(
    candidate: Candidate,
    *,
    output_root: Path,
    phase: str,
    max_windows: int,
    posterior_scenarios: int,
) -> dict[str, Any]:
    run_id = f"{phase}_{candidate.candidate_id}_{candidate.role}"
    run_dir = output_root / run_id
    command = candidate_command(
        candidate,
        output_root=output_root,
        run_id=run_id,
        max_windows=max_windows,
        posterior_scenarios=posterior_scenarios,
    )
    if (run_dir / "portfolio_summary.csv").exists():
        status = "cached"
        stdout_tail = ""
        stderr_tail = ""
    else:
        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        status = "complete" if result.returncode == 0 else "failed"
        stdout_tail = result.stdout[-4000:]
        stderr_tail = result.stderr[-4000:]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "agent_subprocess_stdout_tail.txt").write_text(stdout_tail)
        (run_dir / "agent_subprocess_stderr_tail.txt").write_text(stderr_tail)
    row = {
        **asdict(candidate),
        "phase": phase,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": status,
        "command": " ".join(command),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    if status in {"complete", "cached"}:
        row.update(read_metrics(run_dir))
    return row


def run_phase(
    candidates: list[Candidate],
    *,
    output_root: Path,
    phase: str,
    max_windows: int,
    posterior_scenarios: int,
    workers: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_candidate,
                candidate,
                output_root=output_root,
                phase=phase,
                max_windows=max_windows,
                posterior_scenarios=posterior_scenarios,
            )
            for candidate in candidates
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            print(f"{phase}: {len(rows)}/{len(candidates)} complete", flush=True)
    return pd.DataFrame(rows)


def plot_leaderboard(output_dir: Path, scout: pd.DataFrame, full: pd.DataFrame) -> None:
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    top = full.sort_values("cumulative_return", ascending=False).head(15)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].barh(top["candidate_id"], top["cumulative_return"] * 100, color="#4C78A8")
    axes[0].axvline(24.2315, color="#222222", linestyle="--", linewidth=1, label="Incumbent")
    axes[0].axvline(20.3470, color="#777777", linestyle=":", linewidth=1, label="Equal-weight")
    axes[0].set_title("Promoted Full Runs: Return (%)")
    axes[0].legend(fontsize=8)
    axes[1].scatter(scout["cumulative_return"], scout["mean_ic"], alpha=0.7, s=24)
    axes[1].axvline(0.20347, color="#777777", linestyle=":", linewidth=1)
    axes[1].axhline(0, color="#777777", linewidth=1)
    axes[1].set_title("Scout Universe")
    axes[1].set_xlabel("3-window cumulative return")
    axes[1].set_ylabel("Mean IC")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "broad_kernel_search_leaderboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if INCUMBENT_RUN.exists():
        fig, ax = plt.subplots(figsize=(10, 6))
        inc_returns = pd.read_csv(INCUMBENT_RUN / "portfolio_returns.csv")
        inc = inc_returns.loc[inc_returns["strategy"].eq("gp_scenarios_riskfolio")].copy()
        inc["date"] = pd.to_datetime(inc["date"])
        inc["equity"] = (1.0 + inc["return"]).cumprod()
        ax.plot(inc["date"], inc["equity"], label="Incumbent", linewidth=2.8, color="#222222")
        for _, row in top.head(8).iterrows():
            returns = pd.read_csv(Path(row["run_dir"]) / "portfolio_returns.csv")
            gp = returns.loc[returns["strategy"].eq("gp_scenarios_riskfolio")].copy()
            gp["date"] = pd.to_datetime(gp["date"])
            gp["equity"] = (1.0 + gp["return"]).cumprod()
            ax.plot(gp["date"], gp["equity"], label=row["candidate_id"], linewidth=1.3)
        ax.set_title("Top Promoted Equity Curves")
        ax.set_ylabel("Growth of $1")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncols=2)
        fig.tight_layout()
        fig.savefig(figures / "broad_kernel_search_equity_curves.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_report(output_dir: Path, candidates: list[Candidate], scout: pd.DataFrame, full: pd.DataFrame) -> None:
    scout_sorted = scout.sort_values("cumulative_return", ascending=False)
    full_sorted = full.sort_values("cumulative_return", ascending=False)
    (output_dir / "search_space.json").write_text(json.dumps([asdict(candidate) for candidate in candidates], indent=2))
    scout_sorted.to_csv(output_dir / "scout_leaderboard.csv", index=False)
    full_sorted.to_csv(output_dir / "promoted_full_leaderboard.csv", index=False)
    manifest = {
        "run_id": output_dir.name,
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "complete",
        "candidate_count": len(candidates),
        "scout_count": int(len(scout)),
        "promoted_count": int(len(full)),
        "incumbent_run": str(INCUMBENT_RUN),
        "scout_leaderboard": "scout_leaderboard.csv",
        "promoted_full_leaderboard": "promoted_full_leaderboard.csv",
        "plots": [
            "figures/broad_kernel_search_leaderboard.png",
            "figures/broad_kernel_search_equity_curves.png",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Broad Kernel Search",
        "",
        "Fixed evaluation target: U.S. equity ETF monthly GP Riskfolio ranking/portfolio behavior.",
        "",
        "Scout stage: 100 candidates, 3 scored windows, 64 posterior scenarios.",
        "",
        "Promoted stage: top scout candidates rerun for 12 scored windows, 256 posterior scenarios.",
        "",
        "## Top promoted full runs",
        "",
        full_sorted[
            [
                "candidate_id",
                "role",
                "feature_set",
                "covar_grouping_proposal",
                "macro_component_proposal",
                "gp_experiment",
                "time_modulation_mode",
                "maxiter",
                "cumulative_return",
                "mean_ic",
                "rmse",
                "mean_gaussian_nll",
            ]
        ]
        .head(15)
        .to_markdown(index=False, floatfmt=".4f"),
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="broad_kernel_search_20260706T0056Z")
    parser.add_argument("--candidate-count", type=int, default=100)
    parser.add_argument("--promote-count", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scout-windows", type=int, default=3)
    parser.add_argument("--full-windows", type=int, default=12)
    parser.add_argument("--fixed-feature-set", default=None)
    parser.add_argument("--etf-unsplit-macro-only", action="store_true")
    args = parser.parse_args()

    output_dir = RUNS_DIR / args.run_id
    scout_root = output_dir / "scout_runs"
    full_root = output_dir / "full_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    scout_root.mkdir(exist_ok=True)
    full_root.mkdir(exist_ok=True)

    candidates = stable_candidates(
        args.candidate_count,
        fixed_feature_set=args.fixed_feature_set,
        etf_unsplit_macro_only=args.etf_unsplit_macro_only,
    )
    scout = run_phase(
        candidates,
        output_root=scout_root,
        phase="scout",
        max_windows=args.scout_windows,
        posterior_scenarios=64,
        workers=args.workers,
    )
    complete_scout = scout[scout["status"].isin(["complete", "cached"])].copy()
    promoted_ids = (
        complete_scout.sort_values(["cumulative_return", "mean_ic"], ascending=[False, False])
        .head(args.promote_count)["candidate_id"]
        .tolist()
    )
    promoted = [candidate for candidate in candidates if candidate.candidate_id in set(promoted_ids)]
    full = run_phase(
        promoted,
        output_root=full_root,
        phase="full",
        max_windows=args.full_windows,
        posterior_scenarios=256,
        workers=max(1, min(args.workers, 4)),
    )
    plot_leaderboard(output_dir, complete_scout, full)
    write_report(output_dir, candidates, complete_scout, full)

    OUTBOX.mkdir(parents=True, exist_ok=True)
    for name in ["broad_kernel_search_leaderboard.png", "broad_kernel_search_equity_curves.png"]:
        source = output_dir / "figures" / name
        if source.exists():
            shutil.copy2(source, OUTBOX / name)
    print(f"Wrote broad search to {output_dir}")


if __name__ == "__main__":
    main()
