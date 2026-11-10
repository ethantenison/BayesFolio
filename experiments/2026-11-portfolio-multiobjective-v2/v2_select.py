"""Freeze development robustness, matched controls and final confirmation menu.

Isolated evidence orchestration; no fitting or trading. Uses only development41
outcomes until SHORTLIST_FROZEN exists. Returns/drawdowns are decimal fractions.
Usage: uv run --locked --no-sync python <this_file> --freeze
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import numpy as np
from v2_common import HERE, digest, read, save, status, verify_hashes
from v2_driver import controls, distance
from v2_evaluate import policy
from v2_protocol import prepare
from v2_research import development_records, frontier
from v2_robustness import assess


def rank2_counterpart(case: dict) -> dict:
    """Construct the exact rank2 counterpart without changing any other setting.

    Args:
        case: Complete policy that may use compact rank1 predictions/covariance.

    Returns:
        Content-addressed policy with compactrank1 replaced by compactrank2 only.
    """
    return policy("exact_rank2_counterpart", np.asarray(case["prediction_fractions"]),
        nea=case["minimum_effective_assets"], risk=case["risk_multiplier"],
        covariance=case["covariance_fractions"], equity_cap=case["equity_cap"],
        prediction_experts=[name.replace("compact_r1_i15_exp", "compact_r2_i15_exp")
                            for name in case["prediction_experts"]],
        covariance_experts=[name.replace("compact_r1_i15_exp", "compact_r2_i15_exp")
                            for name in case["covariance_experts"]])


def matched_coverage(rows: list[dict]) -> dict:
    """Record exact pair coverage and a conservative metadata-only restriction.

    Args:
        rows: Every preserved development policy including controls.

    Returns:
        Exact pair mapping. Unmatched exploratory leaves are not promotable.
    """
    ids = {row["id"] for row in rows}
    matches = {}
    for row in rows:
        case = row["policy"]
        if "compact_r1_i15_exp" in case["prediction_experts"] + case["covariance_experts"]:
            other = rank2_counterpart(case)["id"]
            matches[row["id"]] = {"rank2_id": other, "exact_counterpart_scored": other in ids}
    result = {"registered_before_confirmation": True, "rank1_policies": len(matches), "pairs": matches,
        "exploratory_nonpromotable_ids": [identity for identity, pair in matches.items()
                                         if not pair["exact_counterpart_scored"]],
        "interpretation": "Conservative post-registration metadata-only design restriction. Preserve all "
            "unclosed leaves, but do not promote or use them for rank attribution. Round4 promised full "
            "pair coverage; its16 unclosed exposure/covariance leaves are a documented coverage limitation, "
            "not erased or called complete rank experiments. Exact pairs are required for rank finalists."}
    save(HERE / "MATCHED_CONTROL_AUDIT.json", result)
    return result


def complexity(case: dict) -> tuple:
    """Use declared recipe structure for deterministic simplicity ties.

    Args:
        case: Complete static policy tuple.

    Returns:
        Ordered count of GP variants, nonzero blend terms and exposure constraints.
    """
    return (int("compact_r1_i15_exp" in case["prediction_experts"] + case["covariance_experts"]),
            int(np.count_nonzero(case["prediction_fractions"])) + int(np.count_nonzero(case["covariance_fractions"])),
            int(case["equity_cap"] is not None))


def development_key(row: dict, reference: dict, robustness: dict | None = None) -> tuple:
    """Apply frozen distance/stability/Sharpe/IC/simplicity ordering.

    Args:
        row: One complete development record.
        reference: Original fixed development-best actual control.
        robustness: Optional saved assessment providing annual stability count.

    Returns:
        Lower-is-better deterministic selection key.
    """
    count = (robustness["annual_blocks_within_tolerance"] if robustness else
             sum(distance(a) <= distance(b) + .25 for a, b in zip(
                 row["annual_development_blocks"], reference["annual_development_blocks"], strict=True)))
    return (round(distance(row["summary"]), 8), -count, -(row["summary"]["sharpe"] or -1e6),
            -row["summary"]["rank_ic"], complexity(row["policy"]), row["id"])


def freeze() -> dict:
    """Complete development checks and freeze no more than ten final policies.

    Returns:
        Immutable shortlist, hash and development gate evidence. The existence
        of this passed file is the sole authorization to score latest12 outcomes.
    """
    prepare()
    target = HERE / "SHORTLIST_FROZEN.json"
    if target.exists():
        if digest(target) != (HERE / "SHORTLIST_FROZEN.sha256").read_text().strip():
            raise ValueError("Frozen shortlist changed")
        return read(target)
    verify_hashes(read(HERE / "MATH_CONTRACT.json")["files_sha256"])
    rows = development_records()
    registry = {row["id"]: row for row in rows}
    reference = registry[read(HERE / "CONTROLS.json")["best_development_control"]]
    coverage = matched_coverage(rows)
    excluded = set(coverage["exploratory_nonpromotable_ids"])
    available = [row for row in rows if row["complete"] and row["id"] not in excluded]
    front = frontier(available)
    ordered = sorted(available, key=lambda r: development_key(r, reference))
    menu = {row["id"]: row for row in [*front, *ordered[:48]]}
    first_round_ids = {case["id"] for case in read(HERE / "rounds/ROUND01.json")["cases"]}
    coarse = min((row for row in available if row["id"] in first_round_ids),
                 key=lambda r: development_key(r, reference))
    rank1 = min((row for row in available if "compact_r1_i15_exp" in row["policy"]["prediction_experts"]),
                key=lambda r: development_key(r, reference))
    menu.update({coarse["id"]: coarse, rank1["id"]: rank1})
    if rank1["id"] in coverage["pairs"]:
        other = coverage["pairs"][rank1["id"]]["rank2_id"]
        menu[other] = registry[other]
    save(HERE / "ROBUSTNESS_MENU.json", {"ids": sorted(menu), "fixed_reference_id": reference["id"],
        "draws": 512, "seed": 20261101, "block_lengths": [3, 6],
        "source_sha256": digest(HERE / "v2_robustness.py"),
        "registered_before_checks_utc": datetime.now(UTC).isoformat(),
        "stability_tie": "Number of three annual development blocks within fixedcontrol distance+0.25, "
                         "the same predeclared stability property used by the annual gate"})
    robust = {}
    anchors = read(HERE / "ANCHORS.json")
    for identity in sorted(menu):
        path = HERE / "outputs/robustness" / f"{identity}.json"
        if path.exists():
            assessed = read(path)
        else:
            assessed = assess({**menu[identity], "feasibility_complete": True},
                              {**reference, "feasibility_complete": True}, anchors)
            save(path, assessed)
        robust[identity] = assessed
    eligible = [row for row in menu.values() if robust[row["id"]]["eligible"]]
    eligible_front = frontier(eligible)
    ranked = sorted(eligible_front, key=lambda r: development_key(r, reference, robust[r["id"]]))
    picks = [registry[case["id"]] for case in controls()]

    def add(row: dict) -> None:
        if row["id"] not in {case["id"] for case in picks} and len(picks) < 10:
            picks.append(row)

    for row in ranked[:3]:
        add(row)
    if front:
        add(max(front, key=lambda r: (r["summary"]["cagr"], r["id"])))
        add(max(front, key=lambda r: (r["summary"]["max_drawdown"], r["id"])))
    add(rank1)
    add(registry[coverage["pairs"][rank1["id"]]["rank2_id"]])
    add(coarse)
    for row in [*ranked, *sorted(front, key=lambda r: development_key(r, reference))]:
        add(row)
    identities = {row["id"] for row in picks}
    for row in picks:
        if row["id"] in coverage["pairs"] and coverage["pairs"][row["id"]]["rank2_id"] not in identities:
            raise ValueError("A shortlisted compactrank1 policy lacks its exact rank2 companion")
    result = {"passed": True, "frozen_at_utc": datetime.now(UTC).isoformat(),
        "ids": [row["id"] for row in picks], "cases": [row["policy"] for row in picks],
        "control_ids": [case["id"] for case in controls()], "fixed_development_control": reference["id"],
        "eligible_development_ids": [row["id"] for row in eligible],
        "eligible_development_frontier": [row["id"] for row in eligible_front],
        "development_gate_evidence": robust, "confirmation_used": False,
        "source_sha256": {str(path): digest(path) for path in [HERE / "SPLITS.json", HERE / "ANCHORS.json",
            HERE / "SELECTION_RULE.json", HERE / "MATH_CONTRACT.json", HERE / "MATCHED_CONTROL_AUDIT.json",
            HERE / "ROBUSTNESS_MENU.json", HERE / "v2_select.py", HERE / "v2_robustness.py"]},
        "all_registered_results_preserved": len(rows), "new_policy_count": len(rows) - 2,
        "design_coverage_limitation_count": len(excluded)}
    save(target, result)
    save(HERE / "SHORTLIST_FROZEN.sha256", (digest(target) + "\n").encode())
    status("shortlist_frozen", "Score only frozen finalists on latest12 once, then select without retuning",
           new_policy_count=len(rows) - 2, finalists=len(picks), confirmation_locked=False,
           fit_attempts_used=6, fit_attempts_remaining=1194, design_coverage_limitations=len(excluded))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", required=True)
    parser.parse_args()
    result = freeze()
    print({"finalists": len(result["ids"]), "eligible_development": len(result["eligible_development_ids"])})
