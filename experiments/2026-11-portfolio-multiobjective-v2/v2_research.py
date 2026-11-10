"""Development-only adaptive menus and primary return/drawdown frontier evidence.

This orchestration reads previously scored development41 policies, writes the
next menu before scoring, and never inspects locked confirmation portfolios.
Returns/drawdowns are decimals. No GP fitting, trading, or production mutation.
Usage: uv run --locked --no-sync python <this_file> --round2 --workers 14
"""

from __future__ import annotations

import argparse

import numpy as np
from v2_common import COMPONENTS, HERE, V1, read, save
from v2_driver import controls, distance, register_round, run_cases
from v2_evaluate import policy


def development_records() -> list[dict]:
    """Read complete development outcomes only.

    Returns:
        Registered policy records, including explicit ineligible infeasibilities.
    """
    return [read(path) for path in sorted((HERE / "outputs/development").glob("*.json"))]


def frontier(rows: list[dict]) -> list[dict]:
    """Find nondominated CAGR/drawdown points without candidate normalization.

    Args:
        rows: Complete, same-panel outcome records.

    Returns:
        All nondominated records, retaining tied policy identities.
    """
    complete = [row for row in rows if row["complete"]]
    values = np.array([[row["summary"]["cagr"], row["summary"]["max_drawdown"]] for row in complete])
    return [row for i, row in enumerate(complete) if not np.any(
        np.all(values >= values[i] - 1e-12, axis=1) & np.any(values > values[i] + 1e-12, axis=1))]


def hypervolume(rows: list[dict]) -> float:
    """Compute two-objective dominated area using immutable reference anchors.

    Args:
        rows: Same-panel complete outcome records.

    Returns:
        Nonnegative area in squared return units, no dynamic scaling.
    """
    anchors = read(HERE / "ANCHORS.json")
    x0, y0 = anchors["hypervolume_reference_CAGR"], -anchors["hypervolume_reference_drawdown_loss"]
    points = sorted({(row["summary"]["cagr"], row["summary"]["max_drawdown"])
                     for row in frontier(rows) if row["summary"]["cagr"] > x0
                     and row["summary"]["max_drawdown"] > y0})
    area, prior = 0., x0
    for i, (x, _) in enumerate(points):
        height = max(y for _, y in points[i:]) - y0
        area += max(0., x - prior) * height
        prior = x
    return float(area)


def review(round_number: int) -> dict:
    """Save immutable development-only interpretation and frontier summary.

    Args:
        round_number: Just-completed registered round.

    Returns:
        Summary showing new evidence and next hypotheses without confirmation.
    """
    path = HERE / "rounds" / f"REVIEW{round_number:02d}.json"
    if path.exists():
        return read(path)
    rows = development_records()
    complete = [row for row in rows if row["complete"]]
    front = frontier(complete)
    leader = min(complete, key=lambda r: (distance(r["summary"]), r["id"]))
    result = {"round": round_number, "scored_total": len(rows), "complete": len(complete),
        "infeasible": len(rows) - len(complete), "frontier_ids": [r["id"] for r in front],
        "hypervolume": hypervolume(complete), "best_distance": distance(leader["summary"]),
        "balanced_leader": {"id": leader["id"], "policy": leader["policy"], "summary": leader["summary"]},
        "covariance_constraint_active_cases": sum(row["risk_binding_fraction"] > 0 for row in complete),
        "latest12_scored": False}
    save(path, result)
    return result


def _structural(row: dict) -> tuple:
    case = row["policy"]
    return (case["minimum_effective_assets"], case["risk_multiplier"],
            tuple(case["covariance_fractions"]), case["equity_cap"])


def round2() -> list[dict]:
    """Jointly retune static family blends for matched compact ranks1 and2.

    Returns:
        At most256 preregistered recipes, all from development evidence.
    """
    rows = [row for row in development_records() if row["complete"]]
    front = frontier(rows)
    balanced = sorted(front, key=lambda r: (distance(r["summary"]), r["id"]))
    grow = max(front, key=lambda r: (r["summary"]["cagr"], r["id"]))
    defensive = max(front, key=lambda r: (r["summary"]["max_drawdown"], r["id"]))
    structures = {}
    for row in [balanced[0], grow, defensive, *rows]:
        if row in [balanced[0], grow, defensive] or row["policy"]["control"]:
            structures.setdefault(_structural(row), row["policy"])
    for row in balanced:
        if len(structures) >= 4:
            break
        structures.setdefault(_structural(row), row["policy"])
    structures = list(structures.values())[:4]
    base = np.asarray(controls()[0]["prediction_fractions"])
    menu = [np.asarray(case["prediction_fractions"]) for case in controls()]
    old_menu = read(V1 / "PREREGISTRATION.json")["candidate_menu"]
    menu.extend(np.asarray(case["fractions"]) for case in old_menu if case["label"].startswith("prior48"))
    menu.extend([np.full((4, 3), 1 / 3), np.tile([.5, .5, 0], (4, 1))])
    for family in range(4):
        for expert in range(3):
            recipe = base.copy()
            recipe[family] = np.eye(3)[expert]
            menu.append(recipe)
    generator = np.random.default_rng(20261102)
    while len(menu) < 32:
        center = base if len(menu) % 2 else np.full((4, 3), 1 / 3)
        menu.append(np.stack([generator.dirichlet(row * 12 + .5) for row in center]))
    cases = []
    for structure in structures:
        for number, fractions in enumerate(menu):
            for compact in ["compact_r2_i15_exp", "compact_r1_i15_exp"]:
                prediction = [COMPONENTS[0], compact, COMPONENTS[2]]
                cases.append(policy(f"R2_jointblend_{number}_{compact}", fractions,
                    nea=structure["minimum_effective_assets"], risk=structure["risk_multiplier"],
                    covariance=structure["covariance_fractions"], equity_cap=structure["equity_cap"],
                    prediction_experts=prediction, covariance_experts=COMPONENTS))
    return register_round(2, cases, "Round1 found active risk ceilings at low NE; compare four "
        "development-selected structural settings. Retune all family3wayprediction simplexes jointly "
        "for compactrank1 andrank2 under MATCHED fixedreferencecovariance; rankvariant predictionsonly. "
        "32coarse deterministic jointrecipes perstructure/rank, no latest12 scoring.")


def round3() -> list[dict]:
    """Refine joint blends and active covariance regions identified in round2.

    Returns:
        Registered local and covariance challengers, at most256 complete tuples.
    """
    rows = [row for row in development_records() if row["complete"]]
    front = frontier(rows)
    balanced = sorted(front, key=lambda r: (distance(r["summary"]), r["id"]))
    ranked = sorted(rows, key=lambda r: (distance(r["summary"]), r["id"]))
    rank1 = next(row for row in ranked if row["policy"]["prediction_experts"][1] == "compact_r1_i15_exp")
    bases = []
    for row in [balanced[0], rank1, max(front, key=lambda r: r["summary"]["cagr"]), *balanced, *ranked]:
        if not any(np.array_equal(row["policy"]["prediction_fractions"], prior["policy"]["prediction_fractions"])
                   and _structural(row) == _structural(prior)
                   and row["policy"]["prediction_experts"] == prior["policy"]["prediction_experts"] for prior in bases):
            bases.append(row)
        if len(bases) == 4:
            break
    cases = []
    generator = np.random.default_rng(20261103)
    for number, base in enumerate(bases):
        old = base["policy"]
        fractions = np.asarray(old["prediction_fractions"])
        for trial in range(32):
            if trial < 24:
                family, edge = divmod(trial, 6)
                source, dest = [(a, b) for a in range(3) for b in range(3) if a != b][edge]
                changed = fractions.copy()
                amount = min(.125, changed[family, source])
                changed[family, source] -= amount
                changed[family, dest] += amount
            else:
                changed = np.stack([generator.dirichlet(row * 64 + .25) for row in fractions])
            cases.append(policy(f"R3_local{number}_{trial}", changed,
                nea=old["minimum_effective_assets"], risk=old["risk_multiplier"],
                covariance=old["covariance_fractions"], equity_cap=old["equity_cap"],
                prediction_experts=old["prediction_experts"], covariance_experts=old["covariance_experts"]))
        for cov in [[.75, .25, 0, 0], [.25, .75, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0],
                    [.375, .375, .25, 0], [.375, .375, 0, .25], [.25, .25, 0, .5], [0, 0, 0, 1]]:
            for risk in [.5, .75, 1.]:
                cases.append(policy(f"R3_cov{number}_{cov}_{risk}", fractions,
                    nea=old["minimum_effective_assets"], risk=risk, covariance=cov,
                    equity_cap=old["equity_cap"], prediction_experts=old["prediction_experts"],
                    covariance_experts=old["covariance_experts"]))
        if old["prediction_experts"][1] == "compact_r1_i15_exp":
            for cov in [[.5, .5, 0, 0], [.25, .75, 0, 0], [0, 1, 0, 0]]:
                for risk in [.5, .75, 1.]:
                    cases.append(policy(f"R3_r1_prediction_and_cov{number}_{cov}_{risk}", fractions,
                        nea=old["minimum_effective_assets"], risk=risk, covariance=cov,
                        equity_cap=old["equity_cap"], prediction_experts=old["prediction_experts"],
                        covariance_experts=old["prediction_experts"]))
    return register_round(3, cases, "Round2 development evidence guides local12.5pp familycoefficient "
        "transfers plusjointDirichlet refinements and active-constraint covariance mixtures. Explicitly "
        "separate compactrank1 prediction-only and prediction+covariance proposals; fixeddates/seeds. "
        "No kernel/prior change justified by low amplitude alone or unoptimized15iteration telemetry.")


def round4() -> list[dict]:
    """Close matched rank controls and test promising risk/exposure neighborhoods.

    Returns:
        Final bounded development menu, frozen before its scores are computed.
    """
    rows = [row for row in development_records() if row["complete"]]
    existing = {row["id"] for row in rows}
    candidates = []
    # Complete exact rank2 controls for any rank1-specific local/covariance
    # refinements. No rank attribution is made from an unmatched comparison.
    for row in rows:
        old = row["policy"]
        if "compact_r1_i15_exp" not in old["prediction_experts"] + old["covariance_experts"]:
            continue
        prediction = [name.replace("compact_r1_i15_exp", "compact_r2_i15_exp")
                      for name in old["prediction_experts"]]
        covariance_names = [name.replace("compact_r1_i15_exp", "compact_r2_i15_exp")
                            for name in old["covariance_experts"]]
        paired = policy("R4_exact_rank2_control_for_" + row["id"], np.array(old["prediction_fractions"]),
            nea=old["minimum_effective_assets"], risk=old["risk_multiplier"],
            covariance=old["covariance_fractions"], equity_cap=old["equity_cap"],
            prediction_experts=prediction, covariance_experts=covariance_names)
        if paired["id"] not in existing:
            candidates.append(paired)
    front = frontier(rows)
    ranked = sorted(rows, key=lambda r: (distance(r["summary"]), r["id"]))
    rank1 = next(row for row in ranked if row["policy"]["prediction_experts"][1] == "compact_r1_i15_exp")
    bases = []
    proposed = [min(front, key=lambda r: (distance(r["summary"]), r["id"])),
                max(front, key=lambda r: r["summary"]["cagr"]), rank1,
                *sorted(front, key=lambda r: (distance(r["summary"]), r["id"]))]
    for row in proposed:
        if row["id"] not in {base["id"] for base in bases}:
            bases.append(row)
        if len(bases) == 4:
            break
    generator = np.random.default_rng(20261104)
    for number, base in enumerate(bases):
        old = base["policy"]
        fractions = np.asarray(old["prediction_fractions"])
        for nea in [3, 4]:
            for risk in [.625, .75, .875, 1.]:
                for cap in [.8, .9]:
                    candidates.append(policy(f"R4_risk_exposure{number}_{nea}_{risk}_{cap}", fractions,
                        nea=nea, risk=risk, covariance=old["covariance_fractions"], equity_cap=cap,
                        prediction_experts=old["prediction_experts"], covariance_experts=old["covariance_experts"]))
        for trial in range(16):
            changed = np.stack([generator.dirichlet(row * 128 + .125) for row in fractions])
            for compact in ["compact_r2_i15_exp", "compact_r1_i15_exp"]:
                candidates.append(policy(f"R4_matched_jointfine{number}_{trial}_{compact}", changed,
                    nea=old["minimum_effective_assets"], risk=old["risk_multiplier"],
                    covariance=old["covariance_fractions"], equity_cap=old["equity_cap"],
                    prediction_experts=[COMPONENTS[0], compact, COMPONENTS[2]],
                    covariance_experts=COMPONENTS))
    # Prioritize exact missing matched controls, then the registered parameter
    # neighborhood. Deduplication never adds an unregistered hidden product.
    unique = {}
    for case in candidates:
        if case["id"] not in existing:
            unique.setdefault(case["id"], case)
    return register_round(4, list(unique.values())[:256], "Close exact rank2 companions for allrank1 "
        "refinements before interpretation; test NE3/4 active-riskmultiplier neighborhoods andstatic "
        "80/90% equityceilings. Finejointmean neighborhoods remain matched between compactranks1/2. "
        "Low broad forecast amplitude andmaxiterstop do not justify a new kernel/prior mechanism; "
        "no additional GPspecification is admitted. Last adaptive round, latest12stilllocked.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round2", action="store_true")
    parser.add_argument("--round3", action="store_true")
    parser.add_argument("--round4", action="store_true")
    parser.add_argument("--review", type=int)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    if args.review:
        result = review(args.review)
        print({key: result[key] for key in ["round", "complete", "hypervolume", "best_distance"]})
    for number, enabled, factory in [(2, args.round2, round2), (3, args.round3, round3),
                                     (4, args.round4, round4)]:
        if enabled:
            cases = factory()
            result = run_cases(cases, workers=args.workers)
            print(f"Round{number} complete {sum(row['complete'] for row in result)}/{len(result)}")
            review(number)
