"""Matched cached pool-first portfolio clarification and frozen November choice.

Research I/O only, no GP fitting or fresh October outcomes. All candidates use
the same observed covariance pool, allocator, dates, and causal risk budget.
Returns are decimal, weights fractional and NAV dollars; evidence is retrospective.
Usage: uv run --locked --no-sync python <this_file>
"""

from __future__ import annotations

import numpy as np
from common import CHRONO, COMPONENTS, HERE, arrays, digest, read, save, snapshot, status, verify_hashes
from decision_math import BLOCKS, metrics, pool, promotion
from policy import allocate_path
from prepare_decision import prepare


def _score_path(context: dict, mean: np.ndarray, covariance: np.ndarray) -> tuple[dict, dict, list]:
    weights, numerical = allocate_path(mean, covariance, context["risk_budget"])
    actual, rf = context["actual"], context["risk_free"]
    overall, paths = metrics(actual, rf, mean, weights)
    earlier, _ = metrics(actual[:24], rf[:24], mean[:24], weights[:24])
    later, _ = metrics(actual[24:], rf[24:], mean[24:], weights[24:])
    blocks = []
    for start, end in BLOCKS:
        score, _ = metrics(actual[start:end], rf[start:end], mean[start:end], weights[start:end])
        blocks.append(score)
    excess = paths["excess_returns"]
    leave_one = []
    for month in range(52):
        retained = np.delete(excess, month)
        leave_one.append(float(np.sqrt(12) * retained.mean() / retained.std(ddof=1)))
    evidence = {"all52": overall, "earlier24": earlier, "later28": later, "blocks": blocks,
                "block_mean_excess": [float(excess[start:end].mean()) for start, end in BLOCKS],
                "leave_one_out_sharpe": leave_one}
    paths.update(mean=mean, covariance=covariance, weights=weights, raw_weights=weights.copy(),
                 actual=actual, risk_free=rf, dates=context["dates"], assets=context["assets"],
                 risk_budget=context["risk_budget"],
                 predicted_variance=np.einsum("ta,tab,tb->t", weights, covariance, weights))
    return evidence, paths, numerical


def _choose(records: list[dict]) -> tuple[dict, list[str]]:
    incumbent = next(row for row in records if row["recipe"]["label"] == "incumbent")
    eligible = [row for row in records if row["promotion"]["eligible"]]
    frontier = []
    for candidate in eligible:
        values = np.array([candidate["all52"][key] for key in ["sharpe", "total_return", "rank_ic"]])
        dominated = False
        for other in eligible:
            alternatives = np.array([other["all52"][key] for key in ["sharpe", "total_return", "rank_ic"]])
            if np.all(alternatives >= values) and np.any(alternatives > values + 1e-12):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    base = np.array(incumbent["recipe"]["fractions"])
    winner = min(frontier, key=lambda row: (
        float(np.abs(np.array(row["recipe"]["fractions"]) - base).sum()),
        -row["all52"]["sharpe"], -row["all52"]["rank_ic"], -row["all52"]["total_return"], row["id"]
    )) if frontier else incumbent
    return winner, [row["id"] for row in frontier]


def evaluate() -> dict:
    """Evaluate the preregistered batch and freeze the historical decision.

    Returns:
        Immutable decision independent of newly appended October returns.
    """
    plan = prepare()
    target = HERE / "HISTORICAL_DECISION.json"
    if target.exists():
        value = read(target)
        verify_hashes(value["artifact_sha256"])
        return value
    context = arrays(CHRONO / "context52.npz")
    components = [arrays(CHRONO / "components" / f"{name}.npz") for name in COMPONENTS]
    for panel in components:
        for axis in ["dates", "assets", "seeds"]:
            np.testing.assert_array_equal(panel[axis], context[axis])
    means = np.stack([item["mean"] for item in components])
    covariance = np.stack([item["covariance"] for item in components])
    records = []
    for index, case in enumerate(plan["candidate_menu"], 1):
        path = HERE / "outputs/historical" / f"{case['id']}.json"
        if path.exists():
            row = read(path)
            if row["recipe"] != case or digest(path.with_suffix(".npz")) != row["arrays_sha256"]:
                raise ValueError("Historical checkpoint differs")
        else:
            mean, cov = pool(means, covariance, np.array(case["fractions"]))
            scores, paths, numerical = _score_path(context, mean, cov)
            snapshot(path.with_suffix(".npz"), paths)
            row = {"id": case["id"], "recipe": case, **scores, "numerical": numerical,
                   "arrays_sha256": digest(path.with_suffix(".npz")),
                   "preregistration_sha256": digest(HERE / "PREREGISTRATION.json")}
            save(path, row)
        records.append(row)
        status("cached_clarifications", "Complete fixed batch and apply frozen promotion gates",
               completed_configurations=index, total_configurations=len(plan["candidate_menu"]),
               new_configurations=plan["batch1_new_recipes"], batches_used=1,
               fit_attempts_used=0, fit_attempts_remaining=128, failures=[])
    incumbent = next(row for row in records if row["recipe"]["label"] == "incumbent")
    for row in records:
        row["promotion"] = promotion(row, incumbent)
    winner, frontier = _choose(records)
    mandatory = [row["id"] for row in records if row["recipe"]["control"]]

    def slack(row: dict) -> float:
        a, b = row["all52"], incumbent["all52"]
        recent, ref = row["later28"], incumbent["later28"]
        return min((a["sharpe"] - b["sharpe"] - .05) / .05,
                   (a["rank_ic"] - b["rank_ic"] - .01) / .01,
                   (a["total_return"] - b["total_return"]) / .01,
                   (recent["sharpe"] - ref["sharpe"] + .05) / .05,
                   (recent["rank_ic"] - ref["rank_ic"] + .01) / .01,
                   (recent["total_return"] - ref["total_return"] + .01) / .01,
                   (a["max_drawdown"] - b["max_drawdown"] + .01) / .01,
                   float(np.min(np.array(a["family_ic"]) - b["family_ic"]) + .05) / .05,
                   (row["promotion"]["positive_blocks"] - 4) / 4,
                   row["promotion"]["minimum_leave_one_out_sharpe_advantage"] / .05)

    others = sorted([row for row in records if row["id"] not in mandatory],
                    key=lambda row: (-slack(row), -row["all52"]["sharpe"],
                                     -row["all52"]["rank_ic"], -row["all52"]["total_return"], row["id"]))
    shortlist = mandatory + [row["id"] for row in others[:10]]
    if winner["id"] not in shortlist:
        shortlist[-1] = winner["id"]
    save(HERE / "outputs/historical/ALL_RESULTS.json", records)
    result = {"passed": True, "winner": winner["recipe"], "incumbent_retained": winner["id"] == incumbent["id"],
              "eligible_candidates": [row["id"] for row in records if row["promotion"]["eligible"]],
              "eligible_pareto_ids": frontier, "shortlist": shortlist, "batches_used": 1,
              "configurations_scored": len(records), "new_configurations_scored": plan["batch1_new_recipes"],
              "batch2_used": False, "reason_for_stopping": "Predefined clarifications and promotion decision complete; "
              "do not add candidates merely because a challenger failed to win.",
              "October_outcomes_used_for_selection": False, "pristine_holdout": False,
              "preregistration_sha256": digest(HERE / "PREREGISTRATION.json"),
              "artifact_sha256": {str(path): digest(path) for path in sorted((HERE / "outputs/historical").glob("*"))}}
    save(target, result)
    status("historical_decision_frozen", "Validate October data; fit only September/October shared experts",
           selected_recipe=winner["recipe"]["label"], incumbent_retained=result["incumbent_retained"],
           shortlist_count=len(shortlist), batches_used=1, fit_attempts_used=0,
           fit_attempts_remaining=128, failures=[])
    return result


if __name__ == "__main__":
    evaluate()
