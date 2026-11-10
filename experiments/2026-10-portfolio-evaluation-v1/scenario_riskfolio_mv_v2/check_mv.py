"""Preflight the no-CVaR comparator on the three previously infeasible solves.

Isolated research orchestration: reads the same saved GP moments and 5000 joint
scenarios, changes only Riskfolio rm from CVaR to MV, and checks whether maximum
Sharpe becomes feasible. No fitting, downloads, constraint relaxation or fallback
allocation. Outputs only a new preflight record; a failure is not a completed
backtest. Returns are decimals, covariance decimal squared and weights fractions.

Usage from repository root:
    uv run --locked --no-sync python <this_file>
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bayesfolio-scenario-riskfolio-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/bayesfolio-scenario-riskfolio-cache")
HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parent
PRIOR = FRAMEWORK / "scenario_riskfolio_v1/outputs"
sys.path.extend([str(FRAMEWORK), str(FRAMEWORK / "scenario_riskfolio_v1")])

import cvxpy as cp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import riskfolio as rp  # noqa: E402
from scenario_allocator import audit_weights, sample_joint  # noqa: E402

from artifacts import Reader  # noqa: E402


def _check() -> None:
    if rp.__version__ != "7.2.1":
        raise ValueError("The comparator requires the locked Riskfolio 7.2.1")
    reader = Reader()
    policy = reader.read(HERE / "policy.json")
    reader.read(Path(__file__))
    baseline = reader.read(FRAMEWORK / "CURRENT_BASELINES.json")
    reference = next(row for row in baseline["baselines"] if row["id"] == "sharpe_leader")
    current_path = FRAMEWORK / reference["evidence"]["npz"]
    current = reader.read(current_path)
    if reader.hashes[str(current_path.resolve())] != reference["evidence"]["npz_sha256"]:
        raise ValueError("Current GP identity changed")
    panel = reader.read(PRIOR / "input_panel.npz")
    old = reader.read(PRIOR / "infeasibility.json")
    np.testing.assert_array_equal(current["mean"], panel["mean"])
    np.testing.assert_array_equal(current["covariance"], panel["covariance"])
    if policy["CVaR"] or policy["risk_measure"] != "MV" or policy["objective"] != "Sharpe":
        raise ValueError("Only the user-requested CVaR removal is authorized")
    rows = []
    for failure in old["failed_allocations"]:
        seed, origin = failure["seed"], failure["forecast_origin"]
        seed_index = panel["seeds"].tolist().index(seed)
        month_index = panel["dates"].tolist().index(origin)
        source = PRIOR / "failures" / f"s{seed}_{origin}.npz"
        scenarios = reader.read(source)["scenarios"]
        if reader.hashes[str(source.resolve())] != failure["scenario_sha256"]:
            raise ValueError("Previously retained scenarios changed")
        mean, covariance = panel["mean"][seed_index, month_index], panel["covariance"][seed_index, month_index]
        np.testing.assert_array_equal(scenarios, sample_joint(mean, covariance, seed=seed, month=origin))
        portfolio = rp.Portfolio(
            returns=pd.DataFrame(scenarios, columns=panel["assets"].tolist()), nea=policy["minimum_effective_assets"]
        )
        portfolio.upperlng = policy["ETF_position_cap"]
        portfolio.lowerlng = 0.0
        portfolio.card = None
        portfolio.solvers = ["CLARABEL"]
        portfolio.sol_params = {
            "CLARABEL": {
                "tol_gap_abs": 1e-10,
                "tol_gap_rel": 1e-10,
                "tol_feas": 1e-10,
                "max_iter": 1000,
            }
        }
        portfolio.assets_stats(method_mu=policy["method_mu"], method_cov=policy["method_cov"])
        captured = []
        original_solve = cp.Problem.solve

        def _tracked_solve(problem: cp.Problem, *args, **kwargs):
            captured.append(problem)
            return original_solve(problem, *args, **kwargs)

        with warnings.catch_warnings(record=True) as caught, patch.object(cp.Problem, "solve", _tracked_solve):
            result = portfolio.optimization(
                model=policy["model"],
                rm=policy["risk_measure"],
                obj=policy["objective"],
                rf=policy["rf"],
                hist=policy["hist"],
            )
        if len(captured) != 1:
            raise ValueError("Expected exactly one unrelaxed solver call")
        # Independently certify the best feasible mean under the unchanged constraints.
        weights = cp.Variable(len(mean))
        certificate = cp.Problem(
            cp.Maximize(scenarios.mean(axis=0) @ weights),
            [
                weights >= 0,
                cp.sum(weights) == 1,
                weights <= policy["ETF_position_cap"],
                cp.sum_squares(weights) <= 1 / policy["minimum_effective_assets"],
            ],
        )
        certificate.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
        if certificate.status != cp.OPTIMAL or weights.value is None:
            raise ValueError("Feasible-mean certificate did not solve optimally")
        audit_weights(weights.value)
        row = {
            "seed": seed,
            "forecast_origin": origin,
            "risk_measure": policy["risk_measure"],
            "solver_status": captured[0].status,
            "returned_weights": result is not None,
            "maximum_feasible_scenario_mean": float(certificate.value),
            "largest_individual_scenario_mean": float(scenarios.mean(axis=0).max()),
            "warnings": [str(item.message) for item in caught],
        }
        if captured[0].status != cp.INFEASIBLE or result is not None or certificate.value >= 0:
            raise ValueError(f"Negative-feasible-mean diagnosis changed; inspect before continuation: {row}")
        rows.append(row)
    reader.verify()
    report = {
        "schema": policy["schema"],
        "status": "incomplete_requires_negative_forecast_policy",
        "CVaR_removed": True,
        "GP_refits": 0,
        "scenarios_reused_exactly": True,
        "allocation_checks": rows,
        "new_full_backtest_completed": False,
        "finding": "Removing CVaR does not remove the infeasible expected-return normalization "
        "in Riskfolio's Sharpe formulation. The same three month/seed allocations fail with MV.",
        "fallback_weights": False,
        "constraint_relaxation": False,
        "input_sha256": reader.hashes,
    }
    path = HERE / "outputs/preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("Refusing to overwrite previous preflight evidence")
    path.write_bytes(payload)
    print(json.dumps({key: value for key, value in report.items() if key != "input_sha256"}, indent=2))


if __name__ == "__main__":
    _check()
