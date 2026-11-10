"""Integration check of training-only transforms and actual model-prior construction.

The audit constructs models but performs zero optimizer calls or new backtests.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-repaired-global-feature-selection/test_model_contract.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_actual_construction_and_adversarial_evaluation_isolation() -> None:
    """Every approved task universe shares the documented likelihood and causal transforms."""
    here = Path(__file__).resolve().parent
    result = subprocess.run([sys.executable, str(here / "audit_model_contract.py")],
                            cwd=here.parents[1], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {"passed": True, "constructed_models": 5, "optimizer_calls": 0}
    report = json.loads((here / "results/model_contract_audit.json").read_text())
    assert {row["tasks"] for row in report["constructed_unfitted_models"]} == {3, 4, 15}
    assert all(row["initial_training_months"] == 84 for row in report["constructed_unfitted_models"])
    assert all(row["completed_seed27_first_window_likelihood_and_transform_match"]
               for row in report["constructed_unfitted_models"])
