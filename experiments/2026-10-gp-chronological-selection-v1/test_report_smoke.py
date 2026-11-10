"""Offline report smoke test using sealed forecasts as a local regression fixture.

All generated evidence goes to pytest's temporary directory, never research
outputs. This tests reporting/allocator wiring, not new investment findings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import run_study as study
from report import create_report
from selection_math import recipe


def test_complete_report_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full reporting preserves baseline parity and seals all expected artifacts."""
    original = study.arrays(study.CAMPAIGN / "outputs/context.npz")
    contract = study.read(study.OUTPUT / "CONTRACT.json")
    names = list(contract["configs"])
    means = np.repeat(original["baseline_mean"][None], len(names), axis=0)
    for family in ["broad", "compact"]:
        values = study.arrays(study.CAMPAIGN / "outputs/components" / f"{family}_r2_i15_exp.npz")
        means[names.index(f"{family}_r2_i15_exp")] = values["mean"]
    extended = np.concatenate([means[:, :, :24], means], axis=2)
    covariance = np.concatenate([original["baseline_covariance"][:, :24],
                                 original["baseline_covariance"]], axis=1)
    context = {key: original[key] for key in ["assets", "seeds"]}
    for key in ["dates", "actual", "risk_free", "risk_budget"]:
        context[key] = np.concatenate([original[key][:24], original[key]])
    counts = np.zeros((4, len(names)), dtype=int)
    counts[:, names.index("compact_r2_i15_exp")] = [32, 0, 24, 16]
    counts[:, names.index("broad_r2_i15_exp")] = 64 - np.array([32, 0, 24, 16])
    incumbent = recipe(counts)
    monkeypatch.setattr(study, "OUTPUT", tmp_path)
    for name in ["CONTRACT.json", "FITS_COMPLETED.json", "FEATURE_AUDIT.json"]:
        study.save(tmp_path / name, {"fixture_only": True})
    study.save(tmp_path / "SELECTIONS_FROZEN.json", {"selection_sha256": {}})
    selections = []
    for fold in range(4):
        study.save(tmp_path / "selections" / f"fold_{fold}.json", {"fixture_fold": fold})
        selections.append({"fold": fold, "ledger": {"last_origin": "fixture-earlier-only"},
                           "test_origins": original["dates"][fold * 7:(fold + 1) * 7].tolist(),
                           "selected": {key: {"recipe": incumbent} for key in
                                        ["sharpe", "ranking", "compromise"]}})
    create_report({"configs": contract["configs"], "input_sha256": {}}, context,
                  extended, covariance, selections, incumbent)
    seal = study.read(tmp_path / "COMPLETED.json")
    assert seal["passed"]
    assert seal["complete_cases"] == 29
    study.verify_hashes(seal["artifact_sha256"])
    assert (tmp_path / "COMPARISON.png").stat().st_size > 1000
