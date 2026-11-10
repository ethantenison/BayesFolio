"""Paths, sealed evidence readers and scope planning for approved research.

Usage: imported by run_study.py before numerical libraries. Only the new study
directory is writable; existing datasets, fit sources and completed studies are
read-only. Returns are discounted-excess decimals and covariance decimal squared.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parent
REPO = HERE.parents[2]
ROOT = HERE / "outputs"
FITS = FRAMEWORK / "full_rerun_v1/outputs"
SELECTION = FRAMEWORK / "selection_refresh_v1/outputs"
POLICY = FRAMEWORK / "portfolio_policy_comparison_v1/outputs"
LEGACY = REPO / "experiments/2026-10-repaired-global-feature-selection"
BASELINE = "global_e6c41f2020fd"
US_MASK = "us_equity_86c90676a157"
SEEDS = [27, 127, 227]
JOINT_REFERENCES = [
    BASELINE,
    "global_d8737126b4e7",
    "global_bb1b8f9500f2",
    "global_4a6348639308",
    "global_b48732adb186",
]
for _key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[_key] = "1"
sys.path.extend(
    [
        str(FRAMEWORK),
        str(FRAMEWORK / "full_rerun_v1"),
        str(FRAMEWORK / "selection_refresh_v1"),
        str(FRAMEWORK / "portfolio_policy_comparison_v1"),
    ]
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(hashes: dict) -> None:
    for path, expected in hashes.items():
        if _hash(Path(path)) != expected:
            raise ValueError(f"Changed protected evidence: {path}")


def _save(path: Path, value: object, immutable: bool = False) -> None:
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Attempt to write outside the new study outputs")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if immutable and path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Conflicting checkpoint: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def _context() -> dict:
    import numpy as np

    context = json.loads((FITS / "context.json").read_text())
    with np.load(FITS / "context_arrays.npz", allow_pickle=False) as values:
        for key in ["actual", "baseline_mean", "historical_covariance", "risk_budget", "risk_free"]:
            context[key] = values[key].copy()
        context["fixed_probes"] = {
            key.removeprefix("probe_"): values[key].copy() for key in values.files if key.startswith("probe_")
        }
    return context


def _command(config: dict, name: str, assets: list[str], global_feature: str) -> dict:
    registered = copy.deepcopy(config)
    registered.pop("old_run_root", None)
    registered["tickers"] = assets
    command = registered["command"]
    for flag, value in [
        ("--tickers", ",".join(assets)),
        ("--feature-path", global_feature),
        ("--output-root", str(ROOT / "runs")),
        ("--run-id", f"{name}_s27"),
        ("--upperlng", "1.0"),
    ]:
        command[command.index(flag) + 1] = value
    return registered


def _groups(context: dict) -> dict:
    families = list(context["families"])
    groups = {}
    for count in range(1, 5):
        for combination in itertools.combinations(families, count):
            chosen = set().union(*(set(context["families"][family]) for family in combination))
            assets = [asset for asset in context["assets"] if asset in chosen]
            name = "matched__" + "__".join(combination)
            groups[name] = {"families": list(combination), "assets": assets}
    return groups


def _portfolio_cases(model: dict) -> list[dict]:
    rows = []
    if model["native_covariance"]:
        for minimum, factor, source in itertools.product(
            [None, 3, 5, 7, 10, 12], [0.5, 1.0, 1.5], ["historical", "GP"]
        ):
            rows.append({"minimum": minimum, "budget_factor": factor, "risk_source": source, "objective": "rank"})
        for minimum, source in itertools.product(
            [None, 10], ["GP_prequential_scale", "GP_blend_0.25", "GP_blend_0.5", "GP_blend_0.75", "GP_diagonal"]
        ):
            rows.append({"minimum": minimum, "budget_factor": 1.0, "risk_source": source, "objective": "rank"})
        for minimum, source, objective in itertools.product(
            [None, 10], ["historical", "GP"], ["smooth_rank", "shrunk_return"]
        ):
            rows.append({"minimum": minimum, "budget_factor": 1.0, "risk_source": source, "objective": objective})
    else:
        for minimum in [None, 10]:
            rows.append({"minimum": minimum, "budget_factor": 1.0, "risk_source": "historical", "objective": "rank"})
    return [{"configuration": model["name"], **row} for row in rows]
