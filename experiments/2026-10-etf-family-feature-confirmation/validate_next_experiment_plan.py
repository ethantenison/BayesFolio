"""Validate Qwen's proposed next deterministic U.S. experiment batch.

This validator is intentionally independent of Qwen. It checks the evidence
binding, required strategy mix, fixed experiment controls, exact run budget,
and prohibited concepts before a human reviews or authorizes any new fits.

Usage:
    uv run --locked python \
      experiments/2026-10-etf-family-feature-confirmation/validate_next_experiment_plan.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
PLANNING_DIR = EXPERIMENT_DIR / "planning"
EVIDENCE_PATH = PLANNING_DIR / "step4_evidence.json"
PLAN_PATH = PLANNING_DIR / "NEXT_EXPERIMENT_PLAN.md"
CANDIDATES_PATH = PLANNING_DIR / "next_experiment_candidates.json"
EXPECTED_CONFIGURATIONS = {"U-AC", "U-C005", "U-C009", "U-ALL-MATCH"}
EXPECTED_SEEDS = [27, 127, 227]
EXPECTED_DIRECTIONS = {"increase", "decrease", "neutral", "uncertain"}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_keys(payload: dict[str, Any], keys: set[str], *, label: str) -> list[str]:
    """Return missing-key failures for one JSON object."""
    return [f"{label}: missing {key}" for key in sorted(keys.difference(payload))]


def validate_candidate(candidate: dict[str, Any], *, allowlist: set[str]) -> list[str]:
    """Validate one exact candidate definition and fixed comparison controls."""
    required = {
        "id",
        "mode",
        "priority",
        "parent_configurations",
        "hypothesis",
        "frontier_gap_targeted",
        "exact_feature_changes",
        "exact_model_changes",
        "exact_runner_args",
        "cached_columns_required",
        "evidence",
        "expected_direction",
        "falsification_condition",
        "risk",
        "compute",
    }
    failures = require_keys(candidate, required, label="candidate")
    if failures:
        return failures
    identifier = candidate["id"]
    if not isinstance(identifier, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", identifier):
        failures.append(f"candidate {identifier!r}: invalid id")
    if candidate["mode"] not in {"exploit", "bridge", "explore"}:
        failures.append(f"candidate {identifier}: invalid mode")
    parents = set(candidate["parent_configurations"])
    if not parents or not parents.issubset(EXPECTED_CONFIGURATIONS):
        failures.append(f"candidate {identifier}: invalid parent configurations")
    feature_changes = candidate["exact_feature_changes"]
    if not isinstance(feature_changes, dict) or set(feature_changes) != {"add", "remove"}:
        failures.append(f"candidate {identifier}: feature changes must contain exactly add and remove")
    runner_args = candidate["exact_runner_args"]
    if not isinstance(runner_args, list) or not all(isinstance(value, str) for value in runner_args):
        failures.append(f"candidate {identifier}: exact_runner_args must be a string list")
    compute = candidate["compute"]
    if compute != {"seeds": EXPECTED_SEEDS, "windows": 24, "run_count": 3}:
        failures.append(f"candidate {identifier}: fixed compute panel")
    expected = candidate["expected_direction"]
    if set(expected) != {"mean_ic", "mean_gaussian_nll", "sharpe"}:
        failures.append(f"candidate {identifier}: expected directions must cover all objectives")
    elif any(direction not in EXPECTED_DIRECTIONS for direction in expected.values()):
        failures.append(f"candidate {identifier}: invalid expected direction")
    targets = set(candidate["frontier_gap_targeted"])
    if not targets or not targets.issubset({"mean_ic", "mean_gaussian_nll", "sharpe"}):
        failures.append(f"candidate {identifier}: invalid frontier gap")
    evidence = candidate["evidence"]
    if not evidence:
        failures.append(f"candidate {identifier}: evidence is required")
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"path", "finding"} or item["path"] not in allowlist:
            failures.append(f"candidate {identifier}: evidence must use the research allowlist")
    serialized = json.dumps(candidate).lower()
    prohibited = ["ijr", "turnover", "exclude-time-index", "external refresh", "fetch external"]
    if any(term in serialized for term in prohibited):
        failures.append(f"candidate {identifier}: prohibited concept")
    for flag, expected_value in {
        "--mtgp-rank": "2",
        "--maxiter": "15",
        "--posterior-scenarios": "256",
    }.items():
        if flag not in runner_args:
            failures.append(f"candidate {identifier}: missing fixed {flag}")
            continue
        index = runner_args.index(flag)
        if index + 1 >= len(runner_args) or runner_args[index + 1] != expected_value:
            failures.append(f"candidate {identifier}: {flag} must remain {expected_value}")
    return failures


def main() -> None:
    """Validate Qwen planning artifacts and print a compact success record."""
    missing = [str(path) for path in [EVIDENCE_PATH, PLAN_PATH, CANDIDATES_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing planning artifacts: {missing}")
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    payload = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    failures = require_keys(
        payload,
        {
            "schema",
            "based_on",
            "objectives",
            "batch_design",
            "candidates",
            "execution_order",
            "stopping_rules",
            "deferred_ideas",
        },
        label="root",
    )
    if payload.get("schema") != "bayesfolio.etf_family_next_experiment_candidates.v1":
        failures.append("root: invalid schema")
    based_on = payload.get("based_on", {})
    if based_on.get("evidence_sha256") != sha256_file(EVIDENCE_PATH):
        failures.append("root: evidence hash mismatch")
    if based_on.get("current_hypervolume") != evidence["hypervolume"]["value"]:
        failures.append("root: hypervolume mismatch")
    if based_on.get("reference_point") != evidence["hypervolume"]["reference_point_raw_objectives"]:
        failures.append("root: reference point mismatch")
    if based_on.get("seeds") != EXPECTED_SEEDS:
        failures.append("root: seed panel mismatch")
    source_hashes = evidence.get("source_sha256", {})
    for relative, expected_hash in source_hashes.items():
        source = REPO_ROOT / relative
        if not source.exists() or sha256_file(source) != expected_hash:
            failures.append(f"root: step-3 source changed after evidence capture: {relative}")
    objectives = payload.get("objectives", {})
    if objectives != {
        "maximize": ["mean_ic", "sharpe"],
        "minimize": ["mean_gaussian_nll"],
        "diagnostic_only": ["cumulative_return", "rmse"],
        "turnover_excluded": True,
    }:
        failures.append("root: objective contract mismatch")

    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        failures.append("root: candidates must be a list")
        candidates = []
    if not 4 <= len(candidates) <= 8:
        failures.append("root: candidate count must be between 4 and 8")
    candidate_objects = [candidate for candidate in candidates if isinstance(candidate, dict)]
    if len(candidate_objects) != len(candidates):
        failures.append("root: every candidate must be an object")
    identifiers = [candidate.get("id") for candidate in candidate_objects]
    if len(identifiers) != len(set(identifiers)):
        failures.append("root: candidate IDs must be unique")
    modes = [candidate.get("mode") for candidate in candidate_objects]
    for required_mode in ["exploit", "bridge", "explore"]:
        if required_mode not in modes:
            failures.append(f"root: missing {required_mode} candidate")
    priorities = [candidate.get("priority") for candidate in candidate_objects]
    priorities_valid = len(priorities) == len(candidates) and all(isinstance(value, int) for value in priorities)
    if not priorities_valid or sorted(priorities) != list(range(1, len(candidates) + 1)):
        failures.append("root: priorities must be unique and contiguous from 1")
    expected_order = (
        [item["id"] for item in sorted(candidate_objects, key=lambda item: item["priority"])]
        if priorities_valid and all(isinstance(item.get("id"), str) for item in candidate_objects)
        else []
    )
    if payload.get("execution_order") != expected_order:
        failures.append("root: execution order must match priority")
    if not payload.get("stopping_rules"):
        failures.append("root: stopping rules are required")
    allowlist = set(evidence["research_allowlist"])
    for candidate in candidate_objects:
        failures.extend(validate_candidate(candidate, allowlist=allowlist))
    batch_design = payload.get("batch_design", {})
    expected_counts = {
        "candidate_count": len(candidates),
        "exploit_count": modes.count("exploit"),
        "bridge_count": modes.count("bridge"),
        "explore_count": modes.count("explore"),
    }
    if any(batch_design.get(key) != value for key, value in expected_counts.items()):
        failures.append("root: batch design counts do not match candidates")
    if not isinstance(batch_design.get("selection_logic"), str) or not batch_design["selection_logic"].strip():
        failures.append("root: batch selection logic is required")

    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    required_plan_terms = ["hypervolume", "exploitation", "exploration", "Pareto", "stopping"]
    if any(term.lower() not in plan_text.lower() for term in required_plan_terms):
        failures.append("plan: missing required design discussion")
    if failures:
        raise ValueError("Next-experiment plan validation failed:\n- " + "\n- ".join(failures))
    print(
        json.dumps(
            {
                "status": "passed",
                "candidate_count": len(candidates),
                "modes": {mode: modes.count(mode) for mode in ["exploit", "bridge", "explore"]},
                "evidence_sha256": sha256_file(EVIDENCE_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
