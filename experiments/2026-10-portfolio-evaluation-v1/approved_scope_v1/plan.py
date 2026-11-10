"""Freeze existing input masks, task partitions and evidence for approved scope.

Usage: called by run_study.py --prepare. Research I/O is restricted to outputs/;
no source datasets or previous studies are modified. All numeric arrays retain
the original discounted-excess decimal target units and chronological axes.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from study_common import (
    BASELINE,
    FITS,
    HERE,
    JOINT_REFERENCES,
    POLICY,
    ROOT,
    SEEDS,
    SELECTION,
    US_MASK,
    _command,
    _context,
    _groups,
    _hash,
    _save,
    _verify,
)


def _snapshot(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(values) or any(not np.array_equal(old[k], values[k]) for k in values):
                raise ValueError(f"Conflicting immutable array snapshot: {path}")
        return
    np.savez(path, **values)


def _prepare() -> dict:

    path = ROOT / "contract.json"
    if path.exists():
        contract = json.loads(path.read_text())
        _verify(contract["input_sha256"])
        return contract
    full = json.loads((FITS / "contract.json").read_text())
    old = json.loads((SELECTION / "fit_contract.json").read_text())
    policy = json.loads((POLICY / "policy_contract.json").read_text())
    full_seal = json.loads((FITS / "COMPLETED.json").read_text())
    old_seal = json.loads((SELECTION / "COMPLETED.json").read_text())
    policy_seal = json.loads((POLICY / "COMPLETED.json").read_text())
    if not all(s["passed"] for s in [full_seal, old_seal, policy_seal]):
        raise ValueError("Upstream study incomplete")
    for contract, location, seal in [
        (full, FITS, full_seal),
        (old, SELECTION, old_seal),
        (policy, POLICY, policy_seal),
    ]:
        field = (
            "fit_contract.json"
            if location == SELECTION
            else "policy_contract.json"
            if location == POLICY
            else "contract.json"
        )
        expected = seal.get("signature", seal.get("fit_signature"))
        if expected is not None and expected != _hash(location / field):
            raise ValueError("Upstream contract seal differs")
        _verify(contract["source_sha256"])
    evidence = {}

    def checked(path: Path, expected: dict | None = None) -> None:
        digest = _hash(path)
        if expected is not None and digest != expected[str(path.resolve())]:
            raise ValueError(f"Unsealed source input: {path}")
        evidence[str(path.resolve())] = digest

    for folder, name in [
        (FITS, "contract.json"),
        (FITS, "COMPLETED.json"),
        (FITS, "context.json"),
        (FITS, "context_arrays.npz"),
        (SELECTION, "fit_contract.json"),
        (SELECTION, "COMPLETED.json"),
        (POLICY, "policy_contract.json"),
        (POLICY, "COMPLETED.json"),
    ]:
        checked(folder / name)
    csv = POLICY / "all_499_policy_metrics.csv"
    checked(csv, policy_seal["artifact_sha256"])
    metrics = pd.read_csv(csv)
    context = _context()
    global_config = full["configs"][BASELINE]
    global_feature = global_config["command"][global_config["command"].index("--feature-path") + 1]
    checked(Path(global_feature), full["source_sha256"])
    global_frame = pd.read_parquet(global_feature)
    if global_frame.duplicated(["date", "asset_id"]).any():
        raise ValueError("Duplicate target/input rows")
    target = (
        global_frame.pivot(index="date", columns="asset_id", values="y_excess_lead")
        .reindex(index=pd.to_datetime(context["dates"]), columns=context["assets"])
        .to_numpy()
    )
    if not np.allclose(target, context["actual"], rtol=0, atol=1e-12):
        raise ValueError("Forecast-origin/next-month target alignment differs")
    for name in [BASELINE, US_MASK]:
        config = full["configs"][name]
        selected = global_frame[config["inputs"] + ["y_excess_lead"]].to_numpy()
        if not np.isfinite(selected).all():
            raise ValueError("Unavailable registered inputs or labels; no imputation")
    own_us = full["configs"][US_MASK]
    family_feature = Path(own_us["command"][own_us["command"].index("--feature-path") + 1])
    checked(family_feature, full["source_sha256"])
    family_frame = pd.read_parquet(family_feature).set_index(["date", "asset_id"]).sort_index()
    matched = (
        global_frame[global_frame.asset_id.isin(context["families"]["us_equity"])]
        .set_index(["date", "asset_id"])
        .sort_index()
    )
    cols = own_us["inputs"] + ["y_excess_lead"]
    if not family_frame.index.equals(matched.index) or not np.allclose(
        family_frame[cols].to_numpy(), matched[cols].to_numpy(), atol=1e-12, rtol=0
    ):
        raise ValueError("US-only and global datasets do not contain identical matched inputs")
    components, models, new_configs = {}, [], {}

    def reuse(name: str, source_name: str, folder: Path, config: dict, source_seal: dict) -> None:
        if name in components:
            return
        arrays = []
        for seed in SEEDS:
            source = folder / "analytic" / source_name / f"s{seed}.npz"
            if folder == FITS:
                stamp = folder / "evaluation_checkpoints" / f"{source_name}.json"
                checked(stamp, source_seal["artifact_sha256"])
                hashes = json.loads(stamp.read_text())["result"]["artifact_sha256"]
            else:
                hashes = source_seal["artifact_sha256"]
            checked(source, hashes)
            with np.load(source, allow_pickle=False) as values:
                if values["dates"].tolist() != context["dates"] or values["assets"].tolist() != config["tickers"]:
                    raise ValueError("Reused predictor axes differ")
                mean, covariance = values["mean"].copy(), values["covariance"].copy()
                if not np.allclose(
                    np.diagonal(covariance, axis1=-2, axis2=-1), values["std"] ** 2, rtol=1e-8, atol=1e-12
                ):
                    raise ValueError("Reused observed covariance differs from marginal SD")
                np.linalg.cholesky(covariance)
                arrays.append((mean, covariance))
        destination = ROOT / "components" / f"{name}.npz"
        _snapshot(
            destination,
            mean=np.stack([a[0] for a in arrays]),
            covariance=np.stack([a[1] for a in arrays]),
            assets=np.array(config["tickers"]),
            dates=np.array(context["dates"]),
        )
        checked(destination)
        components[name] = {
            "assets": config["tickers"],
            "inputs": config["inputs"],
            "reuse": True,
            "source_name": source_name,
            "source_root": str(folder),
            "config": config,
        }

    for name in JOINT_REFERENCES:
        reuse(name, name, FITS, full["configs"][name], full_seal)
        models.append({"name": name, "components": [name], "native_covariance": True, "kind": "joint_reference"})
    choices = {}
    for family in context["families"]:
        frame = metrics[metrics.family.eq(family)]
        ranking = (
            frame.sort_values(["mean_ic", "input_count_excluding_time", "configuration"], ascending=[False, True, True])
            .iloc[0]
            .configuration
        )
        portfolio = (
            frame.sort_values(
                [
                    "historical__risk_budget_plus_effective__conventional_sharpe",
                    "input_count_excluding_time",
                    "configuration",
                ],
                ascending=[False, True, True],
            )
            .iloc[0]
            .configuration
        )
        if ranking == portfolio:
            raise ValueError("Two distinct registered family choices required")
        choices[family] = [ranking, portfolio]
        for name in choices[family]:
            reuse(name, name, FITS, full["configs"][name], full_seal)
    for bits in itertools.product([0, 1], repeat=4):
        names = [choices[family][bit] for family, bit in zip(context["families"], bits, strict=True)]
        models.append(
            {
                "name": "family_assembly_" + "".join(map(str, bits)),
                "components": names,
                "native_covariance": False,
                "kind": "selected_family_assembly",
            }
        )
    groups = _groups(context)
    for name, group in groups.items():
        families, assets = group["families"], group["assets"]
        config = _command(global_config, name, assets, global_feature)
        old_name = None
        if len(families) == 1:
            old_name = "common_" + families[0]
        elif set(families) == {"us_equity", "us_fixed_income"}:
            old_name = "common_us_region"
        elif set(families) == {"international_equity", "international_fixed_income"}:
            old_name = "common_international_region"
        if len(families) == 4:
            reuse(name, BASELINE, FITS, global_config, full_seal)
        elif old_name:
            reuse(name, old_name, SELECTION, old["configs"][old_name], old_seal)
        else:
            components[name] = {"assets": assets, "inputs": config["inputs"], "reuse": False, "config": config}
            new_configs[name] = config
    family_names = list(context["families"])
    singles = [name for name, g in groups.items() if len(g["families"]) == 1]
    models.append(
        {
            "name": "matched_four_family_assembly",
            "components": singles,
            "native_covariance": False,
            "kind": "matched_grouping",
        }
    )
    for name, group in groups.items():
        group_set = set(group["families"])
        complement = set(family_names) - group_set
        if len(group_set) not in {2, 3}:
            continue
        other = next(n for n, g in groups.items() if set(g["families"]) == complement)
        if len(group_set) == 2 and name > other:
            continue
        models.append(
            {
                "name": "partition__" + name.removeprefix("matched__"),
                "components": [name, other],
                "native_covariance": False,
                "kind": "matched_grouping",
            }
        )
    name = "joint_existing_us_mask"
    config = _command(own_us, name, context["assets"], global_feature)
    components[name] = {"assets": context["assets"], "inputs": config["inputs"], "reuse": False, "config": config}
    new_configs[name] = config
    models.append({"name": name, "components": [name], "native_covariance": True, "kind": "matched_US_mask"})
    audit_configs = {}
    for reference in [BASELINE, US_MASK]:
        name = "audit__" + reference
        config = _command(full["configs"][reference], name, full["configs"][reference]["tickers"], global_feature)
        config["reference_component"] = reference
        audit_configs[name] = config
    if len(new_configs) != 9 or len(models) != 30 or len(groups) != 15:
        raise ValueError("Approved task-partition plan count differs")
    for name, config in {**new_configs, **audit_configs}.items():
        if "hy_spread_chg_1p" in config["inputs"] and len(config["tickers"]) == 15:
            raise ValueError("Excluded joint HY-change input")
        if "t_index" not in config["inputs"]:
            raise ValueError("Missing retained time input")
        if config["command"][config["command"].index("--maxiter") + 1] != "15":
            raise ValueError("Iteration guardrail changed")
    for path in [
        *HERE.glob("*.py"),
        HERE / "specification.json",
        *map(Path, full["source_sha256"]),
        *map(Path, old["source_sha256"]),
    ]:
        checked(path)
    contract = {
        "schema": "bayesfolio.approved_scope.v1",
        "approved_items": [1, 2, 3, 4, 5, 6, 8],
        "excluded_items": [7, 9],
        "models": models,
        "components": components,
        "new_configs": new_configs,
        "audit_configs": audit_configs,
        "groups": groups,
        "family_choices": choices,
        "rank": 2,
        "maxiter": 15,
        "seeds": SEEDS,
        "initial_training_months": 84,
        "scored_months": 24,
        "fresh_component_fits": 648,
        "fresh_audit_replay_fits": 4,
        "worker_ceiling": min(13, max(1, (os.cpu_count() or 16) - 2)),
        "memory_budget_gib": 48,
        "memory_safety_multiplier": 1.25,
        "ETF_position_cap": None,
        "CVaR": False,
        "fallback_weights": False,
        "constraint_relaxation": False,
        "turnover_objective": False,
        "automatic_promotion": False,
        "input_masks": "existing_registered_only_no_new_feature_search",
        "chronology": "unchanged_84_plus_index_training_24_reused_origins",
        "US_dataset_parity": "identical_raw_inputs_and_targets_family_vs_global",
        "calibration": "EW_error_scale_prior_strength_6_earlier_origins_only",
        "legacy_portfolio_optimizer_invoked": False,
        "input_sha256": evidence,
    }
    _verify(evidence)
    _save(ROOT / "contract.json", contract, immutable=True)
    return contract


def _assemble(contract: dict) -> list[dict]:
    context = _context()
    available = []
    for model in contract["models"]:
        paths = [ROOT / "components" / f"{name}.npz" for name in model["components"]]
        if not all(path.exists() for path in paths):
            continue
        mean = np.empty((3, 24, 15))
        covered, native = set(), None
        for path in paths:
            with np.load(path, allow_pickle=False) as values:
                assets = values["assets"].tolist()
                if covered.intersection(assets):
                    raise ValueError("Assembly components overlap")
                covered.update(assets)
                indices = [context["assets"].index(asset) for asset in assets]
                mean[:, :, indices] = values["mean"]
                if model["native_covariance"]:
                    if len(paths) != 1 or assets != context["assets"]:
                        raise ValueError("Invalid full GP covariance provenance")
                    native = values["covariance"].copy()
        if covered != set(context["assets"]):
            raise ValueError("Incomplete common-universe assembly")
        fields = {"mean": mean, "assets": np.array(context["assets"]), "dates": np.array(context["dates"])}
        if native is not None:
            fields["covariance"] = native
        _snapshot(ROOT / "panels" / f"{model['name']}.npz", **fields)
        available.append(model)
    return available
