"""Local verification of cache compatibility and causal v2 worker boundaries."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import v2_experts as experts
from v2_common import ASSETS
from v2_expert_worker import training_panel


def test_rank1_cache_exact_parity_and_missing_jobs(tmp_path, monkeypatch):
    """Rank1 reuses52 complete origins and requires exactly six new fits."""
    monkeypatch.setattr(experts, "checkpoint_path", lambda job: tmp_path / "missing_checkpoints" /
                        f"{job['component']}_s{job['seed']}_{job['date']}.json")
    report = experts.cache_inventory(["compact_r1_i15_exp"])["compact_r1_i15_exp"]
    assert report["cached_complete_origins"] == 52
    assert report["missing_fits"] == 6
    assert report["prior28_bitwise_compatible"]
    jobs = experts.missing_jobs(experts.registered_configs(["compact_r1_i15_exp"]))
    assert len(jobs) == 6
    assert {job["index"] for job in jobs} == {28, 29}
    assert {job["seed"] for job in jobs} == {27, 127, 227}


def test_reference_caches_complete_and_registered_specs_unchanged():
    """All three references need zero new fits and retain exact original masks."""
    configs = experts.registered_configs(experts.COMPONENTS)
    assert experts.missing_jobs(configs) == []
    original = experts.read(experts.CHRONO / "CONTRACT.json")["configs"]
    for name, config in configs.items():
        assert config["inputs"] == original[name]["inputs"]
        assert config["tickers"] == ASSETS
        assert config["rank"] == 2
        assert config["maxiter"] == 15
        assert config["command"][config["command"].index("--feature-path") + 1] == str(experts.DATASET)


@pytest.mark.parametrize("names,dates,root", [
    (["broad_r5_i15_exp"], None, None),
    (["compact_r1_i15_exp"], ["2026-10-31"], None),
    (["compact_r1_i15_exp"], ["2026-10-30", "2026-09-30"], None),
    (["compact_r1_i15_exp"], None, Path("/private/tmp/outside-v2")),
])
def test_registration_rejects_out_of_scope_changes(names, dates, root):
    """No implicit new feature/rank/date/output specification is admitted."""
    with pytest.raises(ValueError):
        experts.registered_configs(names, dates, root)


def test_training_panel_retains_unknown_live_labels_and_all_tasks():
    """November training uses113 labeled months and never its unknown target."""
    frame = pd.read_parquet(experts.DATASET)
    train, evaluation = training_panel(frame, "2026-10-30", 29, ASSETS, None)
    assert train.date.nunique() == 113
    assert len(train) == 113 * 15
    assert train.date.max() == pd.Timestamp("2026-09-30")
    assert evaluation.y_excess_lead.isna().all()
    assert set(train.asset_id) == set(ASSETS)
    rolling, _ = training_panel(frame, "2026-10-30", 29, ASSETS, 48)
    assert rolling.date.nunique() == 48
    earliest, _ = training_panel(frame, "2022-05-31", -24, ASSETS, None)
    assert earliest.date.nunique() == 60


def test_training_panel_rejects_known_live_target_missing_tasks_and_wrong_index():
    """Leakage and incomplete causal panels are fatal, without label substitution."""
    frame = pd.read_parquet(experts.DATASET)
    leaked = frame.copy()
    leaked.loc[leaked.date.eq(pd.Timestamp("2026-10-30")), "y_excess_lead"] = 0.0
    with pytest.raises(ValueError, match="cannot be known"):
        training_panel(leaked, "2026-10-30", 29, ASSETS, None)
    with pytest.raises(ValueError, match="Incomplete"):
        training_panel(frame[frame.asset_id.ne("MGK")], "2026-10-30", 29, ASSETS, None)
    with pytest.raises(ValueError, match="Unregistered"):
        training_panel(frame, "2026-10-30", 28, ASSETS, None)


def test_corrupt_covariance_or_changed_axes_are_rejected(tmp_path):
    """A bad covariance cache cannot enter a model/blend panel."""
    source = experts.arrays(experts.CHRONO / "components/compact_r1_i15_exp.npz")
    values = copy.deepcopy(source)
    values["covariance"][0, 0, 0, 0] = -1.0
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **values)
    with pytest.raises(np.linalg.LinAlgError):
        experts._checked_cache(path)
    values = copy.deepcopy(source)
    values["assets"][1] = "IJR"
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="Incompatible"):
        experts._checked_cache(path)


def test_worker_initialize_maps_original_indexes_and_rejects_disguised_feature_change(monkeypatch):
    """Initialization enforces registration before any numerical fitter runs."""
    import v2_expert_worker

    observed = {}
    fake = SimpleNamespace(initialize=lambda configs, framework: observed.update(configs=configs, framework=framework))
    monkeypatch.setitem(sys.modules, "tuning_worker", fake)
    configs = experts.registered_configs(["compact_r1_i15_exp"], ["2022-05-31", "2026-10-30"])
    v2_expert_worker.initialize(configs, "framework")
    assert observed["configs"]["compact_r1_i15_exp"]["dates"] == {-24: "2022-05-31", 29: "2026-10-30"}
    assert fake.training_panel is training_panel
    changed = copy.deepcopy(configs)
    command = changed["compact_r1_i15_exp"]["command"]
    command[command.index("--macro-component-proposal") + 1] = "matern_linear"
    with pytest.raises(ValueError, match="Unregistered kernel/feature command"):
        v2_expert_worker.initialize(changed, "framework")


def test_actual_task_prior_telemetry_does_not_query_distribution_variance():
    """An actual LKJ prior can be inspected without fitting or undefined moments."""
    import json

    import torch
    from gpytorch.kernels import IndexKernel
    from gpytorch.priors import LKJCovariancePrior, LogNormalPrior
    from torch import nn
    from v2_expert_worker import _model_telemetry

    model = nn.Module()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        model.covar_module = IndexKernel(num_tasks=15, rank=1,
                                        prior=LKJCovariancePrior(n=15, eta=2.0, sd_prior=LogNormalPrior(0.0, 0.5)))
    model.named_priors = model.covar_module.named_priors
    before = torch.get_num_threads()
    report = _model_telemetry(model)
    json.dumps(report, allow_nan=False)
    assert report["nodes"]["covar_module"]["class"] == "IndexKernel"
    assert np.asarray(report["nodes"]["covar_module"]["covar_factor"]).shape == (15, 1)
    assert any(prior["class"] == "LKJCovariancePrior" for prior in report["priors"].values())
    assert torch.get_num_threads() == before
