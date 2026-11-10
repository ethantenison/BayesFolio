"""Fast deterministic pipeline tests; synthetic GP readouts never optimize a fit.

Usage from repository root:
    PYTHONPATH=experiments/2026-10-portfolio-evaluation-v1 .venv/bin/python -m pytest \
        -q experiments/2026-10-portfolio-evaluation-v1/parallel_pipeline_v1
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import run_pipeline
import workers


def test_dense_readout_matches_manual_conditioning_and_raw_outcome_scaling():
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize

    torch.manual_seed(41)
    x = torch.linspace(-2, 2, 12, dtype=torch.float64)[:, None]
    y = 80 + 12 * torch.sin(x)
    model = SingleTaskGP(x, y, input_transform=Normalize(d=1), outcome_transform=Standardize(m=1))
    model.likelihood.noise = 0.05
    model.eval()
    model.likelihood.eval()
    test = torch.tensor([[-1.3], [0.6]], dtype=torch.float64)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    result = workers.analytic_readout(model, test)
    with torch.no_grad():
        train = model.train_inputs[0]
        transformed = model.transform_inputs(test)
        noise = float(model.likelihood.noise.item())
        covariance = model.covar_module(train).to_dense().numpy() + np.eye(12) * noise
        cross = model.covar_module(transformed, train).to_dense().numpy()
        test_covariance = model.covar_module(transformed).to_dense().numpy()
        offset = (model.train_targets - model.mean_module(train)).numpy()
        mean = model.mean_module(transformed).numpy() + cross @ np.linalg.solve(covariance, offset)
        observed_covariance = test_covariance - cross @ np.linalg.solve(covariance, cross.T) + np.eye(2) * noise
        scale = float(model.outcome_transform.stdvs.item())
        center = float(model.outcome_transform.means.item())
    np.testing.assert_allclose(result["mean"], mean * scale + center, rtol=0, atol=1e-8)
    np.testing.assert_allclose(result["covariance"], observed_covariance * scale**2, rtol=0, atol=1e-8)
    np.testing.assert_allclose(result["likelihood_noise_diagonal"], noise * scale**2, rtol=0, atol=1e-8)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("policy", ["legacy_scenario_maxret", "risk_rank_cap35", "risk_rank_no_cap"])
def test_strict_solvers_respect_budget_diversification_and_caps(policy):
    mean = np.linspace(-0.02, 0.03, 15)
    vol = np.linspace(0.01, 0.06, 15)
    covariance = np.diag(vol**2)
    first, diagnostics = workers.solve_weights(mean, covariance, policy)
    second, _ = workers.solve_weights(mean, covariance, policy)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)
    np.testing.assert_allclose(first.sum(), 1, rtol=0, atol=1e-7)
    assert diagnostics["effective_assets"] >= 3 - 1e-6
    assert first.max() <= (1 if policy == "risk_rank_no_cap" else 0.35) + 1e-7
    assert not diagnostics["fallback_used"]
    assert not diagnostics["relaxations_used"]
    if policy.startswith("risk_"):
        assert diagnostics["variance_ratio_to_equal_weight"] <= 1 + 1e-6


def test_solver_failure_raises_without_equal_weight_or_previous_weight_fallback(monkeypatch):
    import cvxpy as cp

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(cp.Problem, "solve", fail)
    with pytest.raises(RuntimeError, match="synthetic solver failure"):
        workers.solve_weights(np.arange(15, dtype=float), np.eye(15), "risk_rank_cap35")


def test_memory_and_cpu_limits_are_global_not_multiplied_by_seed():
    assert run_pipeline.worker_limit(14, 16) == 14
    assert run_pipeline.worker_limit(14, 16, 8 * 1024**3) == 4
    with pytest.raises(ValueError):
        run_pipeline.worker_limit(15, 16)


def test_jobs_preserve_original_month_indices_and_independent_seed_paths(tmp_path):
    dates = ["2024-05-31", "2024-06-28", "2024-07-31"]
    jobs = run_pipeline._jobs(tmp_path, dates)
    assert len(jobs) == 9
    assert {row["seed"] for row in jobs} == {27, 127, 227}
    assert all(row["index"] == dates.index(row["date"]) for row in jobs)
    assert len({(row["run"], row["date"]) for row in jobs}) == 9


def test_unordered_month_completion_resumes_only_hash_valid_jobs(tmp_path, monkeypatch):
    calls = []

    def fake_fit(job):
        calls.append(job["date"])
        path = tmp_path / f"{job['seed']}_{job['date']}.txt"
        path.write_text(job["date"])
        return {"seed": job["seed"], "date": job["date"], "paths": {"data": str(path)}}

    monkeypatch.setattr(workers, "fit_job", fake_fit)
    jobs = list(reversed(run_pipeline._jobs(tmp_path, ["2024-05-31", "2024-06-28"])))
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = run_pipeline._stage(tmp_path, pool, jobs, "fit", "registered-signature")
        second = run_pipeline._stage(tmp_path, pool, jobs, "fit", "registered-signature")
    assert first == second
    assert len(calls) == 6
    assert [(row["seed"], row["date"]) for row in first] == sorted((row["seed"], row["date"]) for row in first)
    (tmp_path / "27_2024-05-31.txt").write_text("mutation")
    with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(ValueError, match="changed"):
        run_pipeline._stage(tmp_path, pool, jobs, "fit", "registered-signature")


def test_invalid_covariance_is_not_repaired_for_allocation():
    with pytest.raises(np.linalg.LinAlgError):
        workers.solve_weights(np.arange(15, dtype=float), np.zeros((15, 15)), "risk_rank_cap35")
