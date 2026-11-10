"""Local deterministic tests for isolated ETF evaluation, with no GP fits/network.

Usage:
    .venv/bin/python -m pytest -q experiments/2026-10-portfolio-evaluation-v1
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from evaluate import evaluate_model
from metrics import (
    calibration_summary,
    forecast_skill,
    gaussian_scores,
    joint_gaussian_scores,
    paired_block_bootstrap,
    portfolio_performance,
    project_portfolios,
    rank_ic,
)
from run_evaluation import _write_immutable
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm, spearmanr

from artifacts import Reader, aligned_matrix, load_context, load_covariance, verify_legacy_audit


def _fixture() -> tuple[dict, dict, np.ndarray]:
    rng = np.random.default_rng(732)
    assets = [f"ETF{i}" for i in range(15)]
    mean = rng.normal(0, 0.01, (24, 15))
    covariance = np.broadcast_to(np.eye(15) * 0.02**2, (24, 15, 15)).copy()
    actual = mean + rng.normal(0, 0.02, mean.shape)
    families = {f"family{i}": assets[start:end] for i, (start, end) in enumerate([(0, 4), (4, 8), (8, 12), (12, 15)])}
    run = {"actual": actual, "mean": mean, "std": np.full_like(mean, 0.02),
           "legacy_weights": np.ones_like(mean) / 15}
    context = {"assets": assets, "dates": pd.date_range("2020-01-31", periods=24, freq="BME").strftime(
        "%Y-%m-%d").tolist(), "families": families, "baseline_mean": np.zeros_like(mean),
        "historical_covariance": covariance, "fixed_probes": {"equal_weight": np.ones(15) / 15},
        "risk_free": np.full(24, 0.003)}
    return run, context, covariance


@pytest.mark.parametrize("z", [-4.0, -1.0, 0.0, 0.8, 3.0])
def test_gaussian_nll_and_crps_against_independent_references(z):
    mean, std = 0.01, 0.04
    actual = mean + z * std
    scores = gaussian_scores(np.array([actual]), np.array([mean]), np.array([std]))
    assert scores["nll"][0] == pytest.approx(-norm.logpdf(actual, mean, std), abs=1e-12)
    integrated = quad(lambda x: (norm.cdf(x, mean, std) - float(x >= actual))**2,
                      -np.inf, np.inf, points=None, epsabs=1e-10)[0]
    assert scores["crps"][0] == pytest.approx(integrated, abs=1e-8)
    assert scores["crps"][0] >= 0


def test_joint_nll_matches_scipy_and_batch_equals_loop():
    rng = np.random.default_rng(123)
    mean, actual = rng.normal(size=(7, 4)), rng.normal(size=(7, 4))
    factor = rng.normal(size=(7, 4, 4))
    covariance = factor @ factor.swapaxes(-1, -2) + np.eye(4)[None] * 0.1
    result = joint_gaussian_scores(actual, mean, covariance)
    expected = [-multivariate_normal.logpdf(y, mu, cov) for y, mu, cov in zip(actual, mean, covariance, strict=True)]
    np.testing.assert_allclose(result["joint_nll"], expected, rtol=0, atol=1e-10)


def test_joint_nll_detects_dependence_that_marginal_nll_cannot():
    actual, mean = np.array([[1.0, 1.0]]), np.zeros((1, 2))
    independent = np.eye(2)[None]
    correlated = np.array([[[1.0, 0.8], [0.8, 1.0]]])
    marginal = gaussian_scores(actual, mean, np.ones((1, 2)))["nll"].sum()
    assert joint_gaussian_scores(actual, mean, independent)["joint_nll"][0] == pytest.approx(marginal)
    assert joint_gaussian_scores(actual, mean, correlated)["joint_nll"][0] < marginal


@pytest.mark.parametrize("covariance", [np.array([[[1.0, 2.0], [0.0, 1.0]]]),
    np.array([[[1.0, 1.0], [1.0, 1.0]]]), np.array([[[1.0, 2.0], [2.0, 1.0]]]),
    np.array([[[np.nan, 0.0], [0.0, 1.0]]])])
def test_invalid_covariance_raises_without_jitter_or_diagonal_fallback(covariance):
    with pytest.raises(ValueError):
        joint_gaussian_scores(np.zeros((1, 2)), np.zeros((1, 2)), covariance)


def test_portfolio_projection_preserves_covariance_not_just_diagonal():
    covariance = np.array([[[0.04, 0.01], [0.01, 0.09]]])
    weights = np.array([[[0.5, 0.5], [1.0, -1.0]]])
    result = project_portfolios(np.array([[0.1, 0.2]]), np.array([[0.02, 0.04]]), covariance, weights)
    np.testing.assert_allclose(result["mean"], [[0.03, -0.02]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["std"]**2, [[0.0375, 0.11]], rtol=0, atol=1e-12)


def test_calibration_uses_forecast_errors_not_volatility_of_raw_returns():
    mean = np.array([-0.1, 0.0, 0.1, 0.2])
    actual = mean + np.array([-0.01, 0.01, -0.01, 0.01])
    result = calibration_summary(gaussian_scores(actual, mean, np.full(4, 0.01)))
    assert result["mean_squared_standardized_error"] == pytest.approx(1)
    assert result["error_rms_to_predicted_sd_rms"] == pytest.approx(1)
    assert result["mean_standardized_error"] == pytest.approx(0, abs=1e-12)
    assert sum(result[f"pit_bin_{index}_count"] for index in range(5)) == 4


def test_calibration_exposes_bias_and_understated_uncertainty():
    scores = gaussian_scores(np.full(24, 0.02), np.zeros(24), np.full(24, 0.01))
    result = calibration_summary(scores)
    assert result["mean_standardized_error"] == pytest.approx(2)
    assert result["mean_squared_standardized_error"] == pytest.approx(4)
    assert result["error_rms_to_predicted_sd_rms"] == pytest.approx(2)
    assert result["coverage_95"] == pytest.approx(0)


def test_rank_ic_ties_match_scipy_and_constant_is_explicitly_undefined():
    actual = np.array([[1, 2, 2, 4], [1, 2, 3, 4]], dtype=float)
    mean = np.array([[2, 2, 1, 4], [0, 0, 0, 0]], dtype=float)
    result = rank_ic(actual, mean)
    assert result[0] == pytest.approx(spearmanr(actual[0], mean[0]).statistic)
    assert np.isnan(result[1])


def test_r_squared_compares_expanding_mean_not_zero_or_test_mean():
    result = forecast_skill(np.array([[1.0], [2.0]]), np.array([[1.0], [1.0]]), np.zeros((2, 1)))
    assert result["oos_r_squared_vs_expanding_mean"][0] == pytest.approx(0.8)
    assert result["rmse"][0] == pytest.approx(np.sqrt(0.5))


def test_return_conversion_sharpe_and_nav_include_rf_once():
    relative = np.array([0.02, -0.03, 0.01])
    rf = np.full(3, 0.004)
    total = (1 + relative) * (1 + rf) - 1
    result = portfolio_performance(total, rf)
    excess = relative * (1 + rf)
    assert result["conventional_sharpe"] == pytest.approx(excess.mean() / excess.std(ddof=1) * np.sqrt(12))
    assert result["compounded_total_return"] == pytest.approx(np.prod(1 + total) - 1)
    assert result["max_drawdown_total_nav"] == pytest.approx(total[1])
    assert result["cost_basis"] == "gross_costs_not_supplied"


def test_costs_are_deducted_once_and_labelled_net_without_turnover_objective():
    total, rf, costs = np.array([0.03, -0.01, 0.02]), np.full(3, 0.002), np.full(3, 0.001)
    result = portfolio_performance(total, rf, costs)
    assert result["compounded_total_return"] == pytest.approx(np.prod(1 + total - costs) - 1)
    assert result["cost_basis"] == "net_explicit_costs"


def test_bootstrap_is_paired_reproducible_and_does_not_triple_market_months():
    rng = np.random.default_rng(88)
    left = rng.normal(0.01, 0.03, (24, 3))
    before = np.random.get_state()
    first = paired_block_bootstrap(left, left, samples=200)
    second = paired_block_bootstrap(left, left, samples=200)
    assert first == second
    assert first["months"] == 24
    assert first["seeds_not_independent_months"] == 3
    np.testing.assert_allclose(first["mean_difference_ci95"], [0, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(first["sharpe_difference_ci95"], [0, 0], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])


def test_reader_and_immutable_writer_reject_changed_sources_and_overwrites(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"value": 1}')
    reader = Reader()
    assert reader.read(path) == {"value": 1}
    reader.verify()
    path.write_text('{"value": 2}')
    with pytest.raises(ValueError, match="changed"):
        reader.verify()
    output = tmp_path / "output.json"
    _write_immutable(output, b"original")
    _write_immutable(output, b"original")
    with pytest.raises(ValueError, match="Conflicting"):
        _write_immutable(output, b"replacement")


@pytest.mark.parametrize("problem", ["duplicate", "missing", "unexpected", "nonfinite"])
def test_alignment_rejects_silent_row_dropping_or_imputation(problem):
    frame = pd.DataFrame({"date": ["2020-01-31"] * 2, "asset_id": ["A", "B"], "value": [0.1, 0.2]})
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "missing":
        frame = frame.iloc[:1]
    elif problem == "unexpected":
        frame.loc[1, "asset_id"] = "C"
    else:
        frame.loc[1, "value"] = np.nan
    with pytest.raises(ValueError):
        aligned_matrix(frame, "value", ["2020-01-31"], ["A", "B"])


def test_alignment_uses_explicit_order_not_input_position():
    frame = pd.DataFrame({"date": ["2020-01-31"] * 2, "asset_id": ["B", "A"], "value": [0.2, 0.1]})
    np.testing.assert_allclose(aligned_matrix(frame, "value", ["2020-01-31"], ["A", "B"]), [[0.1, 0.2]])


def test_none_covariance_never_fabricates_joint_scores():
    run, context, _ = _fixture()
    result = evaluate_model(run, context, None, {"source": "unavailable", "exact_gp_joint_nll_available": False})
    assert result["joint_distribution"]["status"] == "unavailable_no_complete_covariance"
    assert "joint_nll" not in result["joint_distribution"]
    assert all(row["covariance_source"] == "common_training_ledoit_wolf_not_gp"
               for row in result["portfolio_distributions"])


def test_full_evaluation_distinguishes_fixed_probes_adaptive_decisions_and_common_covariance():
    run, context, covariance = _fixture()
    result = evaluate_model(run, context, covariance, {"source": "analytic_gp_observed_predictive",
                                                      "exact_gp_joint_nll_available": True})
    assert result["joint_distribution"]["exact_gp_joint_nll_available"]
    assert len(result["ranking"]) == 6
    assert len(result["etfs"]) == 15
    records = result["portfolio_distributions"]
    assert {row["comparison_role"] for row in records} == {"fixed_reference", "adaptive_decision_diagnostic"}
    assert {row["covariance_source"] for row in records} == {
        "analytic_gp_observed_predictive", "common_training_ledoit_wolf_not_gp"}
    assert json.loads(json.dumps(result, allow_nan=False))["mean_marginal_nll"] == pytest.approx(
        result["mean_marginal_nll"])


def test_analytic_covariance_requires_matching_units_axes_prediction_hash_and_observed_noise(tmp_path):
    run, context, covariance = _fixture()
    run["run"] = tmp_path / "run"
    reader = Reader()
    prediction_path = str((run["run"] / "gp_window_predictions.csv").resolve())
    reader.hashes[prediction_path] = hashlib.sha256(b"test-predictions").hexdigest()
    arrays = {"dates": np.array(context["dates"]), "assets": np.array(context["assets"]),
              "mean": run["mean"], "covariance": covariance}
    np.savez(tmp_path / "s27.npz", **arrays)
    meta = {"kind": "observed_predictive", "source": "analytic_gp", "units": "discounted_excess_decimal_squared",
            "prediction_sha256": reader.hashes[prediction_path]}
    (tmp_path / "s27.json").write_text(json.dumps(meta))
    loaded, label = load_covariance(reader, run, context, "analytic", tmp_path, 27)
    np.testing.assert_allclose(loaded, covariance, rtol=0, atol=1e-12)
    assert label["exact_gp_joint_nll_available"]
    meta["kind"] = "latent_without_likelihood_noise"
    (tmp_path / "s27.json").write_text(json.dumps(meta))
    fresh = Reader()
    fresh.hashes[prediction_path] = reader.hashes[prediction_path]
    with pytest.raises(ValueError, match="provenance"):
        load_covariance(fresh, run, context, "analytic", tmp_path, 27)


def test_analytic_mode_never_falls_back_to_scenarios_when_missing(tmp_path):
    run, context, _ = _fixture()
    with pytest.raises(FileNotFoundError):
        load_covariance(Reader(), run, context, "analytic", tmp_path, 27)


def test_scenario_covariance_is_explicitly_an_approximation(tmp_path):
    rng = np.random.default_rng(401)
    context = {"dates": ["2020-01-31"], "assets": ["A", "B"]}
    values = rng.normal(0, 0.01, (256, 2))
    pd.DataFrame(values, columns=["A", "B"]).to_csv(tmp_path / "gp_scenarios_2020-01-31.csv", index=False)
    covariance, meta = load_covariance(Reader(), {"run": tmp_path}, context, "scenarios", None, 27)
    np.testing.assert_allclose(covariance, np.cov(values, rowvar=False, ddof=1)[None], rtol=0, atol=1e-15)
    assert meta["monte_carlo_approximation"]
    assert not meta["exact_gp_joint_nll_available"]
    assert meta["scenario_counts"] == [256]


def test_covariance_loader_rejects_exact_diagonal_inconsistent_with_observation_sd(tmp_path):
    run, context, covariance = _fixture()
    run["run"] = tmp_path
    prediction_path = str((tmp_path / "gp_window_predictions.csv").resolve())
    reader = Reader()
    reader.hashes[prediction_path] = "test"
    np.savez(tmp_path / "s27.npz", dates=context["dates"], assets=context["assets"],
             mean=run["mean"], covariance=covariance * 0.1)
    (tmp_path / "s27.json").write_text(json.dumps({"kind": "observed_predictive", "source": "analytic_gp",
        "units": "discounted_excess_decimal_squared", "prediction_sha256": "test"}))
    with pytest.raises(ValueError, match="diagonal"):
        load_covariance(reader, run, context, "analytic", tmp_path, 27)


def test_context_training_baselines_do_not_see_current_or_future_labels(tmp_path, monkeypatch):
    root = tmp_path / "experiments" / "old"
    dates = pd.date_range("2017-05-31", periods=108, freq="BME").strftime("%Y-%m-%d").tolist()
    assets = [f"ETF{i}" for i in range(15)]
    rng = np.random.default_rng(123)
    values = rng.normal(0, 0.02, (108, 15))
    grid = pd.MultiIndex.from_product([dates, assets], names=["date", "asset_id"])
    panel = pd.DataFrame({"y_excess_lead": values.reshape(-1)}, index=grid).reset_index()
    study = {"families": {"us_equity": assets[:4], "international_equity": assets[4:8],
                          "us_fixed_income": assets[8:12], "international_fixed_income": assets[12:]}}
    yields = pd.DataFrame({"Adj Close": 3.0}, index=pd.bdate_range("2017-01-02", "2026-05-29"))
    reader = Reader()

    def fake_read(path):
        name = str(path.relative_to(root))
        reader.hashes[str(path.resolve())] = "fixture"
        if name == "study.json":
            return study
        if name.endswith("scored_dates_24.txt"):
            return "\n".join(dates[84:])
        if name.endswith("global.parquet"):
            return panel.copy()
        if name.endswith("sources.json"):
            return {"^IRX": {"sha256": "fixture"}}
        return yields.copy()

    monkeypatch.setattr(reader, "read", fake_read)
    first = load_context(reader, root)
    panel.loc[panel.date.ge(dates[84]), "y_excess_lead"] += 0.5
    second = load_context(reader, root)
    np.testing.assert_allclose(first["baseline_mean"][0], second["baseline_mean"][0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(first["historical_covariance"][0], second["historical_covariance"][0],
                               rtol=0, atol=1e-15)
    assert not np.allclose(first["baseline_mean"][1], second["baseline_mean"][1])
    assert len(first["risk_free"]) == 24
    np.testing.assert_allclose(first["fixed_probes"]["fixed_60_40"][:8].sum(), 0.6, rtol=0, atol=1e-15)


def test_existing_audit_cannot_be_accepted_after_source_mutation(tmp_path):
    root = tmp_path / "experiments" / "old"
    audit_path = root / "followup/results/all_joint/independent_audit.json"
    audit_path.parent.mkdir(parents=True)
    source = root / "source.csv"
    source.write_bytes(b"original")
    audit_path.write_text(json.dumps({"passed": True, "unresolved": [], "source_sha256": {
        "experiments/old/source.csv": hashlib.sha256(b"original").hexdigest()}}))
    assert verify_legacy_audit(Reader(), root) == 1
    source.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="changed"):
        verify_legacy_audit(Reader(), root)


def test_all_constant_forecasts_have_explicit_missing_ic_not_artificial_zero():
    run, context, _ = _fixture()
    run["mean"] = np.zeros_like(run["mean"])
    result = evaluate_model(run, context, None, {"source": "unavailable"})
    all_scope = next(row for row in result["ranking"] if row["scope"] == "all")
    assert all_scope["mean_ic"] is None
    assert all_scope["defined_ic_months"] == 0
    assert not all_scope["complete_ic"]


def test_framework_has_no_fitting_network_or_old_experiment_imports():
    import ast

    for name in ["metrics.py", "artifacts.py", "evaluate.py", "run_evaluation.py"]:
        tree = ast.parse((Path(__file__).parent / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert not any(value.split(".")[0] in {"torch", "botorch", "gpytorch", "yfinance", "requests",
                    "urllib", "repair_dataset", "ranking_evaluation", "run_study"} for value in imported)
