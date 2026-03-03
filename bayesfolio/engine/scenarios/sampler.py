from __future__ import annotations

import numpy as np

from bayesfolio.contracts.commands.scenario import ScenarioCommand
from bayesfolio.contracts.results.forecast import ForecastResult


def sample_joint_scenarios(
    forecast: ForecastResult,
    n_scenarios: int,
    seed: int = 1,
    jitter: float = 1e-8,
) -> ScenarioCommand:
    """Sample scenarios from a joint Gaussian using forecast mean and covariance."""

    rng = np.random.default_rng(seed)
    mean = np.asarray(forecast.mean, dtype=float)
    covariance = np.asarray(forecast.covariance, dtype=float)
    covariance = covariance + np.eye(covariance.shape[0], dtype=float) * jitter
    samples = rng.multivariate_normal(mean=mean, cov=covariance, size=n_scenarios)

    return ScenarioCommand(
        asset_order=forecast.asset_order,
        n_scenarios=n_scenarios,
        values=samples.astype(float).tolist(),
        return_unit=forecast.return_unit,
    )
