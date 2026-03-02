from __future__ import annotations

import numpy as np

from bayesfolio.schemas.common import SchemaMetadata
from bayesfolio.schemas.contracts.forecast import ForecastPayload
from bayesfolio.schemas.contracts.scenarios import ScenarioPanel


def sample_joint_scenarios(
    forecast: ForecastPayload,
    n_scenarios: int,
    seed: int = 1,
    jitter: float = 1e-8,
) -> ScenarioPanel:
    """Sample scenarios from a joint Gaussian using forecast mean and covariance."""

    rng = np.random.default_rng(seed)
    mean = np.asarray(forecast.mean, dtype=float)
    covariance = np.asarray(forecast.covariance, dtype=float)
    covariance = covariance + np.eye(covariance.shape[0], dtype=float) * jitter
    samples = rng.multivariate_normal(mean=mean, cov=covariance, size=n_scenarios)

    return ScenarioPanel(
        metadata=SchemaMetadata(**forecast.metadata.model_dump()),
        asset_order=forecast.asset_order,
        n_scenarios=n_scenarios,
        values=samples.astype(float).tolist(),
        return_unit=forecast.return_unit,
    )
