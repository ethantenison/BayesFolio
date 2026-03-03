from __future__ import annotations

import numpy as np

from bayesfolio.contracts.results.forecast import ForecastResult


def build_forecast_payload(
    asset_order: list[str],
    mean: np.ndarray,
    covariance: np.ndarray,
    return_unit: str = "decimal",
) -> ForecastResult:
    """Build forecast transport payload from numerical model outputs."""

    return ForecastResult(
        asset_order=asset_order,
        mean=mean.astype(float).tolist(),
        covariance=covariance.astype(float).tolist(),
        return_unit=return_unit,
    )
