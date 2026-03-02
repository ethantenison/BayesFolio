from __future__ import annotations

import numpy as np

from bayesfolio.schemas.common import SchemaMetadata
from bayesfolio.schemas.contracts.forecast import ForecastPayload


def build_forecast_payload(
    asset_order: list[str],
    mean: np.ndarray,
    covariance: np.ndarray,
    metadata: SchemaMetadata,
    return_unit: str = "decimal",
) -> ForecastPayload:
    """Build forecast transport payload from numerical model outputs."""

    return ForecastPayload(
        metadata=metadata,
        asset_order=asset_order,
        mean=mean.astype(float).tolist(),
        covariance=covariance.astype(float).tolist(),
        return_unit=return_unit,
    )
