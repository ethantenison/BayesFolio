from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from bayesfolio.core.settings import Horizon

logger = logging.getLogger(__name__)


class ReturnsProvider:
    """Transitional provider for long-format excess return labels.

    This provider intentionally does not import engine modules. A composition
    root can inject a legacy fetch callable during transitional migration.
    """

    def __init__(
        self,
        fetcher: Callable[..., pd.DataFrame] | None = None,
    ) -> None:
        """Initialize provider with an injected fetch callable.

        Args:
            fetcher: Callable returning ``[date, asset_id, y_excess_lead]``.
        """

        self._fetcher = fetcher

    def get_y_excess_lead_long(
        self,
        tickers: list[str],
        start: str,
        end: str,
        horizon: Horizon,
    ) -> pd.DataFrame:
        """Fetch long-format target labels in decimal units.

        Args:
            tickers: Asset tickers.
            start: Inclusive start date in ISO format.
            end: Inclusive end date in ISO format.
            horizon: Frequency code (for example ``BME``).

        Returns:
            DataFrame with columns ``date``, ``asset_id``, ``y_excess_lead``
            where returns are decimal (``0.02`` means ``2%``).

        Raises:
            ValueError: If no fetcher is configured.
        """

        if self._fetcher is None:
            msg = "ReturnsProvider requires a fetcher callable."
            raise ValueError(msg)

        logger.info("Fetching return labels for %d tickers from %s to %s.", len(tickers), start, end)
        try:
            return self._fetcher(tickers=tickers, start=start, end=end, horizon=horizon)
        except TypeError:
            return self._fetcher(tickers, start, end, horizon)
