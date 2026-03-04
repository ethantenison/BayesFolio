from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from bayesfolio.core.settings import Horizon

logger = logging.getLogger(__name__)


class EtfFeaturesProvider:
    """Transitional provider for long-format ETF features."""

    def __init__(
        self,
        fetcher: Callable[..., pd.DataFrame] | None = None,
    ) -> None:
        """Initialize provider with an injected fetch callable.

        Args:
            fetcher: Callable returning long ETF feature dataframe.
        """

        self._fetcher = fetcher

    def get_etf_features_long(
        self,
        tickers: list[str],
        start: str,
        end: str,
        horizon: Horizon,
    ) -> pd.DataFrame:
        """Fetch ETF predictors in long format.

        Args:
            tickers: Asset tickers.
            start: Inclusive start date in ISO format.
            end: Inclusive end date in ISO format.
            horizon: Frequency code (for example ``BME``).

        Returns:
            DataFrame with ``date``, ``asset_id``, and ETF predictor columns.

        Raises:
            ValueError: If no fetcher is configured.
        """

        if self._fetcher is None:
            msg = "EtfFeaturesProvider requires a fetcher callable."
            raise ValueError(msg)

        logger.info("Fetching ETF features for %d tickers from %s to %s.", len(tickers), start, end)
        try:
            return self._fetcher(tickers=tickers, start=start, end=end, horizon=horizon)
        except TypeError:
            return self._fetcher(tickers, start, end, horizon)
