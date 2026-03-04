from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from bayesfolio.core.settings import Horizon

logger = logging.getLogger(__name__)


class MacroProvider:
    """Transitional provider for date-indexed macro features."""

    def __init__(
        self,
        fetcher: Callable[..., pd.DataFrame] | None = None,
    ) -> None:
        """Initialize provider with an injected fetch callable.

        Args:
            fetcher: Callable returning macro features with a ``date`` column.
        """

        self._fetcher = fetcher

    def get_macro_features(self, start: str, end: str, horizon: Horizon) -> pd.DataFrame:
        """Fetch macro predictors.

        Args:
            start: Inclusive start date in ISO format.
            end: Inclusive end date in ISO format.
            horizon: Frequency code (passed when supported by fetcher).

        Returns:
            DataFrame with ``date`` plus macro feature columns.

        Raises:
            ValueError: If no fetcher is configured.
        """

        if self._fetcher is None:
            msg = "MacroProvider requires a fetcher callable."
            raise ValueError(msg)

        logger.info("Fetching macro features from %s to %s.", start, end)
        try:
            return self._fetcher(start=start, end=end, horizon=horizon)
        except TypeError:
            return self._fetcher(start=start, end=end)
