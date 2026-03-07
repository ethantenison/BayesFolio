"""IO provider for long-format ETF feature data.

Boundary responsibility: this module handles IO-layer retrieval and cache
orchestration for ETF features, without engine business logic.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from bayesfolio.core.settings import Horizon
from bayesfolio.io.providers._cache_frame_ops import (
    concat_frames,
    dedupe_rows,
    missing_tickers,
    normalize_asset_id_column,
    normalize_date_column,
    slice_requested,
)

logger = logging.getLogger(__name__)


class EtfFeaturesProvider:
    """Transitional provider for long-format ETF features."""

    def __init__(
        self,
        fetcher: Callable[..., pd.DataFrame] | None = None,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize provider with an injected fetch callable.

        Args:
            fetcher: Callable returning long ETF feature dataframe.
            cache_dir: Optional local directory for parquet cache files.
        """

        self._fetcher = fetcher
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

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

        normalized_tickers = [str(ticker).upper() for ticker in tickers]
        if self._cache_dir is None:
            logger.info("Fetching ETF features for %d tickers from %s to %s.", len(tickers), start, end)
            return self._call_fetcher(tickers=normalized_tickers, start=start, end=end, horizon=horizon)

        cache_frame = self._read_cache_frame(horizon)
        requested_cached = slice_requested(
            frame=cache_frame,
            tickers=normalized_tickers,
            start=start,
            end=end,
        )
        missing_ticker_values = missing_tickers(
            cache_frame=cache_frame,
            tickers=normalized_tickers,
            start=start,
            end=end,
            freq=horizon.value,
        )

        if not missing_ticker_values:
            logger.info(
                "Using cached ETF features for %d tickers from %s to %s.",
                len(normalized_tickers),
                start,
                end,
            )
            return requested_cached.sort_values(["date", "asset_id"]).reset_index(drop=True)

        logger.info(
            "ETF feature cache partial/miss for %d tickers; fetching live data for %d tickers.",
            len(normalized_tickers),
            len(missing_ticker_values),
        )
        fetched = self._call_fetcher(tickers=missing_ticker_values, start=start, end=end, horizon=horizon)
        merged_request = concat_frames(requested_cached, fetched)
        merged_request = dedupe_rows(merged_request, subset=["date", "asset_id"], sort_by=["date", "asset_id"])
        merged_request = slice_requested(
            frame=merged_request,
            tickers=normalized_tickers,
            start=start,
            end=end,
        )

        updated_cache = dedupe_rows(
            concat_frames(cache_frame, fetched),
            subset=["date", "asset_id"],
            sort_by=["date", "asset_id"],
        )
        self._write_cache_frame(frame=updated_cache, horizon=horizon)
        return merged_request.sort_values(["date", "asset_id"]).reset_index(drop=True)

    def _call_fetcher(
        self,
        *,
        tickers: list[str],
        start: str,
        end: str,
        horizon: Horizon,
    ) -> pd.DataFrame:
        try:
            frame = self._fetcher(tickers=tickers, start=start, end=end, horizon=horizon)
        except TypeError:
            frame = self._fetcher(tickers, start, end, horizon)

        required = {"date", "asset_id"}
        missing = required - set(frame.columns)
        if missing:
            msg = f"ETF features fetcher missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        return frame

    def _cache_file_path(self, horizon: Horizon) -> Path:
        safe_horizon = str(horizon.value).replace("-", "_").lower()
        return self._cache_dir / f"etf_features_{safe_horizon}.parquet"

    def _read_cache_frame(self, horizon: Horizon) -> pd.DataFrame:
        cache_path = self._cache_file_path(horizon)
        if not cache_path.exists():
            return pd.DataFrame(columns=["date", "asset_id"])

        frame = pd.read_parquet(cache_path)
        return normalize_asset_id_column(normalize_date_column(frame))

    def _write_cache_frame(self, frame: pd.DataFrame, horizon: Horizon) -> None:
        if self._cache_dir is None:
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_file_path(horizon)
        frame.to_parquet(cache_path, index=False)
