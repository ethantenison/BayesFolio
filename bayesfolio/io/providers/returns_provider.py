"""IO provider for long-format excess return labels.

Boundary responsibility: this module handles retrieval and cache orchestration for
return labels in the IO layer, without engine business logic.
Returns are expressed in decimal units (0.02 means 2%).
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


class ReturnsProvider:
    """Transitional provider for long-format excess return labels.

    This provider intentionally does not import engine modules. A composition
    root can inject a legacy fetch callable during transitional migration.
    """

    def __init__(
        self,
        fetcher: Callable[..., pd.DataFrame],
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize provider with an injected fetch callable.

        Args:
            fetcher: Callable returning ``[date, asset_id, y_excess_lead]``.
            cache_dir: Optional local directory for parquet cache files.
        """

        self._fetcher: Callable[..., pd.DataFrame] = fetcher
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

    def get_y_excess_lead_long(
        self,
        tickers: list[str],
        start: str,
        end: str,
        horizon: Horizon,
        include_unlabeled_tail: bool = False,
    ) -> pd.DataFrame:
        """Fetch long-format target labels in decimal units.

        Args:
            tickers: Asset tickers.
            start: Inclusive start date in ISO format.
            end: Inclusive end date in ISO format.
            horizon: Frequency code (for example ``BME``).
            include_unlabeled_tail: If True, preserve the final period with NaN
                returns for forecasting workflows. Defaults to False (drops
                unlabeled tail for training).

        Returns:
            DataFrame with columns ``date``, ``asset_id``, ``y_excess_lead``
            where returns are decimal (``0.02`` means ``2%``).

        Raises:
            ValueError: If no fetcher is configured.
        """

        if self._fetcher is None:
            msg = "ReturnsProvider requires a fetcher callable."
            raise ValueError(msg)

        normalized_tickers = [str(ticker).upper() for ticker in tickers]
        if self._cache_dir is None:
            logger.info("Fetching return labels for %d tickers from %s to %s.", len(tickers), start, end)
            return self._call_fetcher(
                tickers=normalized_tickers,
                start=start,
                end=end,
                horizon=horizon,
                include_unlabeled_tail=include_unlabeled_tail,
            )

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
                "Using cached return labels for %d tickers from %s to %s.",
                len(normalized_tickers),
                start,
                end,
            )
            return requested_cached.sort_values(["date", "asset_id"]).reset_index(drop=True)

        logger.info(
            "Return cache partial/miss for %d tickers; fetching live data for %d tickers.",
            len(normalized_tickers),
            len(missing_ticker_values),
        )
        fetched = self._call_fetcher(
            tickers=missing_ticker_values,
            start=start,
            end=end,
            horizon=horizon,
            include_unlabeled_tail=include_unlabeled_tail,
        )
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
        include_unlabeled_tail: bool = False,
    ) -> pd.DataFrame:
        try:
            frame = self._fetcher(
                tickers=tickers,
                start=start,
                end=end,
                horizon=horizon,
                include_unlabeled_tail=include_unlabeled_tail,
            )
        except TypeError:
            try:
                frame = self._fetcher(tickers, start, end, horizon, include_unlabeled_tail)
            except TypeError:
                frame = self._fetcher(tickers, start, end, horizon)

        required = {"date", "asset_id", "y_excess_lead"}
        missing = required - set(frame.columns)
        if missing:
            msg = f"Returns fetcher missing required columns: {sorted(missing)}"
            raise ValueError(msg)
        return frame

    def _cache_file_path(self, horizon: Horizon) -> Path:
        assert self._cache_dir is not None
        safe_horizon = str(horizon.value).replace("-", "_").lower()
        return self._cache_dir / f"returns_{safe_horizon}.parquet"

    def _read_cache_frame(self, horizon: Horizon) -> pd.DataFrame:
        cache_path = self._cache_file_path(horizon)
        if not cache_path.exists():
            return pd.DataFrame(columns=["date", "asset_id", "y_excess_lead"])

        frame = pd.read_parquet(cache_path)
        return normalize_asset_id_column(normalize_date_column(frame))

    def _write_cache_frame(self, frame: pd.DataFrame, horizon: Horizon) -> None:
        if self._cache_dir is None:
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_file_path(horizon)
        frame.to_parquet(cache_path, index=False)
