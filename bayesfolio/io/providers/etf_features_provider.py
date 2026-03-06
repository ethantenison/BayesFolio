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
        requested_cached = self._slice_requested(
            frame=cache_frame,
            tickers=normalized_tickers,
            start=start,
            end=end,
        )
        missing_tickers = self._missing_tickers(
            cache_frame=cache_frame,
            tickers=normalized_tickers,
            start=start,
            end=end,
            horizon=horizon,
        )

        if not missing_tickers:
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
            len(missing_tickers),
        )
        fetched = self._call_fetcher(tickers=missing_tickers, start=start, end=end, horizon=horizon)
        merged_request = self._concat_frames(requested_cached, fetched)
        merged_request = self._dedupe_rows(merged_request)
        merged_request = self._slice_requested(
            frame=merged_request,
            tickers=normalized_tickers,
            start=start,
            end=end,
        )

        updated_cache = self._dedupe_rows(self._concat_frames(cache_frame, fetched))
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
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        if "asset_id" in frame.columns:
            frame["asset_id"] = frame["asset_id"].astype(str).str.upper()
        return frame

    def _write_cache_frame(self, frame: pd.DataFrame, horizon: Horizon) -> None:
        if self._cache_dir is None:
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_file_path(horizon)
        frame.to_parquet(cache_path, index=False)

    def _slice_requested(
        self,
        *,
        frame: pd.DataFrame,
        tickers: list[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame

        output = frame.copy()
        output["date"] = pd.to_datetime(output["date"])
        output["asset_id"] = output["asset_id"].astype(str).str.upper()
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return output[
            output["asset_id"].isin(tickers) & (output["date"] >= start_ts) & (output["date"] <= end_ts)
        ].copy()

    def _missing_tickers(
        self,
        *,
        cache_frame: pd.DataFrame,
        tickers: list[str],
        start: str,
        end: str,
        horizon: Horizon,
    ) -> list[str]:
        if cache_frame.empty:
            return tickers

        frame = cache_frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["asset_id"] = frame["asset_id"].astype(str).str.upper()
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        expected_dates = pd.DatetimeIndex(pd.date_range(start=start_ts, end=end_ts, freq=horizon.value))

        missing: list[str] = []
        for ticker in tickers:
            per_ticker = frame.loc[
                (frame["asset_id"] == ticker) & (frame["date"] >= start_ts) & (frame["date"] <= end_ts),
                "date",
            ]
            if per_ticker.empty:
                missing.append(ticker)
                continue

            if len(expected_dates) == 0:
                continue
            available = pd.DatetimeIndex(per_ticker.drop_duplicates().sort_values())
            if not expected_dates.isin(available).all():
                missing.append(ticker)
        return missing

    def _dedupe_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        output = frame.copy()
        output["date"] = pd.to_datetime(output["date"])
        output["asset_id"] = output["asset_id"].astype(str).str.upper()
        output = output.drop_duplicates(subset=["date", "asset_id"], keep="last")
        return output.sort_values(["date", "asset_id"]).reset_index(drop=True)

    def _concat_frames(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        if left.empty:
            return right.copy()
        if right.empty:
            return left.copy()
        return pd.concat([left, right], ignore_index=True)
