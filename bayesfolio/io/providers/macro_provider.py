"""IO provider for date-indexed macro feature data.

Boundary responsibility: this module handles IO-layer retrieval, retry/fallback,
and cache orchestration for macro features, without engine business logic.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import requests

from bayesfolio.core.settings import Horizon

logger = logging.getLogger(__name__)


class MacroProvider:
    """Transitional provider for date-indexed macro features."""

    def __init__(
        self,
        fetcher: Callable[..., pd.DataFrame] | None = None,
        *,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        fallback_csv_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize provider with an injected fetch callable.

        Args:
            fetcher: Callable returning macro features with a ``date`` column.
            max_retries: Number of retry attempts for transient failures.
            retry_backoff_seconds: Linear backoff base in seconds between attempts.
            fallback_csv_path: Optional local CSV snapshot path used if live fetch
                fails after retries.
            cache_dir: Optional local directory for parquet cache files.
        """

        self._fetcher = fetcher
        self._max_retries = max(0, max_retries)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._fallback_csv_path = Path(fallback_csv_path) if fallback_csv_path is not None else None
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

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

        if self._cache_dir is None:
            logger.info("Fetching macro features from %s to %s.", start, end)
            return self._fetch_with_retries_or_fallback(start=start, end=end, horizon=horizon)

        cache_frame = self._read_cache_frame(horizon=horizon)
        if self._has_coverage(frame=cache_frame, start=start, end=end, horizon=horizon):
            logger.info("Using cached macro features from %s to %s.", start, end)
            return self._slice_requested(frame=cache_frame, start=start, end=end)

        logger.info("Macro cache miss/partial for %s to %s; fetching live data.", start, end)
        fetched = self._fetch_with_retries_or_fallback(start=start, end=end, horizon=horizon)
        updated_cache = self._dedupe_rows(self._concat_frames(cache_frame, fetched))
        self._write_cache_frame(frame=updated_cache, horizon=horizon)
        return self._slice_requested(frame=updated_cache, start=start, end=end)

    def _fetch_with_retries_or_fallback(self, *, start: str, end: str, horizon: Horizon) -> pd.DataFrame:
        logger.info("Fetching macro features from %s to %s.", start, end)

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._call_fetcher(start=start, end=end, horizon=horizon)
            except (requests.RequestException, TimeoutError) as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                sleep_seconds = self._retry_backoff_seconds * float(attempt + 1)
                logger.warning(
                    "Macro fetch timed out/failed (%s). Retrying in %.2fs (attempt %d/%d).",
                    exc,
                    sleep_seconds,
                    attempt + 1,
                    self._max_retries + 1,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            except Exception as exc:  # pragma: no cover - defensive pass-through
                last_error = exc
                if attempt >= self._max_retries:
                    break
                sleep_seconds = self._retry_backoff_seconds * float(attempt + 1)
                logger.warning(
                    "Macro fetch failed (%s). Retrying in %.2fs (attempt %d/%d).",
                    exc,
                    sleep_seconds,
                    attempt + 1,
                    self._max_retries + 1,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

        fallback = self._read_fallback_csv(start=start, end=end)
        if fallback is not None:
            logger.warning(
                "Using fallback macro CSV at %s after live fetch failure.",
                self._fallback_csv_path,
            )
            return fallback

        if last_error is not None:
            raise last_error
        msg = "Macro fetch failed without a captured error."
        raise RuntimeError(msg)

    def _call_fetcher(self, start: str, end: str, horizon: Horizon) -> pd.DataFrame:
        try:
            return self._fetcher(start=start, end=end, horizon=horizon)
        except TypeError:
            return self._fetcher(start=start, end=end)

    def _read_fallback_csv(self, start: str, end: str) -> pd.DataFrame | None:
        if self._fallback_csv_path is None or not self._fallback_csv_path.exists():
            return None

        frame = pd.read_csv(self._fallback_csv_path)
        if "date" not in frame.columns:
            return frame

        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        filtered = frame[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)].copy()
        return filtered.reset_index(drop=True)

    def _cache_file_path(self, horizon: Horizon) -> Path:
        safe_horizon = str(horizon.value).replace("-", "_").lower()
        return self._cache_dir / f"macro_{safe_horizon}.parquet"

    def _read_cache_frame(self, *, horizon: Horizon) -> pd.DataFrame:
        cache_path = self._cache_file_path(horizon)
        if not cache_path.exists():
            return pd.DataFrame(columns=["date"])

        frame = pd.read_parquet(cache_path)
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def _write_cache_frame(self, *, frame: pd.DataFrame, horizon: Horizon) -> None:
        if self._cache_dir is None:
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_file_path(horizon)
        frame.to_parquet(cache_path, index=False)

    def _slice_requested(self, *, frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        if frame.empty or "date" not in frame.columns:
            return frame.copy()

        output = frame.copy()
        output["date"] = pd.to_datetime(output["date"])
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return output[(output["date"] >= start_ts) & (output["date"] <= end_ts)].copy().reset_index(drop=True)

    def _has_coverage(self, *, frame: pd.DataFrame, start: str, end: str, horizon: Horizon) -> bool:
        if frame.empty or "date" not in frame.columns:
            return False

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        expected_dates = pd.DatetimeIndex(pd.date_range(start=start_ts, end=end_ts, freq=horizon.value))
        if len(expected_dates) == 0:
            return True

        available = pd.DatetimeIndex(pd.to_datetime(frame["date"]).drop_duplicates().sort_values())
        return bool(expected_dates.isin(available).all())

    def _dedupe_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        output = frame.copy()
        if "date" in output.columns:
            output["date"] = pd.to_datetime(output["date"])
            output = output.drop_duplicates(subset=["date"], keep="last")
            output = output.sort_values("date")
        return output.reset_index(drop=True)

    def _concat_frames(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        if left.empty:
            return right.copy()
        if right.empty:
            return left.copy()
        return pd.concat([left, right], ignore_index=True)
