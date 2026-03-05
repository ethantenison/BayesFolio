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
    ) -> None:
        """Initialize provider with an injected fetch callable.

        Args:
            fetcher: Callable returning macro features with a ``date`` column.
            max_retries: Number of retry attempts for transient failures.
            retry_backoff_seconds: Linear backoff base in seconds between attempts.
            fallback_csv_path: Optional local CSV snapshot path used if live fetch
                fails after retries.
        """

        self._fetcher = fetcher
        self._max_retries = max(0, max_retries)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._fallback_csv_path = Path(fallback_csv_path) if fallback_csv_path is not None else None

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
