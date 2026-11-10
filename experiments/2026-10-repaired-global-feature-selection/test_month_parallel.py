"""Guard global concurrency, original month identities and strict portfolio failure."""

from __future__ import annotations

import month_parallel_worker as worker
import numpy as np
import pandas as pd
import pytest
import run_month_parallel as pipeline


def test_global_worker_ceiling() -> None:
    """Fourteen is a global limit on a sixteen-core machine, never per seed."""
    assert pipeline.worker_limit(14, 16) == 14
    with pytest.raises(ValueError):
        pipeline.worker_limit(15, 16)
    assert pipeline.worker_limit(14, 16, 4 * 1024**3) == 12


def test_small_machine_respects_headroom() -> None:
    """The requested pool must not consume more than CPU count minus two."""
    with pytest.raises(ValueError):
        pipeline.worker_limit(14, 8)
    assert pipeline.worker_limit(6, 8) == 6


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2017-05-31", periods=108, freq="BME")
    return pd.DataFrame([{"date": date, "asset_id": ticker, "y_excess_lead": 0.01}
                         for date in dates for ticker in pipeline.data.TICKERS])


def test_asof_slices_and_original_window_index() -> None:
    """Each job uses only its original 84+k historical dates, not scheduling order."""
    frame = _panel()
    train, evaluation = worker.training_panel(frame, "2024-07-31", 2)
    assert train.date.nunique() == 86 and len(train) == 86 * 15 and len(evaluation) == 15
    assert train.date.max() < evaluation.date.min()
    future = frame.date.ge(pd.Timestamp("2024-07-31"))
    frame.loc[future, "y_excess_lead"] = 1e9
    other, _ = worker.training_panel(frame, "2024-07-31", 2)
    pd.testing.assert_frame_equal(train, other)


def test_wrong_index_fails_closed() -> None:
    """Renumbering a subset of months cannot silently change the training contract."""
    with pytest.raises(ValueError, match="84.k"):
        worker.training_panel(_panel(), "2024-07-31", 0)


def test_all_seed_month_jobs_are_unique() -> None:
    """The scheduler has 72 distinct jobs with stable original date indices."""
    dates = pd.date_range("2024-05-31", periods=24, freq="BME").strftime("%Y-%m-%d").tolist()
    jobs = pipeline._jobs("baseline", dates)
    assert len(jobs) == len({(job["seed"], job["date"]) for job in jobs}) == 72
    assert all(job["index"] == dates.index(job["date"]) for job in jobs)


def test_solver_failure_does_not_supply_weights() -> None:
    """Failed solves raise after one call instead of trying a relaxed or fallback portfolio."""
    calls = []

    def failed(panel, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("no solution")

    frame = pd.DataFrame(np.ones((30, 15)), columns=pipeline.data.TICKERS)
    with pytest.raises(RuntimeError, match="no fallback"):
        worker.strict_solve(failed, frame, upperlng=0.35, nea=3)
    assert calls == [{"upperlng": 0.35, "nea": 3}]


def test_invalid_solver_weights_fail_closed() -> None:
    """A malformed solution is never repaired with previous or equal weights."""
    frame = pd.DataFrame(np.ones((30, 15)), columns=pipeline.data.TICKERS)
    with pytest.raises(RuntimeError, match="Invalid portfolio weights"):
        worker.strict_solve(lambda panel: pd.Series(0.0, index=panel.columns), frame)


def test_missing_assets_are_not_dropped() -> None:
    """Portfolio failures must not be hidden by changing the investment universe."""
    frame = pd.DataFrame(np.ones((30, 15)), columns=pipeline.data.TICKERS)
    frame.iloc[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="dropping assets"):
        worker.strict_solve(lambda panel: pd.Series(1 / 15, index=panel.columns), frame)
