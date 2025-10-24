"""Cross Validation"""
from typing import Iterator, Tuple
import numpy as np
import pandas as pd

def rolling_time_splits(
    dates: pd.DatetimeIndex,
    train_min: int,           # min train length in periods
    step: int,                # how far to advance origin each split
    horizon: int = 1,         # forecast horizon (steps ahead)
    embargo: int = 0,         # gap between train end and test start
    train_max: int | None = None,  # optional cap for sliding window
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yields (train_idx, test_idx) index arrays for walk-forward splits.
    Test is a single timestamp at t+h (or a block if you prefer multi-step).
    """
    n = len(dates)
    start = train_min
    while True:
        train_end = start
        # windowing
        if train_max is None:
            train_start = 0                         # expanding
        else:
            train_start = max(0, train_end - train_max)  # sliding
        
        test_time = train_end + embargo + horizon - 1
        if test_time >= n: 
            break
        
        train_idx = np.arange(train_start, train_end)  # [0 ... train_end-1]
        test_idx  = np.array([test_time])              # predict y at t+h
        
        yield train_idx, test_idx
        start += step
        

def rolling_time_splits_multitask(
    df: pd.DataFrame,
    date_col: str,
    asset_col: str,
    train_min: int,
    step: int,
    horizon: int = 1,
    embargo: int = 0,
    train_max: int | None = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Rolling time-series splits for multitask (panel) data.

    Each split includes equal numbers of samples per task (e.g., if there are 3 tasks
    and train_min=60, each split will contain 60 * 3 training rows).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'date' and 'asset_id' columns.
    date_col : str
        Name of the date column.
    asset_col : str
        Name of the task/asset identifier column.
    train_min : int
        Minimum number of time steps per task in training.
    step : int
        How far to move the rolling window each iteration (in time steps).
    horizon : int
        Forecast horizon in time steps.
    embargo : int
        Gap (in time steps) between training end and testing start.
    train_max : int | None
        Optional maximum training window per task (for sliding windows).
    
    Yields
    ------
    (train_idx, test_idx) : Tuple[np.ndarray, np.ndarray]
        Row indices for training and testing in the input DataFrame.
    """

    # Ensure dates are sorted
    df = df.sort_values(by=[date_col, asset_col]).reset_index(drop=True)
    unique_dates = pd.DatetimeIndex(sorted(df[date_col].unique()))
    n_dates = len(unique_dates)

    start = train_min
    while True:
        train_end = start

        if train_max is None:
            train_start = 0
        else:
            train_start = max(0, train_end - train_max)

        test_time = train_end + embargo + horizon - 1
        if test_time >= n_dates:
            break

        # Dates included in this split
        train_dates = unique_dates[train_start:train_end]
        test_date = unique_dates[test_time]

        # Select rows matching the desired time steps
        train_idx = df.index[df[date_col].isin(train_dates)].to_numpy()
        test_idx = df.index[df[date_col] == test_date].to_numpy()

        # Sanity check: equal number of rows per task
        n_train_per_task = df.loc[train_idx, asset_col].value_counts().values
        if not np.all(n_train_per_task == n_train_per_task[0]):
            raise ValueError("Unequal task representation in split. Check date alignment per task.")

        yield train_idx, test_idx
        start += step