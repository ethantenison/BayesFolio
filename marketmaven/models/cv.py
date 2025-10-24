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