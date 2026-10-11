# Failed Run

This smoke run failed before GP scenario artifacts were written.

- Feature set: `momentum_trend`
- Config: signed LKJ MTGP rank 2, one scored window, maxiter 3
- Failure: the reused covariance builder attempted to create a macro block with zero ARD dimensions.
- Error: `ValueError: ard_num_dims must be >= 1`
- Resolution: `run_first_round_walkforward.py` now builds covariance blocks dynamically and skips empty ETF/macro blocks and interactions.

