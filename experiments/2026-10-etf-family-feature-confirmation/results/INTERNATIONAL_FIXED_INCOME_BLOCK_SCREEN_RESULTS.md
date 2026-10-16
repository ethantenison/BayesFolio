# International Fixed Income Block Screen Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:-----------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| INTFI-TREND                  |            3 | 27,127,227 |  0.069444 | 0.133940 |           -2.658105 |           0.008210 | 0.833310 |     0.015839 |                            0.066623 |               0.015975 | True                | False                 |
| INTFI-ALL-LOCAL              |            3 | 27,127,227 | -0.006944 | 0.104859 |           -2.659330 |           0.012283 | 0.830601 |     0.003658 |                            0.066336 |               0.016188 | True                | False                 |
| INTFI-TREND+CREDIT-INFLATION |            3 | 27,127,227 |  0.048611 | 0.060141 |           -2.332169 |           0.000174 | 0.828407 |     0.012013 |                            0.066510 |               0.015137 | True                | False                 |
| INTFI-TREND+MARKET-RISK      |            3 | 27,127,227 |  0.159722 | 0.024056 |           -2.688399 |           0.008227 | 0.847268 |     0.009645 |                            0.067871 |               0.015314 | True                | True                  |
| INTFI-TREND+GLOBAL           |            3 | 27,127,227 |  0.229167 | 0.041667 |           -2.649867 |           0.016120 | 0.849766 |     0.005312 |                            0.068428 |               0.013867 | True                | True                  |
| INTFI-TREND+ALL-MACRO        |            3 | 27,127,227 |  0.243056 | 0.066970 |           -2.556656 |           0.004855 | 0.854791 |     0.012243 |                            0.068591 |               0.015009 | True                | True                  |
| INTFI-ALL-MATCH              |            3 | 27,127,227 |  0.236111 | 0.086736 |           -2.560539 |           0.006161 | 0.861886 |     0.013547 |                            0.069174 |               0.014546 | True                | True                  |

## Nondominated configurations

INTFI-TREND+MARKET-RISK, INTFI-TREND+GLOBAL, INTFI-TREND+ALL-MACRO, INTFI-ALL-MATCH

## Hypervolume

- Observed dominated hypervolume: `0.558858315`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BNDX, VWOB, HYEM form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
