# Us Fixed Income Block Screen Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration               |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:----------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| USFI-TREND                  |            3 | 27,127,227 |  0.147222 | 0.033679 |           -2.673565 |           0.007682 | 0.429514 |     0.024294 |                            0.034860 |               0.015503 | True                | True                  |
| USFI-ALL-LOCAL              |            3 | 27,127,227 |  0.144444 | 0.017347 |           -2.623942 |           0.070102 | 0.404338 |     0.041958 |                            0.032954 |               0.015880 | True                | False                 |
| USFI-TREND+CREDIT-INFLATION |            3 | 27,127,227 |  0.152778 | 0.082215 |           -2.302203 |           0.026303 | 0.358433 |     0.011712 |                            0.029019 |               0.016336 | True                | True                  |
| USFI-TREND+MARKET-RISK      |            3 | 27,127,227 |  0.025000 | 0.008333 |           -2.509732 |           0.014705 | 0.306427 |     0.022732 |                            0.024889 |               0.018342 | True                | False                 |
| USFI-TREND+GLOBAL           |            3 | 27,127,227 |  0.125000 | 0.022048 |           -2.577503 |           0.014902 | 0.363953 |     0.025336 |                            0.028518 |               0.015181 | True                | False                 |
| USFI-TREND+ALL-MACRO        |            3 | 27,127,227 |  0.061111 | 0.004811 |           -2.576329 |           0.008267 | 0.310994 |     0.051263 |                            0.024358 |               0.015712 | True                | False                 |
| USFI-ALL-MATCH              |            3 | 27,127,227 | -0.000000 | 0.041667 |           -2.575279 |           0.016073 | 0.297157 |     0.060957 |                            0.023600 |               0.016002 | True                | False                 |

## Nondominated configurations

USFI-TREND, USFI-TREND+CREDIT-INFLATION

## Hypervolume

- Observed dominated hypervolume: `0.173644663`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BND, IEF, LQD, HYG form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
