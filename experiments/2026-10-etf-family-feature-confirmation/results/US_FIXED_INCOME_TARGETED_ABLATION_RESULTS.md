# Us Fixed Income Targeted Ablation Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration               |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:----------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| USFI-TREND                  |            3 | 27,127,227 |  0.147222 | 0.033679 |           -2.673565 |           0.007682 | 0.429514 |     0.024294 |                            0.034860 |               0.015503 | True                | True                  |
| USFI-TREND+CREDIT-INFLATION |            3 | 27,127,227 |  0.152778 | 0.082215 |           -2.302203 |           0.026303 | 0.358433 |     0.011712 |                            0.029019 |               0.016336 | True                | False                 |
| USFI-TREND+HY-LEVEL         |            3 | 27,127,227 |  0.147222 | 0.017347 |           -2.434908 |           0.010288 | 0.345388 |     0.008899 |                            0.028114 |               0.017910 | True                | False                 |
| USFI-TREND+HY-CHANGE        |            3 | 27,127,227 |  0.055556 | 0.026788 |           -2.259920 |           0.011025 | 0.365980 |     0.049619 |                            0.029988 |               0.014997 | True                | False                 |
| USFI-TREND+HY-Z             |            3 | 27,127,227 |  0.036111 | 0.037577 |           -2.307714 |           0.011114 | 0.394954 |     0.052823 |                            0.032910 |               0.015499 | True                | False                 |
| USFI-TREND+INFLATION        |            3 | 27,127,227 | -0.041667 | 0.036324 |           -2.580415 |           0.009899 | 0.272215 |     0.023357 |                            0.022155 |               0.017482 | True                | False                 |
| USFI-TREND+HY-CHANGE-Z      |            3 | 27,127,227 |  0.063889 | 0.039382 |           -2.186438 |           0.029488 | 0.347171 |     0.075983 |                            0.028495 |               0.015755 | True                | False                 |
| USFI-TREND-NO-LAG           |            3 | 27,127,227 |  0.183333 | 0.054645 |           -2.529196 |           0.016369 | 0.395310 |     0.056405 |                            0.032608 |               0.018035 | True                | False                 |
| USFI-TREND-NO-MOM36         |            3 | 27,127,227 |  0.205556 | 0.055486 |           -2.619566 |           0.048283 | 0.443697 |     0.013657 |                            0.036133 |               0.015978 | True                | True                  |

## Nondominated configurations

USFI-TREND, USFI-TREND-NO-MOM36

## Hypervolume

- Observed dominated hypervolume: `0.242330509`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BND, IEF, LQD, HYG form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
