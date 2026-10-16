# International Fixed Income Refinement Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                                        |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:-----------------------------------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| INTFI-PRIMARY-NO-LAG-Y-EXCESS-LEAD                   |            3 | 27,127,227 |  0.402778 | 0.043368 |           -2.614493 |           0.011814 | 0.877914 |     0.007306 |                            0.070189 |               0.015585 | True                | True                  |
| INTFI-PRIMARY-NO-LAG-NO-GOLD-RATIO                   |            3 | 27,127,227 |  0.388889 | 0.043368 |           -2.623207 |           0.005333 | 0.879767 |     0.013571 |                            0.070392 |               0.015190 | True                | False                 |
| INTFI-PRIMARY-NO-LAG-NO-MA-SIGNAL                    |            3 | 27,127,227 |  0.395833 | 0.041667 |           -2.623165 |           0.013470 | 0.879086 |     0.008619 |                            0.070326 |               0.015358 | True                | True                  |
| INTFI-PRIMARY-NO-LAG-NO-ERP-NO-CPI-MOM-NO-GOLD-RATIO |            3 | 27,127,227 |  0.319444 | 0.043368 |           -2.607666 |           0.001014 | 0.885193 |     0.009149 |                            0.070665 |               0.015074 | True                | True                  |
| INTFI-TREND+MARKET-RISK                              |            3 | 27,127,227 |  0.159722 | 0.024056 |           -2.688399 |           0.008227 | 0.847268 |     0.009645 |                            0.067871 |               0.015314 | True                | True                  |
| INTFI-PRIMARY-NO-LAG-NO-MA-NO-GOLD                   |            3 | 27,127,227 |  0.402778 | 0.043368 |           -2.632185 |           0.009446 | 0.874315 |     0.008222 |                            0.070082 |               0.015078 | True                | True                  |
| INTFI-PRIMARY-NO-LAG-NO-ERP-NO-GOLD                  |            3 | 27,127,227 |  0.312500 | 0.036084 |           -2.612626 |           0.010757 | 0.878214 |     0.016042 |                            0.070243 |               0.015177 | True                | False                 |
| INTFI-PRIMARY-NO-LAG-NO-CPI-NO-GOLD                  |            3 | 27,127,227 |  0.388889 | 0.024056 |           -2.633392 |           0.014193 | 0.883191 |     0.004387 |                            0.070512 |               0.015134 | True                | True                  |
| INTFI-PRIMARY-NO-LAG-NO-MA-NO-ERP                    |            3 | 27,127,227 |  0.409722 | 0.122072 |           -2.618535 |           0.005895 | 0.871866 |     0.013305 |                            0.069719 |               0.015061 | True                | True                  |
| INTFI-PRIMARY-NO-LAG-NO-MA-NO-CPI                    |            3 | 27,127,227 |  0.354167 | 0.036084 |           -2.632567 |           0.004266 | 0.869397 |     0.013678 |                            0.069513 |               0.015210 | True                | False                 |
| INTFI-PRIMARY-NO-LAG-NO-MA-NO-ERP-NO-GOLD            |            3 | 27,127,227 |  0.305556 | 0.052429 |           -2.628490 |           0.004558 | 0.857918 |     0.007086 |                            0.068890 |               0.014974 | True                | False                 |
| INTFI-PRIMARY-NO-LAG-NO-MA-NO-CPI-NO-GOLD            |            3 | 27,127,227 |  0.368056 | 0.012028 |           -2.630708 |           0.009714 | 0.867753 |     0.017028 |                            0.069573 |               0.014873 | True                | False                 |

## Nondominated configurations

INTFI-PRIMARY-NO-LAG-Y-EXCESS-LEAD, INTFI-PRIMARY-NO-LAG-NO-MA-SIGNAL, INTFI-PRIMARY-NO-LAG-NO-ERP-NO-CPI-MOM-NO-GOLD-RATIO, INTFI-TREND+MARKET-RISK, INTFI-PRIMARY-NO-LAG-NO-MA-NO-GOLD, INTFI-PRIMARY-NO-LAG-NO-CPI-NO-GOLD, INTFI-PRIMARY-NO-LAG-NO-MA-NO-ERP

## Hypervolume

- Observed dominated hypervolume: `0.961554445`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BNDX, VWOB, HYEM form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
