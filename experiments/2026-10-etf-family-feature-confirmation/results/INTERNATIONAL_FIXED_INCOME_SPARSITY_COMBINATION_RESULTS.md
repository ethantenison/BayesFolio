# International Fixed Income Refinement Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                                              |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:-----------------------------------------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| INTFI-PRIMARY-NO-LAG-Y-EXCESS-LEAD                         |            3 | 27,127,227 |  0.402778 | 0.043368 |           -2.614493 |           0.011814 | 0.877914 |     0.007306 |                            0.070189 |               0.015585 | True                | True                  |
| INTFI-ALL-MACRO-NO-HY-CHANGE-NO-COPPER                     |            3 | 27,127,227 |  0.354167 | 0.036084 |           -2.601980 |           0.013482 | 0.870139 |     0.006510 |                            0.069877 |               0.015702 | True                | False                 |
| INTFI-TREND+MARKET-RISK                                    |            3 | 27,127,227 |  0.159722 | 0.024056 |           -2.688399 |           0.008227 | 0.847268 |     0.009645 |                            0.067871 |               0.015314 | True                | True                  |
| INTFI-PRIMARY-NO-LAG-NO-ERP                                |            3 | 27,127,227 |  0.430556 | 0.043368 |           -2.607603 |           0.005999 | 0.874801 |     0.013448 |                            0.070008 |               0.015199 | True                | True                  |
| INTFI-PRIMARY-NO-LAG-NO-CPI-MOM                            |            3 | 27,127,227 |  0.388889 | 0.024056 |           -2.627085 |           0.009960 | 0.875343 |     0.007732 |                            0.069971 |               0.015311 | True                | True                  |
| INTFI-PRIMARY-NO-LAG-NO-MA-SIGNAL                          |            3 | 27,127,227 |  0.395833 | 0.041667 |           -2.623165 |           0.013470 | 0.879086 |     0.008619 |                            0.070326 |               0.015358 | True                | True                  |
| INTFI-PRIMARY-NO-LAG-NO-SPY-FLOW                           |            3 | 27,127,227 |  0.430556 | 0.024056 |           -2.605596 |           0.005450 | 0.870346 |     0.006236 |                            0.069460 |               0.015653 | True                | False                 |
| INTFI-PRIMARY-NO-LAG-NO-GOLD-RATIO                         |            3 | 27,127,227 |  0.388889 | 0.043368 |           -2.623207 |           0.005333 | 0.879767 |     0.013571 |                            0.070392 |               0.015190 | True                | True                  |
| INTFI-PRIMARY-NO-LAG-NO-VIX                                |            3 | 27,127,227 |  0.388889 | 0.012028 |           -2.613620 |           0.007692 | 0.873745 |     0.009051 |                            0.069810 |               0.015556 | True                | False                 |
| INTFI-PRIMARY-NO-LAG-NO-ERP-NO-CPI-MOM                     |            3 | 27,127,227 |  0.430556 | 0.104859 |           -2.600355 |           0.006026 | 0.874153 |     0.015987 |                            0.069940 |               0.015306 | True                | False                 |
| INTFI-PRIMARY-NO-LAG-NO-ERP-NO-CPI-MOM-NO-GOLD-RATIO       |            3 | 27,127,227 |  0.319444 | 0.043368 |           -2.607666 |           0.001014 | 0.885193 |     0.009149 |                            0.070665 |               0.015074 | True                | True                  |
| INTFI-PRIMARY-NO-LAG-NO-MA-NO-ERP-NO-CPI-MOM               |            3 | 27,127,227 |  0.402778 | 0.060141 |           -2.614328 |           0.005296 | 0.855468 |     0.010431 |                            0.068712 |               0.014943 | True                | False                 |
| INTFI-PRIMARY-NO-LAG-NO-MA-NO-ERP-NO-CPI-MOM-NO-GOLD-RATIO |            3 | 27,127,227 |  0.298611 | 0.052429 |           -2.617427 |           0.015027 | 0.870999 |     0.020084 |                            0.069866 |               0.014825 | True                | False                 |
| INTFI-PRIMARY-NO-LAG-NO-VIX-NO-SPY-FLOW                    |            3 | 27,127,227 |  0.402778 | 0.031823 |           -2.607325 |           0.008230 | 0.869014 |     0.007110 |                            0.069358 |               0.015452 | True                | False                 |
| INTFI-PRIMARY-NO-LAG-NO-MA-NO-VIX-NO-SPY-FLOW              |            3 | 27,127,227 |  0.381944 | 0.043368 |           -2.615605 |           0.014683 | 0.862654 |     0.005657 |                            0.068905 |               0.015248 | True                | False                 |

## Nondominated configurations

INTFI-PRIMARY-NO-LAG-Y-EXCESS-LEAD, INTFI-TREND+MARKET-RISK, INTFI-PRIMARY-NO-LAG-NO-ERP, INTFI-PRIMARY-NO-LAG-NO-CPI-MOM, INTFI-PRIMARY-NO-LAG-NO-MA-SIGNAL, INTFI-PRIMARY-NO-LAG-NO-GOLD-RATIO, INTFI-PRIMARY-NO-LAG-NO-ERP-NO-CPI-MOM-NO-GOLD-RATIO

## Hypervolume

- Observed dominated hypervolume: `1.006937357`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BNDX, VWOB, HYEM form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
