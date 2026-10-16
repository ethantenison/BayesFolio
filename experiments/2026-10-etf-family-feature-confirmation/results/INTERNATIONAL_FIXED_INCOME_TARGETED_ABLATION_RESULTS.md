# International Fixed Income Targeted Ablation Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                  |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:-------------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| INTFI-TREND+MARKET-RISK        |            3 | 27,127,227 |  0.159722 | 0.024056 |           -2.688399 |           0.008227 | 0.847268 |     0.009645 |                            0.067871 |               0.015314 | True                | True                  |
| INTFI-TREND+GLOBAL             |            3 | 27,127,227 |  0.229167 | 0.041667 |           -2.649867 |           0.016120 | 0.849766 |     0.005312 |                            0.068428 |               0.013867 | True                | True                  |
| INTFI-TREND+ALL-MACRO          |            3 | 27,127,227 |  0.243056 | 0.066970 |           -2.556656 |           0.004855 | 0.854791 |     0.012243 |                            0.068591 |               0.015009 | True                | True                  |
| INTFI-ALL-MATCH                |            3 | 27,127,227 |  0.236111 | 0.086736 |           -2.560539 |           0.006161 | 0.861886 |     0.013547 |                            0.069174 |               0.014546 | True                | True                  |
| INTFI-TREND+MARKET+GLOBAL      |            3 | 27,127,227 |  0.013889 | 0.063647 |           -2.659976 |           0.005034 | 0.818006 |     0.006627 |                            0.065423 |               0.015171 | True                | False                 |
| INTFI-TREND+VIX                |            3 | 27,127,227 |  0.069444 | 0.043368 |           -2.634494 |           0.020434 | 0.845989 |     0.007402 |                            0.067710 |               0.016780 | True                | False                 |
| INTFI-TREND+EQUITY-RISK        |            3 | 27,127,227 |  0.062500 | 0.062500 |           -2.631626 |           0.011108 | 0.850382 |     0.009556 |                            0.068254 |               0.017250 | True                | True                  |
| INTFI-TREND+EM-FX              |            3 | 27,127,227 | -0.055556 | 0.043368 |           -2.633875 |           0.021125 | 0.831965 |     0.021027 |                            0.066725 |               0.016814 | True                | False                 |
| INTFI-TREND+COMMODITY-RET      |            3 | 27,127,227 | -0.083333 | 0.020833 |           -2.710271 |           0.023027 | 0.819405 |     0.012577 |                            0.065428 |               0.015601 | True                | True                  |
| INTFI-TREND+GOLD-OIL           |            3 | 27,127,227 |  0.013889 | 0.043368 |           -2.584233 |           0.007822 | 0.841802 |     0.013893 |                            0.067253 |               0.016372 | True                | False                 |
| INTFI-TREND+GLOBAL+VIX         |            3 | 27,127,227 |  0.125000 | 0.020833 |           -2.662574 |           0.004604 | 0.831339 |     0.010303 |                            0.066734 |               0.014796 | True                | False                 |
| INTFI-TREND+GLOBAL+EQUITY-RISK |            3 | 27,127,227 |  0.152778 | 0.031823 |           -2.635306 |           0.005254 | 0.840890 |     0.004372 |                            0.067155 |               0.015613 | True                | False                 |

## Nondominated configurations

INTFI-TREND+MARKET-RISK, INTFI-TREND+GLOBAL, INTFI-TREND+ALL-MACRO, INTFI-ALL-MATCH, INTFI-TREND+EQUITY-RISK, INTFI-TREND+COMMODITY-RET

## Hypervolume

- Observed dominated hypervolume: `0.558861053`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `094ad6740466edb33fbbfd9c354092c1d8f87ad1a8076d380148f0bb2e01bf33`
- All requested runs passed: `True`
- Failures: `[]`
- BNDX, VWOB, HYEM form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
