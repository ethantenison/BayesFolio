# International Targeted Ablation Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:-----------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| I-BREADTH                    |            3 | 27,127,227 |  0.136111 | 0.041944 |           -1.861688 |           0.021903 | 1.340510 |     0.048731 |                            0.360539 |               0.037982 | True                | True                  |
| I-BREADTH+OIL                |            3 | 27,127,227 |  0.200000 | 0.030046 |           -1.781851 |           0.054191 | 1.342357 |     0.046904 |                            0.360394 |               0.040373 | True                | True                  |
| I-BREADTH+COMMODITIES        |            3 | 27,127,227 |  0.027778 | 0.034694 |           -1.867543 |           0.007396 | 1.213128 |     0.010333 |                            0.317689 |               0.036940 | True                | True                  |
| I-BREADTH+OIL DROP SPY-FLOW  |            3 | 27,127,227 |  0.188889 | 0.041107 |           -1.865049 |           0.074295 | 1.299568 |     0.055807 |                            0.345607 |               0.037545 | True                | True                  |
| I-BREADTH+OIL DROP ERP       |            3 | 27,127,227 |  0.241667 | 0.050690 |           -1.841202 |           0.039735 | 1.333582 |     0.084349 |                            0.361421 |               0.037971 | True                | True                  |
| I-BREADTH+OIL DROP RSP-SPY   |            3 | 27,127,227 |  0.002778 | 0.012729 |           -1.589168 |           0.088826 | 1.209956 |     0.015092 |                            0.315372 |               0.045764 | True                | False                 |
| I-BREADTH+OIL DROP PCT-ABOVE |            3 | 27,127,227 |  0.172222 | 0.055486 |           -1.775647 |           0.065727 | 1.275120 |     0.076445 |                            0.339453 |               0.040445 | True                | False                 |
| I-BREADTH+OIL DROP SPY-RET   |            3 | 27,127,227 |  0.219444 | 0.113141 |           -1.692986 |           0.092487 | 1.353966 |     0.042599 |                            0.367270 |               0.042188 | True                | True                  |
| I-BREADTH+EM-FX              |            3 | 27,127,227 |  0.055556 | 0.062546 |           -1.831303 |           0.021304 | 1.252026 |     0.055556 |                            0.327452 |               0.038906 | True                | False                 |
| I-BREADTH+DXY                |            3 | 27,127,227 |  0.136111 | 0.029266 |           -1.786251 |           0.035898 | 1.248828 |     0.010823 |                            0.337939 |               0.040696 | True                | False                 |

## Nondominated configurations

I-BREADTH, I-BREADTH+OIL, I-BREADTH+COMMODITIES, I-BREADTH+OIL DROP SPY-FLOW, I-BREADTH+OIL DROP ERP, I-BREADTH+OIL DROP SPY-RET

## Hypervolume

- Observed dominated hypervolume: `0.607223669`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `bea704cb64a42d1d9aeb112ed76ddb28e9487ca8cedbfcccd137f2ee993dde13`
- All requested runs passed: `True`
- Failures: `[]`
- VEA, VWO, VSS, and EWX form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
