# International Targeted Ablation Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration                           |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:----------------------------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| I-BREADTH                               |            3 | 27,127,227 |  0.136111 | 0.041944 |           -1.861688 |           0.021903 | 1.340510 |     0.048731 |                            0.360539 |               0.037982 | True                | False                 |
| I-BREADTH+OIL                           |            3 | 27,127,227 |  0.200000 | 0.030046 |           -1.781851 |           0.054191 | 1.342357 |     0.046904 |                            0.360394 |               0.040373 | True                | False                 |
| I-BREADTH+COMMODITIES                   |            3 | 27,127,227 |  0.027778 | 0.034694 |           -1.867543 |           0.007396 | 1.213128 |     0.010333 |                            0.317689 |               0.036940 | True                | False                 |
| I-BREADTH+OIL DROP SPY-FLOW             |            3 | 27,127,227 |  0.188889 | 0.041107 |           -1.865049 |           0.074295 | 1.299568 |     0.055807 |                            0.345607 |               0.037545 | True                | False                 |
| I-BREADTH+OIL DROP ERP                  |            3 | 27,127,227 |  0.241667 | 0.050690 |           -1.841202 |           0.039735 | 1.333582 |     0.084349 |                            0.361421 |               0.037971 | True                | False                 |
| I-BREADTH+OIL DROP SPY-RET              |            3 | 27,127,227 |  0.219444 | 0.113141 |           -1.692986 |           0.092487 | 1.353966 |     0.042599 |                            0.367270 |               0.042188 | True                | False                 |
| I-BREADTH DROP ERP                      |            3 | 27,127,227 |  0.213889 | 0.019245 |           -1.781311 |           0.025996 | 1.395827 |     0.046474 |                            0.372743 |               0.038653 | True                | True                  |
| I-BREADTH+OIL DROP ERP+SPY-RET          |            3 | 27,127,227 |  0.161111 | 0.045896 |           -1.654478 |           0.105273 | 1.351071 |     0.064944 |                            0.367927 |               0.042868 | True                | False                 |
| I-BREADTH+OIL DROP ERP+SPY-FLOW         |            3 | 27,127,227 |  0.252778 | 0.070874 |           -1.917061 |           0.053409 | 1.382449 |     0.039470 |                            0.368493 |               0.036124 | True                | True                  |
| I-BREADTH+OIL DROP ERP+SPY-RET+SPY-FLOW |            3 | 27,127,227 |  0.075000 | 0.036324 |           -1.699727 |           0.121225 | 1.301757 |     0.005006 |                            0.347241 |               0.042262 | True                | False                 |
| I-BREADTH+COMMODITIES DROP ERP          |            3 | 27,127,227 |  0.130556 | 0.029266 |           -1.903091 |           0.021799 | 1.273767 |     0.043088 |                            0.342931 |               0.036107 | True                | False                 |
| I-BREADTH DROP SPY-RET                  |            3 | 27,127,227 |  0.022222 | 0.052924 |           -1.755382 |           0.010761 | 1.303634 |     0.040352 |                            0.349278 |               0.041437 | True                | False                 |

## Nondominated configurations

I-BREADTH DROP ERP, I-BREADTH+OIL DROP ERP+SPY-FLOW

## Hypervolume

- Observed dominated hypervolume: `0.675018493`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `bea704cb64a42d1d9aeb112ed76ddb28e9487ca8cedbfcccd137f2ee993dde13`
- All requested runs passed: `True`
- Failures: `[]`
- VEA, VWO, VSS, and EWX form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
