# International Targeted Ablation Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional
annualized Sharpe. Cumulative return and RMSE are diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration         |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:----------------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| I-BREADTH             |            3 | 27,127,227 |  0.136111 | 0.041944 |           -1.861688 |           0.021903 | 1.340510 |     0.048731 |                            0.360539 |               0.037982 | True                | True                  |
| I-BREADTH+FX          |            3 | 27,127,227 |  0.050000 | 0.044096 |           -1.745432 |           0.069499 | 1.253096 |     0.091994 |                            0.336868 |               0.041354 | True                | False                 |
| I-BREADTH+INFLATION   |            3 | 27,127,227 |  0.050000 | 0.038188 |           -1.753976 |           0.057827 | 1.201915 |     0.037419 |                            0.316682 |               0.039718 | True                | False                 |
| I-BREADTH+CREDIT      |            3 | 27,127,227 |  0.072222 | 0.053576 |           -1.796184 |           0.037034 | 1.263735 |     0.091912 |                            0.344036 |               0.039700 | True                | False                 |
| I-BREADTH+COMMODITIES |            3 | 27,127,227 |  0.027778 | 0.034694 |           -1.867543 |           0.007396 | 1.213128 |     0.010333 |                            0.317689 |               0.036940 | True                | True                  |

## Nondominated configurations

I-BREADTH, I-BREADTH+COMMODITIES

## Hypervolume

- Observed dominated hypervolume: `0.339877795`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- Common date hash: `bea704cb64a42d1d9aeb112ed76ddb28e9487ca8cedbfcccd137f2ee993dde13`
- All requested runs passed: `True`
- Failures: `[]`
- VEA, VWO, VSS, and EWX form the exact task universe; t_index is present throughout.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
