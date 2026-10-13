# U.S. Feature Confirmation Results

Seeds: `27`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional annualized
Sharpe across seeds. The monthly return series is already risk-free adjusted. Return and RMSE are
diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration   |   seed_count |   seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:----------------|-------------:|--------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| U-AC            |            1 |      27 |  0.116667 |      nan |           -1.653393 |                nan | 1.205594 |          nan |                            0.354665 |               0.045245 | True                | True                  |
| U-C005          |            1 |      27 |  0.233333 |      nan |           -1.683705 |                nan | 1.141345 |          nan |                            0.318381 |               0.045599 | True                | True                  |
| U-C009          |            1 |      27 |  0.108333 |      nan |           -1.703157 |                nan | 1.014353 |          nan |                            0.272650 |               0.045852 | True                | True                  |
| U-ALL-MATCH     |            1 |      27 |  0.258333 |      nan |           -1.636080 |                nan | 1.287046 |          nan |                            0.375740 |               0.046269 | True                | True                  |

## Nondominated configurations

U-AC, U-C005, U-C009, U-ALL-MATCH

## Hypervolume

- Observed dominated hypervolume: `0.558925611`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.
- NLL is negated for the maximization-space calculation; the reference is frozen for subsequent rounds.

## Validation

- Common date hash: `bea704cb64a42d1d9aeb112ed76ddb28e9487ca8cedbfcccd137f2ee993dde13`
- All requested runs passed: `True`
- Failures: `[]`
- IJR is excluded and t_index is present in every model.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
