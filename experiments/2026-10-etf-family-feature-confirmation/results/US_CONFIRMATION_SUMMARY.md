# U.S. Feature Confirmation Results

Seeds: `27,127,227`.

Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher mean conventional annualized
Sharpe across seeds. The monthly return series is already risk-free adjusted. Return and RMSE are
diagnostic-only; turnover is excluded.

## Configuration aggregates

| configuration   |   seed_count |      seeds |   mean_ic |   std_ic |   mean_gaussian_nll |   std_gaussian_nll |   sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | validation_passed   | pareto_nondominated   |
|:----------------|-------------:|-----------:|----------:|---------:|--------------------:|-------------------:|---------:|-------------:|------------------------------------:|-----------------------:|:--------------------|:----------------------|
| U-AC            |            3 | 27,127,227 |  0.130556 | 0.031549 |           -1.653460 |           0.026340 | 1.269129 |     0.096272 |                            0.366784 |               0.046139 | True                | False                 |
| U-C005          |            3 | 27,127,227 |  0.272222 | 0.053576 |           -1.705886 |           0.022156 | 1.217001 |     0.104262 |                            0.340540 |               0.045260 | True                | True                  |
| U-C009          |            3 | 27,127,227 |  0.075000 | 0.038188 |           -1.713216 |           0.020266 | 1.151083 |     0.119961 |                            0.307945 |               0.045187 | True                | True                  |
| U-ALL-MATCH     |            3 | 27,127,227 |  0.227778 | 0.029266 |           -1.658456 |           0.019505 | 1.334647 |     0.055214 |                            0.385601 |               0.046199 | True                | True                  |

## Nondominated configurations

U-C005, U-C009, U-ALL-MATCH

## Hypervolume

- Observed dominated hypervolume: `0.610225830`
- Fixed raw-objective reference: IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.
- NLL is negated for the maximization-space calculation; the reference is frozen for subsequent rounds.

## Validation

- Common date hash: `bea704cb64a42d1d9aeb112ed76ddb28e9487ca8cedbfcccd137f2ee993dde13`
- All requested runs passed: `True`
- Failures: `[]`
- IJR is excluded and t_index is present in every model.
- Rank 2, maxiter 15, 256 scenarios, and the 24-date panel are fixed.
