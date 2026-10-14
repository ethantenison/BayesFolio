# Next U.S. ETF Experiment Batch Results

All candidates use the frozen 24-window panel and seeds `27,127,227`.
The Pareto objectives are higher mean IC, lower mean Gaussian NLL, and higher conventional annualized
Sharpe. Cumulative return and RMSE are recorded only as diagnostics.

## Candidate aggregates and individual promotion decisions

| configuration               | mode    |   priority |   seed_count |      seeds |     mean_ic |      std_ic |   mean_gaussian_nll |   std_gaussian_nll |      sharpe |   std_sharpe |   cumulative_return_diagnostic_only |   rmse_diagnostic_only | pareto_nondominated   | nondominated_vs_incumbents   |   hypervolume_with_incumbents |   incremental_hypervolume | promoted   |
|:----------------------------|:--------|-----------:|-------------:|-----------:|------------:|------------:|--------------------:|-------------------:|------------:|-------------:|------------------------------------:|-----------------------:|:----------------------|:-----------------------------|------------------------------:|--------------------------:|:-----------|
| all_match_redundancy_pruned | exploit |          1 |            3 | 27,127,227 | 0.213888889 | 0.029265705 |        -1.664956781 |        0.011603309 | 1.335337977 |  0.053257473 |                         0.386084807 |            0.046288239 | True                  | True                         |                   0.610635327 |               0.000409497 | True       |
| c005_three_way              | exploit |          2 |            3 | 27,127,227 | 0.263888889 | 0.012729377 |        -1.673605278 |        0.021528463 | 1.168121907 |  0.177495651 |                         0.337567053 |            0.045256092 | True                  | False                        |                   0.610225830 |               0.000000000 | False      |
| c005_breadth_flow_bridge    | bridge  |          3 |            3 | 27,127,227 | 0.172222222 | 0.083887049 |        -1.597127148 |        0.047984676 | 1.133362637 |  0.089839483 |                         0.319929413 |            0.048836739 | False                 | False                        |                   0.610225830 |               0.000000000 | False      |
| target_macro_revisit        | explore |          4 |            3 | 27,127,227 | 0.255555556 | 0.058531410 |        -1.643258805 |        0.018357135 | 1.185021694 |  0.025354709 |                         0.333480612 |            0.046456330 | True                  | False                        |                   0.610225830 |               0.000000000 | False      |

A candidate is promoted only when it is nondominated against the incumbents and its individual insertion
strictly increases fixed-reference hypervolume.

## Combined Pareto front

| configuration               |     mean_ic |   mean_gaussian_nll |      sharpe | pareto_nondominated   |
|:----------------------------|------------:|--------------------:|------------:|:----------------------|
| U-C005                      | 0.272222222 |        -1.705885708 | 1.217001314 | True                  |
| U-C009                      | 0.075000000 |        -1.713216015 | 1.151083402 | True                  |
| U-ALL-MATCH                 | 0.227777778 |        -1.658456136 | 1.334647416 | True                  |
| all_match_redundancy_pruned | 0.213888889 |        -1.664956781 | 1.335337977 | True                  |

## Hypervolume

- Step-3 incumbent hypervolume: `0.610225830003`
- Incumbents plus the complete new batch: `0.610635326687`
- Combined improvement: `0.000409496684`
- Reference point remains IC `0.0`, Gaussian NLL `0.0`, Sharpe `0.0`.

## Validation

- All 12 requested runs passed: `True`
- Common date hash: `bea704cb64a42d1d9aeb112ed76ddb28e9487ca8cedbfcccd137f2ee993dde13`
- Promoted candidates: `['all_match_redundancy_pruned']`
- The task universe, `t_index`, rank 2, maxiter 15, and 256 scenarios were verified from every manifest.
