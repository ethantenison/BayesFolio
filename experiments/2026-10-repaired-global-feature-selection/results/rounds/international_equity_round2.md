# International Equity round 2

Cumulative configurations: 161 -> 173.

Fixed-reference hypervolume: 0.595004502 -> 0.620432258.

Stalled: False; two consecutive stalled rounds: False.

| configuration                     | hypothesis_name                            | base_configuration                | category                        |    delta_ic |    delta_nll |   delta_sharpe |   input_count_change |   seed_ic_wins |   seed_nll_wins |   seed_sharpe_wins | on_cumulative_front   |
|:----------------------------------|:-------------------------------------------|:----------------------------------|:--------------------------------|------------:|-------------:|---------------:|---------------------:|---------------:|----------------:|-------------------:|:----------------------|
| international_equity_c5be1947ee1b | risk_without_move_or_slope_change          | international_equity_4c2b72c2c055 | exploitation_combined_pruning   |  0.00277778 | -0.0384554   |    -0.00907238 |                   -1 |              1 |               3 |                  2 | True                  |
| international_equity_41d789e7d4d7 | risk_without_move_or_slope_change_plus_oil | international_equity_4c2b72c2c055 | frontier_bridge                 | -0.0111111  | -0.0573668   |     0.0278419  |                    0 |              1 |               3 |                  2 | True                  |
| international_equity_ed4ce751392c | risk_without_move_plus_oil                 | international_equity_4c2b72c2c055 | frontier_bridge                 | -0.0388889  | -0.0475509   |     0.0131789  |                    1 |              0 |               3 |                  2 | False                 |
| international_equity_d32b62e7dd7c | risk_without_move_plus_local_volatility    | international_equity_4c2b72c2c055 | frontier_bridge                 | -0.0416667  |  0.0141156   |     0.0115284  |                    1 |              0 |               1 |                  1 | False                 |
| international_equity_88902e75894f | risk_without_move_or_global_rank           | international_equity_4c2b72c2c055 | exploitation_local_pruning      | -0.05       | -0.00108622  |    -0.0184458  |                   -1 |              0 |               2 |                  2 | False                 |
| international_equity_78bf31d00d4b | risk_without_move_or_trend_slope           | international_equity_4c2b72c2c055 | exploitation_local_pruning      | -0.025      | -0.0293803   |    -0.0419538  |                   -1 |              0 |               3 |                  1 | False                 |
| international_equity_6a30de6c710d | risk_without_move_or_dealer_proxy          | international_equity_4c2b72c2c055 | exploration_risk_pruning        | -0.0611111  | -0.0303928   |    -0.131135   |                   -1 |              0 |               2 |                  0 | False                 |
| international_equity_7f66d5419f06 | risk_without_move_or_dollar_level          | international_equity_4c2b72c2c055 | exploration_fx_pruning          | -0.133333   | -0.0733884   |    -0.11677    |                   -1 |              0 |               3 |                  0 | True                  |
| international_equity_ebfded375c99 | risk_without_move_or_em_fx                 | international_equity_4c2b72c2c055 | exploration_fx_pruning          | -0.119444   | -0.00331223  |    -0.05969    |                   -1 |              0 |               2 |                  0 | False                 |
| international_equity_c38267888036 | risk_without_move_or_vix_slope_level       | international_equity_4c2b72c2c055 | exploration_risk_representation | -0.0138889  |  0.000360982 |     0.00277093 |                   -1 |              1 |               2 |                  1 | False                 |
| international_equity_54880a9ae051 | broad_without_either_hy_transform          | international_equity_baaff7ca24fa | exploitation_credit_pruning     |  0.0444444  |  0.0378697   |     0.0278694  |                   -1 |              3 |               1 |                  2 | True                  |
| international_equity_3f07351cc740 | broad_without_hy_change_or_credit_change   | international_equity_baaff7ca24fa | exploitation_credit_pruning     |  0.05       |  0.00514248  |     0.00180535 |                   -1 |              3 |               1 |                  2 | True                  |

Deltas are candidate minus its explicit preregistered parent: lower NLL is favorable.
Count, return, RMSE, and interval width are not hidden selection objectives.

Seed repeats measure optimization variability on the same reused dates, not independent economic samples or an untouched holdout.
