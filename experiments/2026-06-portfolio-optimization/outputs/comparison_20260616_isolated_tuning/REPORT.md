# Isolated Portfolio Tuning Comparison

Run root: `/Users/et/Documents/BayesFolio/experiments/2026-06-portfolio-optimization/outputs/comparison_20260616_isolated_tuning`

## Headline Summary

| experiment                  | terminal_value | cagr   | sharpe | max_drawdown | avg_turnover | mean_ic | delta_terminal_value | delta_sharpe | delta_avg_turnover |
| --------------------------- | -------------- | ------ | ------ | ------------ | ------------ | ------- | -------------------- | ------------ | ------------------ |
| signed_lkj_eta_2            | 11997.5170     | 0.1998 | 2.1035 | -0.0473      | 0.3442       | 0.2723  | 99.6478              | 0.1481       | 0.1154             |
| scenario_mean_scale_1.5     | 11906.2789     | 0.1906 | 1.9630 | -0.0555      | 0.2295       | 0.3708  | 8.4097               | 0.0076       | 0.0008             |
| control                     | 11897.8692     | 0.1898 | 1.9554 | -0.0556      | 0.2288       | 0.3708  | 0.0000               | 0.0000       | 0.0000             |
| lengthscale_floor_0.02      | 11892.1694     | 0.1892 | 1.9619 | -0.0561      | 0.2443       | 0.3885  | -5.6999              | 0.0065       | 0.0155             |
| turnover_blend_0.5          | 11891.8013     | 0.1892 | 1.9820 | -0.0545      | 0.1536       | 0.3708  | -6.0679              | 0.0266       | -0.0752            |
| component_outputscale_floor | 11562.0603     | 0.1562 | 1.6694 | -0.0580      | 0.2135       | 0.1441  | -335.8090            | -0.2860      | -0.0153            |

## Monthly Spread Versus Control

| experiment                  | months_beating_control | months_lagging_control | avg_monthly_spread | feb_2026_return | feb_2026_spread_vs_control | best_month_spread | worst_month_spread |
| --------------------------- | ---------------------- | ---------------------- | ------------------ | --------------- | -------------------------- | ----------------- | ------------------ |
| signed_lkj_eta_2            | 6                      | 6                      | 0.0007             | -0.0473         | 0.0083                     | 0.0083            | -0.0086            |
| scenario_mean_scale_1.5     | 8                      | 4                      | 0.0001             | -0.0555         | 0.0001                     | 0.0003            | -0.0001            |
| lengthscale_floor_0.02      | 7                      | 5                      | -0.0000            | -0.0561         | -0.0005                    | 0.0008            | -0.0015            |
| turnover_blend_0.5          | 7                      | 4                      | -0.0001            | -0.0545         | 0.0012                     | 0.0012            | -0.0017            |
| component_outputscale_floor | 0                      | 12                     | -0.0024            | -0.0580         | -0.0024                    | -0.0001           | -0.0067            |

## Readout

- `signed_lkj_eta_2` is the only isolated GP architecture change that clearly beat control on terminal value, Sharpe, and drawdown. It did so despite lower mean IC and higher turnover, which means it likely improved portfolio covariance/rotation more than raw ranking.
- `turnover_blend_0.5` kept nearly the same terminal value as control, improved Sharpe modestly, reduced drawdown slightly, and cut turnover from about `0.229` to `0.154`. In an IRA this is not mainly tax-driven, but it is still a stability win.
- `scenario_mean_scale_1.5` was almost neutral. Simple mean amplification did not materially change optimizer behavior.
- `lengthscale_floor_0.02` improved mean IC but did not improve terminal value or drawdown, so better ranking did not translate cleanly to portfolio value.
- `component_outputscale_floor` damaged both IC and portfolio value. Reject this version for now.

## GP Diagnostic Notes

- The signed LKJ run changed the task covariance geometry substantially: mean off-diagonal task correlation fell from about `0.741` in the control family to about `0.165`, and eig1 share fell from about `0.760` to `0.224`. That is the strongest evidence that the portfolio improvement came from less globally positive task pooling.
- The lengthscale-floor run forced the effective eval lengthscale to the `0.02` floor and improved mean IC, but not portfolio value. This suggests the floor may help ranking while still needing a better portfolio translation or signed covariance pairing.
- The component outputscale-floor run raised suppressed component amplitudes but hurt IC and terminal value. That supports the earlier warning that forcing weak kernel components awake can add noise rather than useful structure.
- Likelihood noise stayed near `0.0065` for most successful runs, so the main differences are covariance geometry, time-warp behavior, and portfolio translation rather than observation-noise instability.

## Suggested Combination Tests

1. `signed_lkj_eta_2 + turnover_blend_0.5`: best architecture improvement plus best translation/stability improvement.
2. `signed_lkj_eta_2 + lengthscale_floor_0.02`: only if we want to see whether the higher-IC floor helps signed covariance without losing its drawdown benefit.
3. Keep `component_outputscale_floor` out of combinations for now.

## IRA Context

Tax drag is not the main concern for the intended IRA use. I still track turnover because high turnover can reveal unstable optimizer translation, concentration churn, and sensitivity to small forecast changes.
