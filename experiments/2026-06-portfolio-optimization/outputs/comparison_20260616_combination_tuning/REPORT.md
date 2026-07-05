# Combination Portfolio Tuning Comparison

Run root: `/Users/et/Documents/BayesFolio/experiments/2026-06-portfolio-optimization/outputs/comparison_20260616_combination_tuning`

## Headline Summary

| experiment                              | terminal_value | cagr   | sharpe | max_drawdown | avg_turnover | mean_ic | delta_control_terminal_value | delta_signed_terminal_value | delta_signed_avg_turnover |
| --------------------------------------- | -------------- | ------ | ------ | ------------ | ------------ | ------- | ---------------------------- | --------------------------- | ------------------------- |
| signed_lkj_eta_2                        | 11997.5170     | 0.1998 | 2.1035 | -0.0473      | 0.3442       | 0.2723  | 99.6478                      | 0.0000                      | 0.0000                    |
| signed_lkj_eta_2_lengthscale_floor_0.02 | 11996.6522     | 0.1997 | 2.1064 | -0.0473      | 0.3435       | 0.2559  | 98.7829                      | -0.8649                     | -0.0007                   |
| signed_lkj_eta_2_turnover_blend_0.5     | 11985.9028     | 0.1986 | 2.0632 | -0.0513      | 0.1936       | 0.2723  | 88.0336                      | -11.6142                    | -0.1507                   |
| control                                 | 11897.8692     | 0.1898 | 1.9554 | -0.0556      | 0.2288       | 0.3708  | 0.0000                       | -99.6478                    | -0.1154                   |
| lengthscale_floor_0.02                  | 11892.1694     | 0.1892 | 1.9619 | -0.0561      | 0.2443       | 0.3885  | -5.6999                      | -105.3477                   | -0.0999                   |
| turnover_blend_0.5                      | 11891.8013     | 0.1892 | 1.9820 | -0.0545      | 0.1536       | 0.3708  | -6.0679                      | -105.7157                   | -0.1907                   |

## Monthly Spread Summary

| experiment                              | terminal_value | months_beating_control | months_beating_signed | avg_spread_vs_control | avg_spread_vs_signed | feb_2026_return | feb_spread_vs_control | feb_spread_vs_signed |
| --------------------------------------- | -------------- | ---------------------- | --------------------- | --------------------- | -------------------- | --------------- | --------------------- | -------------------- |
| signed_lkj_eta_2                        | 11997.5170     | 6                      | 0                     | 0.0007                | 0.0000               | -0.0473         | 0.0083                | 0.0000               |
| signed_lkj_eta_2_lengthscale_floor_0.02 | 11996.6522     | 6                      | 7                     | 0.0007                | -0.0000              | -0.0473         | 0.0083                | 0.0000               |
| signed_lkj_eta_2_turnover_blend_0.5     | 11985.9028     | 7                      | 5                     | 0.0006                | -0.0001              | -0.0513         | 0.0043                | -0.0040              |
| control                                 | 11897.8692     | 0                      | 6                     | 0.0000                | -0.0007              | -0.0556         | 0.0000                | -0.0083              |
| lengthscale_floor_0.02                  | 11892.1694     | 7                      | 6                     | -0.0000               | -0.0007              | -0.0561         | -0.0005               | -0.0088              |
| turnover_blend_0.5                      | 11891.8013     | 7                      | 6                     | -0.0001               | -0.0007              | -0.0545         | 0.0012                | -0.0071              |

## Readout

- `signed_lkj_eta_2` remains the best run by terminal value and drawdown.
- `signed_lkj_eta_2_turnover_blend_0.5` cut turnover materially versus signed-only (`0.194` vs `0.344`) while preserving most of the terminal-value gain, but it did not beat signed-only.
- `signed_lkj_eta_2_lengthscale_floor_0.02` was nearly identical to signed-only and did not add value.
- The clean next candidate for IRA use depends on preference: signed-only maximizes return/drawdown in this 12-month test; signed+turnover blend is smoother and less churny.

## Recommendation

Use `signed_lkj_eta_2` as the current best model configuration for another validation pass. Keep `signed_lkj_eta_2_turnover_blend_0.5` as the practical/stability alternative. Do not pursue the signed+lengthscale-floor combo unless a broader seed/window check says the higher-IC lengthscale floor matters.

