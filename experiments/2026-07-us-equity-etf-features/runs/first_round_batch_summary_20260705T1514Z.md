# First-Round U.S. Equity Batch Summary

Scope: pruned four-task monthly BME artifact (`SPY`, `MGK`, `VTV`, `IWM`), 3 realized windows (`2026-04-30`, `2026-05-29`, `2026-06-30`), rank 2 MTGP, BoTorch Normalize inputs, StratifiedStandardize outputs, 20 fit iterations, 512 posterior scenarios.

## Portfolio Comparison

| run_id                                                     | gp_cum_return | gp_sharpe | gp_mean_ic | gp_median_ic | equal_weight_cum_return | historical_cum_return | gp_avg_turnover |
| ---------------------------------------------------------- | ------------- | --------- | ---------- | ------------ | ----------------------- | --------------------- | --------------- |
| 20260705T1510Z_first_round_control_rank2_3w                | 0.0405        | 1.7784    | 0.0000     | -0.2000      | 0.0499                  | 0.0418                | 0.4667          |
| 20260705T1511Z_first_round_positive_no_prior_rank2_3w      | 0.0693        | 3.8661    | -0.1333    | -0.2000      | 0.0499                  | 0.0418                | 0.4667          |
| 20260705T1512Z_first_round_signed_lkj_rank2_3w             | 0.0534        | 2.6027    | 0.3333     | 0.8000       | 0.0499                  | 0.0418                | 0.4729          |
| 20260705T1513Z_first_round_signed_lkj_noise_floor_rank2_3w | 0.0521        | 2.5766    | 0.1333     | 0.4000       | 0.0499                  | 0.0418                | 0.4815          |

## Forecast-Shape Diagnostics

| run_id                                                     | pred_cross_section_std | pred_cross_section_range | pred_mean_abs | post_std_mean | y_true_cross_section_std |
| ---------------------------------------------------------- | ---------------------- | ------------------------ | ------------- | ------------- | ------------------------ |
| 20260705T1510Z_first_round_control_rank2_3w                | 0.0109                 | 0.0250                   | 0.0121        | 0.0553        | 0.0209                   |
| 20260705T1511Z_first_round_positive_no_prior_rank2_3w      | 0.0080                 | 0.0174                   | 0.0108        | 0.0644        | 0.0209                   |
| 20260705T1512Z_first_round_signed_lkj_rank2_3w             | 0.0098                 | 0.0215                   | 0.0145        | 0.0498        | 0.0209                   |
| 20260705T1513Z_first_round_signed_lkj_noise_floor_rank2_3w | 0.0082                 | 0.0182                   | 0.0110        | 0.0496        | 0.0209                   |

## Readout

- The wrapper/preflight path is working on the pruned U.S. equity artifact; all runs produced manifests, predictions, weights, and diagnostics.
- This is a deliberately tiny first batch, so portfolio rankings are not decision-grade yet.
- Positive no-prior produced the best 3-window cumulative return in this tiny slice, but its mean IC was negative; signed LKJ had the strongest mean/median IC.
- Forecasts are not fully collapsed to a flat mean: cross-sectional prediction standard deviations are small but nonzero and are roughly 15-25% of realized cross-sectional target variation in this slice.
- All variants emitted small Cholesky jitter warnings; keep watching kernel conditioning as we scale window count and iteration budget.

## Next Step

Run the same candidate set over 6-12 windows with a larger optimizer budget, then compare IC stability, prediction dispersion, and posterior noise before trusting portfolio return differences.
