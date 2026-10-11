# Forecast Metric Comparison

Scope: first four 3-window U.S. equity first-round runs. Metrics are computed from `gp_window_predictions.csv`; NLL assumes the model posterior predictive mean/std define a Gaussian predictive density.

## Overall Metrics By Run

| run_id                                                     | rmse   | mae    | mean_gaussian_nll | mean_pred_std | coverage_50 | coverage_80 | coverage_95 | standardized_residual_std |
| ---------------------------------------------------------- | ------ | ------ | ----------------- | ------------- | ----------- | ----------- | ----------- | ------------------------- |
| 20260705T1510Z_first_round_control_rank2_3w                | 0.0310 | 0.0262 | -1.8488           | 0.0553        | 0.7500      | 1.0000      | 1.0000      | 0.5297                    |
| 20260705T1511Z_first_round_positive_no_prior_rank2_3w      | 0.0303 | 0.0256 | -1.7384           | 0.0644        | 0.9167      | 1.0000      | 1.0000      | 0.4500                    |
| 20260705T1512Z_first_round_signed_lkj_rank2_3w             | 0.0286 | 0.0249 | -1.9547           | 0.0498        | 0.7500      | 1.0000      | 1.0000      | 0.5060                    |
| 20260705T1513Z_first_round_signed_lkj_noise_floor_rank2_3w | 0.0292 | 0.0247 | -1.9559           | 0.0496        | 0.7500      | 1.0000      | 1.0000      | 0.5138                    |

## Best Variant By ETF: Gaussian NLL

| asset_id | run_id                                                     | rmse   | mean_gaussian_nll | mean_pred_std | coverage_80 | coverage_95 |
| -------- | ---------------------------------------------------------- | ------ | ----------------- | ------------- | ----------- | ----------- |
| IWM      | 20260705T1513Z_first_round_signed_lkj_noise_floor_rank2_3w | 0.0313 | -1.7575           | 0.0610        | 1.0000      | 1.0000      |
| MGK      | 20260705T1512Z_first_round_signed_lkj_rank2_3w             | 0.0374 | -1.7831           | 0.0554        | 1.0000      | 1.0000      |
| SPY      | 20260705T1512Z_first_round_signed_lkj_rank2_3w             | 0.0174 | -2.2015           | 0.0414        | 1.0000      | 1.0000      |
| VTV      | 20260705T1513Z_first_round_signed_lkj_noise_floor_rank2_3w | 0.0229 | -2.1150           | 0.0415        | 1.0000      | 1.0000      |

## Best Variant By ETF: RMSE

| asset_id | run_id                                         | rmse   | mean_gaussian_nll | mean_pred_std | coverage_80 | coverage_95 |
| -------- | ---------------------------------------------- | ------ | ----------------- | ------------- | ----------- | ----------- |
| IWM      | 20260705T1512Z_first_round_signed_lkj_rank2_3w | 0.0309 | -1.7539           | 0.0612        | 1.0000      | 1.0000      |
| MGK      | 20260705T1512Z_first_round_signed_lkj_rank2_3w | 0.0374 | -1.7831           | 0.0554        | 1.0000      | 1.0000      |
| SPY      | 20260705T1512Z_first_round_signed_lkj_rank2_3w | 0.0174 | -2.2015           | 0.0414        | 1.0000      | 1.0000      |
| VTV      | 20260705T1510Z_first_round_control_rank2_3w    | 0.0184 | -2.0277           | 0.0489        | 1.0000      | 1.0000      |

## Readout

- RMSE and NLL can disagree; NLL rewards calibrated uncertainty, not just point accuracy.
- With only three scored windows, per-ETF metrics are diagnostic smoke evidence, not selection-grade evidence.
- These metrics should be promoted to the standard run report for all larger U.S. equity batches.
