# GP Prior And Configuration EDA
This pass is aimed at choices for GP and multitask GP configuration, not at proving forecast skill.

## Output And Task Structure
- `BME`: average per-task target std `0.0514` (5.14%), average absolute task mean `0.0074` (0.74%).
  Task-correlation PC1 explains `83.50%`; first two PCs explain `95.25%`.
- `3W-FRI`: average per-task target std `0.0446` (4.46%), average absolute task mean `0.0055` (0.55%).
  Task-correlation PC1 explains `82.53%`; first two PCs explain `94.85%`.

Implication: start MTGP coregionalization with rank 1-2, not a fully flexible task covariance. The tasks are related enough for pooling, but style/size differences mean rank 1 alone may be too restrictive.

## Feature Space And Kernel Implications
- `BME` standardized feature-space median pairwise distance is `1.34` with 5-95% range `0.87`-`1.99`.
- `3W-FRI` standardized feature-space median pairwise distance is `1.32` with 5-95% range `0.85`-`2.06`.

Implication: standardize all predictors before GP fitting. Use conservative lengthscale priors or ARD regularization; short lengthscales across all 44 features would overfit these small panels.

## Redundancy Pressure

`3W-FRI` highest absolute feature correlations:
- `mom12m` vs `mom12m_skip1m`: corr `0.960`
- `vol_1m` vs `baspread`: corr `0.948`
- `vol_1m` vs `downside_vol_1m`: corr `0.924`
- `term_spread` vs `tbill3m`: corr `-0.904`
- `downside_vol_1m` vs `baspread`: corr `0.899`

`BME` highest absolute feature correlations:
- `short_reversal_1m` vs `lag2_y_excess_lead`: corr `-0.977`
- `mom12m` vs `mom12m_skip1m`: corr `0.964`
- `vol_1m` vs `baspread`: corr `0.953`
- `vol_1w` vs `vol_1m`: corr `0.951`
- `vol_1m` vs `downside_vol_1m`: corr `0.933`

Implication: the first experiment batch should compare compact blocks against broader blocks. Momentum variants, volatility variants, and curve/macro variables contain overlapping information.

## Tail And Robustness Pressure

`BME` largest feature outliers:
- `pct_above_50dma`: max robust z `24.0`, share >5 `3.12%`
- `ill_log`: max robust z `12.1`, share >5 `9.38%`
- `vol_of_vol`: max robust z `11.7`, share >5 `3.44%`
- `ret_kurt`: max robust z `10.6`, share >5 `5.31%`
- `vol_accel`: max robust z `10.3`, share >5 `1.88%`

`3W-FRI` largest feature outliers:
- `pct_above_50dma`: max robust z `23.2`, share >5 `1.08%`
- `ill_log`: max robust z `16.0`, share >5 `9.46%`
- `vol_accel`: max robust z `11.9`, share >5 `1.72%`
- `vol_of_vol`: max robust z `11.5`, share >5 `3.44%`
- `ret_kurt`: max robust z `11.0`, share >5 `6.24%`

Implication: keep clipping/robust scaling in the feature pipeline and consider Student-t or noise-floor sensitivity for targets when evaluating GP fits.

## Starting Prior Suggestions To Test
- Mean: per-task constant mean near zero, with weak task offsets around the observed 0.5%-1.1% one-period scale.
- Observation noise: initialize around 3%-6% target std by asset/horizon; test a floor so the GP cannot explain all noise as signal.
- Task covariance: LKJ or low-rank coregionalization with moderate shrinkage toward positive correlation; compare rank 1 vs rank 2.
- Kernel: standardized inputs, ARD RBF or Matern, conservative lengthscale prior, and feature-block ablations before all-feature ARD.
- Output scale: regularize strongly enough that posterior means can move, but avoid letting weak feature correlations create spurious sharp forecasts.

## Files
- `figures/gp_target_task_correlations.png`
- `figures/gp_task_correlation_eigens.png`
- `figures/gp_target_mean_vs_std.png`
- `figures/gp_feature_redundancy_top_pairs.png`
- `figures/gp_feature_tail_outliers.png`
- `figures/gp_standardized_feature_distances.png`
