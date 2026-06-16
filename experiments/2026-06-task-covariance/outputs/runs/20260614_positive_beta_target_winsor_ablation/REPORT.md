# Target Winsorization Ablation

Question: does per-ETF training-window target winsorization before StratifiedStandardize help the positive_beta_prior lengthscale-only GP?

Setup: clipping bounds are estimated inside each roll-forward training window only; evaluation targets remain unclipped realized y_excess_lead.

## Summary

| run_label | IC | top_bottom_5_sharpe | top_bottom_5_hit_rate | resid_IC | resid_top_bottom_5_sharpe | resid_top_bottom_5_hit_rate | rmse | nlpd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_lengthscale_only | 0.367 | 3.205 | 0.917 | 0.201 | 1.925 | 0.583 | 0.034 | -2.137 |
| winsor_01 | 0.363 | 3.293 | 0.833 | 0.163 | 1.971 | 0.583 | 0.034 | -2.139 |
| winsor_025 | 0.365 | 3.427 | 0.833 | 0.184 | 1.831 | 0.583 | 0.034 | -2.155 |

## Clipping Footprint

- `winsor_01`: clipped 216 low and 216 high training labels across scored windows; mean std_after/std_before 0.973.
- `winsor_025`: clipped 432 low and 432 high training labels across scored windows; mean std_after/std_before 0.942.

## SPY/VEA Residual Amplitude

| run_label | asset_id | residual_pred_true_std_ratio | residual_corr |
| --- | --- | --- | --- |
| baseline_lengthscale_only | SPY+VEA | 0.090 | 0.500 |
| baseline_lengthscale_only | SPY | 0.089 | 0.477 |
| baseline_lengthscale_only | VEA | 0.091 | 0.520 |
| winsor_01 | SPY+VEA | 0.098 | 0.420 |
| winsor_01 | SPY | 0.087 | 0.346 |
| winsor_01 | VEA | 0.106 | 0.474 |
| winsor_025 | SPY+VEA | 0.119 | 0.414 |
| winsor_025 | SPY | 0.099 | 0.340 |
| winsor_025 | VEA | 0.133 | 0.466 |

## Read

Mild winsorization does not materially improve the model. IC is basically flat versus the unclipped baseline, top-bottom Sharpe rises slightly, but hit rate and residual IC weaken. The stronger 2.5/97.5 variant improves headline top-bottom Sharpe more, but residual Sharpe falls below the unclipped baseline.

Self-critique: this could be noise over only 12 scored windows. The stronger variant may be helping ranking by suppressing noisy training extremes, but the weaker residual diagnostics argue against treating it as a real fix yet.

Decision: keep target winsorization as an available diagnostic, but do not make it the default target treatment from this run alone.
