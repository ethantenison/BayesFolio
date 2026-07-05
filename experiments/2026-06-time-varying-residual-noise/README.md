# June 2026 Time-Varying Residual Noise

This experiment continues the progress from:

`/Users/et/Documents/BayesFolio/experiments/2026-06-heteroskedastic-noise/outputs/sweeps/20260618_residual_noise_sweep/CLOSEOUT.md`

## Research Question

Can BayesFolio improve on the two-stage plug-in residual fixed-noise model by
learning a better time-varying per-observation noise surface,
`sigma(asset, date)`, while preserving leakage control and scenario usefulness?

The distinction matters: the existing Hadamard multitask GP already allowed
task-specific noise, but that noise was effectively constant over time for a
given asset/task. This experiment is about modeling observation noise that
changes across both asset and rebalance date.

## Current Baseline

Primary baseline:

- `signed_lkj_eta_2` multitask GP from
  `/Users/et/Documents/BayesFolio/experiments/2026-06-portfolio-optimization/outputs/runs/20260616_signed_lkj_eta_2_portfolio`

Current best residual-noise candidates:

- Balanced calibration candidate:
  `/Users/et/Documents/BayesFolio/experiments/2026-06-heteroskedastic-noise/outputs/runs/20260618_hnoise_residual_scale_050_portfolio`
- Portfolio-summary candidate:
  `/Users/et/Documents/BayesFolio/experiments/2026-06-heteroskedastic-noise/outputs/runs/20260618_hnoise_residual_scale_025_portfolio`

Reference sweep:

- `/Users/et/Documents/BayesFolio/experiments/2026-06-heteroskedastic-noise/outputs/sweeps/20260618_residual_noise_sweep/REPORT.md`

## Initial Hypothesis

Residual-history fixed noise is directionally right because it targets GP
forecast error rather than raw ETF return volatility. The next gain should come
from choosing or estimating the residual-noise scale using only prior evidence,
not from picking a global scale after the fact.

## Candidate Methods

Start simple and escalate only if diagnostics justify it.

1. Rolling residual-scale selector.
   Select among `{0.25, 0.50, 0.75, 1.00}` using only prior-window calibration
   metrics. Compare against fixed `residual_scale_050` and `residual_scale_025`.

2. Residual EWMA noise model.
   Estimate residual variance with an exponentially weighted history by
   asset/class. Tune half-life using only prior windows.

3. Robust residual scale.
   Use winsorized squared residuals, Huberized residuals, or median absolute
   residuals to reduce sensitivity to one shock window.

4. Improved hierarchical shrinkage.
   Replace hand-tuned asset/class/global shrinkage with a more explicit
   empirical-Bayes or James-Stein style residual variance shrinkage rule.

5. Feature-conditioned residual noise.
   Predict residual scale from rebalance-available features such as trailing
   volatility, drawdown, momentum dispersion, cross-asset correlation, asset
   class, and prior residual state. Keep the first version regularized and
   auditable.

## Selection Metrics

Do not select on return alone.

Primary calibration metrics:

- mean NLL
- CRPS
- standardized residual scale (`z_std`)
- interval coverage error
- PIT/standardized residual diagnostics

Portfolio and scenario metrics:

- cumulative return
- Sharpe
- max drawdown
- turnover
- mean/median IC
- window-level return and IC deltas

## Required Visual Checks

Every serious candidate should write a visual check pack:

- `noise_std_asset_date_heatmap.png`: predicted noise by asset and rebalance
  date.
- `noise_std_by_asset_group_box.png`: distribution by asset class.
- `noise_source_mix_by_window.png`: source/fallback usage over time.
- `noise_std_vs_abs_error_binned.png`: predicted noise versus realized absolute
  forecast error.
- `noise_variance_share_asset_date_heatmap.png`: observation-noise share of
  total predictive variance.
- `window_level_calibration_deltas.png`: candidate versus baseline by rebalance
  date.
- `window_level_portfolio_deltas.png`: return, drawdown, turnover, and IC deltas
  by rebalance date.

Minimum rule: no relevant visual checks, no confident claim.

## Leakage Rules

- Noise estimates for rebalance date `T` may use only residuals from dates
  strictly before `T`.
- Scale selection for date `T` may use only calibration evidence from prior
  rebalance windows.
- Feature-conditioned noise models may use only features known at rebalance
  time.
- Do not tune on the same future window being scored.

## Artifact Contract

Each run should write:

- `manifest.json`
- `REPORT.md`
- `portfolio_summary.csv`
- `portfolio_returns.csv`
- `portfolio_weights.csv`
- `gp_window_predictions.csv`
- `gp_window_ic.csv`
- `noise_model_diagnostics.csv`
- `calibration_summary.csv` or linked calibration audit
- visual check pack

Manifest must include:

- command
- git SHA
- dirty diff summary
- feature/data path and hash when available
- baseline run path
- candidate method and hyperparameters
- residual source path
- scored dates
- output paths

Minimum rule: no manifest, no claim.

## Suggested First Run

Implement and run the rolling residual-scale selector over
`{0.25, 0.50, 0.75, 1.00}`.

Decision target:

- If the rolling selector matches or beats fixed `residual_scale_050` on
  calibration without materially hurting portfolio behavior, promote rolling
  selection as the next baseline for time-varying residual noise.
- If it underperforms, keep fixed `residual_scale_050` as the balanced candidate
  and move next to EWMA or robust residual scale.

## Output Layout

- `outputs/runs/`: individual candidate walk-forward runs.
- `outputs/sweeps/`: multi-candidate comparisons.
- `outputs/calibration/`: calibration audits.
- `outputs/visual_checks/`: standalone visual diagnostics when not tied to one
  run directory.

## Runs

### 20260619 Rolling Residual-Scale Selector

Run:

`/Users/et/Documents/BayesFolio/experiments/2026-06-time-varying-residual-noise/outputs/runs/20260619_rolling_residual_scale_selector_v2`

Method:

- Artifact-level rolling selector over already completed fixed residual-scale
  runs.
- First window fallback: `residual_scale_050`.
- Later windows: choose the scale with best mean prior-window NLL.
- No GP refit; selected predictions, returns, weights, scenarios, noise
  diagnostics, calibration, manifest, report, and visual checks are assembled
  into one run directory.

Readout:

- Improves over the signed baseline on calibration and portfolio summary.
- Does not beat fixed `residual_scale_050` on calibration.
- Selected `residual_scale_025` for most windows, which preserved strong
  portfolio behavior but gave up some calibration quality versus fixed 0.50.
- Conclusion: simple prior-window mean-NLL selection is too data-poor as the
  first selector. Keep `residual_scale_050` as the balanced candidate and test a
  more stable selector or EWMA/robust residual scale next.
