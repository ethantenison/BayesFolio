# Positive Beta Prior Without Time Modulation

## Question

Does removing the time-varying lengthscale improve the flat prediction-amplitude problem for the `positive_beta_prior` multitask GP?

## Run

- Script: `experiments/2026-06-task-covariance/run_task_covariance_rollforward.py`
- Run ID: `20260614_positive_beta_no_time_modulation`
- Variant: `positive_beta_prior`
- Time modulation: `neither`
- Max optimizer iterations: 50
- Output path: `experiments/2026-06-task-covariance/outputs/runs/20260614_positive_beta_no_time_modulation`

Artifacts:

- `manifest.json`
- `variant_summary.csv`
- `window_predictions.csv`
- `window_metrics.csv`
- `task_covariance_diagnostics.csv`
- `live_june_predictions.csv`
- `plots/spy_lengthscale_only_vs_no_time_modulation.png`
- `plots/vea_lengthscale_only_vs_no_time_modulation.png`

## Result

Removing time modulation increases amplitude, but weakens the useful signal.

| Model | IC | Top-bottom Sharpe | Residual IC | Residual Sharpe |
|---|---:|---:|---:|---:|
| lengthscale-only | 0.367 | 3.205 | 0.201 | 1.925 |
| no time modulation | 0.320 | 2.341 | 0.189 | 1.897 |

SPY/VEA amplitude and correlation:

| Model | Pred/true std | Residual pred/true std | Corr | Residual corr |
|---|---:|---:|---:|---:|
| lengthscale-only | 0.101 | 0.090 | 0.362 | 0.500 |
| no time modulation | 0.225 | 0.215 | 0.247 | 0.291 |

## Interpretation

The time-varying lengthscale is suppressing amplitude, but it is also preserving rank signal.

Removing it makes predictions less flat, but the additional movement is not reliably correct. For SPY/VEA, residual amplitude more than doubles, while residual correlation falls sharply.

The better diagnosis is not "remove time-varying lengthscales." It is:

- The current lengthscale-only model is too conservative.
- The no-time model is more expressive, but less aligned with realized returns.
- The architecture needs a way to increase calibrated amplitude without sacrificing the useful rank signal.

## Next Check

Run fake-data recovery with known return-scale amplitude. If the lengthscale-only model cannot recover simulated amplitude, the architecture or priors are too restrictive. If it can recover fake amplitude, the real-data signal is too weak or the feature/target transform is washing it out.
