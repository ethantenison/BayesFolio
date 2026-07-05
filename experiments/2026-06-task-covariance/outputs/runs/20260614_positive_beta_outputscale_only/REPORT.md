# Positive Beta Prior With Outputscale-Only Time Modulation

## Question

What happens if the model uses the time-varying outputscale wrapper without the time-varying lengthscale wrapper?

## Run

- Script: `experiments/2026-06-task-covariance/run_task_covariance_rollforward.py`
- Run ID: `20260614_positive_beta_outputscale_only`
- Variant: `positive_beta_prior`
- Time modulation: `outputscale_only`
- Max optimizer iterations: 50

## Result

Outputscale-only is not the fix.

| Run | IC | Top-bottom Sharpe | Residual IC | Residual Sharpe |
|---|---:|---:|---:|---:|
| `both` | 0.270 | 1.860 | -0.006 | 0.040 |
| `outputscale_only` | 0.235 | 1.146 | 0.071 | -0.117 |
| `lengthscale_only` | 0.367 | 3.205 | 0.201 | 1.925 |
| `neither` | 0.320 | 2.341 | 0.189 | 1.897 |

SPY/VEA amplitude:

| Run | Pred/true std | Residual pred/true std | Corr | Residual corr |
|---|---:|---:|---:|---:|
| `both` | 0.073 | 0.054 | -0.111 | 0.016 |
| `outputscale_only` | 0.182 | 0.174 | -0.290 | -0.249 |
| `lengthscale_only` | 0.101 | 0.090 | 0.362 | 0.500 |
| `neither` | 0.225 | 0.215 | 0.247 | 0.291 |

## Interpretation

Outputscale-only increases movement relative to `both`, but the movement is not useful. For SPY/VEA, correlations are negative.

This suggests:

- the time-varying outputscale wrapper can create amplitude;
- but it does not create calibrated or directionally useful amplitude;
- combining it with the time-varying lengthscale flattens the model further;
- lengthscale-only remains the best current compromise.

## Recommendation

Do not use `outputscale_only` or `both` for the current architecture. Keep `lengthscale_only` as the reference and use fake-data recovery to test whether the model can recover known amplitude.
