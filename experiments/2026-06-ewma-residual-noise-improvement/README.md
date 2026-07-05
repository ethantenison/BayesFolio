# June 2026 EWMA Residual-Noise Improvement

## Purpose

This experiment tries to improve the residual EWMA noise method from the
24-window residual-noise comparison.

Decision question:

Can an EWMA residual-noise variant improve forecast calibration versus
`residual_ewma_hl3_scale_050_24w` without materially hurting the plain multitask
GP portfolio baseline?

## Baselines

- Plain portfolio baseline:
  `experiments/2026-06-portfolio-optimization/outputs/runs/20260621_signed_lkj_eta_2_plain_24w`
- Current EWMA benchmark:
  `experiments/2026-06-heteroskedastic-noise/outputs/runs/20260621_residual_ewma_hl3_scale_050_24w`
- Fixed residual-history benchmark:
  `experiments/2026-06-heteroskedastic-noise/outputs/runs/20260621_residual_history_scale_050_24w`

## Candidate Knobs

Stage 1 varies EWMA half-life at fixed variance scale 0.50:

- `1`, `2`, `3`, `6`, `9`, `12`

This checks whether the current half-life 3 is too reactive or too slow for
monthly ETF residuals.

Stage 2 varies variance scale for the best half-life candidates:

- `0.25`, `0.50`, `0.75`, `1.00`

This separates temporal smoothing from the overall amount of observation noise
injected into the fixed-noise GP.

Optional sensitivity checks:

- class shrinkage values around the default `0.35`
- floor and ceiling raw noise standard-deviation clipping
- robust residual variants if the EWMA sweep appears shock-sensitive

## Promotion Rule

Do not promote a candidate from point estimates alone. A stronger candidate
should:

- improve mean NLL or materially improve coverage / standardized residual scale
  versus current EWMA,
- avoid a meaningful Sharpe, return, drawdown, or turnover regression versus the
  plain multitask GP,
- pass paired 24-window uncertainty checks directionally,
- show visual improvements that are not concentrated in one or two windows.

Minimum rule: no manifest, no visual diagnostics, no paired tests, no claim.
