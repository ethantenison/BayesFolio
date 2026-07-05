# June 2026 GP vs EWMA Portfolio Backtest

## Purpose

Compare the requested end-to-end portfolio policies over the same 24 monthly
holdout windows used in the residual-noise reports.

## Policies

- Plain signed multitask GP scenarios optimized with Riskfolio.
- Best EWMA residual-noise GP scenarios optimized with Riskfolio:
  `half_life=1`, `variance_scale=0.50`.
- Historical excess-return Riskfolio baseline.
- Equal weight baseline.

## Riskfolio Configuration

GP scenario optimization:

- `model="Classic"`
- `rm="CVaR"`
- `obj="Sharpe"`
- `method_mu="hist"`
- `method_cov="hist"`
- `hist=True`
- `upperlng=0.20`
- `nea=10`
- `alpha=0.5`

Historical excess-return optimization:

- `model="Classic"`
- `rm="CVaR"`
- `obj="Sharpe"`
- `method_mu="ewma2"`
- `method_cov="ewma2"`
- `hist=True`
- `upperlng=0.20`
- `nea=10`

## Source Runs

- Plain GP:
  `experiments/2026-06-portfolio-optimization/outputs/runs/20260621_signed_lkj_eta_2_plain_24w`
- EWMA GP:
  `experiments/2026-06-ewma-residual-noise-improvement/outputs/runs/20260621_ewma_hl1_scale_050_24w`

Minimum rule: no manifest, no path diagnostics, no paired-return readout, no
claim.
