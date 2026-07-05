# June 2026 Heteroskedastic Noise Plug-In

This experiment tests whether a separate, leakage-aware noise model can improve
uncertainty calibration while preserving the existing multitask GP latent
covariance and task covariance used for posterior scenario portfolio
optimization.

The first candidate is intentionally simple:

- estimate raw monthly excess-return variance from trailing asset returns,
  shrunk toward asset-class variance;
- fit the signed multitask GP with fixed per-observation training noise;
- draw latent multitask GP posterior scenarios and add predicted diagonal
  heteroskedastic observation noise for realized-return scenarios;
- compare against the existing `signed_lkj_eta_2` baseline on calibration,
  window stability, IC, portfolio performance, equity/drawdown path, and
  turnover.

Outputs are written to versioned directories under `outputs/runs/`.
