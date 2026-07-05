# June 2026 Task Covariance Experiment

Purpose: compare ETF multitask GP task-covariance assumptions before making the
BayesFolio production model explicit.

Data source:

- `/Users/et/.bayesfolio/artifacts/features/portfolio_etf_macro_features_2026_06.parquet`
- June 2026 ETF universe from `notebooks/20260601_portfolio.py`
- 12 scored monthly roll-forward windows ending with the last month that has
  realized labels, plus an unscored June live prediction row.

Variants:

1. `historical_mean`: no-model baseline using each ETF's training-window mean
   realized excess return.
2. `positive_no_prior`: BoTorch `PositiveIndexKernel`, `task_covar_prior=None`
3. `positive_beta_prior`: BoTorch `PositiveIndexKernel`, BoTorch 0.18 default
   `BetaPrior(2.5, 1.5)`
4. `signed_no_prior`: GPyTorch `IndexKernel`, no task covariance prior
5. `signed_lkj_eta_1`: GPyTorch `IndexKernel`, `LKJCovariancePrior(eta=1.0)`
6. `signed_lkj_eta_2`: GPyTorch `IndexKernel`, `LKJCovariancePrior(eta=2.0)`

Fixed modeling choices:

- Rank 5 task covariance.
- Stratified standardization by ETF task.
- Time-varying lengthscale and outputscale wrappers from the GPArchitect
  implementation already present in BayesFolio.
- June monthly GP architecture: time, ETF, and macro blocks plus explicit
  time-ETF, time-macro, and macro-ETF interactions.
- No Riskfolio portfolio optimization. This experiment evaluates model quality
  and top/bottom 5 long-short signal only.
- Multitask constant mean module.
- BoTorch `Normalize` input transform on all non-task feature columns. The
  appended categorical task feature is not normalized.
- Stable per-variant/per-window seeds so adding a new variant does not change
  existing variant initializations.
- Residualized metrics that subtract each ETF's training-window historical mean
  from both realized returns and model predictions.

Outputs:

- Versioned runs are written under `outputs/runs/<run-id>/`.
- Each run includes `manifest.json` with command, git SHA, dirty status, data
  artifact hash, train sizes, variants, dependency versions, seed policy, and
  output path.
- Run files include `window_predictions.csv`, `window_metrics.csv`,
  `variant_summary.csv`, `task_covariance_diagnostics.csv`, and
  `live_june_predictions.csv`.

Run:

```bash
poetry run python experiments/2026-06-task-covariance/run_task_covariance_rollforward.py
```

For a fast smoke test:

```bash
poetry run python experiments/2026-06-task-covariance/run_task_covariance_rollforward.py --max-windows 1 --variants positive_no_prior
```
