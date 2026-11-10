# Portfolio evaluation framework

This isolated research framework evaluates saved ETF forecasts without fitting models or changing earlier experiments. It separates ranking skill, probabilistic forecast quality, and realized portfolio performance. It does not select a winner by IC/NLL/Sharpe hypervolume.

## Metric hierarchy

| Layer | Measures | Interpretation |
| --- | --- | --- |
| Forecast skill | Monthly cross-sectional Spearman IC across all 15 ETFs, each family, and an equal-family average; first/last 12 months and individual seeds | Ordering quality, not the complete portfolio objective. Family IC is coarse with three or four ETFs. Constant cross-sections are explicitly undefined, never silently zero or omitted. |
| Portfolio usefulness | Conventional arithmetic-excess Sharpe, total return, benchmark-relative return and paired Sharpe differences | Same investment universe, forecast months and allocation policy must be used when comparing models. The CLI is gross because trading costs are not supplied. |
| Risk adequacy | Forecast-error calibration, realized volatility, total-NAV drawdown, full joint predictive NLL, fixed-portfolio NLL/CRPS, marginal NLL, coverage and interval width | Good uncertainty is calibrated uncertainty, not simply smaller standard deviation. |
| Parsimony | Input count | Tie-breaker for economically indistinguishable candidates; no automatic trade-off threshold is invented. |

Turnover is not an objective. No investor risk aversion, live volatility target, CVaR objective, or automatic promotion criterion is inferred. The equal-weight and 60/40 references are research benchmarks, not proposed live allocations.

## Portfolio risk calibration

For weights chosen before the outcome, compute predicted mean `w' mu`, variance `w' Sigma w`, and error `realized portfolio return - predicted mean`. The standardized error is `z = error / predicted SD`.

Report mean z (target zero), mean z squared (target one), error RMS divided by predicted-SD RMS (target one), PIT histograms, and 50/80/95% coverage together with interval widths. These measure errors about ex-ante means, not the volatility of raw returns across months. Twenty-four observations support diagnostics, not precise tail-calibration claims.

Adaptive allocations have their own calibration records. Their distribution scores evaluate different portfolios across models and are not used as like-for-like proper scores of the complete predictive distribution.

## Joint predictive NLL and fixed portfolios

Joint Gaussian NLL uses the complete observed predictive covariance for the ETF return vector: the input kernel, task covariance, conditioning on training observations, outcome unscaling, and likelihood noise must already be reflected in the exported matrix. The task-kernel matrix B is not this covariance. Marginal standard deviations do not determine the off-diagonal entries.

The evaluator checks symmetry, positive definiteness, date/task axes, prediction fingerprints, raw units and agreement with marginal observation variances. It never substitutes a diagonal covariance, concatenates independent family scenarios, clips eigenvalues, or adds hidden jitter.

Fixed references are equal-weight across all 15 ETFs, equal-weight within each of the four families, and a fixed 60% equity/40% fixed-income allocation distributed equally within those sleeves. Identical weights are used across models and seeds. Their projected Gaussian NLL and nonnegative CRPS loss evaluate mean, scale and dependence without changing the portfolio being scored. Lower scores are better; NLL can be negative for continuous densities.

The shared training-only Ledoit-Wolf covariance is a separately labeled control. Combining it with GP means evaluates return forecasts with common risk estimates; it does not test the GP's own covariance forecast.

## Covariance precision modes

- `none`: Exact GP covariance is absent. GP joint/projection scores stay explicitly unavailable. Marginal scores and the common historical covariance control still run.
- `scenarios`: Estimate full sample covariance from each saved joint scenario matrix and use the saved analytic GP mean. These are **Gaussian plug-in Monte Carlo approximation scores**, not exact GP joint NLL. The existing baseline has 256 draws per month.
- `analytic`: Consume exact covariance artifacts from a future model-producing workflow. Missing or invalid artifacts stop evaluation; there is no fallback to scenarios.

For analytic mode, each seed requires `s27.npz` and `s27.json` (likewise 127 and 227). The NPZ contains string arrays `dates`, `assets`, `mean` of shape `(24, 15)`, and `covariance` of shape `(24, 15, 15)`. Pickled arrays are forbidden. The JSON must contain:

```json
{
  "kind": "observed_predictive",
  "source": "analytic_gp",
  "units": "discounted_excess_decimal_squared",
  "prediction_sha256": "SHA-256 of the matching gp_window_predictions.csv"
}
```

The producer must export covariance while the fitted model is available. This framework does not refit old models to recover absent matrices.

## Return accounting

The preserved target is `target = (1 + total return) / (1 + risk-free return) - 1`, using the integrated daily continuous ^IRX convention in the original target builder. Forecast scoring remains in those target units.

For fully invested portfolio performance, reconstruct `total = (1 + weighted target) * (1 + realized RF) - 1`, then `arithmetic excess = total - realized RF`. Conventional Sharpe is `sqrt(12) * mean(arithmetic excess) / sample_SD(arithmetic excess, ddof=1)`. Total account wealth and drawdown compound total returns, not excess returns. The old target-unit return path is independently checked against saved portfolio returns before conversion.

Frozen source yields, including the already-frozen final-month endpoint, are read without network refreshes. This reconstructed total-return series preserves old labels and source-vintage limitations; it is not a broker NAV record. RF is an outcome-accounting input, never a predictor derived from the future. The numerical portfolio function supports explicit beginning-wealth trading costs and labels results net only when those costs are supplied.

## Reproducibility and statistical limits

Only rows strictly before the forecast origin enter expanding historical means and covariance. The initial window is 84 months; subsequent windows contain 84+k months. All 24 scored forecast-origin dates and all 15 ETFs must match exactly. Existing primary-solver allocations are validated; missing data, fallback allocations or inconsistent artifacts fail instead of being repaired silently.

Every input and evaluation source is SHA-256 fingerprinted. The original joint evidence audit is verified read-only. Output score files are immutable: matching reruns verify their bytes; conflicting reruns require a new output name. Runtime telemetry is separate from scientific-result fingerprints.

Paired circular block bootstrap intervals use 2,000 replicates, three-month blocks and seed 20261020. The same month indices are resampled across strategies and model seeds. Seed metrics are averaged rather than concatenated into 72 fictitious market months. Intervals are conditional exploratory diagnostics, not selection-adjusted guarantees. A Bartlett HAC long-run Sharpe sensitivity with three lags accompanies conventional Sharpe; it is not asserted to be an exact annual Sharpe.

The evaluation months have already informed many candidate choices. This framework cannot turn them into an untouched holdout or certify a deployment winner.

## Run and verify

From the repository root:

```sh
.venv/bin/python -m pytest -q experiments/2026-10-portfolio-evaluation-v1
.venv/bin/ruff check --no-force-exclude experiments/2026-10-portfolio-evaluation-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/benchmark.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/run_evaluation.py --covariance-mode scenarios --output cached_scenarios_v1
```

The last command evaluates the accepted 14-input-plus-time joint baseline for seeds 27/127/227; it does not add `hy_spread_chg_1p`. Repeat it unchanged to check byte-for-byte result agreement. Use `--covariance-mode none --output cached_marginals_v1` to verify that missing covariance is handled explicitly.

Output directories contain the specification-linked manifest, per-seed JSON/CSV metrics, fixed/adaptive portfolio calibration, monthly IC and return paths, paired benchmark comparisons, a verification seal, and separate runtime telemetry. No fresh GP fits or allocation optimization runs are launched. Linear rank allocations are a deterministic diagnostic; saved MaxRet allocations remain labeled as lacking an explicit variance budget. A new risk-aware allocator comparison belongs to the next experiment.

Numerical scoring is batched across months and assets. Shared training references are computed once, not separately per seed. NumPy/BLAS threading is bounded to avoid nested oversubscription; small evaluation arrays do not justify launching 14 heavyweight worker processes. The benchmark measures kernel time against an independent SciPy loop; it is not a claim about GP-fit acceleration.

All work is contained here. Earlier sources, results, datasets and unfinished ranking-first modules remain untouched and are not imported. Removing this new directory rolls back the framework without altering prior studies. No production package APIs change. Static Pylance diagnostics require a separately available language-server tool; numerical tests and linting are the runnable local gates here.

## Scoring references

- [Gneiting and Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437). CRPS is implemented as a nonnegative loss, opposite the paper's reward-sign convention.
- [SciPy multivariate normal documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html). Independent test reference for joint Gaussian density.
- [Lo, The Statistics of Sharpe Ratios](https://alo.mit.edu/publications/page/18/). Motivation for reporting serial-dependence sensitivity rather than assuming square-root annualization is universally exact.
