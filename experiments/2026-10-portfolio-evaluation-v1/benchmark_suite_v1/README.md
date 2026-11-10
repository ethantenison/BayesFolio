# Standing portfolio benchmarks

Compare saved BayesFolio allocations with five deterministic monthly rebalanced strategies. The default comparator is the approved hard-rank Sharpe leader, with its existing three seed allocations unchanged. The study uses the same 24 realized return months, June 2024 through May 2026, starting from $10,000 with no cash flows and reinvested distributions. This is an isolated research suite, not a change to production allocation or a live trading recommendation.

## Approved benchmark rules

| ID | Rule |
|---|---|
| schwab_aggressive | 50% SPY, 20% IWM, 25% VEA, 5% cash. Reuse the verified publication-aware schedule from the previous Schwab study. |
| equal_weight | 1/N across the same 15 ETFs: 6.667% each, or 53.333% equities and 46.667% fixed income. |
| fixed_80_20 | 80% equally across eight equity ETFs and 20% equally across seven bond ETFs: 10% and 2.857% per member, respectively. |
| inverse_volatility | Normalize 1/sigma across all 15 ETFs, using sample standard deviation of 36 monthly adjusted total returns available through decision month t. No correlations or covariance optimization; not full risk parity. |
| momentum_top7 | Rank adjusted P[t-1]/P[t-12]-1; equally weight seven winners. Ties use alphabetical tickers. This skips the most recent month and uses 11 monthly returns. Stay invested in the best seven even when all scores are negative. |

These are fixed approved rules, not parameters chosen by searching the comparison results. The four same-universe strategies keep IWM and exclude REITs and commodity ETFs. Their universe is SPY, MGK, VTV, IWM, VEA, VWO, VSS, EWX, BND, IEF, LQD, HYG, BNDX, VWOB and HYEM. No additional ETF cap, variance budget, effective-count floor, CVaR, leverage, shorting, turnover objective or fallback is imposed. The GP comparator retains its already-approved allocator settings.

The definitions live in `suite.json`; `strategy_math.py` contains reusable pure numerical functions. Numerical return conversion, conventional Sharpe, drawdown, paired bootstrap, dollar-holdings validation and evidence fingerprinting reuse the existing evaluation framework rather than reimplementing those rules.

## Data and timing

Signals use registered, immutable daily adjusted-price snapshots in the repaired experiment's `artifacts/v1/raw`, reduced to actual last-observed trading dates per month. The default history covers May 2021 through April 2026, giving the first decision 37 prices for 36 returns. Every asset must have the same complete monthly date axis. Missing, nonpositive or nonfinite values, zero volatility and insufficient lookbacks raise errors; no filling, silent dropped months, volatility floors or allocation fallback is allowed.

At decision t, only monthly closes dated at or before t enter the rules. The first volatility window is June 2021 through May 2024 realized returns; the first momentum endpoints are May 2023 and April 2024. Synthetic future-price poisoning tests ensure that prices after a decision cannot affect its signals or weights. All realized evaluation returns use the original frozen targets and RF, not refreshed prices. The total-return formula and all portfolio wealth paths are independently checked against dollar holdings.

This preserves the existing frictionless month-end trading convention; it is not a next-open, transaction-cost or broker-execution simulation. Adjusted prices are a frozen retrospective vintage, not a point-in-time vendor archive. Cash earns the shared frozen risk-free proxy, not Schwab's actual sweep rate. No dated change to these specific Schwab classic weights was verified in the previous study; no annual update is invented. See its preserved [source schedule](../schwab_benchmark_v1/weights_schedule.json) and [findings](../schwab_benchmark_v1/FINDINGS.md).

## One command comparison

From the repository root:

```sh
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync python experiments/2026-10-portfolio-evaluation-v1/benchmark_suite_v1/run_benchmarks.py --execute
```

This computes the deterministic benchmark paths, reproduces the current Sharpe leader, generates the chart and findings, and seals the inputs and outputs under `outputs/current_sharpe_leader`. It does not refit GPs, run an allocation optimizer, refresh data, edit the current baseline or change previous sealed studies. Tiny monthly array operations are vectorized and BLAS threads are bounded to one; there are no expensive fits to distribute across processes.

For visual inspection before sealing, use `--preview`. It performs the same calculations but writes only a PNG in a newly created temporary directory and prints that path. No study output is changed. A repeated `--execute` on a completed run only verifies its evidence; it does not overwrite or rerun the study.

## Comparing another BayesFolio approach

The runner accepts another saved NPZ with `weights`, `dates` and `assets`. Dates are ISO date strings and tickers are strings in non-pickled NumPy arrays. Axes must match this fixed evaluation window and ETF order exactly. Weights must have shape `(24, 15)` or `(runs, 24, 15)`, long-only and fully invested, without a cash column. Missing axes or invalid weights fail explicitly; no normalization or fallback occurs. Cash-inclusive benchmark weights are recorded internally with a final cash column.

```sh
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync python experiments/2026-10-portfolio-evaluation-v1/benchmark_suite_v1/run_benchmarks.py --execute --comparison-npz /absolute/path/saved_weights.npz --label "New BayesFolio approach" --run-id new_approach
```

That writes a separate immutable run. A new evaluation period or universe needs a new versioned context and study rather than silently joining different dates. Reusable numerical functions accept other complete universes/date grids, but this runner intentionally keeps the approved comparison context fixed.

## Evidence and interpretation

Each run contains `paths.npz` (all monthly weights, returns, wealth, historical signal prices, lookback dates, volatility and momentum scores), `results.json` (performance, exposures, effective asset counts and paired exploratory bootstrap differences), `audit.json`, `contract.json`, `growth_10000.png`, `FINDINGS.md` and `COMPLETED.json`.

The GP curve is mean wealth across independently compounded seed paths; its band is their range, not a confidence interval or a new tradable ensemble. Summary risk statistics average the individual-seed statistics, never concatenate seeds into extra months. Fixed allocation benchmarks do not produce return-density forecasts, so no NLL is invented for them. Comparisons emphasize return, conventional Sharpe, volatility and month-end drawdown. All are gross of explicit costs and taxes; costs embedded in adjusted ETF returns are not deducted twice.

Different capital/risk exposures mean that outperformance cannot isolate the value of GP forecasts or covariance alone. The GP was chosen after extensive reuse of these 24 months. Neither the chart nor the non-selection-adjusted paired bootstrap establishes future outperformance. Freeze these benchmark rules for later comparisons instead of tuning them after seeing winners.

## Verification and rollback

```sh
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync ruff check experiments/2026-10-portfolio-evaluation-v1/benchmark_suite_v1
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync ruff format --check experiments/2026-10-portfolio-evaluation-v1/benchmark_suite_v1
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync pytest -q experiments/2026-10-portfolio-evaluation-v1/benchmark_suite_v1 experiments/2026-10-portfolio-evaluation-v1/schwab_benchmark_v1
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync python experiments/2026-10-portfolio-evaluation-v1/benchmark_suite_v1/run_benchmarks.py --verify
```

Pylance diagnostics are unavailable in the current tool environment; passing lint, documentation checks and numerical tests does not assert a completed Pylance gate. No package exports or architecture boundaries changed. All new code and outputs are confined here; preserve desired evidence before removing only this folder. Prior studies and production require no rollback. Input changes or conflicting partial outputs require a new run/version, never overwriting protected evidence.
