# Matched GP and momentum ranking diagnostic

Change only the ranking signal used by the approved Sharpe leader's hard-rank allocator. Replay the GP control, then compare it with 12-minus-1 momentum ranks under the identical seed/month covariance, original absolute variance budget, allocator, numerical settings and investment constraints. Both arms use all 15 ETFs, seeds 27/127/227, and the same 24 realized return months, June 2024 through May 2026. There are no new GP fits or production changes.

## Frozen investment problem

The allocator is the existing `allocate_checked` and `StableScoreAllocator`, not a new implementation. Both arms maximize `scores @ weights - 1e-8 * sum(weights_squared)` subject to long-only, full investment, effective assets at least seven, and the original monthly training-only historical equal-weight variance ceiling. The covariance is exactly the saved 50% broad plus 50% compact observed GP matrix, matched separately for each seed and month. There are no ETF caps, CVaR, fixed stock/bond split, continuous risk penalty, calibration, covariance repair or fallback weights.

GP scores are the saved ascending average ranks of the family-blended GP means, divided by 15. Momentum replaces those scores with ascending average ranks of `adjusted_price[t-1] / adjusted_price[t-12] - 1`, also divided by 15. Ties use average ranks in both arms. Unlike the previous standalone momentum benchmark, this does not select seven ETFs or impose equal holdings. The same optimizer chooses weights from all 15 assets.

The momentum signals and their causal lookback evidence come from the sealed standing benchmark suite. Their ratios are independently reconstructed from its frozen historical prices before allocation. Original GP rank 2, maxiter 15, training window, input features and mean-blend settings remain unchanged; no fits are launched. The GP control's ranks and 50/50 covariance are checked against their original sealed inputs, and replayed weights must match within 1e-7 absolute and portfolio metrics within 1e-8 absolute.

## Execution and outputs

From the repository root:

```sh
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync python experiments/2026-10-portfolio-evaluation-v1/matched_signal_v1/run_matched_signal.py --execute
```

Six independent signal/seed jobs evaluate 24 months each, for 144 monthly allocations including the GP control replays. Processes are limited to the smaller of six jobs, 15 workers and CPU count minus two. BLAS is single-threaded per worker. Successful jobs checkpoint locally; a failed or inaccurate monthly solve blocks the complete comparison. No months are omitted and no weights are clipped, normalized, carried forward or substituted. Existing numerical retries solve the identical original problem; they do not relax it.

The one command writes and verifies an immutable completion seal. Repeating it verifies completed evidence without rerunning. Only this new folder is writable; all previous experiment files and the current baseline manifest are checked against their fingerprints and remain unchanged.

`outputs/` includes the input panel and policy contract, all per-seed solver/preflight records, monthly weights, returns, wealth, rank IC by scope, variance ratios, effective counts, paired exploratory differences, control replay audit, chart and findings. `gp_weights.npz` and `momentum_weights.npz` include explicit `weights`, `dates`, `assets` and `seeds` for reuse in the benchmark suite. For example, from the repository root:

```sh
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync python experiments/2026-10-portfolio-evaluation-v1/benchmark_suite_v1/run_benchmarks.py --execute --comparison-npz experiments/2026-10-portfolio-evaluation-v1/matched_signal_v1/outputs/momentum_weights.npz --label "Momentum with matched hard-rank allocator" --run-id matched_momentum
```

That optional benchmark-suite comparison is not required or automatically launched by this diagnostic.

## Interpretation

The matched comparison measures the effect of replacing this specific ranking signal conditional on the retained GP risk and allocator. It does not test whether the covariance is accurate or whether momentum can work without a GP. Momentum's trailing multi-month magnitude is not a one-month expected-return forecast; no momentum predictive NLL is invented.

Report conventional Sharpe, cumulative total return, ending wealth from $10,000, annualized excess volatility, month-end drawdown, effective assets, equity allocation, binding variance ceilings and rank IC globally and within each family. Equity fractions are unconstrained, exactly as in the GP control. A mean monthly return-gap attribution by ETF family is an accounting identity, not a causal family effect or attribution of Sharpe/compound return.

The earlier top-seven equal-weight momentum curve is reused as explicitly unmatched context only, without new allocations. Its contrast with matched momentum changes the allocation/risk handling package, not the momentum signal. No automatic model promotion or parameter search occurs.

All results use the same repeatedly reused 24 market months. Seeds are initialization replicates, not independent market histories. Three-month paired circular block bootstraps (2,000 samples, seed 20261020) preserve common months across arms/seeds; intervals are conditional and not selection-adjusted. IC intervals summarize mean IC differences, not a Sharpe ratio of IC. Retrospective adjusted-price vintage, frictionless month-end execution and gross-of-explicit-cost limitations remain unchanged. The chart shows mean seed wealth and min-to-max initialization bands, not confidence intervals or a newly tradable ensemble.

## Verification and rollback

```sh
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync ruff check experiments/2026-10-portfolio-evaluation-v1/matched_signal_v1
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync ruff format --check experiments/2026-10-portfolio-evaluation-v1/matched_signal_v1
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync pytest -q experiments/2026-10-portfolio-evaluation-v1/matched_signal_v1
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync python experiments/2026-10-portfolio-evaluation-v1/matched_signal_v1/run_matched_signal.py --verify
```

Pylance diagnostics are unavailable in the current tool environment. Tests, lint and documentation checks do not assert a completed Pylance gate. No maintained package exports or architecture boundaries change. Preserve desired evidence before removing only this new folder; production and prior experiments require no rollback. Changed inputs or conflicting partial results require a new version, never overwriting protected evidence.
