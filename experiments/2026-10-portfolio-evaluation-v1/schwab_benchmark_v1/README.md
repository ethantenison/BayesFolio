# Sharpe leader and Schwab allocation comparison

Compare a $10,000 initial investment in the approved hard-rank Sharpe leader with ETF implementations of Schwab's classic aggressive and moderately aggressive allocation models. All strategies rebalance monthly, reinvest dividends, have no external cash flows, and use the same 24 target-return months from June 2024 through May 2026. There are no new GP fits, optimizer runs or data refreshes.

## Target weights and revision evidence

| Proxy | Aggressive | Moderately aggressive |
|---|---:|---:|
| SPY, U.S. large cap | 50% | 45% |
| IWM, U.S. small cap | 20% | 15% |
| VEA, developed international equity | 25% | 20% |
| BND, U.S. aggregate bonds | 0% | 15% |
| Cash, shared risk-free return proxy | 5% | 5% |

The 2018 Schwab-authored questionnaire, current classic profile table and March 27, 2026 retirement article corroborate these mixes. No dated change to these specific classic target weights was verified. This is not an exhaustive monthly archive, and no calendar-year update is invented. The simulation carries the documented mixes until a verified revision is available. All source references and the older PDF checksum are in `weights_schedule.json`.

If a revision is verified, the scheduler applies it at the first monthly rebalance strictly after publication day and on or after its effective date. It cannot backfill a later announcement into earlier months. The code tests both publication and effective-date conditions. A changed schedule requires a new versioned output study, preserving this comparison.

These are simulated ETF proxies, not actual Schwab account performance or Intelligent Portfolios. VEA is a developed-market proxy and not an exact replication of MSCI EAFE. Cash uses the existing frozen risk-free return series, not Schwab's actual bank-sweep rate. Schwab's historical illustrations use annual rebalancing; this comparison deliberately uses the user's requested monthly rebalancing. No GP effective-count floor or risk budget is imposed on Schwab's published target mixes.

## Return accounting and comparison

The cached target is discounted excess, `(1 + total_return) / (1 + RF) - 1`. Cash has zero discounted excess. Reconstruct portfolio total return as `(1 + weighted_ETF_target) * (1 + RF) - 1`, then compound the $10,000 investment. An independent dollar-holdings recursion checks every monthly account value and the self-financing rebalance.

All three GP seed paths are reproduced without changing their saved weights. The central GP curve is their mean dollar value, with a min-to-max initialization band and faint individual runs. It is not a new tradable ensemble strategy or a statistical confidence interval. GP summary statistics average individual-seed metrics, rather than concatenating seeds into 72 fictitious months. The ranking and Sharpe compromise remains an unchanged baseline but is not added to this requested chart.

Report ending wealth, cumulative return, conventional Sharpe, realized annualized excess volatility, drawdown and exploratory paired block-bootstrap differences. Comparisons are gross of explicit trading costs and taxes; fund-level costs already present in cached ETF returns are not subtracted twice. The selected GP model used this same window repeatedly, so a performance gap is descriptive and cannot establish future skill or selection-adjusted significance. Different asset universes and exposures also prevent attributing all outperformance to the GP or allocator alone.

## Verification and rollback

```sh
.venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/schwab_benchmark_v1
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/schwab_benchmark_v1/compare_schwab.py --execute
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/schwab_benchmark_v1/compare_schwab.py --seal
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/schwab_benchmark_v1/compare_schwab.py --verify
```

Only this new study folder is writable. Prior results and `CURRENT_BASELINES.json` are fingerprinted and remain unchanged. Preserve desired evidence before discarding only this new folder; no production rollback is required.
