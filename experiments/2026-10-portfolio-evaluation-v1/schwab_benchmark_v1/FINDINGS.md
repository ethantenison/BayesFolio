# Sharpe leader versus Schwab portfolio allocations

The approved Sharpe leader grew a simulated $10,000 to $16,026 on average across its three model seeds, versus $14,588 for the Schwab-style aggressive allocation and $13,984 for moderately aggressive. This covers 24 monthly returns from June 2024 through May 2026, with monthly rebalancing, reinvested distributions and no external cash flows. Results are gross of explicit trading costs and taxes, not broker account records.

## Results

| Strategy | Ending value | Cumulative total return | Conventional Sharpe | Annualized excess volatility | Maximum month-end drawdown |
|---|---:|---:|---:|---:|---:|
| Sharpe leader, three-seed mean | $16,026.32 | 60.26% | 1.897 | 10.56% | -6.36% |
| Schwab-style aggressive | $14,587.54 | 45.88% | 1.314 | 11.70% | -6.02% |
| Schwab-style moderately aggressive | $13,984.09 | 39.84% | 1.298 | 10.06% | -4.93% |

The GP strategy ended $1,438.79 ahead of aggressive and $2,042.23 ahead of moderately aggressive. Those gaps equal 14.39 and 20.42 percentage points of cumulative return, respectively. It had lower realized volatility than aggressive but slightly higher volatility than moderately aggressive. Its maximum month-end drawdown was worse than either benchmark, so it did not dominate every risk measure. Intramonth drawdowns are not measured by these monthly observations.

| GP initialization seed | Ending value | Conventional Sharpe |
|---|---:|---:|
| 27 | $15,958.92 | 1.854 |
| 127 | $16,003.25 | 1.890 |
| 227 | $16,116.80 | 1.948 |

Every GP seed finished above both benchmarks. The blue central curve in [the chart](outputs/growth_10000.png) averages these three independently compounded wealth paths; it is not a newly optimized or tradable ensemble. Its band is the initialization range, not a confidence interval. GP risk statistics above average individual-seed statistics rather than scoring the mean wealth curve.

## Schwab weights and the possible 2026 update

The approved proxy allocations are SPY/IWM/VEA/BND/cash: 50/20/25/0/5 for aggressive and 45/15/20/15/5 for moderately aggressive. These simulate Schwab's classic five-asset-class illustrations, not Schwab Intelligent Portfolios or any investor's actual Schwab account.

The [classic profile table](https://hgtoolsweb.schwab.com/pcu/mp/enRiskProfileAnalysisForClassic.action) lists both mixes. Schwab's [March 27, 2026 retirement article](https://www.schwab.com/learn/story/beyond-4-rule-how-much-can-you-spend-retirement) still lists the same moderately aggressive weights. Its discussion of annual updates concerns projected returns, volatility and withdrawal estimates; that is not evidence of a change in target portfolio weights. The [MoneyWise allocation table](https://www.schwabmoneywise.com/essentials/finding-the-right-asset-allocation) also corroborates the aggressive mix.

The final page of the locally preserved [2018 Schwab-authored questionnaire](sources/schwab_investor_profile_2018.pdf), document code 1018-8X5W, shows the same two mixes. That PDF is hosted by a third party; its provenance and checksum are recorded in [the schedule](weights_schedule.json). It establishes that these weights predate the evaluation window, not that every intervening publication has been exhaustively checked.

No dated 2024-2026 revision to these specific target mixes was verified. Consequently, this comparison carries the documented weights forward rather than fabricating a January update. If dated revision evidence becomes available, a new version can apply it at the first monthly rebalance strictly after publication day and on or after its effective date. Same-day releases are conservatively deferred because publication time is unknown; future announcements cannot be backfilled.

VEA and BND are ETF proxies, not exact replications of every index Schwab uses in its illustrations. Cash earns the shared frozen risk-free return series, not a verified Schwab sweep rate. Schwab's published long-history illustrations rebalance annually; this test uses the requested monthly rebalancing. No GP effective-asset floor or variance constraint is imposed on Schwab's fixed weights.

## What this says about the value of the model

The complex strategy performed better in this particular retrospective comparison, including better conventional Sharpe, and not merely a higher return obtained with higher volatility than aggressive. That is useful descriptive evidence, but does not establish that the modeling effort will outperform a simple allocation in new months.

The GP feature sets, blending fractions and allocator were selected through repeated experiments on these same 24 evaluation months. Forecasts were generated chronologically, but strategy selection reused their outcomes. In addition, the GP trades 15 ETFs including emerging markets and several bond categories, while these passive proxies use four ETFs and cash. The gap cannot isolate forecasting skill from asset-universe or exposure differences. Explicit trading costs, spreads, taxes and implementation timing differences are not deducted; cached adjusted prices have the existing source-vintage limitations.

Exploratory paired circular block bootstraps use the existing 2,000 draws, three-month blocks and fixed seed, resampling identical month indices across strategies and model seeds. The GP-minus-benchmark Sharpe intervals are [-0.243, 1.947] against aggressive and [-0.229, 1.913] against moderately aggressive. Both include zero, even before adjusting for adaptive strategy selection. These are not evidence of statistically established or future outperformance. Seeds are initialization replicates, not three independent market histories.

Keep the approved research baselines unchanged. A meaningful next validation would freeze them and evaluate on genuinely new months alongside these simple benchmarks, rather than tuning again to this comparison.

## Implementation and verification

Only this new study folder was added. There were no GP fits, optimizer calls, data refreshes, production changes or git commits. The saved Sharpe leader's original return and Sharpe values were reproduced to absolute tolerance 1e-12. An independent dollar-holdings calculation checks every monthly account value and self-financing rebalance to 1e-7 dollars. The audit includes all 24 months, both benchmarks and all three GP seeds. Input and output checksums protect the previous experiment evidence and current baseline manifest.

Ruff lint and formatting checks passed. The targeted benchmark tests passed 12 cases; the combined relevant blending, covariance, risk, uncertainty-rank and benchmark suite passed 85 tests. The chart was visually inspected for legibility and overlapping labels. No public package API or architecture boundary changed.

From the repository root, the exact verification commands are:

```sh
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync ruff check experiments/2026-10-portfolio-evaluation-v1/schwab_benchmark_v1
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync ruff format --check experiments/2026-10-portfolio-evaluation-v1/schwab_benchmark_v1
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync pytest -q experiments/2026-10-portfolio-evaluation-v1/blend_sleeves_v1 experiments/2026-10-portfolio-evaluation-v1/blend_interactions_v2 experiments/2026-10-portfolio-evaluation-v1/covariance_matching_v1 experiments/2026-10-portfolio-evaluation-v1/rank_risk_v1 experiments/2026-10-portfolio-evaluation-v1/uncertainty_rank_v1 experiments/2026-10-portfolio-evaluation-v1/schwab_benchmark_v1
UV_CACHE_DIR=/private/tmp/bayesfolio-numerics-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync python experiments/2026-10-portfolio-evaluation-v1/schwab_benchmark_v1/compare_schwab.py --verify
```

The immutable outputs include individual monthly weights, returns and wealth, target-weight history, metrics, exploratory bootstrap comparisons and a hash-sealed completion record. A changed weight schedule requires a new version rather than overwriting this result. Prior experiments need no rollback; preserve any desired evidence before removing only this new study folder.
