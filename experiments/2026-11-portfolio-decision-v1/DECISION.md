# November 2026 ETF portfolio decision

**Validated target-weight recommendation.** Use the selected challenger: it passed the frozen conservative promotion rule.

Selected configuration: **average_prior48_sharpe_prior48_return_prior48_ranking** (`recipe_04daf5fc6b9240560445`).

Logical as-of date: `2026-10-31`. Market-data/forecast origin: `2026-10-30`. Forecast month: `2026-11`.

October 31 is a Saturday; the latest completed regular-session bar is October 30. No fictitious Saturday market bar or unavailable November training target is used.

## Decision and evidence

Against the identically pooled incumbent over the 52-month historical decision panel: Sharpe difference +0.109; cumulative-return difference +8.08 percentage points; ranking-IC difference +0.048.

The selection decision was frozen using the 52 earlier completed return months before October outcomes were scored. October is an appended descriptive check, not a reason to change the chosen configuration. Operational gates are conservative decision hurdles, not statistical significance tests or a proof of future superiority.

| Preregistered gate                        | Passed   |
|:------------------------------------------|:---------|
| all52_IC_improvement_0p01                 | True     |
| all52_return_noninferior                  | True     |
| all52_sharpe_improvement_0p05             | True     |
| drawdown_noninferior_1pp                  | True     |
| every_leave_one_month_out_sharpe_positive | True     |
| four_of_six_positive_blocks               | True     |
| later28_IC_noninferior_0p01               | True     |
| later28_return_noninferior_1pp            | True     |
| later28_sharpe_noninferior_0p05           | True     |
| no_family_IC_loss_over_0p05               | True     |

6 finalists passed every frozen gate. The selected recipe is higher than each of the other 5 eligible recipes on all three 52-month objectives: Sharpe, cumulative return and rank IC. It is the sole eligible Pareto candidate; the closest-to-incumbent tie-break was not needed. Ineligible candidates were excluded by the recorded gates, even if an individual objective was higher.

For example, prior48_sharpe has slightly higher 52-month Sharpe (0.7265 versus 0.7213) and cumulative return (70.387% versus 70.177%), but fails `no_family_IC_loss_over_0p05`. prior48_return and prior48_ranking fail that same family noninferiority gate. The frozen rule deliberately rejects those tradeoffs; it does not simply pick the highest Sharpe.

The 15-finalist frozen decision table and exact rejection reasons are in COMPARISON.md; COMPARISON.json preserves the unrounded metrics and gate flags.

| Configuration                                         | Period                        |   Sharpe | Return   |   Rank IC | Max drawdown   | Average equity   |
|:------------------------------------------------------|:------------------------------|---------:|:---------|----------:|:---------------|:-----------------|
| average_prior48_sharpe_prior48_return_prior48_ranking | Jun 2022–May 2024 (24 months) |   -0.066 | 6.12%    |    -0.043 | -12.25%        | 62.06%           |
| average_prior48_sharpe_prior48_return_prior48_ranking | Jun 2024–Sep 2026 (28 months) |    1.591 | 60.36%   |     0.336 | -6.42%         | 78.54%           |
| average_prior48_sharpe_prior48_return_prior48_ranking | Jun 2024–Oct 2026 (29 months) |    1.588 | 61.97%   |     0.336 | -6.42%         | 78.06%           |
| average_prior48_sharpe_prior48_return_prior48_ranking | Jun 2022–Sep 2026 (52 months) |    0.721 | 70.18%   |     0.161 | -12.25%        | 70.93%           |
| average_prior48_sharpe_prior48_return_prior48_ranking | Jun 2022–Oct 2026 (53 months) |    0.727 | 71.88%   |     0.165 | -12.25%        | 70.82%           |
| incumbent                                             | Jun 2022–May 2024 (24 months) |   -0.187 | 2.45%    |    -0.092 | -14.64%        | 63.86%           |
| incumbent                                             | Jun 2024–Sep 2026 (28 months) |    1.531 | 58.22%   |     0.29  | -6.42%         | 80.69%           |
| incumbent                                             | Jun 2024–Oct 2026 (29 months) |    1.502 | 58.84%   |     0.283 | -6.42%         | 80.45%           |
| incumbent                                             | Jun 2022–Sep 2026 (52 months) |    0.613 | 62.09%   |     0.113 | -14.64%        | 72.92%           |
| incumbent                                             | Jun 2022–Oct 2026 (53 months) |    0.608 | 62.73%   |     0.113 | -14.64%        | 72.94%           |

The selected recipe is not a universal benchmark winner: over 53 months its cumulative return is 71.88%, versus 79.66% for Schwab aggressive. Its conventional Sharpe is 0.727, versus 0.676 for Schwab. Higher risk-adjusted performance did not mean higher ending wealth in this panel.

The 52-month family tradeoff is also visible: U.S. fixed-income within-family IC falls from 0.227 to 0.181, within the allowed 0.05 loss, while international fixed-income IC increases from -0.029 to 0.077. The pooled ranking gain is not an improvement in every family.

## Exact live policy

Apply fixed family prediction fractions, average means equally across seeds 27/127/227, average the approved 50/50 broad/compact observed covariance pool equally across those seeds, and allocate once. Every historical finalist is replayed with exactly this single-portfolio policy. The old research curves averaging three separately optimized seed portfolios are not used as its performance record.

GP rank 2/maxiter 15; t_index retained; raw hy_spread and its change excluded. Hard-rank allocation, long-only/full investment, effective assets ≥7, no ETF cap, and the unchanged causal historical-EW variance ceiling. Fixed 50/50 broad/compact covariance. No CVaR, uncertainty recalibration, continuous risk penalty, new forest blender, monthly discretionary blend selection, fallback weights or constraint relaxation.

### Fixed prediction blend

| Family                     | Broad expanding   | Compact expanding   | Broad rolling 48   |
|:---------------------------|:------------------|:--------------------|:-------------------|
| U.S. equity                | 42.7083%          | 50.0000%            | 7.2917%            |
| International equity       | 94.2708%          | 5.2083%             | 0.5208%            |
| U.S. fixed income          | 0.0000%           | 7.2917%             | 92.7083%           |
| International fixed income | 2.6042%           | 2.0833%             | 95.3125%           |

These are prediction coefficients, not ETF portfolio weights. They do not change by month or seed. The covariance pool remains 50/50 even when prediction coefficients differ.

## November target weights

| ETF   |   Weight (decimal) | Weight (%)   |
|:------|-------------------:|:-------------|
| SPY   |          0.111111  | 11.1111%     |
| MGK   |          0.20312   | 20.3120%     |
| VTV   |          0.0421046 | 4.2105%      |
| IWM   |          0.0881089 | 8.8109%      |
| VEA   |          0.0651067 | 6.5107%      |
| VWO   |          0.157115  | 15.7115%     |
| VSS   |          0.134113  | 13.4113%     |
| EWX   |          0.180118  | 18.0118%     |
| BND   |          0         | 0.0000%      |
| IEF   |          0         | 0.0000%      |
| LQD   |          0         | 0.0000%      |
| HYG   |          0         | 0.0000%      |
| BNDX  |          0         | 0.0000%      |
| VWOB  |          0         | 0.0000%      |
| HYEM  |          0.0191024 | 1.9102%      |

All fifteen assets are included, including zero weights. Percentages above are display-rounded; NOVEMBER_WEIGHTS.csv retains raw decimal optimizer weights and is the machine-readable source.

### Family exposure

| Family                     | November target allocation   |
|:---------------------------|:-----------------------------|
| U.S. equity                | 44.4444%                     |
| International equity       | 53.6453%                     |
| U.S. fixed income          | 0.0000%                      |
| International fixed income | 1.9102%                      |

The November target is 98.09% equities and 1.91% fixed income. Minimum effective assets 7 constrains concentration; it does not require an equity/fixed-income balance or a holding in every family. No family allocation floor or extra constraint was imposed. Nonzero fixed-income target holdings: HYEM.

## Validation

| Check                                     |   Recorded value |
|:------------------------------------------|-----------------:|
| Sum of raw weights                        |      1           |
| Minimum raw weight                        |      0           |
| Maximum raw weight                        |      0.20312     |
| Effective asset count                     |      7           |
| Predicted variance (decimal squared)      |      0.000706501 |
| Causal variance ceiling (decimal squared) |      0.000883389 |
| Predicted variance / ceiling              |      0.799762    |

See VALIDATION.json for strict solver acceptance, raw-weight feasibility, independent conic parity, data/target availability, source fingerprints and deterministic reproduction checks. No infeasible month, fit failure or portfolio was silently substituted.

Fit-budget evidence:

```json
{
  "completed_fits": 18,
  "fit_attempts_remaining": 110,
  "fit_attempts_used": 18
}
```

## Uncertainty and limitations

The earlier/later reversal is real in the legacy saved results: different historical periods favored different configurations. Much of the pooled ranking reversal was between ETF families; month-specific exposure changes mattered more than average equity weight alone. The supporting posthoc diagnosis is outputs/reversal_analysis.md.

All examined historical periods and candidate ideas have been inspected during research. These are retrospective comparisons, not a pristine holdout or a claim of a globally best future portfolio. NLL and covariance diagnostics do not independently establish ranking or investment value. Seed variation is numerical sensitivity, not independent market histories or a confidence interval.

The vendor's MGK adjusted-price overlap showed an approximately 0.08247% vintage discrepancy. A documented historical dividend revision was recognized but not back-applied to frozen history. This is a data-vintage limitation, not an additional strategy return or an unrecorded repair.

Backtests are gross of explicit trading costs/taxes and use the established return convention. The plotted fixed-weight Schwab proxy and top7 momentum benchmark are the only standing benchmarks.

## Execution timing and scope

Oct31 logical as-of uses Oct30 closing observations; possible Nov2 execution cannot obtain an Oct30 closing fill after observing that close. Target weights, not orders.

The output is target weights for human review, not orders. No trades or brokerage changes were made; production defaults and prior experiment evidence remain unchanged.

## Reproduce and verify

```sh
uv run --locked --no-sync python experiments/2026-11-portfolio-decision-v1/run_decision.py --verify
```

CONFIG.json records exact model inputs/kernels/windows, pooling, covariance, allocator and cutoff. VALIDATION.json and the reproduction manifest carry the acceptance results and fingerprints.
