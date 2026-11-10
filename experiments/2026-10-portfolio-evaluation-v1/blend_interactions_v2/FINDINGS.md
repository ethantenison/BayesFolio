# Missing family prediction blending interactions

Testing the missing international interactions improved both previously shortlisted candidates. The new highest-Sharpe blend reached 1.8973, compared with 1.8772 previously. A ranking-focused candidate improved from IC 0.3551 and Sharpe 1.8761 to IC 0.3566 and Sharpe 1.8831 simply by adding 6.25% compact international fixed-income predictions.

The gains are modest and exploratory. Every candidate reuses the same 24 market months; expanding the search does not create new validation evidence. The results support retaining a small compact contribution for international fixed income, not claiming a uniquely optimal fraction or an out-of-sample improvement.

## Scope and coverage

This was a complete factorial search over the following compact-model prediction fractions. The remainder always came from the broad model.

| Family | Compact fractions tested |
|---|---|
| U.S. equity | 25%, 37.5%, 50%, 62.5% |
| U.S. fixed income | 0%, 6.25%, 12.5%, 25%, 37.5%, 50% |
| International equity | 0%, 6.25%, 12.5%, 25% |
| International fixed income | 0%, 6.25%, 12.5%, 25%, 37.5%, 50%, 75%, 100% |

The 4 × 6 × 4 × 8 grid contains 768 configurations: 748 new and 20 reused from the completed study. Across three seeds and 24 months, the new configurations produced 53,856 monthly portfolios. All 55,296 portfolios in the full grid, including reused controls, passed the independent audit. Combined with all 151 previous recipes, the comparison now contains 899 distinct recipes. This closes every interaction within the stated grid, not every possible continuous blend or blending method.

The same two joint models supplied all predictions: broad with 14 non-time inputs, and compact with five, each fitted on all 15 ETF tasks and retaining `t_index`. Rank remained 2 and maxiter 15. No new GP fitting, features, calibration or family-specific GP models were introduced.

Covariance was held at the same arithmetic 50/50 average of the two complete observed-return covariance matrices in every case. The prediction-blend formula was unchanged: `(1 − compact_fraction) × broad_mean + compact_fraction × compact_mean`. Fractions were fixed across months and shared within each family. Every portfolio used the existing full 15-ETF rank allocator, combined effective count at least seven, original training-only variance budget, long-only/full investment, no position caps, no CVaR and no fallback weights. This batch did not repeat the 80/20 sleeve experiment.

## Leading candidates

All metrics average seeds 27, 127 and 227 over the same 24 forecast origins, May 2024 through April 2026. Return is cumulative gross total return, not annualized. IC is mean monthly cross-sectional Spearman correlation across 15 ETFs.

| Candidate | Compact U.S. EQ | Compact U.S. FI | Compact Intl EQ | Compact Intl FI | Sharpe | Return | Pooled IC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous Sharpe leader | 50% | 25% | 0% | 0% | 1.8772 | 60.02% | 0.3436 |
| New Sharpe leader | 50% | 37.5% | 0% | 25% | **1.8973** | 60.26% | 0.3489 |
| Previous ranking leader | 37.5% | 25% | 0% | 0% | 1.8761 | 60.00% | 0.3551 |
| Ranking-focused shortlist candidate | 37.5% | 25% | 0% | 6.25% | **1.8831** | **60.20%** | **0.3566** |
| Highest IC anywhere in the new grid | 37.5% | 25% | 25% | 25% | 1.8116 | 58.09% | 0.3580 |

The new Sharpe leader improved Sharpe in all three seeds, by approximately 0.0103, 0.0242 and 0.0256. Its mean return improvement was about 0.25 percentage points, not an improvement in every seed's return. Its annualized excess volatility was 10.56%, versus 10.64% for the old Sharpe leader.

The ranking-focused shortlist candidate improved Sharpe in all three seeds relative to the previous ranking leader. It is a nondominated candidate, not an arbitrary selection from a weaker part of the frontier. It gives up about 0.014 Sharpe against the new Sharpe leader while increasing pooled IC from 0.3489 to 0.3566.

Maximizing IC alone produced a further IC gain of only about 0.0014 over that balanced candidate but reduced Sharpe by approximately 0.072. Given the portfolio objective, the highest-IC recipe is a tradeoff point rather than an unqualified winner.

## What the previously missing comparisons showed

Holding the previous Sharpe leader's U.S. fractions fixed at 50% compact for equities and 25% compact for fixed income, with broad-only international equities:

| Compact international FI fraction | Portfolio Sharpe | Total return | Pooled IC | International FI IC |
|---|---:|---:|---:|---:|
| 0% | 1.8772 | 60.02% | 0.3436 | 0.1250 |
| 6.25% | 1.8865 | 60.28% | 0.3460 | 0.1458 |
| 12.5% | 1.8836 | 60.18% | 0.3454 | 0.1528 |
| 25% | 1.8824 | 60.08% | 0.3494 | 0.1667 |
| 50% | 1.8473 | 58.90% | 0.3291 | 0.2014 |
| 100% | 1.8350 | 57.36% | 0.3070 | 0.0625 |

This directly answers the open question: adding small compact international fixed-income contributions to the final U.S. blend did improve both Sharpe and pooled ranking in several matched comparisons. A higher family-only IC did not guarantee better pooled IC or portfolio performance: 50% compact international fixed income had the strongest international FI IC in this table but weaker portfolio results.

The new overall Sharpe leader further changes U.S. fixed income to 37.5% compact. Its advantage therefore reflects an interaction between U.S. and international fixed-income prediction blends, not solely the addition of international predictions. The previous 25% U.S. fixed-income weight was a tested winner under a narrower search, not a constraint or a universally optimal coefficient.

![Matched international fixed-income prediction blending](outputs/international_fixed_income.png)

Each legend percentage is a compact-model prediction weight, not a portfolio capital allocation. Lines connect the evaluated points for readability; they do not represent continuous weight optimization.

## International equities and the objective tradeoff

Both shortlisted high-performance candidates retained broad-only international-equity predictions. However, the highest pooled-IC configuration used 25% compact international equities. International equity IC increased slightly, from 0.1417 to 0.1500, in that configuration.

The narrower conclusion is that compact international-equity contributions did not enter the leading Sharpe/strong-ranking compromises in this grid. They were not uniformly useless for every objective. Increasing pooled ranking slightly while giving up considerable Sharpe remains possible, which is why the complete Pareto comparison is retained.

Across the old and new studies there are 16 configurations nondominated on Sharpe, pooled IC and Gaussian joint NLL. The two shortlisted candidates above have joint NLL −38.8312 and −38.8493 respectively. Covariance was identical across the new grid, so these differences reflect prediction means, not new covariance blending or uncertainty calibration. Return, RMSE, asset-level diagnostics, fixed-reference portfolio scores, complete covariance, every monthly forecast and all raw weights remain saved per configuration.

## Takeaways

1. **The missing international fixed-income combinations mattered.** A small compact contribution improved both previously shortlisted U.S. configurations.
2. **Retain two candidates rather than collapsing the objectives into one winner.** The 50%/37.5%/0%/25% compact fractions lead Sharpe; the 37.5%/25%/0%/6.25% fractions provide a stronger ranking compromise. The order here is U.S. equity, U.S. fixed income, international equity, international fixed income.
3. **Do not treat the precise fractions as identified truths.** This was another adaptive search on reused months, with no untouched holdout or transaction-cost validation. Three seeds are not independent market histories. No production model or portfolio policy was promoted.

## Artifacts and verification

The [full 768-configuration table](outputs/all_configurations.md) links to each configuration's complete records. The [comparison](outputs/comparison.json) includes all 899 cumulative recipes and the Pareto identifiers. The [frozen plan](outputs/plan.json) distinguishes new evaluations from reused controls.

The [portfolio audit](outputs/audit.json) recomputed performance from raw accepted weights, returns and risk-free values, and checked all axes, full covariance positive definiteness, capital, risk, concentration and solver status. The [panel audit](outputs/panel_audit.json) independently verified all 768 prediction formulas and pooled IC values and confirmed that every case used the same covariance. No allocations failed or used substitute weights. An initial pre-run coverage test caught a grid-count mismatch; it was corrected before the final plan was frozen or any experiment was launched.

All code and results are contained in this new folder; the previous sealed studies remain unchanged. Ruff and the public-docstring gate passed, as did the combined 30-test new/related regression suite. Editor Pylance diagnostics were unavailable, so no Pylance-clean claim is made.

Verification from the repository root:

```sh
.venv/bin/ruff check experiments/2026-10-portfolio-evaluation-v1/blend_interactions_v2
.venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/blend_interactions_v2 experiments/2026-10-portfolio-evaluation-v1/blend_sleeves_v1 experiments/2026-10-portfolio-evaluation-v1/joint_source_hybrids_v2 experiments/2026-10-portfolio-evaluation-v1/minimum_seven_hybrid_v1 experiments/2026-10-portfolio-evaluation-v1/allocation_numerics_v2
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/blend_interactions_v2/run_interactions.py --verify
```

`run_interactions.py --execute` resumes using completed case files; it does not silently overwrite immutable evidence. To discard this research later, preserve any desired artifacts and remove only this new experiment folder. No production rollback is required.
