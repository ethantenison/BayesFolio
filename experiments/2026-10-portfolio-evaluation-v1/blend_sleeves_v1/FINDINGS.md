# Compact and broad joint blending and separate sleeve results

Blending the two joint models improved the leading portfolio in this exploratory evaluation. The highest-Sharpe configuration blends U.S. equity predictions 50% compact and 50% broad, U.S. fixed-income predictions 25% compact and 75% broad, retains broad predictions for international ETFs, and averages the complete covariance matrices equally. Its mean conventional Sharpe is 1.877 and mean cumulative total return is 60.02%.

Fixing capital at 80% equities and 20% fixed income did not improve that result. Separate sleeve optimization returned 53.71% with Sharpe 1.753. A full-portfolio optimizer with the same 80/20 capital constraint produced almost the same outcome, so most of the reduction came from imposing the fixed capital split rather than the separate solvers.

These are candidates from two adaptive searches on the same 24 market months, not untouched validation results. Nothing was promoted to production.

## Scope and unchanged settings

The study evaluated 65 first-round and 86 second-round blending recipes, including endpoint controls that can produce identical forecasts. These are output-blending configurations, not 151 newly fitted feature masks. Every configuration used seeds 27, 127 and 227 and all 24 forecast origins from May 2024 through April 2026; each forecast targets the following month's return. Three additional allocation controls used the selected Sharpe leader. This produced 11,088 seed-month portfolios, all completed and independently audited. There were no new GP fits.

Both source models were fitted on the same 15 ETF tasks. The broad model has 14 non-time inputs; the compact model has five: `mom12m_skip1m`, `mom6m`, `cpi_chg_12p`, `hy_spread_z_12p`, and `term_spread`. Both retain `t_index`. The compact model is **not** the U.S.-equity-only model or the separate ten-input NLL-leading model.

Rank stayed at 2, maxiter at 15, and initial training history at 84 months. The allocator maximized forecast-derived rank scores with the existing tiny numerical tie-break penalty. All portfolios remained long-only and fully invested, with a combined effective-asset floor of seven, no ETF position cap, no CVaR, and no previous-weight fallback. The original training-only variance budget was unchanged. No new HY-spread feature was introduced, and uncertainty-calibration experiments remained deferred.

| Family | ETFs |
|---|---|
| U.S. equities | SPY, MGK, VTV, IWM |
| International equities | VEA, VWO, VSS, EWX |
| U.S. fixed income | BND, IEF, LQD, HYG |
| International fixed income | BNDX, VWOB, HYEM |

The exploratory selection rule was highest mean conventional Sharpe across the three seeds. IC and Gaussian joint NLL were also retained as objectives in a reported Pareto comparison, not used as NLL guardrails. Return, RMSE, drawdown, risk diagnostics, fixed-reference distribution scores, asset-level diagnostics, forecasts, full covariances and monthly weights were saved. All performance is gross because transaction costs were not supplied.

## What the two rounds tested

Round one varied compact prediction fractions by family, including global blends and the previous U.S.-only replacement. It compared two decision methods: ranking blended return forecasts, and Borda blending of the source models' percentile ranks. Borda coefficients are decision scores, not return predictions; forecast-distribution metrics still use the separately saved return means.

Five covariance methods were included:

1. Arithmetic covariance averaging.
2. Precision averaging, followed by inversion.
3. Log-Euclidean covariance averaging.
4. Averaging correlations and standard deviations, then reconstructing covariance.
5. Exact first-two-moment matching of a global two-model mixture, including the variance of model disagreement. Its reported Gaussian NLL scores the moment-matched Gaussian, not the exact mixture density.

All methods retained positive-definite complete covariance matrices. Covariance rows were not spliced, eigenvalues were not clipped, and realized errors did not determine a month's blend weights. Covariance interpolation is an ensemble assumption, not a posterior from a newly fitted single GP. See the [CVX portfolio formulation](https://www.cvxgrp.org/cvx_short_course/docs/applications/notebooks/portfolio_optimization.html) and the [original Log-Euclidean matrix framework](https://www-sop.inria.fr/asclepios/Publications/Vincent.Arsigny/Arsigny_SIAM_tensors_07.pdf) for the underlying mathematical constructions.

Round one's leader used a 50/50 U.S.-equity prediction blend, broad predictions elsewhere, and equal covariance averaging: Sharpe 1.844 and return 59.15%. The previous full compact replacement of U.S. predictions scored 1.780 and 57.70% with broad covariance.

The second plan was frozen after reading those results. It refined U.S. prediction fractions, tested small changes in the other families, used a finer covariance grid, and explored global Borda blends and interactions among family weights. In particular, it crossed the promising 50/50 U.S.-equity blend with compact contributions to U.S. fixed income, because a 25% U.S.-fixed-income contribution had helped in round one under a different U.S.-equity setting. This interaction produced the final leader. The [frozen second-round plan](outputs/round_2_plan.json) records the evidence and all 86 candidates.

## Leading portfolio results

Values below are means across the three seeds. Return is cumulative gross total return over the 24 forecast months, not annualized. IC is mean monthly cross-sectional Spearman correlation across all 15 ETFs.

| Configuration | Sharpe | Total return | Ranking IC | Annualized excess volatility |
|---|---:|---:|---:|---:|
| Broad 14-input joint model everywhere, broad covariance | 1.692 | 56.10% | 0.2965 | 11.08% |
| Previous compact-U.S.-prediction replacement, broad covariance | 1.780 | 57.70% | 0.3116 | 10.81% |
| First-round leader | 1.844 | 59.15% | 0.3219 | 10.68% |
| Final Sharpe leader | **1.877** | **60.02%** | 0.3436 | 10.64% |
| Final ranking leader | 1.876 | 60.00% | **0.3551** | 10.64% |
| Sharpe leader with fixed 80/20 capital, full optimization | 1.757 | 53.77% | 0.3436 | 10.18% |
| Sharpe leader with fixed 80/20 capital, within-sleeve ranks, full optimization | 1.756 | 53.87% | 0.3436 | 10.21% |
| Sharpe leader with separate 80/20 sleeves | 1.753 | 53.71% | 0.3436 | 10.20% |

The ranking leader uses 37.5% compact U.S.-equity predictions rather than 50%; its other prediction fractions and covariance are the same as the Sharpe leader. It sacrifices only 0.0012 Sharpe while improving pooled IC by 0.0116. Given the preference for ranking, **keep this candidate alongside the Sharpe leader rather than treating the tiny Sharpe difference as decisive**. The requested sleeve experiment used the preregistered Sharpe leader.

The Sharpe leader improved on the broad baseline and the previous hybrid in every seed. Its seed Sharpes were 1.843, 1.866 and 1.922. Neighboring settings with 37.5–50% compact U.S.-equity predictions and 25–37.5% compact U.S.-fixed-income predictions scored approximately 1.875–1.877. This is a promising neighborhood, not evidence that the exact coefficients are uniquely optimal.

### Prediction blending mattered more than covariance blending

With the winning predictions, retaining broad covariance alone scored Sharpe 1.8767 and return 59.97%; equal covariance averaging scored 1.8772 and 60.02%. The improvement from covariance averaging is only about 0.0005 Sharpe and 0.04 percentage points of total return. The winning portfolio's full risk ceiling bound in only one of 72 seed-months, whereas its effective-count floor was active throughout.

Matched nonlinear covariance methods near the first-round mean leader produced similarly small differences. They were not exhaustively crossed with every later prediction blend, so this does not establish that arithmetic averaging is universally best. **There is little support here for spending effort tuning covariance fusion under this specific allocator.**

The best tested Borda configuration scored Sharpe 1.767, below the 1.877 return-mean-blending leader. Global moment-mixture candidates also did not lead. Neither ranking the source models first nor assigning one global prediction mixture to every family improved the tested frontier's Sharpe leader.

### Better pooled portfolios do not mean every individual forecast improved

| Prediction configuration | U.S. equity IC | International equity IC | U.S. fixed-income IC | International fixed-income IC |
|---|---:|---:|---:|---:|
| Broad everywhere | 0.1333 | 0.1417 | 0.1250 | 0.1250 |
| Previous full compact replacement for U.S. equities | 0.2583 | 0.1417 | 0.1250 | 0.1250 |
| Final Sharpe leader | 0.1806 | 0.1417 | **0.4361** | 0.1250 |
| Final ranking leader | 0.1639 | 0.1417 | **0.4361** | 0.1250 |

The large within-family ranking gain was in U.S. fixed income. Full compact replacement still produces better U.S.-equity-only ranking than the portfolio leaders. For SPY, temporal Spearman correlation was 0.235 for broad, 0.588 for the previous compact replacement, and 0.496 for the final Sharpe leader. The corresponding prediction-amplitude ratios were 0.334, 0.828 and 0.498.

Therefore, the new portfolio leader does **not** resolve the U.S.-equity flattening mechanism, maximize SPY trend tracking, or prove that every task improved. The results favor balancing information from both models when allocating across asset families. Cross-family forecast levels and relative ordering matter in addition to each family's own ranking.

### NLL remains a tradeoff rather than a winner selector

Nine configurations were nondominated on Sharpe, ranking IC and Gaussian joint NLL. The best-NLL configuration scored joint NLL −39.0717, Sharpe 1.830 and IC 0.3478; it was not the Sharpe or IC leader. The Sharpe leader's NLL was −38.7999, compared with −38.8726 for the broad baseline. Thus its portfolio improvement did not require better joint NLL than the broad model. Lower NLL is better, and the absolute value depends on decimal return units.

The [complete comparison](outputs/comparison.json) contains all Pareto identifiers. [All 151 configurations](outputs/all_configurations.md) are ranked by Sharpe with exact fractions, methods and metrics; each row links to all seeds and monthly records.

## How the separate 80/20 optimizer worked

The same winning 15-task forecasts and full covariance were reused; there were no separate equity and bond GP fits. At every rebalance, one eight-ETF equity portfolio was normalized to 100% within its sleeve, and one seven-ETF fixed-income portfolio was normalized similarly. Their weights were then scaled by 0.8 and 0.2.

The effective-count floor of seven applied to the **combined portfolio**, not to each sleeve. A deterministic scalar concentration price coordinated the two separately solved sleeve objectives until the combined floor was satisfied. This avoids silently forcing seven equally weighted fixed-income holdings or inventing arbitrary sleeve-specific floors. It is separate sleeve optimization with limited constraint coordination, not completely uncoordinated solves.

For risk, each normalized sleeve was constrained to variance at most the original full-portfolio budget B. Since each sleeve volatility is at most √B, combined volatility is at most 0.8√B + 0.2√B = √B, including any admissible cross-sleeve correlation. This is a conservative sufficient condition for the original full risk ceiling; it is **not mathematically equivalent** to optimizing against the combined covariance alone. Combined risk was evaluated with the full covariance, including the equity–fixed-income cross term.

The full-vector 80/20 controls isolate these changes:

- Adding only the fixed 80/20 capital constraint reduced Sharpe from 1.877 to 1.757.
- Switching to within-sleeve ranks changed it to 1.756.
- Using the conservative separate-sleeve risk constraints changed it to 1.753. The equity sleeve risk constraint bound in seven of 72 seed-months; fixed income remained below its ceiling.

The unrestricted winner averaged 79.59% equity exposure but varied from 28.73% to 98.09% across seed-months. Fixing the same average allocation removes substantial monthly freedom and changes security weights too. This explains the controlled comparison; it does not establish a general market-timing advantage outside these reused months. Forecast IC and NLL did not change across allocator controls, because the forecasts did not change. Pooled IC of within-sleeve decision scores is not comparable to global forecast IC and should not be read as forecast deterioration.

![Portfolio wealth and monthly equity exposure](outputs/comparison.png)

## Takeaways and next decision

1. **Retain family-dependent mean blending as the strongest new direction.** Broad versus compact is not an all-or-nothing choice. Keep both the 50% U.S.-equity Sharpe leader and the 37.5% ranking leader, with 25% compact U.S.-fixed-income predictions and broad international predictions.
2. **Do not promote the fixed 80/20 separate-sleeve policy on performance grounds from this test.** It reduced volatility slightly but gave up return and Sharpe. Keep it available if a fixed strategic allocation is a policy preference rather than a performance hypothesis.
3. **Do not overinterpret the covariance blend or exact coefficients.** Broad covariance alone was effectively tied, the risk ceiling was usually inactive, and a small neighborhood of prediction fractions performed similarly.
4. **Portfolio performance and individual-ETF trend fidelity remain different goals.** The previous compact replacement tracked SPY more strongly, but the new blends allocated across families more effectively in this window.

A paired three-month block bootstrap versus broad produced a conditional 95% Sharpe-difference interval of approximately −0.201 to +0.890. It includes zero and does not adjust for searching 151 configurations. Three seeds are three model initializations on the same market history, not 72 independent market months. Treat these findings as exploratory; no untouched holdout, cost validation or uncertainty-calibration study was added.

## Verification and reproducibility

The independent audit recomputed conventional Sharpe, total return, volatility and drawdown from raw weights, returns and risk-free values for all 11,088 portfolios. It checked all asset/month/seed axes, covariance positive definiteness, combined risk budgets, capital constraints, effective counts, strict accepted solver statuses, and absence of fallback weights. The previous hybrid replay matched its saved result within 1e-8. All protected source and experiment hashes verified. Another 72 reference solves checked the separate-sleeve solutions against a mathematically equivalent full-vector convex problem under identical sleeve constraints; those reference weights were not substituted into the experiment. See [sleeve equivalence](outputs/sleeve_equivalence.json).

A macOS multiprocessing launch from a stdin-only adaptive planning command failed before evaluating candidates. A saved main-guarded launcher resumed the exact frozen plan. Transient inaccurate-solve warnings during the sleeve dual-price search were handled by the strict numerical retry routine; only optimal, independently feasible weights were accepted. No constraints were loosened. See [execution recovery](outputs/execution_recovery.json).

All changes are confined to this new experiment folder; previous studies and production code remain unchanged. Ruff and the public-docstring gate passed. The combined new and related numerical regression suites passed 27 tests. Editor Pylance diagnostics were unavailable; no Pylance-clean claim is made.

From the repository root, verification commands are:

```sh
.venv/bin/ruff check experiments/2026-10-portfolio-evaluation-v1/blend_sleeves_v1
.venv/bin/pytest -q experiments/2026-10-portfolio-evaluation-v1/blend_sleeves_v1
.venv/bin/python experiments/2026-10-portfolio-evaluation-v1/blend_sleeves_v1/run_blends.py --stage verify
MPLCONFIGDIR=/private/tmp/bayesfolio-blend-mpl .venv/bin/python experiments/2026-10-portfolio-evaluation-v1/blend_sleeves_v1/outputs/audit_and_plot.py
```

Cached recovery entrypoints are `run_blends.py --stage first`, `outputs/replay_round_two.py`, and `run_blends.py --stage sleeves`. Round two must use its saved replay entrypoint, because its frozen plan includes the targeted additions recorded after reviewing round one. Existing completed case artifacts are reused, not silently overwritten.

The [audit](outputs/audit.json), [frozen policy and source hashes](outputs/contract.json), [comparison](outputs/comparison.json), and per-case JSON/NPZ pairs preserve the result trail. To discard this experiment later, remove only this new folder after preserving any desired reports; no rollback of earlier code or results is needed.
