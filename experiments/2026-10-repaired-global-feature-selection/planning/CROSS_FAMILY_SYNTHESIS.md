# Final component evidence and initial global hypotheses

## Completed component scope

All historical masks and three adaptive rounds per family are complete, with
three seeds and the same repaired 84-initial-month/24-score-date expanding
windows. There are 47 U.S. equity, 185 international equity, 105 U.S.-FI and 125
international-FI configurations: 462 family masks, plus the completed joint
all-input control, for 463 configurations/1,389 seed runs/33,336 scored GP windows.
Each adaptive round contributed twelve genuinely novel masks per family.

| Family | Final Pareto masks | Final fixed-reference HV | Third-round HV gain |
| --- | ---: | ---: | ---: |
| U.S. equity | 9 | 0.861677628 | 3.21% |
| International equity | 8 | 0.701799493 | 13.11% |
| U.S. fixed income | 6 | 0.544561035 | 12.66% |
| International fixed income | 5 | 1.057468608 | 5.27% |

All final rounds improved their fronts. Family exploration ends because the
three-round cap was reached, not because convergence or universal optimality was
shown. All 28 family Pareto points, their exact inputs, three-seed objectives,
paired effects and uncertainty remain available; no single weighted winner is imposed.

Frozen component evidence is preserved in
`results/component_checkpoints/after_family_round3/`. It records 463 exact
one-feature contexts across 55 tested non-time inputs. Feature/family rows without
matched tests remain unknown. The context counts are not independent votes or
importance scores. Block experiments remain block evidence, not individual attribution.

## What the families actually teach us

| Mechanism | Conditional evidence | Consequence for global planning |
| --- | --- | --- |
| Annual/local momentum | Skip momentum resists deletion/substitution in U.S. equity; replacing annual momentum with skip momentum worsens all objectives in every seed in the final foreign-debt context. Wholesale local-basis transfers often fail. | Do not impose one annual definition on every task. Test a compact mixed annual basis, then prune only using joint results. |
| Recent-return lag | Adding lag damages several equity/foreign-debt contexts. Yet in the final U.S.-FI oil/rate model it raises IC in all seeds and aggregate Sharpe, with an NLL cost. Lag removal also helps a different density branch. | Keep lag eligible and include an exact matched global lag contrast. A universal lag-free requirement is not supported. |
| Genuine HY state | HY z-score and curve removals damage compact U.S. equity. HY level versus change creates a high-IC/density tradeoff. Both HY axes can be removed from a competitive foreign-debt parent; broad international equity also benefits from credit pruning. | Credit belongs in competing joint hypotheses, not in a universally necessary or universally harmful category. |
| Nominal/TIPS rate state | Removing nominal yield, one-month SCHP return or trailing TIPS proxy from the final U.S.-FI oil branch worsens all objectives in every seed. Duplicate Treasury naming can be removed in other contexts. | Preserve genuine rate exceptions while testing equivalent representations; do not mistake redundant naming for dispensable economic information. |
| Oil, FX and commodity representation | Oil improves the reduced U.S.-FI rate branch. Oil/EM-FX deletion damages the tested foreign-debt parent, and dollar/EM-FX removal hurts international equity ranking. Gold/oil-ratio deletion is harmful in the broad international equity context, including after HY removal. | Include oil/FX mechanisms, but do not universally delete the ratio based on an earlier small-context result. |
| Market/risk proxies | MOVE and slope-change pruning helps selected international equity contexts. VIX-slope z-score deletion is harmful in the compact oil branch. Rank plus acceleration removal helps foreign debt jointly even when individual removals do not. | Preserve interactions and context, not a univariate importance ranking. Test bounded global blocks and exact refinements. |

There is no compact set proven individually useful in every family. The family
results generate hypotheses about shared mechanisms and exceptions; only actual
15-task fits can test whether a single shared mask works well jointly.

## Disclosed representatives for the literal-union hypothesis

For one exploratory union, choose the highest-aggregate-IC point on each family
front. This is a transparent hypothesis-generation rule, not a new selection
objective or a recommendation to discard other Pareto tradeoffs.

| Family / configuration | Inputs excluding time | IC | Gaussian NLL | Conventional Sharpe |
| --- | ---: | ---: | ---: | ---: |
| U.S. equity / us_equity_8b0e1b779842 | 6 | 0.325000 | -1.639829 | 1.456596 |
| International equity / international_equity_f7933f4054ed | 16 | 0.261111 | -1.645865 | 1.474843 |
| U.S. FI / us_fixed_income_78c395992b67 | 8 | 0.350000 | -2.609230 | 0.564963 |
| International FI / international_fixed_income_ade5ca6498cf | 11 | 0.437500 | -2.735421 | 0.858261 |

Their literal union has 29 non-time inputs, including lag and both identical
Treasury-level names. It is intentionally not called a reduced production mask.
The union is reproducible from the four exact lists in the frozen
`component_exact_inputs.json`; `global_round0.json` records the resolved lists.

Important alternatives remain: U.S. equity has a seven-input higher-Sharpe point
(1.504956) and density-oriented points; international equity has ten-input
risk/oil alternatives with lower NLL; U.S.-FI has a five-input higher-Sharpe point
(0.597448) and no-lag oil/rate points with better NLL. Foreign debt has a different
eleven-input density point that improves its parent in every objective and seed.
The final report must retain these alternatives, not label the union inputs as
the final variables of each family.

## Uncertainty is not synonymous with fewer inputs

For the selected representatives, mean predictive standard deviations are
0.054591, 0.045117, 0.019715 and 0.019576 respectively; 95% coverages are 0.961806,
0.920139, 0.927083 and 0.962963. These are decimal-return spreads and empirical
coverage, averaged over the same three seeds and 24 months. U.S. equity's
same-family all-input spread is 0.051135: its six-input high-IC representative is
not sharper and has slightly worse NLL than that all-input comparator. Thus the
desired lower-uncertainty result has not been established just by reducing inputs.

Use the complete 50/80/95% coverage, width and NLL reports and monthly diagnostics.
Do not optimize raw interval width, call overconfident forecasts improved, or
treat optimization-seed repetition as independent economic confidence.

## Initial actual joint comparison

The joint all-input control uses 27 non-time inputs under the same research GP:
pooled IC 0.108780, NLL -2.141413 and actual portfolio Sharpe 1.127881. Its U.S.-FI
within-family IC is -0.097222; positive pooled IC partly describes between-family
ordering and cannot prove each component forecasts well. Allocation is mostly
equities, so portfolio improvements also need explicit family exposure diagnostics.

Preregister six new shared masks in `global_round0.json`:

| Hypothesis | Non-time inputs | Key question |
| --- | ---: | --- |
| Shared mechanism core | 10 | Can mixed annual/intermediate momentum and compact macro mechanisms suffice? |
| Literal selected-family union | 29 | Does preserving all high-IC exceptions transfer to a shared GP? |
| Evidence-reduced union | 22 | Can selected redundancy/block reductions retain a competitive joint tradeoff? |
| Core plus fixed-income exceptions | 14 | Do MA/nominal/TIPS axes repair missing debt information? |
| Core plus international risk exceptions | 15 | Do local/risk/ratio exceptions help more than broad rate additions? |
| Fixed-income-exception core plus lag | 15 | Exact matched lag contrast against the fourteen-input mask. |

The reduced union removes acceleration, lag, duplicate Treasury naming, breakeven,
monthly CPI, copper and flow. Only the duplicate/acceleration changes are based on
exact source identities; even those change finite-budget GP geometry. Other
reductions are falsifiable context-dependent block hypotheses, not established
irrelevance. The ratio is deliberately retained in this broader representation.

After complete results, use up to three adaptive joint rounds of at most six new
masks each, with hosted interpretation, immutable evidence and deterministic
novelty/fit/result checks. Keep the fixed Pareto/HV policy and two-stalled-round
stop. The six initial masks are not an extra adaptive round. See
`GLOBAL_SELECTION_POLICY.md` for the full decision protocol.

## Limits and protected boundaries

Keep t_index, all approved fifteen targets, seeds/dates, model and allocation
controls unchanged. No IJR, REIT or commodity targets; commodity predictors remain
eligible. No masked kernel, zeroing, new optimizer, formal new holdout, deployment,
trading or production-default changes. Return and RMSE remain diagnostic only;
turnover is excluded.

Results are conditional on the rank-two, fifteen-iteration research GP and task
sharing, not the differently configured deployed production GP. Latest-vintage
adjusted prices/CPI are retrospective, including the explicitly flagged CPI gap.
Adaptive score reuse is not untouched confirmation. The three-target foreign-debt
allocation cap compresses Sharpe differences. Remaining gaps may involve kernel
geometry, sharing or finite optimization, not exclusively noisy feature information.
