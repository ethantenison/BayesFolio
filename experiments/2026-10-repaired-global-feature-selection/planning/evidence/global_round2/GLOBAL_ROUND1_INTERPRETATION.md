# First adaptive global round: completed evidence

All six masks completed all three seeds. Independent objective/allocation
arithmetic passed all 39 registered joint seed runs. The two restored historical
macro-only baselines also completed their six seeds; neither changes a family
front. Current total: 477 configurations, 1,431 seed runs, 34,344 scored GP windows.

Global hypervolume rose from 0.822258382 to 0.877450650 (6.71%). The round is not
stalled. Two adaptive global rounds remain permitted; the goal is not complete.

## Current joint front

| Configuration | Non-time inputs | IC | Gaussian NLL | Conventional Sharpe |
| --- | ---: | ---: | ---: | ---: |
| global_15b18df4a34b, rank-free international exceptions | 14 | 0.266617 | -2.232702 | 1.469891 |
| global_43cd2137e85f, international exceptions plus lag | 16 | 0.262054 | -2.239092 | 1.468439 |

Both dominate the initial ten/fifteen-input front in aggregate; the ten-input
core is now dominated because the lagged model also slightly improves its NLL.
The fourteen-input point has higher mean IC/Sharpe and two fewer inputs than the
sixteen-input point, but worse NLL. Small differences between these front points
are not robust economic confidence: seed IC SDs are 0.015501/0.016304 and Sharpe
SDs 0.111265/0.077520 on the same reused 24 dates.

## Hypotheses that survived or failed

- Rank removal from the original fifteen-input front improves IC 0.008185,
  NLL 0.013999 and Sharpe 0.044851; favorable seed counts are 2/3 for each metric.
- Adding lag to that original front improves IC 0.003621, NLL 0.020389 and Sharpe
  0.043399; favorable seed counts are 3/3, 3/3 and 2/3. A feature weak in several
  separately fitted families can therefore help a shared model conditionally.
  This does not establish beneficial cross-task causality or gains for every family.
- Removing unskipped annual momentum slightly improves NLL but slightly worsens
  IC/Sharpe; Sharpe falls in all seeds. It is not on the cumulative front. Its
  effect after rank removal or in a lagged representation remains unresolved.
- Removing gold/oil ratio worsens IC, NLL and Sharpe in every seed. The broad
  international-family finding transfers to this tested shared context, unlike
  a universal rule that oil-only should win.
- Adding HY level to the ten-input core raises pooled IC 0.056696 in all seeds,
  but worsens NLL 0.017501 in all seeds and is dominated. It does not restore
  U.S.-equity ranking in this context: within-family IC falls from 0.080556 to
  0.025000, while U.S.-FI IC rises from 0.119444 to 0.169444. A later bridge must
  acknowledge that falsification instead of repeating the original U.S.-equity story.
- Adding dealer proxy alone to the ten-input core hurts IC/NLL in every seed,
  despite a small aggregate Sharpe gain. It cannot explain the five-axis block
  improvement alone; its interaction/conditional contribution remains unresolved.

## Family outcomes, not concealed by pooled IC

| Family | Fourteen-input IC | Sixteen-input IC | Fourteen-input NLL | Sixteen-input NLL |
| --- | ---: | ---: | ---: | ---: |
| U.S. equity | 0.113889 | 0.175000 | -1.694543 | -1.731983 |
| International equity | 0.294444 | 0.186111 | -1.763787 | -1.743046 |
| U.S. FI | 0.061111 | 0.083333 | -2.820699 | -2.835733 |
| International FI | 0.125000 | 0.069444 | -2.791472 | -2.781111 |

Lag restores U.S.-equity mean IC approximately to the all-input control's
0.172222, but sacrifices international ranking versus the fourteen-input branch.
Foreign-debt IC remains below its all-input control in both front points. These
are reported diagnostic tradeoffs, not hidden new objectives or vetoes.

Coverage is also mixed. For the fourteen-input model, 95% coverage by the above
family order is 0.940972/0.892361/0.927083/0.953704; the lagged model has
0.954861/0.909722/0.951389/0.958333. U.S.-equity spreads widen in the lagged model
while calibration improves. Do not equate a smaller mask or narrower spread with
uniformly lower uncertainty. Full 50/80/95% and monthly evidence is preserved.

## Round-two reasoning

Six explicit hypotheses will test:

1. The combined rank removal plus lag addition (remove rank from the sixteen-input
   density front). The cumulative pair table will also compare this mask against
   the fourteen-input no-lag front, resolving whether the two successful changes combine.
2. Annual-momentum removal after rank removal, a smaller thirteen-input context.
3. Trend-slope removal from the fourteen-input front, testing a remaining local exception.
4. VIX-level removal while retaining normalized term-structure state.
5. Oil-return removal while retaining gold/oil ratio, separating commodity representations.
6. HY-level addition to the fourteen-input front, a ranking/density bridge motivated
   by the ten-input result but without claiming it already helped U.S.-equity IC.

This combines exploitation with conditional substitution/block-representation
exploration. Do not repeat ratio deletion without a new mechanism merely to seek
a preferred answer, and do not enforce a lag-free result. Kernel, likelihood,
preprocessing, seeds, dates, objectives and the fifteen-fit total limit are unchanged.

The recovered full scope is GOAL_CONTRACT.md. Historical macro-only inventory
correction and legacy prior/group-label caveats are documented separately; all
existing fits and as-of round comparisons remain preserved. Final global
recommendations and the requirement-by-requirement completion audit remain due.
