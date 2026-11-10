# Complete initial actual joint comparison

All six new shared masks completed seeds 27, 127 and 227, 24 scored months and
the same 84-initial-month expanding history. Together with the all-input control,
there are seven actual fifteen-task configurations/21 seed runs. The independent
joint arithmetic audit passed every seed, reconstructing IC, marginal Gaussian
NLL, weights-times-realized-excess-return paths, conventional Sharpe and family
calibration without the reporting metric calculators. No run remains pending.

The overall study now has 469 configurations/1,407 seed runs/33,768 scored GP
windows. Final family evidence remains unchanged. The initial global stage is
round zero, not one of the three permitted adaptive global rounds.

## Joint objectives

| Configuration | Non-time inputs | IC | Gaussian NLL | Conventional Sharpe | Current front |
| --- | ---: | ---: | ---: | ---: | --- |
| All-input control | 27 | 0.108780 | -2.141413 | 1.127881 | No |
| Shared mechanism core | 10 | 0.204514 | -2.238293 | 1.288369 | Yes |
| Literal family union | 29 | 0.096478 | -2.106803 | 1.178543 | No |
| Reduced family union | 22 | 0.182192 | -2.157635 | 1.296970 | No |
| Core plus fixed-income exceptions | 14 | 0.114286 | -2.093847 | 1.288462 | No |
| Core plus international exceptions | 15 | 0.258433 | -2.218703 | 1.425040 | Yes |
| Fixed-income-exception core plus lag | 15 | 0.121677 | -2.158251 | 1.299081 | No |

The fifteen-input international-exception mask improves IC, NLL and Sharpe versus
the all-input control in every optimization seed. This is repeatable optimizer
evidence on the same reused economic dates, not independent held-out confirmation.
Its mask is 44.4% smaller excluding time. The ten-input core is smaller still,
with better NLL but lower IC/Sharpe; its Sharpe improvement over the control occurs
in two seeds. The two fronts are genuine competing operating points.

Fixed-reference hypervolume rises from 0.262731421 to 0.822258382. The literal
union fails to transfer its component high-IC results; the reduced union is
better than that literal union but dominated by the international-exception mask.
Keeping every family exception is therefore not a demonstrated global strategy.

Return and RMSE remain diagnostic only: cumulative excess-return path measures
are 0.347077, 0.365231 and 0.447144 for control, core and international exceptions;
RMSE values are 0.033576, 0.032169 and 0.032921 respectively. Neither metric chose
the front, and no turnover metric was consumed.

## Exact current front masks

- Ten-input core, global_b48732adb186: local mom12m, mom12m_skip1m, mom6m;
  macro cpi_chg_12p, hy_spread_z_12p, term_spread, oil_ret, em_fx_ret, dxy,
  vix_ts_z_12p. Always add t_index.
- Fifteen-input international exceptions, global_389a69848aa0: the same core
  plus cs_mom_rank, trend_slope, dealer_gamma_proxy, vix, gold_crude_ratio.
  Always add t_index. These are not family-masked inputs; all tasks receive them.

No individual effect can be assigned to those five additions from the block
comparison alone. The next round must separate mechanisms, not treat the block
as five universally important features.

## Family breakdown: improvement is not uniform

| Family | Control IC | Ten-input IC | Fifteen-input IC | Control NLL | Fifteen-input NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| U.S. equity | 0.172222 | 0.080556 | 0.088889 | -1.621425 | -1.684795 |
| International equity | 0.041667 | 0.158333 | 0.269444 | -1.700185 | -1.731850 |
| U.S. FI | -0.097222 | 0.119444 | 0.083333 | -2.668109 | -2.822332 |
| International FI | 0.222222 | 0.180556 | 0.180556 | -2.720776 | -2.774881 |

The fifteen-input model improves aggregate family NLL for all four families, but
its U.S.-equity IC falls in every seed and foreign-debt mean IC also falls. Pooled
IC includes between-family ordering; it is not an average of the family values.
Capital remains predominantly equities. These are diagnostic tradeoffs, not a
new hidden family guardrail, and they prevent a claim that every task improved.

Predictive standard deviations do not uniformly shrink. Against the same joint
control, the fifteen-input model slightly widens U.S.-equity and foreign-debt
spreads while improving their 95% coverages; it narrows the other two. Better NLL
and coverage should be described as better predictive quality, not blanket lower
interval width. Full 50/80/95% and monthly diagnostics are preserved.

## Exact lag contrast

Adding lag to the fourteen-input fixed-income-exception core improves aggregate
IC by 0.007391, NLL by 0.064404 and Sharpe by 0.010620. The corresponding favorable
seed counts are 1/3, 3/3 and 2/3, not unanimous three-objective gains in every seed.
The resulting model remains dominated. Lag is helpful conditionally here, but its
value on the stronger fifteen-input international-exception branch is unknown.

## First adaptive global round

Preregister six novel masks, all with the same dates/seeds/controls:

1. Remove global rank from the fifteen-input front: probe whether rank is part
   of the lost U.S. ordering and extra density cost.
2. Remove unskipped annual momentum from that front: test whether a common annual
   definition is really needed once corrected skip/intermediate momentum coexist.
3. Remove the gold/oil ratio from that front: do not transfer the broad international
   family's ratio benefit blindly to this different shared risk context.
4. Add lag to that front: test the newly observed global lag benefit in a different,
   actually competitive macro/local context.
5. Add absolute HY level to the ten-input density front: transfer the U.S.-equity
   ranking mechanism without restoring a broad union or all credit transforms.
6. Add dealer gamma proxy alone to the ten-input front: attempt to obtain part of
   the international block's IC/Sharpe benefit without all five additional axes.

This covers both observed front branches, exact reductions, a lag interaction,
and motivated cross-family transfers. Each is falsifiable; none is assumed to
improve every objective or every family. Later rounds can combine supported
changes and test remaining local/risk/FX/oil mechanisms. Keep the cap of six new
global masks per round and stop after two stalled adaptive rounds or round three.

Immutable authoritative sources: results/rounds/global_round0.json and its
cumulative CSV, the copied cumulative evidence in the next registered round,
and results/component_checkpoints/after_global_round0/, including the passed
independent audit. These dates remain adaptively reused; no untouched holdout,
calibrated BO, production change or global-optimality claim is introduced.
