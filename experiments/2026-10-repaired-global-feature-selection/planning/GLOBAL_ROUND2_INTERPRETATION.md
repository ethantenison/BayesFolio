# Second adaptive global round: completed evidence

All six new masks completed all three seeds. Independent objective/allocation
arithmetic now passes all 57 registered joint seed runs. There are 483 complete
configurations/1,449 seed runs/34,776 scored GP windows across the corrected study.
Frozen post-round-two evidence is in results/component_checkpoints/after_global_round2/.

Fixed-reference global hypervolume increased 0.877450650 -> 0.970672195 (10.62%).
Both completed adaptive global rounds improved the frontier. The final third
round remains authorized and required for this execution; no fourth round is allowed.

## Full cumulative joint front

| ID | Non-time inputs | IC | Gaussian NLL | Conventional Sharpe |
| --- | ---: | ---: | ---: | ---: |
| global_15b18df4a34b | 14 | 0.266617 | -2.232702 | 1.469891 |
| global_43cd2137e85f | 16 | 0.262054 | -2.239092 | 1.468439 |
| global_52d9b7e07091 | 15 | 0.258085 | -2.251246 | 1.526414 |
| global_c0bc5df949cf | 13 | 0.275397 | -2.236822 | 1.448330 |
| global_a8a6b8394d6e | 13 | 0.272470 | -2.228913 | 1.567890 |
| global_d062f4d835aa | 15 | 0.274107 | -2.233158 | 1.452375 |

These are objective tradeoffs, not six independently validated economic winners.
Several differences are small relative to seed dispersion. The new credit-level
point adds only a very small new hypervolume region and has favorable seed counts
2/3 IC, 1/3 NLL, 1/3 Sharpe versus its parent; do not overstate its reliability.

## Conditional findings

- Combining rank removal and lag gives the fifteen-input density/Sharpe front.
  Against the prior sixteen-input lag parent, NLL improves 0.012154 in every seed
  and aggregate Sharpe improves 0.057975, while IC falls 0.003968. The combination
  is useful, but the IC gains of each single change did not simply add.
- Annual-momentum removal from the rank-free fourteen-input parent improves IC
  0.008780 in every seed and aggregate NLL 0.004120, but reduces Sharpe 0.021561 in
  every seed. This is a smaller thirteen-input tradeoff, not universal redundancy.
- VIX-level removal improves IC 0.005853 and Sharpe 0.097999 (Sharpe improves in
  every seed), with an NLL cost of 0.003789. Its thirteen-input mask is the highest
  aggregate-Sharpe point so far.
- Trend removal worsens all three aggregate objectives and is not competitive.
- Oil-return removal worsens IC, NLL and Sharpe in every seed. Oil remains useful
  conditionally even when the gold/oil ratio is retained. Coupled with the prior
  ratio-removal result, deleting either commodity representation is unsupported
  in these tested shared contexts.
- Absolute HY-level addition slightly improves IC/NLL and costs Sharpe on the
  stronger rank-free parent. Its U.S.-equity and U.S.-FI mean ICs improve to
  0.127778 and 0.108333, unlike the earlier ten-input context. This is another
  representation interaction, not a blanket statement that the level is needed.

## Family breakdowns constrain interpretation

| Model | U.S. equity IC | International equity IC | U.S. FI IC | International FI IC |
| --- | ---: | ---: | ---: | ---: |
| All-input control | 0.172222 | 0.041667 | -0.097222 | 0.222222 |
| Rank-free plus lag, 15 inputs | 0.194444 | 0.177778 | 0.077778 | 0.041667 |
| No annual momentum, 13 inputs | 0.094444 | 0.263889 | 0.069444 | 0.145833 |
| No VIX level, 13 inputs | 0.108333 | 0.288889 | 0.055556 | 0.125000 |
| Added HY level, 15 inputs | 0.127778 | 0.277778 | 0.108333 | 0.118056 |

The best-density lagged model restores U.S.-equity ranking but further reduces
foreign-debt ranking. Pooled improvements cannot be described as gains in every
family. In the above new models, all four family mean NLLs are better than their
joint-control counterparts; predictive quality and within-family ranking still
tell different stories. Keep these as reported diagnostics, not new hidden vetoes.

Coverage remains heterogeneous: the no-VIX model has 95% coverage
0.930556/0.888889/0.937500/0.958333 in the family order above. The lagged density
point has 0.944444/0.906250/0.954861/0.949074. Interval width, seed SD and finite
performance-sample uncertainty are distinct. No untouched holdout claim is made.

## Final six hypotheses

Use the highest-Sharpe thirteen-input no-VIX point and the fifteen-input lagged
density point to test:

1. Combine annual-momentum and VIX-level removals: a twelve-input shared mask.
2. Combine lag and VIX removal while retaining annual momentum: fourteen inputs.
3. Remove annual momentum in that combined risk/lag representation: thirteen inputs.
   These three form matched contexts after aggregation: lag addition to candidate
   one, and annual removal from candidate two. They test substitutions/interactions,
   not new kernel interaction terms.
4. Remove dealer proxy from the no-VIX model. Dealer alone failed in the compact
   core, but it might be essential after removing the absolute VIX level.
5. Remove HY z-score from the no-VIX model. Repaired credit state helped U.S.
   equity yet could be deleted in foreign debt; probe that conflict in the joint GP.
6. Combine HY-level addition with VIX removal. This may retain the credit branch's
   ranking information and recover the stronger-Sharpe risk representation.

All six are novel and retain t_index, oil, ratio and the frozen experimental
controls. The last round mixes motivated combinations and unresolved conditional
risk/credit removals. It does not force lag or annual momentum to disappear.

After the third round, stop new launches at the cap even if it improves. Final
reporting must distinguish the tested shared masks from untested submasks and
universal-importance claims. Remaining direct global tests (for example CPI,
curve, individual FX axes, alternative local volatility or other candidate
representations) must be listed as unresolved future work, not silently added
as another batch. Calibrated BO, new formal holdout design and production
promotion remain separate future decisions.
