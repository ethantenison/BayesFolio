# Final global round interpretation

All six masks completed seeds 27, 127 and 227, with runner exit zero, exact
fingerprints and 84+k training-month checks. The dispatcher exited. Independent
objective/allocation arithmetic passed all 75 joint seeds. Across the full
study, 489 universe/mask configurations produce 1,467 valid seed runs and
35,208 scored GP windows. There are no incomplete registered outputs.

Global hypervolume rose 0.970672195 to 1.016778459, a 4.75 percent final-round
gain. All three adaptive global rounds improved. Stop at the three-round cap;
this is not evidence of convergence or an exhaustive optimum.

## Final joint Pareto front

| Configuration | Non-time inputs | IC | NLL | Conventional Sharpe |
| --- | ---: | ---: | ---: | ---: |
| global_52d9b7e07091 | 15 | 0.258085 | -2.251246 | 1.526414 |
| global_c0bc5df949cf | 13 | 0.275397 | -2.236822 | 1.448330 |
| global_a8a6b8394d6e | 13 | 0.272470 | -2.228913 | 1.567890 |
| global_9e7175a3ef4a | 12 | 0.274901 | -2.234172 | 1.544716 |
| global_33e57e1c2370 | 13 | 0.264881 | -2.250209 | 1.622642 |
| global_da43a19fc3dd | 12 | 0.236359 | -2.262087 | 1.381369 |
| global_fdb5fb8b385f | 14 | 0.278026 | -2.232245 | 1.528323 |

All seven remain available. Highlight four research alternatives: the highest
Sharpe thirteen-input mask; its twelve-input no-lag parent; the highest-IC
fourteen-input credit mask; and the twelve-input best-NLL dealer-free mask.
This is a disclosed reporting preference, not a fourth objective or weighted
scalar winner. The other three points retain legitimate small tradeoffs.

## What the last experiments taught us

- Combining annual-momentum and VIX-level removals gives the competitive
  twelve-input mask. Against its no-VIX thirteen-input parent, IC rises 0.002431
  and NLL improves 0.005259, while Sharpe falls 0.023174. Each objective's
  favorable seed count is two of three; aggregate direction is not a guarantee.
- Removing VIX from the lagged density parent improves mean Sharpe 0.058854
  in every seed, with worse NLL in every seed. Its fourteen-input result is
  dominated after also deleting annual momentum.
- The combined no-VIX/no-annual, lagged thirteen-input mask has the highest
  mean Sharpe. Against its newly tested twelve-input no-lag parent, adding lag
  changes IC -0.010020, NLL -0.016037 and Sharpe +0.077926. Favorable seed counts
  are only 1/3, 2/3 and 1/3. This does not establish stable lag-driven Sharpe gains.
  Against the fourteen-input lag/no-VIX model, annual-momentum removal improves
  all three aggregate objectives; NLL and Sharpe improve in every seed.
- Dealer removal from the no-VIX thirteen-input parent improves NLL 0.033174
  in every seed but reduces IC 0.036111 and Sharpe 0.186521 in every seed. Dealer
  is not universally noise; its removal creates a density-oriented alternative.
- HY z-score removal from that same parent worsens all three objectives in
  every seed: IC -0.077778, NLL +0.074632 and Sharpe -0.269194. This is strong
  conditional credit evidence, not proof for every family or every shared mask.
- HY level plus VIX removal gives the highest pooled IC. Exact comparisons
  retain its density/Sharpe tradeoffs; the level is not required in the primary
  thirteen-input lagged mask.

## Shared research lead and family limitations

The thirteen-input Sharpe-oriented research lead retains time plus lag,
skip momentum, six-month momentum, trend, annual CPI, dealer proxy, DXY,
EM-FX return, gold/oil ratio, HY z-score, oil return, term spread and normalized
VIX slope. The last thirteen names exclude time. This is a tested shared mask,
not an untested intersection. Relative to the current twenty-seven inputs it
removes twenty and adds six validated representations, retaining seven plus time.

Against the joint all-input control, IC improves 0.156101, NLL 0.108795 and
Sharpe 0.494761; each improves in every seed. Its seed SD is 0.003324 IC,
0.002646 NLL and 0.094196 Sharpe. Seed repetition is not economic confidence.

Within-family IC changes from control to lead are:
U.S. equity 0.172222 to 0.208333; international equity 0.041667 to 0.183333;
U.S. fixed income -0.097222 to 0.038889; international fixed income 0.222222 to
0.090278. All family mean NLLs improve. International debt ranking remains an
unresolved cost, not a hidden guardrail violation or a reason to redefine fronts.

Mean family capital weights for the lead are approximately 58.94 percent U.S.
equity, 36.16 percent international equity, 1.72 percent U.S. debt and 3.18 percent
international debt. Actual joint Sharpe is not the mean of family Sharpes; its
improvement is largely an equity-allocation result. Inactive sleeves have null
full-path diagnostic Sharpe, not fabricated zero or shortened-path estimates.

Predictive standard deviation rises from 0.026795 to 0.027480, while 95 percent
coverage rises from 0.909259 to 0.940741 and NLL improves. Better uncertainty
estimation is supported more directly than uniformly narrower distributions.
Coverage remains heterogeneous: 0.940972/0.913194/0.951389/0.962963 in approved
family order. The best-NLL twelve-input model has mean std 0.026683 and coverage
0.939815: a separate, slightly sharper tradeoff, not a universal uncertainty win.

## Closed search and unresolved hypotheses

No new fits after this round. Direct global tests of CPI, curve, individual FX,
skip/intermediate momentum, slope normalization and alternative local volatility
remain incomplete. Retention does not prove these inputs individually essential.
Nominal/TIPS exceptions and international-debt ranking deserve further explicitly
authorized research. A masked kernel, new calibrated optimizer, formal new holdout
and production promotion remain separate decisions. This recommendation applies
only to the fifteen approved ETF targets and the fixed research kernel/budget.
