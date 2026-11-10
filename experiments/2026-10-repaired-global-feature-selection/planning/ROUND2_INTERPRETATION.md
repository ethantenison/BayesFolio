# Interpretation after all second family rounds

All four batches completed 12 novel masks with all three seeds. This adds 48
configurations/144 seed runs, bringing completed evidence to 415 configurations,
1,245 seed runs and 29,880 scored GP windows. All results use the repaired common
panel, 84 initial months, shared 24 scored dates and unchanged research controls.

| Family | Hypervolume before | Hypervolume after | Relative gain |
| --- | ---: | ---: | ---: |
| U.S. equity | 0.708220358 | 0.834919079 | 17.89% |
| International equity | 0.595004502 | 0.620432258 | 4.27% |
| U.S. fixed income | 0.431890306 | 0.483386142 | 11.92% |
| International fixed income | 0.945671955 | 1.004573515 | 6.23% |

These are within-family fixed-reference changes, not directly comparable
economic values across families. None of the rounds stalled. Every family can
therefore receive its third and final permitted round.

## Conditional findings

- U.S. equity: replacing HY change with the absolute HY level improves IC by
  0.075 in all three seeds and costs only 0.008830 aggregate Sharpe, but worsens
  NLL by 0.125351. This is a real three-objective tradeoff. Trend removal creates
  a different six-input frontier. Replacing curve with nominal yield, adding lag,
  local volatility or flow does not create a competitive front in that context.
- International equity: combining MOVE and slope-change removals with oil yields
  an eleven-input frontier. Relative to the no-MOVE parent, NLL improves 0.057367,
  Sharpe improves 0.027842 and IC falls 0.011111. Deleting either FX representation
  hurts IC in all seeds. Broad-model HY-z or other credit-change removal improves
  IC/Sharpe with density costs; these contexts disagree with a blanket credit rule.
- U.S. fixed income: oil alone added to the reduced no-lag rate model improves
  aggregate IC 0.041667, NLL 0.155328 and Sharpe 0.005563. IC/NLL improve in every
  seed, Sharpe in one. This seven-input front has IC 0.311111, NLL -2.639529 and
  Sharpe 0.553264. The five-input credit branch retains lag and has Sharpe 0.597448;
  lag is neither universally necessary nor universally harmful. Removing lag from
  the reduced density anchor improves IC/Sharpe in all seeds, with a small NLL cost.
- International fixed income: jointly removing rank and acceleration from the
  no-HY parent yields a twelve-input point (IC 0.416667, NLL -2.716427, Sharpe
  0.879944). The individual removals were not equivalently useful. Oil and EM-FX
  removals hurt ranking; lag addition is worse on every objective in every seed.
  Small Sharpe changes still need the three-target allocation-cap caveat.

## Third-round reasoning

The four round-three JSON proposals explicitly specify their parent, operations,
mechanism, expected tradeoff and falsification. They target reduced local/risk
bases, oil/rate/credit bridges, and matched lag or annual-momentum substitutions.
The deterministic registrar confirmed 12 novel masks per family. No fourth family
round will be added. Stopping at the cap is not proof of global optimality.

## What the global loop must not assume

The post-round-two synthesis records 404 exact one-feature contexts across 55
tested non-time inputs, with explicit untested feature/family combinations. This
is not a frequency ranking. There is still no feature set proven useful to every
family: some local representations have not been tested as matched substitutions
everywhere; HY, lag and market-risk effects depend on context.

The actual joint all-input control allocates most capital to equities and has
within-family U.S.-FI IC -0.097222 despite positive pooled IC 0.108780. A higher
joint Sharpe must therefore be reported alongside family forecast diagnostics,
not taken as evidence of improvement for each sleeve. Narrower intervals count
as better uncertainty only when calibration/NLL support the interpretation.

After the final family round, refresh component synthesis, then preregister up
to six initial joint hypotheses spanning shared-core, selected-family union,
reduced-union and exception representations. Fit the actual 15-task GP. Do not
substitute union/intersection logic, zero masking, averaged family Sharpes or
adaptive historical dates described as an untouched holdout.

Numerical sources: `results/rounds/*_round2.json`, adjacent cumulative tables,
and the preserved `results/component_checkpoints/after_family_round2/` snapshot.
