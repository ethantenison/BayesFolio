# Evidence interpretation after all family round-one results

All observations below are conditional on the repaired common panel, fixed
research kernel, three optimization seeds, and reused 24 scored dates. They are
not universal feature-importance statements or independent holdout evidence.

## U.S. equity

- Dropping cross-sectional rank from the eight-input parent improves NLL by
  0.026103 and Sharpe by 0.069870 while reducing IC by 0.025000. It is a new
  seven-input frontier point, not proof that rank is always harmful.
- Adding recent-return lag improves NLL by 0.053008 but reduces IC by 0.108333
  and Sharpe by 0.096297. This is a retained density/ranking tradeoff.
- HY z-score and term spread removals are strongly worse on all three objectives;
  their repaired historical values matter in this compact context.
- The alternate bond-style local basis is worse on all three metrics. Oil return
  and local realized volatility additions are not on the cumulative front here.
- Round-two direction: combine rank removal with carefully selected local
  simplifications; test whether lag's density benefit can coexist with the smaller
  rank-free model; explore a different credit representation without discarding
  the useful credit/curve state wholesale.

## International equity

- Removing the TLT volatility proxy from the new risk-state parent improves all
  three objectives: IC +0.022222, NLL -0.025987, Sharpe +0.017664.
- Removing the one-month VIX-slope change improves IC/NLL but slightly reduces
  Sharpe. Oil return and task-level volatility additions create other front points.
- Removing the gold/oil ratio from the broad rate/credit/commodity parent now
  worsens all three objectives. This is the opposite of earlier small-context
  tests: the value of the representation depends on the rest of the inputs.
- Removing HY change from that broad parent creates a useful tradeoff: lower
  NLL and higher Sharpe with modestly lower IC.
- Alternative local-basis transfer and adding recent-return lag do not create
  competitive front points in this tested risk context.
- Round-two direction: combine the promising MOVE/slope-change removals with
  oil/volatility transfer; target redundant risk representations and broad-model
  credit pruning rather than repeating the already answered whole-block tests.

## U.S. fixed income

- Rank removal from the best-Sharpe credit parent improves IC and Sharpe but
  worsens NLL; removing long momentum creates a different smaller front point.
- Removing lag from that parent improves IC/Sharpe but worsens NLL and is not
  on the cumulative front. Do not call lag universally needed or irrelevant.
- Removing return autocorrelation from the no-lag rate/TIPS parent improves all
  three objectives and reduces the input count from seven to six.
- Adding HY z-score or task-level volatility to the rate parent creates new
  tradeoffs. Removing nominal yield or one-month SCHP return badly hurts ranking.
- Removing the duplicate nominal-yield name and algebraically redundant bill
  level from the density anchor improves IC, NLL, and Sharpe. This was tested,
  not inferred solely from the source identities.
- Round-two direction: combine successful local pruning with the credit/rate
  bridge, re-test lag in changed contexts, and retain genuine rate information
  while reducing redundant rate representations.

## International fixed income

- Removing both HY level and z-score from the high-IC parent improves all three
  aggregate objectives (IC +0.055556, NLL -0.039943, Sharpe +0.000395), reducing
  the parent from sixteen to fourteen inputs. Genuine HY data are not automatically
  useful simply because the old zeros were repaired.
- Removing only HY z-score yields the highest IC of these credit removals, with
  a small Sharpe tradeoff. Removing flow, rank, or acceleration creates other
  useful sparse front points with different objective tradeoffs.
- Removing annual CPI or VIX slope z-score is not supported in this context.
  Removing both slope transforms is particularly harmful to ranking.
- Adding nominal U.S. yield/curve state or transferring the four-input equity
  local basis does not create a competitive frontier point.
- Round-two direction: combine successful HY/flow/rank/acceleration reductions,
  re-test local momentum and risk representations on the reduced parents, and
  keep the three-ETF allocation cap's small Sharpe variation in perspective.

## Implications for the still-required global selection loop

1. No four-variable local basis has been shown universally transferable. The
   tested transfers often underperform. Do not choose a shared core by wishful
   intersection or by counting how frequently a feature appears.
2. HY z-score is very useful in the tested compact U.S. equity context, yet
   deleting both HY features improves the tested international-debt parent.
3. Oil/ratio/FX/risk-state effects change with context. New global hypotheses must
   preserve these competing explanations rather than applying one blanket rule.
4. Actual 15-task evaluations remain essential. The completed joint all-input
   control has positive pooled IC but negative U.S.-FI within-family IC; a good
   pooled score cannot stand in for family outcomes.
5. The next family rounds should sharpen these hypotheses before the bounded
   global initial and adaptive batches. Do not implement masked kernels or
   introduce a calibrated optimizer/new holdout program in this goal.

Authoritative numeric evidence: the four `results/rounds/*_round1.json` files,
their adjacent tables, and `results/cumulative/<family>/`.
