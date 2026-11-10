# First joint follow up interpretation

All eighteen seed runs completed with exit zero and passed signatures, actual
84+k training histories and independent IC/NLL/allocation/Sharpe arithmetic.
The dispatcher session 39907 exited zero. The cumulative comparison now has
31 joint masks and 93 seed runs. Original evidence and data remain unchanged.

Fixed-reference hypervolume increased from 1.016778459 to 1.187618053 (16.80%).
The cumulative front has only two configurations: the previous twelve-input
dealer-free density model and the new fourteen-input SPY-proxy addition.

| Addition to the prior thirteen-input lead | IC | NLL | Conventional Sharpe | On cumulative front |
| --- | ---: | ---: | ---: | --- |
| HY level | 0.263740 | -2.241139 | 1.631714 | No |
| HY change | 0.277728 | -2.242970 | 1.643740 | No |
| SPY proxy | 0.296528 | -2.258812 | 1.771496 | Yes |
| Monthly CPI | 0.266667 | -2.238164 | 1.610998 | No |
| SCHP return | 0.167708 | -2.164351 | 1.475153 | No |
| Nominal yield | 0.253720 | -2.225665 | 1.432244 | No |

The SPY-proxy addition improves pooled IC 0.031647, NLL 0.008603 and Sharpe
0.148854 versus its parent; every objective improves in every seed. Its Sharpe
seed SD is 0.033957, but that is not an economic confidence interval. It uses the
frozen legacy spy_ret formula, not a redefined monthly total SPY return.

Within-family IC changes for that model are U.S. equity 0.208333 to 0.133333,
international equity 0.183333 to 0.141667, U.S. debt 0.038889 to 0.125000 and
international debt 0.090278 to 0.125000. Thus pooled improvement is not improvement
inside every family. Foreign-debt IC remains below the all-input control's
0.222222. Equity-family mean NLL changes are tiny and slightly unfavorable;
debt-family NLL improves. Coverage remains heterogeneous.

HY change improves pooled IC in every seed and mean Sharpe, with an NLL cost;
the stronger SPY candidate dominates it. Absolute HY does not transfer the
earlier highest-IC branch's benefit to this temporal representation. Monthly
CPI is a weak standalone addition here and is also dominated.

The nominal-yield singleton is worse on every global objective in every seed.
SCHP worsens every aggregate objective, even though U.S.-debt within-family IC
rises to 0.247222. These are substantive counterexamples to inferring joint input
importance directly from the strong standalone debt ablations. Neither result
establishes global irrelevance or information redundancy under another context.

## Authorized second batch

Run the six previously proposed and now explicitly requested masks unchanged:
market proxy plus monthly CPI; nominal yield plus SCHP return; remove six-month
momentum; replace six-month with annual momentum; replace trend with MA signal;
and remove DXY. All remain relative to the original thirteen-input parent.

The market/inflation pair now directly tests whether CPI adds to the successful
SPY branch, using both completed singleton controls. The rate pair is less
promising as an optimizer candidate, but remains an informative approved
interaction test: both singleton controls are complete and strong family support
did not transfer separately. Do not describe it as a likely winner.

The temporal and FX tests still address the approved original-parent hypotheses.
Do not silently rebase them onto the new SPY mask; that would create different
unapproved masks and break the fixed twelve-candidate contract. Their results
cannot establish the effect of pruning those features from the SPY model.

After batch two, stop new fits at this campaign's two-batch cap and recompute
the full 37-mask/111-seed joint comparison, retaining all objective tradeoffs,
family IC/NLL/coverage/allocation and exact conditional contexts. Calibrated BO,
new formal holdout design and production promotion remain deferred. No extra
family, time-index removal, architecture change or local LLM is introduced.
