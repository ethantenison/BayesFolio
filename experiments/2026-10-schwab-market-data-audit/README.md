# VTV Schwab vs Yahoo Market-Data Audit

Status: Completed

Run date: 2026-10-11

## Question

Can Schwab Market Data replace the Yahoo Finance daily data currently used by
BayesFolio for VTV prices, volume, and monthly return construction?

## Method

`compare_vtv.py` retrieves 20 years of daily Schwab candles and the matching
Yahoo history with `auto_adjust=False` and corporate actions enabled. It outer
joins observations by New York trading date and compares:

- raw open, high, low, close, and volume;
- missing or vendor-only dates;
- Yahoo adjusted close and recorded dividends;
- month-end raw-price returns and adjusted total returns.

Run with:

```bash
uv run --locked python experiments/2026-10-schwab-market-data-audit/compare_vtv.py
```

Generated daily, monthly, and JSON outputs are written to the ignored
`outputs/` directory.

## Results

### Coverage and raw-price agreement

- Common trading dates: 5,032 from 2006-10-09 through 2026-10-09.
- Yahoo-only dates: 0.
- Schwab-only dates: 1 (`2007-01-06`, a Saturday duplicate of the prior
  Friday candle rather than a genuine trading session).
- Median absolute raw-close difference: approximately $0.0000018.
- 95th-percentile absolute raw-close difference: approximately $0.0000067.
- Raw-close differences exceeded $0.01 on 33 dates and $0.05 on 11 dates.
- The largest raw-close difference was $0.55 on 2008-12-22, a Yahoo-recorded
  ex-dividend date.

The vendors agree almost exactly on ordinary raw closes, but neither should be
treated as infallible. Notable historical outliers include a $3.82 disagreement
in the intraday low on the 2010-05-06 flash-crash date and a $1.83 disagreement
in the open on 2008-12-16.

### Volume agreement

- Median relative difference: approximately 0.0029%.
- Days differing by more than 1%: 19.
- Days differing by more than 10%: 5.
- Largest relative difference: approximately 71.8% on 2009-04-21.

Most volume observations agree closely, with a small number of large vendor
corrections or data-quality disagreements.

### Return consequences

| Measure | Schwab raw | Yahoo raw | Yahoo adjusted |
|---|---:|---:|---:|
| 20-year endpoint return | 242.96% | 242.96% | 477.66% |
| Annualized endpoint return | 6.36% | 6.36% | 9.16% |

Yahoo reported 80 VTV dividend events totaling $50.162 per share and no split
events during the comparison window. There were exactly 80 months in which the
Schwab raw-price return differed from Yahoo's adjusted return by more than 10
basis points.

Monthly Schwab-versus-Yahoo raw returns were normally nearly identical: the
median absolute difference was effectively zero and the 95th percentile was
approximately 1.9 basis points. In contrast, the mean absolute difference
between Schwab raw returns and Yahoo adjusted returns was approximately 21.9
basis points per month, with a maximum of approximately 116.4 basis points.

## Decision

Do not replace Yahoo adjusted close with Schwab raw candles for BayesFolio
training labels or total-return backtests. Doing so would materially understate
VTV's return by omitting distributions.

Use Schwab immediately for:

- live quote and liquidity validation;
- option-derived data;
- reference identifiers and trading eligibility;
- dividend-event metadata;
- raw-candle and volume cross-checks.

Keep Yahoo adjusted close as the current total-return source until BayesFolio
has a tested corporate-action adjustment pipeline. Any future Schwab migration
should filter non-trading dates and quarantine vendor disagreements rather than
silently choosing one source.
