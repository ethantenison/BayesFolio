# Fixed-Income Cached Data Audit

The source is the cached 18-ETF common-history artifact; no external data was refreshed.
Both families use the same final 24 complete monthly score dates and retain all prior rows for training.

## U.S. Fixed Income

Universe: `BND, IEF, LQD, HYG`.
Complete history: `2013-08-30` through `2026-04-30` (`153` months).
Scored panel: `2024-05-31` through `2026-04-30` (24 months).

### Coverage

| asset_id   |   rows | first_date          | last_date           |   target_count |
|:-----------|-------:|:--------------------|:--------------------|---------------:|
| BND        |    153 | 2013-08-30 00:00:00 | 2026-04-30 00:00:00 |            153 |
| HYG        |    153 | 2013-08-30 00:00:00 | 2026-04-30 00:00:00 |            153 |
| IEF        |    153 | 2013-08-30 00:00:00 | 2026-04-30 00:00:00 |            153 |
| LQD        |    153 | 2013-08-30 00:00:00 | 2026-04-30 00:00:00 |            153 |

### Target correlation

| asset_id   |   BND |   HYG |   IEF |   LQD |
|:-----------|------:|------:|------:|------:|
| BND        | 1     | 0.565 | 0.915 | 0.941 |
| HYG        | 0.565 | 1     | 0.283 | 0.737 |
| IEF        | 0.915 | 0.283 | 1     | 0.759 |
| LQD        | 0.941 | 0.737 | 0.759 | 1     |

## International Fixed Income

Universe: `BNDX, VWOB, HYEM`.
Complete history: `2013-08-30` through `2026-04-30` (`153` months).
Scored panel: `2024-05-31` through `2026-04-30` (24 months).

### Coverage

| asset_id   |   rows | first_date          | last_date           |   target_count |
|:-----------|-------:|:--------------------|:--------------------|---------------:|
| BNDX       |    153 | 2013-08-30 00:00:00 | 2026-04-30 00:00:00 |            153 |
| HYEM       |    153 | 2013-08-30 00:00:00 | 2026-04-30 00:00:00 |            153 |
| VWOB       |    153 | 2013-08-30 00:00:00 | 2026-04-30 00:00:00 |            153 |

### Target correlation

| asset_id   |   BNDX |   HYEM |   VWOB |
|:-----------|-------:|-------:|-------:|
| BNDX       |  1     |  0.395 |  0.678 |
| HYEM       |  0.395 |  1     |  0.868 |
| VWOB       |  0.678 |  0.868 |  1     |

## Decisions

- Keep `BND, IEF, LQD, HYG` together despite BND redundancy; task-universe sensitivity is deferred.
- Keep `BNDX, VWOB, HYEM` as the international family; its three-task IC is intrinsically coarse.
- Keep `t_index` in every configuration.
- Exclude REIT and commodity ETF families from this program. Commodity variables remain eligible inputs.
