# July 2026 International Equity ETF Features

This is the canonical experiment folder for the July 2026 international equity ETF feature screens.

Universe:

- `VEA`
- `VWO`
- `VSS`
- `EWX`

Primary decision metric:

- 12-window GP scenario portfolio return under Riskfolio CVaR/MaxRet, with forecast metrics and IC used as checks.

Canonical corrected runs:

- `runs/international_equity_dynamic_feature_screen_20260708T2115Z/`
- `runs/international_equity_remaining_feature_screen_20260708T2320Z/`

Historical note:

The first international runs were originally written under
`experiments/2026-07-us-equity-etf-features/` because they reused the U.S. equity experiment wrapper.
That was poor experiment organization. They have been copied here so future international equity work has
its own experiment root. The original run manifests and command strings may still reference the old location
because that is where those processes actually executed.

Do not treat the earlier `split_groups` international feature-combination screen as evidence against
international features. That path hardcoded covariance blocks and ignored selected custom macro columns.
Use the corrected dynamic screen instead.
