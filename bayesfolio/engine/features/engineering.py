from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_liquidity_features(
    frame: pd.DataFrame,
    ill_col: str = "ill",
    dolvol_col: str = "dolvol",
    q: float = 0.99,
) -> pd.DataFrame:
    """Add log-transformed liquidity features with upper-tail clipping.

    The transformation follows the legacy feature preparation behavior:
    ``ill`` and ``dolvol`` are scaled by ``1e9``, transformed with ``log1p``,
    clipped at the requested quantile, and written to ``ill_log`` and
    ``dolvol_log``.

    Args:
        frame: Long-format dataframe with columns including date/asset keys and
            liquidity inputs.
        ill_col: Column name for Amihud illiquidity proxy.
        dolvol_col: Column name for dollar volume.
        q: Upper quantile in ``(0, 1]`` for clipping log-liquidity tails.

    Returns:
        DataFrame in long format. If source liquidity columns exist, the output
        includes ``ill_log`` and ``dolvol_log`` and drops intermediate/raw
        ``ill`` and ``dolvol`` columns. All return values remain decimal units.

    Raises:
        ValueError: If ``q`` is outside ``(0, 1]``.
    """

    if q <= 0.0 or q > 1.0:
        msg = "q must be in (0, 1]."
        raise ValueError(msg)

    output = frame.copy()

    if ill_col in output.columns:
        output["ill_adj"] = output[ill_col] * 1e9
        output["ill_log"] = np.log(output["ill_adj"] + 1.0)
        upper_ill = output["ill_log"].quantile(q)
        output["ill_log"] = output["ill_log"].clip(upper=upper_ill)

    if dolvol_col in output.columns:
        output["dolvol_adj"] = output[dolvol_col] * 1e9
        output["dolvol_log"] = np.log(output["dolvol_adj"] + 1.0)
        upper_dolvol = output["dolvol_log"].quantile(q)
        output["dolvol_log"] = output["dolvol_log"].clip(upper=upper_dolvol)

    drop_cols = [col for col in [ill_col, "ill_adj", dolvol_col, "dolvol_adj"] if col in output.columns]
    if drop_cols:
        output = output.drop(columns=drop_cols)

    return output


def add_cross_sectional_momentum_rank(
    frame: pd.DataFrame,
    momentum_col: str = "mom12m",
    out_col: str = "cs_mom_rank",
) -> pd.DataFrame:
    """Add cross-sectional momentum percentile rank by date.

    Args:
        frame: Long-format dataframe with at least ``date`` and the momentum
            feature column.
        momentum_col: Momentum feature used for ranking.
        out_col: Output column for percentile ranks in ``[0, 1]``.

    Returns:
        DataFrame including ``out_col`` ranked cross-sectionally for each date.

    Raises:
        ValueError: If required columns are missing.
    """

    if "date" not in frame.columns or momentum_col not in frame.columns:
        msg = f"Missing required columns for momentum rank: date and {momentum_col}."
        raise ValueError(msg)

    output = frame.copy()
    output[out_col] = output.groupby("date")[momentum_col].rank(pct=True, method="average")
    return output


def add_target_lags(
    frame: pd.DataFrame,
    target_col: str = "y_excess_lead",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged target columns grouped by ``asset_id``.

    Args:
        frame: Long-format dataframe containing ``asset_id``, ``date``, and
            target return column.
        target_col: Target return column in decimal units.
        lags: Positive lag periods; defaults to ``[1, 2]``.

    Returns:
        DataFrame with additional lag columns named ``lag_<target>`` for lag 1
        and ``lagN_<target>`` for lag ``N > 1``. Return values stay in decimal
        units.

    Raises:
        ValueError: If required columns are missing or lags are invalid.
    """

    requested_lags = lags or [1, 2]
    if "asset_id" not in frame.columns or target_col not in frame.columns or "date" not in frame.columns:
        msg = "frame must include date, asset_id, and target columns."
        raise ValueError(msg)

    if not requested_lags or any(lag <= 0 for lag in requested_lags):
        msg = "lags must contain positive integers."
        raise ValueError(msg)

    output = frame.sort_values(["asset_id", "date"]).copy()
    grouped = output.groupby("asset_id", sort=False)[target_col]
    for lag in requested_lags:
        if lag == 1:
            col_name = f"lag_{target_col}"
        else:
            col_name = f"lag{lag}_{target_col}"
        output[col_name] = grouped.shift(lag)

    return output.sort_values(["date", "asset_id"]).reset_index(drop=True)


def build_t_index(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Build deterministic integer time index from sorted unique dates.

    Args:
        frame: Long-format dataframe containing a date column.
        date_col: Date-like column used to generate the integer index.

    Returns:
        DataFrame with ``t_index`` inserted as the first column. The index is
        0-based and stable for sorted date order.

    Raises:
        ValueError: If ``date_col`` is missing.
    """

    if date_col not in frame.columns:
        msg = f"Missing date column: {date_col}."
        raise ValueError(msg)

    output = frame.copy()
    output[date_col] = pd.to_datetime(output[date_col])
    output = output.sort_values([date_col, "asset_id"]).reset_index(drop=True)
    output["t_index"] = pd.factorize(output[date_col])[0]

    ordered_cols = ["t_index", *[col for col in output.columns if col != "t_index"]]
    return output.loc[:, ordered_cols]
