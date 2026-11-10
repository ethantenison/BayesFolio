"""Pure monthly target-weight scheduling and self-financing wealth calculations.

Research-only numerical layer; no network, fitting, allocation optimizer or I/O.
Returns are decimals, weights are fractions, and investment values are dollars.
Usage: imported by compare_schwab.py and synthetic tests.
"""

from __future__ import annotations

import numpy as np


def scheduled_weights(dates: list[str], initial: list[float], revisions: list[dict]) -> dict:
    """Apply only target revisions available at a monthly decision date.

    Args:
        dates: Strictly increasing actual rebalance dates, ISO YYYY-MM-DD.
        initial: Pre-window target fractions in proxy order, including cash.
        revisions: Verified records with published_on, effective_on, weights and
            source_url. Unknown intraday timing means same-day release is deferred.

    Returns:
        Monthly fractions, active revision indices (-1 denotes initial weights),
        and applied revision count. No date is omitted or future revision backfilled.
    """
    axis = np.asarray(dates, dtype="datetime64[D]")
    if axis.ndim != 1 or not len(axis) or np.isnat(axis).any() or (np.diff(axis) <= np.timedelta64(0, "D")).any():
        raise ValueError("Complete strictly increasing rebalance dates required")
    original = np.asarray(initial, dtype=float)
    candidates = [original]
    available = []
    for revision in revisions:
        if not revision.get("source_url"):
            raise ValueError("Every revision requires a source")
        published = np.datetime64(revision["published_on"], "D")
        effective = np.datetime64(revision["effective_on"], "D")
        if np.isnat(published) or np.isnat(effective):
            raise ValueError("Every revision requires finite publication and effective dates")
        available.append((published, effective))
        candidates.append(np.asarray(revision["weights"], dtype=float))
    for weights in candidates:
        if weights.ndim != 1 or weights.shape != original.shape or not np.isfinite(weights).all():
            raise ValueError("Complete matching target weight vectors required")
        if (weights < 0).any() or abs(weights.sum() - 1) > 1e-12:
            raise ValueError("Target weights must be long-only and sum to one; no normalization")
    indices = []
    for date in axis:
        eligible = [i for i, (published, effective) in enumerate(available) if published < date and effective <= date]
        chosen = max(eligible, key=lambda i: (max(available[i]), available[i][0], i)) if eligible else -1
        indices.append(chosen)
    weights = np.array([candidates[index + 1] for index in indices])
    return {"weights": weights, "active_revision": indices, "applied_revision_count": len(set(indices) - {-1})}


def portfolio_returns(actual: np.ndarray, risk_free: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reconstruct fully invested total returns from discounted-excess targets.

    Args:
        actual: Realized ETF targets, (months, assets), decimal discounted excess.
        risk_free: Matching realized monthly arithmetic risk-free returns, not yields.
        weights: (months, assets+1) fractions, with cash last. Tiny solver residuals
            have the original 1e-7 budget and 1e-8 negativity tolerances; no clipping.

    Returns:
        Monthly total returns, decimals. Cash has zero discounted excess and earns
        risk_free. The formula preserves the sealed GP return-accounting convention.
    """
    if actual.ndim != 2 or risk_free.shape != actual.shape[:1] or weights.shape != (len(actual), actual.shape[1] + 1):
        raise ValueError("Monthly returns, cash-inclusive weights and RF must align")
    if not all(np.isfinite(value).all() for value in [actual, risk_free, weights]):
        raise ValueError("Complete finite observations required; no dropped months")
    if (actual <= -1).any() or (risk_free <= -1).any():
        raise ValueError("Invalid decimal returns")
    if weights.min() < -1e-8 or np.max(np.abs(weights.sum(axis=1) - 1)) > 1e-7:
        raise ValueError("Fully invested long-only weights required; no repair")
    discounted_excess = np.sum(weights[:, :-1] * actual, axis=1)
    return (1 + discounted_excess) * (1 + risk_free) - 1


def wealth_path(total_returns: np.ndarray, initial_dollars: float = 10000.0) -> np.ndarray:
    """Compound total returns with no contributions or withdrawals.

    Args:
        total_returns: (..., months) decimal total-return arrays.
        initial_dollars: Strictly positive initial investment, in dollars.

    Returns:
        (..., months+1) investment values including the initial dollar amount.
    """
    if total_returns.ndim < 1 or not np.isfinite(total_returns).all() or (total_returns <= -1).any():
        raise ValueError("Finite valid total returns required")
    if not np.isfinite(initial_dollars) or initial_dollars <= 0:
        raise ValueError("Positive finite initial investment required")
    initial = np.ones((*total_returns.shape[:-1], 1))
    return initial_dollars * np.concatenate([initial, np.cumprod(1 + total_returns, axis=-1)], axis=-1)
