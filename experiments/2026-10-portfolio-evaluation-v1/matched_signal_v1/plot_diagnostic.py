"""Render matched signal portfolio wealth and ranking diagnostics to PNG bytes.

Research presentation only; inputs are already scored monthly IC and dollar NAV.
No fitting, allocation, network, file writes or portfolio performance calculations.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


def render_diagnostic(arrays: dict[str, np.ndarray]) -> bytes:
    """Plot matched allocation outcomes and monthly cross-sectional ranking IC.

    Args:
        arrays: Dated GP/momentum wealth (runs, months+1), pooled monthly IC
            (runs, months), and cached simple-momentum wealth in dollars.

    Returns:
        PNG bytes. Bands are initialization ranges, not confidence intervals.
        The cached top-seven line is explicitly a different allocation rule.
    """
    origins = np.asarray(arrays["dates"], dtype="datetime64[D]")
    targets = (origins.astype("datetime64[M]") + 2).astype("datetime64[D]") - 1
    timeline = np.r_[origins[:1], targets]
    figure, axes = plt.subplots(
        2, 1, figsize=(13.5, 9), sharex=True, layout="constrained", gridspec_kw={"height_ratios": [2, 1]}
    )
    for key, label, color in [
        ("gp", "GP ranking + unchanged hard-rank allocator", "#2058aa"),
        ("momentum", "Momentum ranking + identical allocator and GP risk", "#c24f4d"),
    ]:
        wealth, ic = arrays[f"{key}_wealth"], arrays[f"{key}_ic_pooled"]
        axes[0].plot(
            timeline, wealth.mean(axis=0), color=color, linewidth=2.5, label=f"{label} | ${wealth[:, -1].mean():,.0f}"
        )
        axes[0].fill_between(timeline, wealth.min(axis=0), wealth.max(axis=0), color=color, alpha=0.12)
        axes[1].plot(targets, ic.mean(axis=0), color=color, linewidth=1.5, marker="o", markersize=3, label=key)
        axes[1].fill_between(targets, ic.min(axis=0), ic.max(axis=0), color=color, alpha=0.12)
    simple = arrays["simple_momentum_wealth"]
    axes[0].plot(
        timeline,
        simple,
        color="#6d747c",
        linestyle="--",
        linewidth=1.7,
        label=f"Earlier momentum top-7 equal weight — context only | ${simple[-1]:,.0f}",
    )
    axes[0].set_title(
        "Matched ranking-signal diagnostic: GP versus momentum\n"
        "June 2024–May 2026 | Same 15 ETFs, allocator, covariance and constraints",
        pad=12,
    )
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("Investment value ($)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    axes[1].axhline(0, color="#777777", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("Monthly rank IC\n(Spearman, 15 ETFs)")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_xlabel("Realized return month")
    for axis in axes:
        axis.grid(alpha=0.18)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].set_xlim(timeline[0], timeline[-1] + np.timedelta64(10, "D"))
    figure.supxlabel(
        "Mean of 3 seed wealth paths; shading is their range, not confidence. "
        "Gross of explicit costs and taxes.\n"
        "Momentum still uses the same GP covariance; this is not a GP-free strategy or a new holdout.",
        fontsize=9,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=170)
    plt.close(figure)
    return buffer.getvalue()
