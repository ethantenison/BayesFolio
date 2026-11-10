"""Render static benchmark wealth comparisons without file or network access.

Research presentation layer: accepts precomputed dollar wealth and monthly dates,
returns PNG bytes, and never computes investment allocations or performance.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


def render_growth(arrays: dict[str, np.ndarray], comparison_label: str, labels: dict[str, str]) -> bytes:
    """Render the growth of a self-financing initial investment.

    Args:
        arrays: Decision dates, (seeds, months+1) comparison_wealth and matching
            <benchmark_id>_wealth dollar arrays, including initial investment.
        comparison_label: Descriptive portfolio name, not a performance claim.
        labels: Benchmark IDs mapped to chart labels in approved display order.

    Returns:
        PNG bytes. Seed band reflects initialization variation, not statistical
        confidence. Central curve averages wealth paths, not portfolio weights.
    """
    origins = np.asarray(arrays["dates"], dtype="datetime64[D]")
    targets = (origins.astype("datetime64[M]") + 2).astype("datetime64[D]") - 1
    timeline = np.r_[origins[:1], targets]
    paths = arrays["comparison_wealth"]
    figure, axis = plt.subplots(figsize=(13.8, 8), layout="constrained")
    for path in paths:
        axis.plot(timeline, path, color="#204e9c", linewidth=0.8, alpha=0.18, zorder=4)
    axis.fill_between(timeline, paths.min(axis=0), paths.max(axis=0), color="#204e9c", alpha=0.1)
    axis.plot(
        timeline,
        paths.mean(axis=0),
        color="#204e9c",
        linewidth=3,
        zorder=5,
        label=f"{comparison_label}  |  ${paths[:, -1].mean():,.0f}",
    )
    styles = [("#bd752d", "-"), ("#8355a7", "-"), ("#15816c", "--"), ("#59636e", ":"), ("#c94b50", "-.")]
    for (name, label), (color, style) in zip(labels.items(), styles, strict=True):
        values = arrays[f"{name}_wealth"]
        axis.plot(timeline, values, color=color, linestyle=style, linewidth=2, label=f"{label}  |  ${values[-1]:,.0f}")
    first, last = np.datetime_as_string(targets[0], unit="M"), np.datetime_as_string(targets[-1], unit="M")
    axis.set_title(
        f"Growth of $10,000: BayesFolio and five simple benchmarks\n"
        f"{first} through {last} returns | Monthly rebalancing | Distributions reinvested",
        pad=12,
    )
    axis.legend(loc="upper left", ncol=2, fontsize=9.4, framealpha=0.96)
    axis.set_xlim(timeline[0], timeline[-1] + np.timedelta64(12, "D"))
    axis.set_xlabel("Time — month-end investment value")
    axis.set_ylabel("Investment value ($)")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.grid(alpha=0.18)
    figure.supxlabel(
        f"BayesFolio: mean wealth of {len(paths)} saved run(s); shading is their range, not a confidence interval.\n"
        "Gross of explicit trading costs and taxes. Schwab is an ETF proxy simulation; cash earns the shared RF proxy.",
        fontsize=9,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=170)
    plt.close(figure)
    return buffer.getvalue()
