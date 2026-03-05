"""Core primitives and settings shared across BayesFolio.

This package contains foundational types/configuration only.
"""

from bayesfolio.core.settings import Horizon, Interval
from bayesfolio.core.types import HorizonDays, ReturnDecimal, Ticker, Weight

__all__ = [
    "Horizon",
    "HorizonDays",
    "Interval",
    "ReturnDecimal",
    "Ticker",
    "Weight",
]
