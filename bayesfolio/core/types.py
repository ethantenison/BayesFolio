from __future__ import annotations

from typing import Annotated

from annotated_types import Ge, Gt, Le
from pydantic import StringConstraints

# Ticker: non-empty string, max 10 chars, uppercase
Ticker = Annotated[str, StringConstraints(min_length=1, max_length=10, strip_whitespace=True)]

# Weight: float in [0, 1]
Weight = Annotated[float, Ge(0.0), Le(1.0)]

# HorizonDays: positive integer
HorizonDays = Annotated[int, Gt(0)]

# ReturnDecimal: float representing a return as decimal (0.02 = 2%)
ReturnDecimal = Annotated[float, Ge(-1.0)]
