"""
Pydantic Configs used for MLFlow tracking
"""
from pydantic import BaseModel, Field
from datetime import date
from enum import StrEnum


    
class Interval(StrEnum):
    """Supported Yahoo Finance fetch intervals. Granularity of raw data coming from Yahoo. This is the data vendor frequency."""
    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"
    QUARTERLY = "3mo"
    
class Horizon(StrEnum):
    """Supported prediction horizons."""
    DAILY = "B" # Business day
    WEEKLY = "W-FRI" # Friday is often considered end of trading week
    MONTHLY = "BM" # Business month end, aligns with last trading day of the month
    QUARTERLY = "BQ" # Business quarter end
    YEARLY = "BA" # Business year end

class TestingConfig(BaseModel):
    start_date: date = Field(..., description="Start date for the data range")
    end_date: date = Field(..., description="End date for the data range")
    interval: Interval = Field(Interval.DAILY, description="Data frequency")
    tickers: list[str] = Field(..., description="List of asset tickers")
    horizon: Horizon = Field(Horizon.MONTHLY, description="Prediction horizon")
