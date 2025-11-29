import mlflow
import pandas as pd
import warnings
import os
from joblib import Parallel, delayed
from pydantic import BaseModel
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from marketmaven.configs import TickerConfig, Interval, Horizon, CVConfig
from marketmaven.asset_prices import build_long_panel
import numpy as np
import torch
from marketmaven.configs import (
    RiskfolioConfig, OptModel, RiskMeasure, Objective, MuEstimator, CovEstimator)
from marketmaven.visualization.eda import correlation_matrix
from marketmaven.gp_data_prep import prepare_multitask_gp_data
from marketmaven.models.cv import rolling_time_splits_multitask
from marketmaven.models.scaling import MultitaskScaler
device = torch.device("cpu")
from marketmaven.models.gp import train_model_hadamard
from math import log, sqrt
from marketmaven.evaluate import evaluate_asset_pricing
from marketmaven.utils import check_equal_occurrences
from marketmaven.visualization.evaluation import plot_ls_cumulative_compare, plot_actual_vs_pred_matrix
from marketmaven.portfolio.helpers import assessing_long_short_performance, long_short_returns
from marketmaven.models.kernels import MeanF, KernelType, initialize_mean, initialize_kernel, adaptive_lengthscale_prior
from marketmaven.mlflow.helpers import (
    KernelConfig, MultiTaskConfig, long_to_panel, compute_benchmark_panel, r2_os, log_r2_os,
    model_error_by_time_index, log_kernel_to_mlflow, log_gpytorch_state_dict
)
import random
import itertools

warnings.filterwarnings(
    "ignore",
    message=".*torch.sparse.SparseTensor.*is deprecated.*"
)
warnings.filterwarnings("ignore", category=Warning, message=".*not p.d., added jitter.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*not p.d., added jitter.*")

# Setup

etf_tickers = [
    "SPY", # total US market big cap
    "MGK", # US growth
    "VTV", # US value
    "IJR", # US small cap
    "VNQ", # REIT ETF US centric
    "VNQI", # international REIT ETF
    "VEA", # developed international equity
    "VWO", # AVEM actually is better than VWO but not enough history
    "VSS", # forein small/mid cap
    "BND", # total bond market ETF US centric
    "IEF", # 7-10 year treasury bond ETF US centric
    "BNDX", # total international bond market ETF, USD hedged, but actually developed markets only
    # # "IBND", # international corporate bond market ETF unhedged
    # # "ISHG", # international high yield bond ETF unhedged
    "LQD", # investment grade bond ETF US centric
    "HYG", # High yield bond ETF US centric 
    "TIP", # Treasury inflation protected securities ETF US centric
    "EMB", # emerging market bond ETF USD hedged
    "EWX", # emerging market small cap dividend ETF
    #"PDBC", # Commodities ETF
    #"BIL", # 1-3 month us treasuries 
    
]


tickers = TickerConfig(
    start_date="2013-06-01",
    end_date="2025-11-29",
    interval=Interval.DAILY,
    tickers=etf_tickers,
    horizon=Horizon.MONTHLY,
    lookback_date="2010-06-01"
)