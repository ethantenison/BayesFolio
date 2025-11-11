import torch
from scipy.stats import spearmanr
import gpytorch

import math

def gaussian_nlpd(y_true: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor, eps: float = 1e-12) -> float:
    var = (y_std.clamp_min(eps))**2
    return float(0.5 * ( ((y_true - y_mean)**2) / var + torch.log(2 * torch.pi * var) ).mean())

def mse(y_true: torch.Tensor, y_mean: torch.Tensor) -> float:
    return float(((y_true - y_mean)**2).mean())

def rmse(y_true: torch.Tensor, y_mean: torch.Tensor) -> float:
    return float(torch.sqrt(((y_true - y_mean)**2).mean()))

def r2_score(y_true: torch.Tensor, y_mean: torch.Tensor) -> float:
    y_bar = y_true.mean()
    ss_tot = ((y_true - y_bar)**2).sum()
    ss_res = ((y_true - y_mean)**2).sum()
    return float(1 - ss_res / ss_tot)