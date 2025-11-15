"""Scaling utilities for MarketMaven models."""
import torch
from typing import Optional

class MultitaskScaler:
    """
    Reusable scaler for multitask Gaussian Process training and inference.

    Handles:
      • Input feature scaling (optionally excluding time column)
      • Output target scaling (global or per-task)
      • Inverse transforms for predictions and standard deviations

    Parameters
    ----------
    scale_y : {"none", "global", "per_task"}
        Type of target scaling to apply.
    eps : float, default=1e-8
        Numerical stability term to prevent division by zero.
    exclude_time_col : bool, default=True
        If True, the first column of X (time) is *not* scaled.

    Attributes
    ----------
    x_mean : torch.Tensor
        Mean of training inputs (used for feature scaling)
    x_std : torch.Tensor
        Standard deviation of training inputs
    y_mean : torch.Tensor
        Global target mean (if global scaling used)
    y_std : torch.Tensor
        Global target std (if global scaling used)
    y_mean_k : dict[int, torch.Tensor]
        Per-task means (if per_task scaling used)
    y_std_k : dict[int, torch.Tensor]
        Per-task stds (if per_task scaling used)
    """

    def __init__(self, scale_y: str = "per_task", eps: float = 1e-8, exclude_time_col: bool = False):
        self.scale_y = scale_y
        self.eps = eps
        self.exclude_time_col = exclude_time_col

        # containers
        self.x_mean: Optional[torch.Tensor] = None
        self.x_std: Optional[torch.Tensor] = None
        self.y_mean = None
        self.y_std = None
        self.y_mean_k = {}
        self.y_std_k = {}
        self.global_mu = None
        self.global_sd = None

    # -----------------------------------------------
    # Feature scaling
    # -----------------------------------------------
    def fit_x(self, X: torch.Tensor) -> None:
        """Compute mean/std for inputs, optionally excluding the first (time) column."""
        if self.exclude_time_col and X.shape[1] > 1:
            feat_mask = torch.arange(X.shape[1]) != 0
            self.x_mean = X[:, feat_mask].mean(0, keepdim=True)
            self.x_std  = X[:, feat_mask].std(0, unbiased=False, keepdim=True).clamp_min(self.eps)
        else:
            self.x_mean = X.mean(0, keepdim=True)
            self.x_std  = X.std(0, unbiased=False, keepdim=True).clamp_min(self.eps)

    def transform_x(self, X: torch.Tensor) -> torch.Tensor:
        """Scale features using previously fitted statistics."""
        if self.x_mean is None:
            raise RuntimeError("Must call fit_x() before transform_x().")

        Xs = X.clone()
        if self.exclude_time_col and X.shape[1] > 1:
            feat_mask = torch.arange(X.shape[1]) != 0
            Xs[:, feat_mask] = (X[:, feat_mask] - self.x_mean) / self.x_std
        else:
            Xs = (X - self.x_mean) / self.x_std
        return Xs

    # -----------------------------------------------
    # Target scaling
    # -----------------------------------------------
    def fit_y(self, y: torch.Tensor, I: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute target scaling (global or per task)."""
        if self.scale_y == "none":
            return y

        if self.scale_y == "global":
            self.y_mean = y.mean()
            self.y_std = y.std(unbiased=False).clamp_min(self.eps)
            return (y - self.y_mean) / self.y_std

        elif self.scale_y == "per_task":
            if I is None:
                raise ValueError("Must pass task indices I for per-task scaling.")

            y_scaled = torch.empty_like(y)
            task_vals = I.view(-1).to(torch.long)
            for k in torch.unique(task_vals):
                mask = task_vals == k
                mu = y[mask].mean()
                sd = y[mask].std(unbiased=False).clamp_min(self.eps)
                y_scaled[mask] = (y[mask] - mu) / sd
                self.y_mean_k[int(k.item())] = mu
                self.y_std_k[int(k.item())] = sd

            # global fallback for unseen tasks
            self.global_mu = y.mean()
            self.global_sd = y.std(unbiased=False).clamp_min(self.eps)
            return y_scaled

        else:
            raise ValueError(f"Unknown scale_y type: {self.scale_y}")

    # -----------------------------------------------
    # Inverse transforms
    # -----------------------------------------------
    def inverse_y(self, yhat: torch.Tensor, I: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inverse transform predictions back to original target space."""
        if self.scale_y == "none":
            return yhat

        if self.scale_y == "global":
            return yhat * self.y_std + self.y_mean

        elif self.scale_y == "per_task":
            if I is None:
                raise ValueError("Must pass task indices I for per-task inverse transform.")
            mu = torch.tensor(
                [self.y_mean_k.get(int(t), self.global_mu) for t in I.view(-1).tolist()],
                dtype=yhat.dtype, device=yhat.device
            )
            sd = torch.tensor(
                [self.y_std_k.get(int(t), self.global_sd) for t in I.view(-1).tolist()],
                dtype=yhat.dtype, device=yhat.device
            )
            return yhat * sd + mu

    def inverse_std(self, s: torch.Tensor, I: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inverse transform predicted standard deviations."""
        if self.scale_y == "none":
            return s
        if self.scale_y == "global":
            return s * self.y_std
        elif self.scale_y == "per_task":
            if I is None:
                raise ValueError("Must pass task indices I for per-task inverse transform.")
            sd = torch.tensor(
                [self.y_std_k.get(int(t), self.global_sd) for t in I.view(-1).tolist()],
                dtype=s.dtype, device=s.device
            )
            return s * sd