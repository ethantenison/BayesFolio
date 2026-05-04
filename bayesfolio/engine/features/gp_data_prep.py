"""
This module contains functions for preparing and processing financial market data
"""

import numpy as np
import pandas as pd
import torch


def prepare_multitask_gp_data(
    df: pd.DataFrame,
    target_col: str = "target",
    asset_col: str = "asset_id",
    drop_cols: list[str] = ["date"],
    dtype: torch.dtype = torch.float32,
):
    """
    Convert a multitask DataFrame into tensors for Hadamard multitask GP in GPyTorch,
    preserving the original row order (no per-task concatenation).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing target, features, and task identifiers.
    target_col : str
        Column name for the prediction target.
    asset_col : str
        Column name identifying the asset/task.
    drop_cols : list[str]
        Columns to drop before tensor conversion (e.g., date).
    dtype : torch.dtype
        Torch dtype for output tensors.

    Returns
    -------
    full_train_x : torch.Tensor
        Feature tensor (N x D).
    full_train_i : torch.Tensor
        Task index tensor (N x 1).
    full_train_y : torch.Tensor
        Target tensor (N).
    task_map : dict
        Mapping of asset_id → integer task index.
    """

    # Validate required columns
    if asset_col not in df.columns:
        raise ValueError(f"Expected column '{asset_col}' not found in DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Expected column '{target_col}' not found in DataFrame.")

    # Create mapping and apply it (preserves df order)
    unique_assets = df[asset_col].unique()
    task_map = {asset: idx for idx, asset in enumerate(unique_assets)}
    df = df.copy()
    df["__task_idx__"] = df[asset_col].map(task_map)

    # Drop columns not used for training
    df_proc = df.drop(columns=drop_cols + [asset_col], errors="ignore")

    if df_proc.isnull().any().any():
        bad_cols = df_proc.columns[df_proc.isnull().any()].tolist()
        raise ValueError(f"NaNs detected in columns {bad_cols}. Clean data before training.")

    # Extract target, features, and task index
    y_np = df_proc[target_col].to_numpy()
    feature_df = df_proc.drop(columns=[target_col, "__task_idx__"], errors="ignore")
    x_np = feature_df.to_numpy()
    i_np = df["__task_idx__"].to_numpy().reshape(-1, 1)

    # Convert to tensors
    full_train_x = torch.tensor(x_np, dtype=dtype)
    full_train_y = torch.tensor(y_np, dtype=dtype)
    full_train_i = torch.tensor(i_np, dtype=dtype)

    return full_train_x, full_train_i, full_train_y, task_map


def prepare_multitask_gp_data_with_task_feature(
    df: pd.DataFrame,
    target_col: str = "target",
    asset_col: str = "asset_id",
    drop_cols: list[str] = ["date"],
    dtype: torch.dtype = torch.float32,
):
    """
    Prepare multitask GP data for BoTorch, with task index as a feature.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        asset_col: Name of the asset/task column.
        drop_cols: Columns to drop before tensor conversion.
        dtype: Torch dtype for output tensors.

    Returns:
        full_train_x: Feature tensor (N x D+1), last column is task index.
        full_train_y: Target tensor (N).
        task_map: Mapping of asset_id → integer task index.
    """
    if asset_col not in df.columns:
        raise ValueError(f"Expected column '{asset_col}' not found in DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Expected column '{target_col}' not found in DataFrame.")

    unique_assets = df[asset_col].unique()
    task_map = {asset: idx for idx, asset in enumerate(unique_assets)}
    df = df.copy()
    df["__task_idx__"] = df[asset_col].map(task_map)

    df_proc = df.drop(columns=drop_cols + [asset_col], errors="ignore")

    # if df_proc.isnull().any().any():
    #     bad_cols = df_proc.columns[df_proc.isnull().any()].tolist()
    #     raise ValueError(f"NaNs detected in columns {bad_cols}. Clean data before training.")

    feature_df = df_proc.drop(columns=[target_col, "__task_idx__"], errors="ignore")
    x_np = feature_df.to_numpy()
    task_idx_np = df["__task_idx__"].to_numpy().reshape(-1, 1)
    x_with_task = np.concatenate([x_np, task_idx_np], axis=1)
    y_np = df_proc[target_col].to_numpy()

    full_train_x = torch.tensor(x_with_task, dtype=dtype)
    full_train_y = torch.tensor(y_np, dtype=dtype).unsqueeze(-1)

    return full_train_x, full_train_y, task_map
