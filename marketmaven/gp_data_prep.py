"""
This module contains functions for preparing and processing financial market data
"""
import torch
import pandas as pd

def prepare_multitask_gp_data(
    df: pd.DataFrame,
    target_col: str = "target",
    asset_col: str = "asset_id",
    drop_cols: list[str] = ["date"],
    dtype: torch.dtype = torch.float64
):
    """
    Convert a multi-asset DataFrame into tensors for Hadamard multitask GP in GPyTorch.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing target, features, and asset identifier.
    target_col : str, default="target"
        Column name for the prediction target (e.g., excess return).
    asset_col : str, default="asset_id"
        Column name identifying the asset/task.
    drop_cols : list[str], default=["date"]
        Columns to drop before tensor conversion.
    dtype : torch.dtype, default=torch.float64
        Torch dtype for output tensors.
    
    Returns
    -------
    full_train_x : torch.Tensor
        Feature tensor concatenated across all assets.
    full_train_i : torch.Tensor
        Task index tensor (N x 1) indicating asset membership.
    full_train_y : torch.Tensor
        Target tensor concatenated across all assets.
    task_map : dict
        Mapping of asset_id to task index.
        
    Example
    -------
    full_train_x, full_train_i, full_train_y, task_map = prepare_multitask_gp_data(
    df,
    target_col="y_excess_lead",   # your dependent variable
    asset_col="asset_id",
    drop_cols=["date", "asset_id"]
    )

    print(task_map)
    # {'EFA': 0, 'SCZ': 1, 'SPY': 2, 'SCHA': 3}

    print(full_train_x.shape, full_train_i.shape, full_train_y.shape)
    """
    # Ensure consistent column order
    if asset_col not in df.columns:
        raise ValueError(f"Expected column '{asset_col}' not found in DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Expected column '{target_col}' not found in DataFrame.")
    
    unique_assets = df[asset_col].unique()
    task_map = {asset: idx for idx, asset in enumerate(unique_assets)}
    
    tensors_x, tensors_i, tensors_y = [], [], []
    
    for asset in unique_assets:
        sub_df = df[df[asset_col] == asset].reset_index(drop=True)
        sub_df = sub_df.drop(columns=drop_cols, errors="ignore")
        
        if sub_df.isnull().any().any():
            raise ValueError(f"NaNs detected in asset {asset} data. Clean before training.")
        
        tensor = torch.tensor(sub_df.values, dtype=dtype)
        
        # Assume first column = target, rest = features
        y = tensor[:, 0]
        x = tensor[:, 1:]
        i = torch.full((x.shape[0], 1), fill_value=task_map[asset], dtype=dtype)
        
        tensors_x.append(x)
        tensors_y.append(y)
        tensors_i.append(i)
    
    full_train_x = torch.cat(tensors_x)
    full_train_i = torch.cat(tensors_i)
    full_train_y = torch.cat(tensors_y)
    
    return full_train_x, full_train_i, full_train_y, task_map
