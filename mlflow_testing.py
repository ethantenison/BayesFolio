"""
Testing out a GP for time series data.

"""
from astropy import conf
import pandas as pd
import yfinance as yf
import warnings
from marketmaven.asset_prices import build_long_panel
from marketmaven.market_fundamentals import fetch_vix_term_structure
import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, PeriodicKernel, LinearKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from math import sqrt, log
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from pydantic import BaseModel, Field
from typing import List, Literal, Tuple
from datetime import date
import mlflow
from sklearn.preprocessing import MinMaxScaler, StandardScaler

torch.set_default_dtype(torch.float64)

# MLFlow Configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "GP_Model_Experiments"

# Set up MLFlow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.3}'.format

# -------------------------
# 1. Configuration Class
# -------------------------

class TestingConfig(BaseModel):
    start_date: date = Field(..., description="Start date for the data range")
    end_date: date = Field(..., description="End date for the data range")
    interval: Literal["1d", "1w", "1m"] = Field("1d", description="Data frequency")
    tickers: List[str] = Field(..., description="List of asset tickers")
    horizon: Literal["weekly", "monthly", "fixed"] = Field("monthly", description="Prediction horizon")
    fixed_h_days: int = Field(5, description="Fixed horizon days (if horizon is 'fixed')")


# Example usage
config = TestingConfig(
    start_date="2019-06-28",
    end_date="2025-09-01",
    interval="1d",
    tickers=["AVEM", "XMMO", "ESGD", "BYLD", "ISCF", "HYEM", "VNQI", "VNQ", "SNPE", "IEF", "VBK"],
    horizon="monthly",
    fixed_h_days=5
)


# -------------------------
# 2. Data Loading
# -------------------------
def load_data(config: TestingConfig) -> pd.DataFrame:
    """Load and preprocess data."""
    df = build_long_panel(config.tickers, config.start_date, config.end_date, horizon=config.horizon, fixed_h_days=config.fixed_h_days)
    vix = fetch_vix_term_structure(start=config.start_date, end=config.end_date, freq="BM")
    vix_core = vix[['Date', 'vix', 'vix_ts_level']]
    df = df.merge(vix_core, left_on='date', right_on='Date', how='left').drop(columns=['Date'])
    return df

# -------------------------
# 3. Data Filtering
# -------------------------
def filter_asset_data(df: pd.DataFrame, asset_id: str) -> pd.DataFrame:
    """Filter data for a specific asset."""
    return df[df['asset_id'] == asset_id].reset_index(drop=True)

# -------------------------
# 4. Feature Engineering
# -------------------------
def prepare_features(df: pd.DataFrame, input_vars: List[str], output_var: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare input and output variables."""
    X = df[input_vars]
    y = df[[output_var]]
    return X, y

# -------------------------
# 5. Scaling
# -------------------------
def scale_data(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, StandardScaler]:
    """Scale input and output data."""
    scaler_X = MinMaxScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    return X_scaled, y_scaled, scaler_X, scaler_y

# -------------------------
# 6. Train-Test Split
# -------------------------
def train_test_split_data(X: np.ndarray, y: np.ndarray, n_month: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets."""
    X_train = X[:-n_month]
    X_test = X[-n_month:]
    y_train = y[:-n_month]
    y_test = y[-n_month:]
    return X_train, X_test, y_train, y_test

# -------------------------
# 7. MLflow Experiment Logging
# -------------------------
def log_experiment(config: TestingConfig, model, metrics: dict, scaler_X, scaler_y):
    """Log configuration, model, and metrics to MLflow."""
    with mlflow.start_run():
        # Log configuration
        mlflow.log_params(config.to_dict())

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model
        mlflow.pytorch.log_model(model, "gp_model")

        # Save scalers as artifacts
        mlflow.log_artifact("scaler_X.pkl")
        mlflow.log_artifact("scaler_y.pkl")
        
# -------------------------
# 8. Plotting Utility
# -------------------------
def plot_predictions(x_tensor, y_tensor, observed_pred, title: str):
    """Plot predictions with confidence intervals."""
    lower, upper = observed_pred.confidence_region()
    plt.figure(figsize=(10, 6))
    plt.plot(x_tensor[:, 0].numpy(), y_tensor.numpy().flatten(), 'k*', label="Observed Data")
    plt.plot(x_tensor[:, 0].numpy(), observed_pred.mean.numpy(), 'b', label="Mean Prediction")
    plt.fill_between(x_tensor[:, 0].flatten().numpy(), lower.numpy(), upper.numpy(), alpha=0.5, label="Confidence Interval")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
# -------------------------
# 9. Main Pipeline
# -------------------------
def main():
    # Configuration
    config = GPConfig(
        start_date="2019-06-28",
        end_date="2025-09-01",
        tickers=["AVEM", "XMMO", "ESGD", "BYLD", "ISCF", "HYEM", "VNQI", "VNQ", "SNPE", "IEF", "VBK"],
        horizon="monthly",
        fixed_h_days=5
    )

    # Load data
    df = load_data(config)

    # Filter for ESGD asset
    esgd_data = filter_asset_data(df, "ESGD")

    # Prepare features
    input_vars = ['vix', 'vix_ts_level']
    output_var = 'y_excess_lead'
    X, y = prepare_features(esgd_data, input_vars, output_var)

    # Scale data
    X_scaled, y_scaled, scaler_X, scaler_y = scale_data(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y_scaled, n_month=2)

    # Convert to tensors
    train_x = torch.tensor(X_train, dtype=torch.float64)
    train_y = torch.tensor(y_train, dtype=torch.float64).flatten()
    test_x = torch.tensor(X_test, dtype=torch.float64)
    test_y = torch.tensor(y_test, dtype=torch.float64).flatten()

    # Train GP model (single-task example)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(500):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # Evaluate model
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    # Plot predictions
    plot_predictions(test_x, test_y, observed_pred, "GP Model Predictions")

    # Log experiment
    metrics = {"final_loss": loss.item()}
    log_experiment(config, model, metrics, scaler_X, scaler_y)

if __name__ == "__main__":
    main()