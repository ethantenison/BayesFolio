"""
Testing out a GP for time series data.

"""
import pandas as pd
import yfinance as yf
import warnings
from marketmaven.utils import get_current_date
import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.3}'.format

# Date range
start = '2018-01-01'
end = get_current_date()
interval = "1d"  # Daily data

# Tickers of assets 
#assets = ['AVEM', 'XMMO', 'ESGD', 'BYLD', 'ISCF', 'HYEM', 'VNQI', 'VNQ', 'SNPE', 'IEF', 'VBK'] # , 'AGEPX', 'SWSSX', 'SWVXX' mutual funds redundant 'AVES','VWOB',
TICKERS = ['AVEM', 'XMMO', 'ESGD', 'BYLD', 'ISCF', 'HYEM', 'VNQI', 'VNQ', 'SNPE', 'IEF', 'VBK']
TICKERS.sort()
# ==== CONFIG ====

START = start     # adjust as needed
END = get_current_date()                # None = up to today
HORIZON = "monthly"        # options: "weekly", "monthly", or "fixed"
FIXED_H_DAYS = 5          # only used if HORIZON == "fixed"

# ==== CODE ====

def fetch_prices(tickers, start, end=None):
    """
    Returns a tidy df with MultiIndex (date, ticker) and column 'Adj Close'.
    Adjusted Close includes dividends & splits => total-return compatible.
    """
    px = yf.download(
        tickers=tickers, start=start, end=end, interval="1d",
        group_by="ticker", auto_adjust=False, progress=False
    )
    # Normalize shape across yfinance versions
    if isinstance(px.columns, pd.MultiIndex):
        # Multi-index columns: level 0 = ticker, level 1 = field
        adj = []
        for tk in tickers:
            if (tk, 'Adj Close') in px.columns:
                s = px[(tk, 'Adj Close')].rename(tk)
                adj.append(s)
        adj = pd.concat(adj, axis=1)
    else:
        # Single-index columns: use 'Adj Close' directly (single ticker)
        adj = px[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
    adj = adj.dropna(how="all")
    adj.index = pd.to_datetime(adj.index)
    return adj

def fetch_rf_daily(start, end=None):
    """
    Fetch ^IRX (13-week T-bill), robust to MultiIndex columns and auto_adjust True/False.
    Returns a daily series of *continuous* daily rate ≈ (annual_fraction / 252).
    """
    rf = yf.download("^IRX", start=start, end=end, interval="1d", progress=False, auto_adjust=False)

    if rf.empty:
        raise RuntimeError("Could not download ^IRX. Try expanding the date range.")

    # Handle both single-index and MultiIndex column shapes
    if isinstance(rf.columns, pd.MultiIndex):
        col = None
        candidates = [
            ("Adj Close", "^IRX"), ("Close", "^IRX"),
            ("^IRX", "Adj Close"), ("^IRX", "Close")
        ]
        for c in candidates:
            if c in rf.columns:
                col = rf.loc[:, c].astype(float)
                break
        if col is None:
            # Fallback: slice by level name, then pick the first column
            if "Adj Close" in rf.columns.get_level_values(0):
                col = rf.xs("Adj Close", level=0, axis=1).iloc[:, 0].astype(float)
            else:
                col = rf.xs("Close", level=0, axis=1).iloc[:, 0].astype(float)
    else:
        if "Adj Close" in rf.columns:
            col = rf["Adj Close"].astype(float)
        elif "Close" in rf.columns:
            col = rf["Close"].astype(float)
        else:
            raise KeyError(f"^IRX: Neither 'Adj Close' nor 'Close' in columns: {rf.columns.tolist()}")

    # Convert annualized percent -> fraction
    ann_frac = col / 100.0
    rf_daily_cont = ann_frac / 252.0
    rf_daily_cont.index = pd.to_datetime(rf_daily_cont.index)

    # Fill to business-day grid and forward-fill small gaps
    all_bd = pd.date_range(rf_daily_cont.index.min(), rf_daily_cont.index.max(), freq="B")
    rf_daily_cont = rf_daily_cont.reindex(all_bd).ffill()
    rf_daily_cont.name = "rf_daily_cont"
    return rf_daily_cont

def compute_excess_future_return_calendar(adj_close: pd.Series,
                                          rf_daily_cont: pd.Series,
                                          freq: str = "W-FRI"):
    """
    Calendar-aligned future excess return for one asset.
    y_excess = exp( log(P_{t1}) - log(P_t) - ∑_{(t,t1]} rf_daily_cont ) - 1
    Returns a Series indexed by the start-of-period date t.
    """
    # Resample to period-end closes
    px_period = adj_close.resample(freq).last().dropna()
    period_end = px_period.index
    px_fwd = px_period.shift(-1)

    # Price log-return for t -> t1
    log_r_price = (np.log(px_fwd) - np.log(px_period)).iloc[:-1]

    # Integrate RF over (t, t1] by summing continuous daily rates
    rf_log = []
    for t, t1 in zip(period_end[:-1], period_end[1:]):
        days = rf_daily_cont.loc[(rf_daily_cont.index > t) & (rf_daily_cont.index <= t1)]
        rf_log.append(days.sum() if not days.empty else 0.0)
    rf_log = pd.Series(rf_log, index=log_r_price.index)

    # Use numpy arrays to avoid dtype gotchas, then rebuild Series
    delta = (log_r_price.values - rf_log.values)
    y_excess_vals = np.exp(delta) - 1.0
    y_excess = pd.Series(y_excess_vals, index=log_r_price.index, name="y_excess_lead")
    return y_excess

def compute_excess_future_return_fixed(adj_close: pd.Series,
                                       rf_daily_cont: pd.Series,
                                       h_days: int = 5):
    """
    Fixed trading-day horizon (t -> t+h).
    y_excess = exp( log(P_{t+h}) - log(P_t) - sum_{k=0..h-1} rf_daily_cont[t+k] ) - 1
    """
    # Ensure we have RF on the same business-day index
    s = adj_close.dropna()
    idx = s.index
    rf_on_px = rf_daily_cont.reindex(idx).ffill()

    log_p = np.log(s)
    log_r_price = log_p.shift(-h_days * -1) - log_p  # log(P_{t+h}) - log(P_t); shift(-h) forward
    # Sum RF daily cont rates over next h days, aligned at t
    rf_log = rf_on_px.shift(-h_days + 1).rolling(h_days).sum()  # sum of future h days
    # Target
    y_excess = np.exp(log_r_price - rf_log) - 1.0
    y_excess = y_excess.iloc[:-h_days]  # drop last h because of lookahead
    y_excess.name = "y_excess_lead"
    return y_excess

def build_long_panel(tickers, start, end=None, horizon="weekly", fixed_h_days=5):
    prices = fetch_prices(tickers, start, end)
    rf_daily_cont = fetch_rf_daily(start, end)

    # Choose calendar frequency
    if horizon.lower() == "weekly":
        freq = "W-FRI"
        compute_fn = lambda s: compute_excess_future_return_calendar(s, rf_daily_cont, freq=freq)
    elif horizon.lower() == "monthly":
        freq = "BM" # Business Month-end
        compute_fn = lambda s: compute_excess_future_return_calendar(s, rf_daily_cont, freq=freq)
    elif horizon.lower() == "fixed":
        compute_fn = lambda s: compute_excess_future_return_fixed(s, rf_daily_cont, h_days=fixed_h_days)
    else:
        raise ValueError("horizon must be one of {'weekly','monthly','fixed'}")

    rows = []
    for tk in prices.columns:
        try:
            y_excess = compute_fn(prices[tk].dropna())
            df_tk = y_excess.to_frame()
            df_tk["asset_id"] = tk
            df_tk = df_tk.rename_axis("date").reset_index()[["date", "asset_id", "y_excess_lead"]]
            rows.append(df_tk)
        except Exception as e:
            print(f"[WARN] Skipping {tk}: {e}")

    panel = pd.concat(rows, axis=0).sort_values(["date", "asset_id"]).reset_index(drop=True)
    return panel

df = build_long_panel(TICKERS, START, END, horizon=HORIZON, fixed_h_days=FIXED_H_DAYS)



# Filter the DataFrame for the "ESGD" asset
esgd_data = df[df['asset_id'] == 'ESGD'].reset_index(drop=True)

# Plot the line graph
plt.figure(figsize=(10, 6))
plt.plot(esgd_data['date'], esgd_data['y_excess_lead'], label='ESGD', color='blue')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Line Graph of ESGD Asset')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# take esgd and add the index as a column
esgd = esgd_data.reset_index()

esgd = esgd[['index', 'y_excess_lead']]
X = esgd[['index']]
y = esgd[['y_excess_lead']]

# create a test train split 
from sklearn.model_selection import train_test_split
n_month = 3
X_train = X[:-n_month]
X_test = X[-n_month:]
y_train = y[:-n_month]
y_test = y[-n_month:]

# use a minmax scaler from sklearn to scale the X and a zscore to scale the y 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler_X = MinMaxScaler()
scaler_y = StandardScaler()
scaler_X.fit(X)
scaler_y.fit(y)
X_scaled = scaler_X.transform(X_train)
y_scaled = scaler_y.transform(y_train)
train_x = torch.tensor(X_scaled, dtype=torch.float64)
train_y = torch.tensor(y_scaled, dtype=torch.float64).flatten()

X_test_scaled = scaler_X.transform(X_test)
test_x = torch.tensor(X_test_scaled, dtype=torch.float64)
y_test_scaled = scaler_y.transform(y_test)
test_y = torch.tensor(y_test_scaled, dtype=torch.float64).flatten()

from gpytorch.kernels import RBFKernel, ScaleKernel, PeriodicKernel, LinearKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from math import sqrt, log
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
torch.set_default_dtype(torch.float64)

SQRT2 = sqrt(2)
SQRT3 = sqrt(3)

ard_num_dims = 1
lengthscale_prior = LogNormalPrior(loc=SQRT2 + log(ard_num_dims) * 0.5, scale=SQRT3)
lengthscale_constraint = GreaterThan(2.5e-2, initial_value=lengthscale_prior.mode)
MIN_INFERRED_NOISE_LEVEL = 1e-4  # Minimum noise level to avoid numerical issues
# We will use the simplest form of GP model, exact inference
class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        kernel = MaternKernel(
            nu=0.5,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint
             ) + PeriodicKernel(
                period_length_prior=LogNormalPrior(loc=0.0, scale=1.5),
                lengthscale_prior=lengthscale_prior,lengthscale_constraint=lengthscale_constraint) #+ LinearKernel()
        kernel_scale = ScaleKernel(kernel)
        kernel_scale.base_kernel.kernels[1].initialize(period_length=12.0)
        self.covar_module = ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
likelihood =  GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(
            MIN_INFERRED_NOISE_LEVEL,
            initial_value=noise_prior.mode,
        ),
)
model = ExactGPModel(train_x , train_y, likelihood)
model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(MIN_INFERRED_NOISE_LEVEL))

# traing the model
training_iter = 500  # Number of training iterations
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Early stopping variables
best_loss = float("inf")
patience_counter = 0
patience = 10

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscalerbf: %.3f noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        #model.covar_module.base_kernel.lengthscale.item(),
        model.covar_module.base_kernel.kernels[0].lengthscale.item(),
        # model.covar_module.base_kernel.kernels[1].lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

    # Early stopping logic
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0  # Reset patience counter if improvement
    else:
        patience_counter += 1  # Increment patience counter if no improvement

    if patience_counter >= patience:
        break  # Stop training early if patience is exceeded

X_full = scaler_X.transform(X)
x_tensor = torch.tensor(X_full, dtype=torch.float64)
y_full = scaler_y.transform(y)
y_tensor = torch.tensor(y_full, dtype=torch.float64).flatten()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = x_tensor
    observed_pred = likelihood(model(test_x))


with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(x_tensor.numpy(), y_tensor.numpy().flatten(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.flatten().numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

