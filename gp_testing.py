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
from gpytorch.kernels import RBFKernel, ScaleKernel, PeriodicKernel, LinearKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from math import sqrt, log
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
torch.set_default_dtype(torch.float64)


warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.3}'.format

# Date range
start = '2019-06-28'
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
snpe_data = df[df['asset_id'] == 'SNPE'].reset_index(drop=True)

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

# Plot the line graph
plt.figure(figsize=(10, 6))
plt.plot(snpe_data['date'], snpe_data['y_excess_lead'], label='SNPE', color='blue')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Line Graph of SNPE Asset')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# take esgd and add the index as a column
esgd = esgd_data.reset_index()
snpe = snpe_data.reset_index()

esgd = esgd[['index', 'y_excess_lead']]
snpe = snpe[['index', 'y_excess_lead']]
X = esgd[['index']]
y = esgd[['y_excess_lead']]
y2 = snpe[['y_excess_lead']]

# create a test train split 
from sklearn.model_selection import train_test_split
n_month = 2
X_train = X[:-n_month]
X_test = X[-n_month:]
y_train = y[:-n_month]
y_test = y[-n_month:]

y_train2 = y2[:-n_month]
y_test2 = y2[-n_month:]

# use a minmax scaler from sklearn to scale the X and a zscore to scale the y 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler_X = MinMaxScaler()
scaler_y = StandardScaler()
scaler_y2 = StandardScaler()
scaler_X.fit(X)
scaler_y.fit(y)
scaler_y2.fit(y2)
X_scaled = scaler_X.transform(X_train)
y_scaled = scaler_y.transform(y_train)
y2_scaled = scaler_y2.transform(y_train2)
train_x = torch.tensor(X_scaled, dtype=torch.float64)
train_y = torch.tensor(y_scaled, dtype=torch.float64).flatten()
train_y2 = torch.tensor(y2_scaled, dtype=torch.float64).flatten()

X_test_scaled = scaler_X.transform(X_test)
test_x = torch.tensor(X_test_scaled, dtype=torch.float64)
y_test_scaled = scaler_y.transform(y_test)
y_test_scaled2 = scaler_y2.transform(y_test2)
test_y = torch.tensor(y_test_scaled, dtype=torch.float64).flatten()
test_y2 = torch.tensor(y_test_scaled2, dtype=torch.float64).flatten()

##################Single Task GP######################

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
                period_length_prior=LogNormalPrior(loc=0.0, scale=0.5),
                lengthscale_prior=lengthscale_prior,lengthscale_constraint=lengthscale_constraint) #+ LinearKernel()
        kernel_scale = ScaleKernel(kernel)
        kernel_scale.base_kernel.kernels[1].initialize(period_length=1.0)
        self.covar_module = ScaleKernel(kernel_scale)

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
    print('Iter %d/%d - Loss: %.3f    noise: %.3f' % ( #lengthscalerbf: %.3f periodlength: %.3f
        i + 1, training_iter, loss.item(),
        #model.covar_module.base_kernel.lengthscale.item(),
        #model.covar_module.base_kernel.kernels[0].lengthscale.item(),
        #model.covar_module.base_kernel.kernels[1].period_length.item(),
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
    
    
############## Multitask GP     ######################
class MultitaskGPModel(ExactGP):
    """
    Multitask GP with Hadamard kernel for combining task and input covariances.
    
    Notes: 	The Hadamard (Linear Model of Coregionalization, LMC) generalizes this:
    multiple latent kernels + task mixing, which lets different tasks have different smoothness / lengthscales.
    """
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        kernel = MaternKernel(
            nu=0.5,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint
             ) + PeriodicKernel(
                period_length_prior=LogNormalPrior(loc=0.0, scale=0.5),
                lengthscale_prior=lengthscale_prior,lengthscale_constraint=lengthscale_constraint) #+ LinearKernel()
        kernel_scale = ScaleKernel(kernel)
        self.covar_module = ScaleKernel(kernel_scale)
        
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    def forward(self,x,i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    
# initialize likelihood and model
noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
likelihood =  GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(
            MIN_INFERRED_NOISE_LEVEL,
            initial_value=noise_prior.mode,
        ),
)

train_i_task1 = torch.full((train_x.shape[0],1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x.shape[0],1), dtype=torch.long, fill_value=1)
full_train_x = torch.cat([train_x, train_x])
full_train_i = torch.cat([train_i_task1, train_i_task2])
full_train_y = torch.cat([train_y, train_y2])
model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood)
model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(MIN_INFERRED_NOISE_LEVEL))

# traing the model
training_iter = 500 
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Early stopping variables
best_loss = float("inf")
patience_counter = 0
patience = 10

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(full_train_x, full_train_i)
    loss = -mll(output, full_train_y)
    loss.backward()
    print('Iter %d/500 - Loss: %.3f' % (i + 1, loss.item()))
    optimizer.step()

    # Early stopping logic
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0  # Reset patience counter if improvement
    else:
        patience_counter += 1  # Increment patience counter if no improvement

    if patience_counter >= patience:
        break  # Stop training early if patience is exceeded

# make predictions

# Initialize plots1
model.eval()
likelihood.eval()
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(12, 5))


X_full = scaler_X.transform(X)
x_tensor = torch.tensor(X_full, dtype=torch.float64)
test_i_task1 = torch.full((x_tensor.shape[0],1), dtype=torch.long, fill_value=0)
test_i_task2 = torch.full((x_tensor.shape[0],1), dtype=torch.long, fill_value=1)
y_full = scaler_y.transform(y)
y_full2 = scaler_y2.transform(y2)
y_tensor = torch.tensor(y_full, dtype=torch.float64).flatten()
y_tensor2 = torch.tensor(y_full2, dtype=torch.float64).flatten()


# Make predictions - one task at a time
# We control the task we cae about using the indices

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_y1 = likelihood(model(x_tensor, test_i_task1))
    observed_pred_y2 = likelihood(model(x_tensor, test_i_task2))
    
# Define plotting function
def ax_plot(ax, train_y, train_x, rand_var, title):
    # Get lower and upper confidence bounds
    lower, upper = rand_var.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.detach().flatten().numpy(), train_y.detach().numpy(), 'k*')
    # Predictive mean as blue line
    ax.plot(train_x.detach().flatten().numpy(), rand_var.mean.detach().numpy(), 'b')
    # Shade in confidence
    ax.fill_between(train_x.detach().flatten().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(title)

# Plot both tasks
ax_plot(y1_ax, y_tensor, x_tensor, observed_pred_y1, 'Observed Values (Likelihood)')
ax_plot(y2_ax, y_tensor2, x_tensor, observed_pred_y2, 'Observed Values (Likelihood)')


#calculating results
y_full = scaler_y.transform(y)
y_full2 = scaler_y2.transform(y2)
y1_scaled = pd.DataFrame(y_full, columns=['y1'])
y2_scaled = pd.DataFrame(y_full2, columns=['y2'])
y1_pred = pd.DataFrame(observed_pred_y1.mean.numpy().reshape(-1, 1), columns=['y1_pred'])
y2_pred = pd.DataFrame(observed_pred_y2.mean.numpy().reshape(-1, 1), columns=['y2_pred'])
y_all = pd.concat([y1_scaled, y1_pred, y2_scaled, y2_pred], axis=1)
y_all

##################### Kronecker Multitask GP ######################
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        kernel = MaternKernel(
            nu=0.5,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint
             ) + PeriodicKernel(
                period_length_prior=LogNormalPrior(loc=0.0, scale=0.5),
                lengthscale_prior=lengthscale_prior,lengthscale_constraint=lengthscale_constraint) #+ LinearKernel()
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            kernel, num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

train_y_comb = torch.stack([train_y, train_y2], -1)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
model = MultitaskGPModel(train_x, train_y_comb, likelihood)
# model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(MIN_INFERRED_NOISE_LEVEL))

# Find optimal model hyperparameters
model.train()
likelihood.train()
training_iter =500
# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Early stopping variables
best_loss = float("inf")
patience_counter = 0
patience = 10

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y_comb)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()
    
    # Early stopping logic
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0  # Reset patience counter if improvement
    else:
        patience_counter += 1  # Increment patience counter if no improvement

    if patience_counter >= patience:
        break  # Stop training early if patience is exceeded

    
model.eval()
likelihood.eval()
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(12, 5))

X_full = scaler_X.transform(X)
x_tensor = torch.tensor(X_full, dtype=torch.float64)
y_full = scaler_y.transform(y)
y_full2 = scaler_y2.transform(y2)
y_tensor = torch.tensor(y_full, dtype=torch.float64).flatten()
y_tensor2 = torch.tensor(y_full2, dtype=torch.float64).flatten()
y_tensor_comb = torch.stack([y_tensor, y_tensor2], -1)

# Make predictions
# This contains predictions for both tasks, flattened out
# The first half of the predictions is for the first task
# The second half is for the second task
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(x_tensor))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()
    
# Plot training data as black stars
y1_ax.plot(x_tensor.detach().flatten().numpy(), y_tensor_comb[:, 0].detach().numpy(), 'k*')
# Predictive mean as blue line
y1_ax.plot(x_tensor.flatten().numpy(), mean[:, 0].numpy(), 'b')
# Shade in confidence
y1_ax.fill_between(x_tensor.flatten().numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
y1_ax.set_ylim([-3, 3])
y1_ax.legend(['Observed Data', 'Mean', 'Confidence'])
y1_ax.set_title('Observed Values (Likelihood)')

# Plot training data as black stars
y2_ax.plot(x_tensor.detach().flatten().numpy(), y_tensor_comb[:, 1].detach().numpy(), 'k*')
# Predictive mean as blue line
y2_ax.plot(x_tensor.flatten().numpy(), mean[:, 1].numpy(), 'b')
# Shade in confidence
y2_ax.fill_between(x_tensor.flatten().numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
y2_ax.set_ylim([-3, 3])
y2_ax.legend(['Observed Data', 'Mean', 'Confidence'])
y2_ax.set_title('Observed Values (Likelihood)')


# Single task need to be updated. 
#the scale for y1 y1_pred was different but the multitask still beat out the singletask
# Old y2 y2_pred with just singletask
# Obs	Pred
# 0	0.137	0.134
# 1	-0.576	-0.571
# 2	0.145	0.143
# 3	0.188	0.188
# 4	0.509	0.507
# ...	...	...
# 69	-0.624	-0.621
# 70	0.827	0.822
# 71	0.869	0.868
# 72	0.2	0.616
# 73	0.228	0.466

# together #hadamard results 
	y1	y1_pred	y2	y2_pred
0	-0.539	-0.463	0.138	0.0617
1	-0.476	-0.493	-0.575	-0.538
2	0.501	0.46	0.147	0.172
3	0.605	0.54	0.189	0.247
4	0.0689	0.158	0.511	0.429
...	...	...	...	...
69	0.558	0.464	-0.623	-0.532
70	0.757	0.794	0.828	0.773
71	0.338	0.401	0.87	0.809
72	-0.649	0.313	0.201	0.572
73	0.793	0.252	0.14	0.418


# TAKE AWAY: The multitask is 10% better for y2

### krokecker results 

y1	y1_pred	y2	y2_pred
0	-0.539	-0.489	0.138	-0.423
1	-0.476	-0.477	-0.575	-0.413
2	0.501	0.467	0.147	0.4
3	0.605	0.571	0.189	0.488
4	0.0689	0.113	0.511	0.0952
...	...	...	...	...
69	0.558	0.467	-0.623	0.397
70	0.757	0.761	0.828	0.651
71	0.338	0.386	0.87	0.329
72	-0.649	0.328	0.201	0.279
73	0.793	0.292	0.14	0.248

#Takeaway comparing the Hadamard vs the Krokecker:
# Kronecker Slightly better for y1, Kronecker WAYY better for y2 66% better. 
# Kronecker more realistically displays accurate uncertainty. SNPE does infact vary
# quite a bit more than ESGD. 