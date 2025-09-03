"""
Testing out a GP for time series data.

"""
from types import CapsuleType
import pandas as pd
import yfinance as yf
import warnings
from marketmaven.asset_prices import build_long_panel
from marketmaven.market_fundamentals import fetch_vix_term_structure
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


df = build_long_panel(TICKERS, START, END, horizon=HORIZON, fixed_h_days=FIXED_H_DAYS)


# Filter the DataFrame for the "ESGD" asset
esgd_data = df[df['asset_id'] == 'ESGD'].reset_index(drop=True)
snpe_data = df[df['asset_id'] == 'SNPE'].reset_index(drop=True)
byld_data = df[df['asset_id'] == 'BYLD'].reset_index(drop=True)
avem_data = df[df['asset_id'] == 'AVEM'].reset_index(drop=True)
vbk_data = df[df['asset_id'] == 'VBK'].reset_index(drop=True)
iscf_data = df[df['asset_id'] == 'ISCF'].reset_index(drop=True)


def plot_asset_data(asset_data, asset_name):
    plt.figure(figsize=(10, 6))
    plt.plot(asset_data['date'], asset_data['y_excess_lead'], label=asset_name, color='blue')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'Line Graph of {asset_name} Asset')
    plt.legend()
    plt.grid(True)
    plt.show()
    
assets = [('ESGD', esgd_data), ('SNPE', snpe_data), ('BYLD', byld_data), ('AVEM', avem_data), ('VBK', vbk_data), ('ISCF', iscf_data)]
for asset_name, asset_data in assets:
    plot_asset_data(asset_data, asset_name)
    

# take esgd and add the index as a column
esgd = esgd_data.reset_index()
snpe = snpe_data.reset_index()
byld = byld_data.reset_index()
avem = avem_data.reset_index()
vbk = vbk_data.reset_index()
iscf = iscf_data.reset_index()

esgd = esgd[['index', 'y_excess_lead']]
snpe = snpe[['index', 'y_excess_lead']]
byld = byld[['index', 'y_excess_lead']]
avem = avem[['index', 'y_excess_lead']]
vbk = vbk[['index', 'y_excess_lead']]
iscf = iscf[['index', 'y_excess_lead']]
both = pd.concat([esgd, snpe, byld, avem, vbk, iscf], axis=1)
both.columns = ['index', 'esgd', 'index2', 'snpe', 'index3', 'byld', 'index4', 'avem', 'index5', 'vbk', 'index6', 'iscf']

both = both[['esgd', 'snpe', 'byld', 'avem', 'vbk', 'iscf']]
print(f'Correlation Matrix\n: {both.corr()}')
#Take away: AVEM is highly uncorrelated with the others. 

##### Need to compute the rolling correlation and volitility 
#INcrease rank for modeling more complex relationships


##### VIX market data #####
#Chatgpt: 🔮 For a 1-month ahead excess returns model, I’d recommend adding vix_ts_level and vix as your core features, and optionally vix_ts_chg_1m if you want to capture regime dynamics.
vix = fetch_vix_term_structure(start=START, end=END, freq="BM")

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
    Intrinsic Coregionalization Model (ICM) Multi-task GP with Hadamard structure.

    Overview
    --------
    This model implements a multi-task GP where the covariance between two *tasked*
    observations ((x, i), (x', j)) factorizes as a Hadamard (elementwise) product:

        k( (x,i), (x',j) ) = k_X(x, x') * B[i, j]

    • k_X(x, x') is the **data kernel** over inputs (here: Matern(ν=1/2) + Periodic,
      wrapped in ScaleKernel).  
    • B is the **task covariance matrix** learned by an IndexKernel with low rank
      (rank=1 in this example).

    This is the classic **Intrinsic Coregionalization Model (ICM)**. It assumes all
    tasks share the same input kernel shape (lengthscales etc.), while the IndexKernel
    captures how strongly tasks correlate (and their relative marginal variances).

    Why ICM / Hadamard?
    -------------------
    • Simple and data-efficient: few extra parameters, good when tasks are
      moderately to highly correlated and you expect *shared smoothness/periodicity*.
    • Stable & fast: Kronecker-free (no large Kronecker algebra), works well with
      mid-sized panels.  
    • If you need *task-specific* input kernels or multiple latent processes mixed
      per task, consider upgrading to an **LCM** (Linear Coregionalization Model).

    Inputs & Shapes
    ---------------
    We pass a tuple of inputs to the model:
      - x:  Tensor of shape [N, D]   (features / time index, etc.)
      - i:  LongTensor of shape [N, 1] with task IDs in [0, num_tasks-1]
      - y:  Tensor of shape [N]

    For two tasks with aligned inputs, you can stack like:
      full_x = concat([x_task0, x_task1])      -> [2N, D]
      full_i = concat([zeros(N,1), ones(N,1)]) -> [2N, 1]
      full_y = concat([y_task0,   y_task1])    -> [2N]

    Kernels & Priors
    ----------------
    Data kernel:  k_X = Scale(Matern(ν=1/2) + Periodic) wrapped again in ScaleKernel.
    Task kernel:  IndexKernel(num_tasks=2, rank=1) learns B ≽ 0 (task variances +
                  correlations). rank controls the capacity of inter-task structure:
                  rank=1 is ICM with one latent coregionalization component.

    Likelihood
    ----------
    GaussianLikelihood with LogNormal prior and positivity constraint on noise.
    You can:
      • Provide fixed noise (known observation noise), or
      • Infer homoskedastic noise (as done here), or
      • Extend to task-specific noise using MultitaskGaussianLikelihood.

    Training
    --------
    We optimize the Exact Marginal Log-Likelihood (mll) with Adam. Early stopping
    is optional; it’s helpful if the loss plateaus.

    Notes & Tips
    ------------
    • Standardize targets per task (zero mean, unit variance) for stable fits.
    • Normalize inputs (e.g., to [0,1]) so lengthscale priors are well-behaved.
    • If tasks are weakly related or have different smoothness, consider:
        – Increasing IndexKernel rank (>=2), or
        – LCM with multiple latent kernels, or
        – KroneckerMultiTaskGP when inputs are shared on a grid and you want
          exact Kronecker algebra speedups.
    • PeriodicKernel period prior (LogNormal(0, 0.5)) implies a median period ≈ exp(0)=1
      in input units — adjust to match your calendar (e.g., monthly seasonality).

    Returns
    -------
    MultivariateNormal over full stacked observations; predictions for each task
    can be obtained by passing the corresponding (x, i) pairs.
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
    """
    Multi-task GP (ICM, Kronecker-structured) with Matern+Periodic data kernel

    What this is
    ------------
    A 2-task Intrinsic Coregionalization Model (ICM) implemented via
    `gpytorch.kernels.MultitaskKernel`. The full covariance factorizes as:

        K( (x, t), (x', t') ) = B[t, t'] ⊗ K_X(x, x')

    where:
      - K_X is the input (data) kernel: Matern(ν=0.5) + Periodic
      - B is a learned task covariance (low-rank + diagonal; controlled by `rank`)
      - ⊗ denotes the Kronecker product

    This is the standard “ICM with Kronecker structure” used in GPyTorch’s
    multitask stack (paired with MultitaskMean and MultitaskGaussianLikelihood).

    When to use this model
    ----------------------
    - Tasks share the *same* input locations (aligned panel): same timestamps or
      feature vectors for every task. This enables Kronecker structure internally.
    - You expect tasks to be correlated (learned by the low-rank coregionalization B).
    - You want per-task noise handled natively (`MultitaskGaussianLikelihood`).

    When NOT to use this model
    --------------------------
    - Tasks are observed on *different* input grids (missing / misaligned data).
      In that case, prefer the Hadamard ICM with `IndexKernel` and stacking (x, task_id).
    - You need different data kernels per task or significantly different smoothness
      per task. Then consider an LCM (Q > 1) or separate single-task GPs.

    Data / Target shapes
    --------------------
    - train_x : Tensor[N, D]
    - train_y : Tensor[N, T]  (T = num_tasks, here 2)
    - likelihood : MultitaskGaussianLikelihood(num_tasks=T)

    Model parts
    -----------
    mean_module:
        MultitaskMean(ConstantMean(), num_tasks=T)
        One mean per task.

    covar_module:
        MultitaskKernel(
            base_kernel = MaternKernel(nu=0.5, ...) + PeriodicKernel(...),
            num_tasks = T,
            rank = 1,
        )
        - `rank` controls the rank of the task mixing (B ≈ W W^T + diag(τ)).
          rank=1 is ICM with a single latent function; increase for richer
          inter-task structure (at computational / statistical cost).

    likelihood:
        MultitaskGaussianLikelihood(num_tasks=T)
        - Learns per-task homoskedastic noise.

    Priors / constraints (recommended)
    ----------------------------------
    - Lengthscale priors for both Matern and Periodic (e.g., LogNormal)
    - Period-length prior for Periodic (e.g., LogNormal centered at a plausible cycle)
    - Outputscale prior / constraints (ScaleKernel) if you find amplitude drifting
    - Consider standardizing inputs to [0,1]^D and targets to zero-mean, unit-variance

    Training tips
    -------------
    - Normalize features and standardize each task column of train_y. Undo at inference.
    - Start with moderate LR (e.g., 0.05–0.1) and 200–500 iters; add early stopping if needed.
    - Watch the learned task covariance B: `model.covar_module.task_covar_module.covar_matrix`
      should become positive-definite with sensible off-diagonals (not all ~0).
    - If tasks are strongly correlated, rank=1 often suffices; if residual correlation remains
      unexplained, try rank=2.
    - If you see underfitting (over-smoothing), check lengthscales (too large) or allow
      a LinearKernel addend for trend; if overfitting, use stronger priors / increase noise.

    Forecasting / inference
    -----------------------
    - At prediction, pass a test_x of shape [M, D]; the model returns a
      MultitaskMultivariateNormal with mean of shape [M, T].
    - You can request only one task’s marginal by indexing the output:
        preds.mean[:, task_idx]
    - If you standardized y, re-scale predictions back to original units.

    Pros vs the Hadamard (IndexKernel) version
    ------------------------------------------
    + Efficient Kronecker algebra on aligned inputs
    + Clean per-task noise via MultitaskGaussianLikelihood
    − Requires aligned inputs across tasks
    − Less flexible for irregular, sparse multi-task panels

    Extending the model
    -------------------
    - Richer inter-task: increase `rank` (ICM→LCM-like as rank grows)
    - Different smoothness regimes: replace Matern ν, add LinearKernel, or use spectral mixtures
    - Seasonality: keep PeriodicKernel; consider multiple periodic components if needed

    References
    ----------
    - ICM / Coregionalization: Bonilla et al., 2008; Álvarez et al., 2012 (GPs for multi-output)
    - GPyTorch MultitaskKernel docs
    """
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


###### Next steps 
"""
- add small Caps
- add byld and emerging market

Review paper "Portfolio Constraints: An Empirical Analysis"
- takeaways: for long-term portfolios, min sample time window of 60 months
they use an out-of-sample window of 12 months
-THey used a rolling window
-Metrics: Sharpe ratio, variance, turnover, certainty-equivalent returns, and short interest.
-Sharpe, vol, turnover, CEQ, sometimes max drawdown, and compare to 1/N and market-cap benchmarks.
-Transaction costs are a huge limiting factor 
Investors should choose models based on objectives:

If minimizing risk → GMV with constraints.

If maximizing efficiency → MSR with TEV or equal-weight bounds.
"""
