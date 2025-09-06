import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, PeriodicKernel, LinearKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from math import sqrt, log
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan

SQRT2 = sqrt(2)
SQRT3 = sqrt(3)

ard_num_dims = 3
lengthscale_prior = LogNormalPrior(loc=SQRT2 + log(ard_num_dims) * 0.5, scale=SQRT3)
lengthscale_constraint = GreaterThan(2.5e-2, initial_value=lengthscale_prior.mode)
MIN_INFERRED_NOISE_LEVEL = 1e-4  # Minimum noise level to avoid numerical issues


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
    
    
class HadamardModel(ExactGP):
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
        super(HadamardModel, self).__init__(train_x, train_y, likelihood)
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
    
    
class KroneckerModel(gpytorch.models.ExactGP):
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
        super(KroneckerModel, self).__init__(train_x, train_y, likelihood)
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
