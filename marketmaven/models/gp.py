import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, PeriodicKernel, LinearKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood, HadamardGaussianLikelihood
from gpytorch.models import ExactGP
from math import sqrt, log
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
import torch

SQRT2 = sqrt(2)
SQRT3 = sqrt(3)

ard_num_dims = 3
lengthscale_prior = LogNormalPrior(loc=SQRT2 + log(ard_num_dims) * 0.5, scale=SQRT3)
lengthscale_constraint = GreaterThan(2.5e-2, initial_value=lengthscale_prior.mode)
MIN_INFERRED_NOISE_LEVEL = 1e-5  # Minimum noise level to avoid numerical issues


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
    
def as_task_vec(i: torch.Tensor) -> torch.Tensor:
  """Make sure task indices are (N,) long (for IndexKernel)."""
  return i.view(-1).to(torch.long)

def as_task_col(i: torch.Tensor) -> torch.Tensor:
    """Make sure task indices are (N,1) long (for likelihood)."""
    if i.dim() == 1:
        i = i.unsqueeze(-1)
    return i.to(torch.long)
  
def get_hadamard_gaussian_likelihood_with_lognormal_prior(
    num_tasks: int,
    task_feature_index: int = 1,
    batch_shape: torch.Size | None = None,
) -> HadamardGaussianLikelihood:
    """Hadamard Gaussian Likelihood with independent LogNormal(-4, 1) priors per task.

    Args:
        num_tasks: Number of tasks in the multi-output GP.
        batch_shape: Optional batch shape for noise parameterization.

    Returns:
        HadamardGaussianLikelihood configured with per-task priors and constraints.
    """
    batch_shape = torch.Size() if batch_shape is None else batch_shape

    noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
    noise_constraint = GreaterThan(
        MIN_INFERRED_NOISE_LEVEL,
        transform=None,
        initial_value=noise_prior.mode,
    )

    return HadamardGaussianLikelihood(
        num_tasks=num_tasks,
        noise_prior=noise_prior,
        noise_constraint=noise_constraint,
        batch_shape=batch_shape,
        task_feature_index=task_feature_index,
    )

class HadamardMultiTaskGP(ExactGP):
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
    def __init__(self, train_x, train_y, likelihood, mean_f, kernel, num_tasks, rank):
        super(HadamardMultiTaskGP, self).__init__(train_x, train_y, likelihood)
        #d = train_x[0].shape[1]
        self.mean_module = mean_f # gpytorch.means.ZeroMean()
        # kernel = MaternKernel(
        #     nu=0.5,
        #     lengthscale_prior=lengthscale_prior,
        #     lengthscale_constraint=lengthscale_constraint,
        #     ard_num_dims=d,
        #     active_dims=list(range(d)),
        #      ) + PeriodicKernel(
        #         period_length_prior=LogNormalPrior(loc=0.0, scale=0.5),
        #         lengthscale_prior=lengthscale_prior,lengthscale_constraint=lengthscale_constraint,
        #                     ard_num_dims=d,
        #     active_dims=list(range(d)))
        self.covar_module = kernel
        self._register_lengthscale_constraints(self.covar_module)
        
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank)

    def forward(self,x,i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        i_vec   = as_task_vec(i)                   
        covar_i = self.task_covar_module(i_vec)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    
    def _register_lengthscale_constraints(self, kernel):
      """
      Recursively registers constraints for raw_lengthscale on all sub-kernels.

      Args:
          kernel: The kernel or combined kernel to register constraints on.
      """
      if hasattr(kernel, "base_kernel"):
          # If the kernel has a base kernel, register constraints on it
          self._register_lengthscale_constraints(kernel.base_kernel)

      if hasattr(kernel, "kernels"):
          # If the kernel is a combination (e.g., additive or product), recurse into sub-kernels
          for sub_kernel in kernel.kernels:
              self._register_lengthscale_constraints(sub_kernel)

      if hasattr(kernel, "raw_lengthscale"):
          # Register the constraint for raw_lengthscale
          kernel.register_constraint("raw_lengthscale", GreaterThan(2.5e-2))
    
def train_model_hadamard(train_data, rank, mean_f, kernel, training_iterations=500, patience=50, visualize: bool = False):

    (train_x, train_i), train_y = train_data
    train_i_col = as_task_col(train_i)        

    num_tasks = len(torch.unique(train_i))
    likelihood = get_hadamard_gaussian_likelihood_with_lognormal_prior(num_tasks=num_tasks, task_feature_index=1)
    model = HadamardMultiTaskGP((train_x, train_i_col), train_y, likelihood, mean_f, kernel, num_tasks, rank)
    model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(MIN_INFERRED_NOISE_LEVEL))
    
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = float("inf")
    patience_counter = 0
    for it in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x, train_i_col)         # model will squeeze internally
        loss = -mll(output, train_y, [train_i_col])  # ✅ tutorial style: list of (N,1)
        loss.backward()
        if visualize:
          if (it + 1) % 25 == 0:
              print(f'Iter {it+1}/{training_iterations} - Loss: {loss.item():.3f}')
        optimizer.step()

        if loss.item() + 1e-4 < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break

    model.eval()
    likelihood.eval()
    return model, likelihood

      
    
    
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
