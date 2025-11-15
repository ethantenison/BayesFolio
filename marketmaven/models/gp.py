import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, PeriodicKernel, LinearKernel, MaternKernel, IndexKernel
from gpytorch.likelihoods import GaussianLikelihood, HadamardGaussianLikelihood
from gpytorch.models import ExactGP
from math import sqrt, log
from gpytorch.priors import LogNormalPrior, GammaPrior, LKJCovariancePrior
from gpytorch.constraints import GreaterThan
import torch
from marketmaven.models.kernels import PositiveIndexKernel
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from gpytorch.means import MultitaskMean
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)

ard_num_dims = 3
lengthscale_prior = LogNormalPrior(loc=SQRT2 + log(ard_num_dims) * 0.5, scale=SQRT3)
lengthscale_constraint = GreaterThan(2.5e-2, initial_value=lengthscale_prior.mode)
MIN_INFERRED_NOISE_LEVEL = 1e-3  # Minimum noise level to avoid numerical issues


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

    #noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
    noise_prior = LogNormalPrior(loc=-0.69, scale=0.5)
    noise_constraint = GreaterThan(
        #MIN_INFERRED_NOISE_LEVEL,
        0.01,
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
    Intrinsic Coregionalization (ICM) multitask GP with Hadamard structure.

    Supports either a single mean module (e.g. ConstantMean, ZeroMean)
    or a gpytorch.means.MultitaskMean with distinct per-task means.

    The task index must be included in X as the last column, similar
    to BoTorch's MultiTaskGP.

    Covariance:
        k((x,t), (x',t')) = k_X(x,x') * B[t,t']

    Mean:
        m(x,t) = mean_module(x, task=t)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: HadamardGaussianLikelihood,
        mean_module: gpytorch.means.Mean,
        covar_module: gpytorch.kernels.Kernel,
        num_tasks: int,
        rank: int = 1,
        task_feature: int = -1,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        # Move inputs and likelihood to the same device/dtype before calling super()
        train_X = train_X.to(device=device, dtype=dtype)
        train_Y = train_Y.to(device=device, dtype=dtype)
        likelihood = likelihood.to(device=device, dtype=dtype)
        
        super().__init__(train_X, train_Y, likelihood)

        self.mean_module = mean_module.to(device=device, dtype=dtype)
        self.covar_module = covar_module.to(device=device, dtype=dtype)
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")
        self._register_lengthscale_constraints(self.covar_module)

        # --- Task kernel (IndexKernel) with LKJ prior for task covariance
        sd_prior = GammaPrior(1.0, 0.15)
        sd_prior._event_shape = torch.Size([num_tasks])
        eta = 0.5
        task_covar_prior = LKJCovariancePrior(num_tasks, eta, sd_prior)

        # self.task_covar_module = IndexKernel(
        #     num_tasks=num_tasks, rank=rank, prior=task_covar_prior
        # )
        self.task_covar_module = PositiveIndexKernel(
            num_tasks=num_tasks,
            rank=rank,
            task_prior=task_covar_prior,
            #active_dims=[task_feature],
        ).to(device=device, dtype=dtype)

        # Identify which column is the task feature
        self.task_feature = task_feature if task_feature >= 0 else train_X.shape[-1] + task_feature

        # Store task count
        self.num_tasks = num_tasks

    # ----------------------------------------------------------------------
    # Internal utilities
    # ----------------------------------------------------------------------
    def _split_inputs(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (x_before, task_idcs, x_after)."""
        x_before = X[..., : self.task_feature]
        task_idcs = X[..., self.task_feature].view(-1, 1).to(torch.long)
        x_after = X[..., (self.task_feature + 1):]
        return x_before, task_idcs, x_after

    def _register_lengthscale_constraints(self, kernel):
        """Recursively registers constraints for raw_lengthscale on all sub-kernels."""
        if hasattr(kernel, "base_kernel"):
            self._register_lengthscale_constraints(kernel.base_kernel)
        if hasattr(kernel, "kernels"):
            for sub_kernel in kernel.kernels:
                self._register_lengthscale_constraints(sub_kernel)
        if hasattr(kernel, "raw_lengthscale"):
            kernel.register_constraint("raw_lengthscale", GreaterThan(2.5e-2))

    # ----------------------------------------------------------------------
    # Core forward logic
    # ----------------------------------------------------------------------
    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Forward pass — computes mean and covariance for input X
        where task index is embedded as a column.
        """
        x_before, task_idcs, x_after = self._split_inputs(X)
        t_vec = task_idcs.view(-1)

        # ---- Mean (BoTorch-style)
        if isinstance(self.mean_module, MultitaskMean):
            # Mean sees only non-task features, returns [..., n, num_tasks]
            x_mean = torch.cat([x_before, x_after], dim=-1)
            mean_all = self.mean_module(x_mean)
            # Gather the appropriate task mean for each row -> [..., n]
            mean_x = mean_all.gather(-1, task_idcs.long()).squeeze(-1)
        else:
            # Single mean gets task as a feature so it can depend on it
            # (cast to float so concat works with feature dtype)
            x_mean = torch.cat([x_before, task_idcs.to(X.dtype), x_after], dim=-1)
            mean_x = self.mean_module(x_mean)

        # ---- Covariance (Hadamard factorization)
        # Your factorization is fine: k_X(x,x') ⊙ B[t,t']
        x_features = torch.cat([x_before, x_after], dim=-1)
        covar_x = self.covar_module(x_features)
        covar_i = self.task_covar_module(t_vec)
        covar = covar_x.mul(covar_i)

        return MultivariateNormal(mean_x, covar)
      
def train_model_hadamard(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    rank: int,
    mean_f,
    kernel,
    training_iterations: int = 500,
    patience: int = 50,
    visualize: bool = False,
    task_feature: int = -1,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
):
    """
    Train a HadamardMultiTaskGP where the task index is embedded
    as one of the columns in train_X.

    Args
    ----
    train_X : torch.Tensor
        Shape [N, D+1], where one column (specified by task_feature)
        contains the integer task indices in [0, num_tasks-1].
    train_Y : torch.Tensor
        Shape [N], containing the target values.
    rank : int
        Rank of the task covariance (IndexKernel).
    mean_f : gpytorch.means.Mean
        Mean module (e.g., ZeroMean, ConstantMean, MultitaskMean).
    kernel : gpytorch.kernels.Kernel
        Covariance kernel for input features (excluding task index).
    training_iterations : int, optional
        Maximum number of training iterations (default: 500).
    patience : int, optional
        Early stopping patience (default: 50).
    visualize : bool, optional
        Print training progress every 25 iterations if True.
    task_feature : int, optional
        Column index of the task feature in train_X. Defaults to -1
        (i.e., last column).

    Returns
    -------
    model : HadamardMultiTaskGP
        Trained model.
    likelihood : HadamardGaussianLikelihood
        Associated likelihood.
    """
    # ----------------------------------------------------------------------
    # Identify number of tasks and initialize likelihood
    # ----------------------------------------------------------------------
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
    # Ensure tensors are on the same device/dtype
    train_X = train_X.to(device=device, dtype=dtype)
    train_Y = train_Y.to(device=device, dtype=dtype)
    
    task_col = train_X[:, task_feature].long()
    num_tasks = len(torch.unique(task_col))

    likelihood = get_hadamard_gaussian_likelihood_with_lognormal_prior(
        num_tasks=num_tasks, task_feature_index=task_feature
    ).to(device=device, dtype=dtype)
    
    # Mean & kernel to device/dtype BEFORE passing into the model
    mean_f = mean_f.to(device=device, dtype=dtype)
    kernel = kernel.to(device=device, dtype=dtype)

    # ----------------------------------------------------------------------
    # Initialize model
    # ----------------------------------------------------------------------
    model = HadamardMultiTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        likelihood=likelihood,
        mean_module=mean_f,
        covar_module=kernel,
        num_tasks=num_tasks,
        rank=rank,
        task_feature=task_feature,
    ).to(device=device, dtype=dtype)

    # Ensure positive noise constraint
    model.likelihood.noise_covar.register_constraint(
        "raw_noise", GreaterThan(MIN_INFERRED_NOISE_LEVEL)
    )

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # ----------------------------------------------------------------------
    # Training loop with early stopping
    # ----------------------------------------------------------------------
    best_loss = float("inf")
    patience_counter = 0

    for it in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_X)
        loss = -mll(output, train_Y, train_X)  # pass X so likelihood can read task indices
        loss.backward()

        if visualize and (it + 1) % 25 == 0:
            print(f"Iter {it+1}/{training_iterations} - Loss: {loss.item():.3f}")

        optimizer.step()

        # Early stopping check
        if loss.item() + 1e-4 < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            if visualize:
                print(f"Early stopping at iteration {it+1}")
            break

    # ----------------------------------------------------------------------
    # Finalize model
    # ----------------------------------------------------------------------
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
