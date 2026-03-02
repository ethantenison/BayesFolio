import gpytorch
from gpytorch.likelihoods import HadamardGaussianLikelihood
from gpytorch.models import ExactGP
from math import sqrt
from gpytorch.priors import LogNormalPrior, GammaPrior, LKJCovariancePrior
from gpytorch.constraints import GreaterThan
import torch
from bayesfolio.ml.legacy.old_kernels import PositiveIndexKernel
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from gpytorch.means import MultitaskMean
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)

# ard_num_dims = 3
# lengthscale_prior = LogNormalPrior(loc=SQRT2 + log(ard_num_dims) * 0.5, scale=SQRT3)
# lengthscale_constraint = GreaterThan(2.5e-2, initial_value=lengthscale_prior.mode)
MIN_INFERRED_NOISE_LEVEL = 5e-3 #1e-5  # Minimum noise level to avoid numerical issues

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
    min_noise: float = MIN_INFERRED_NOISE_LEVEL,
) -> HadamardGaussianLikelihood:
    """Hadamard Gaussian Likelihood with independent LogNormal(-4, 1) priors per task.

    Args:
        num_tasks: Number of tasks in the multi-output GP.
        batch_shape: Optional batch shape for noise parameterization.

    Returns:
        HadamardGaussianLikelihood configured with per-task priors and constraints.
    """
    batch_shape = torch.Size() if batch_shape is None else batch_shape

    noise_prior = LogNormalPrior(loc=-4.0, scale=1.0) # loc=-3.2, scale=0.45
    noise_constraint = GreaterThan(
        min_noise,
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

        self.task_covar_module = PositiveIndexKernel(
            num_tasks=num_tasks,
            rank=rank,
            task_prior=task_covar_prior,
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
            kernel.register_constraint("raw_lengthscale", GreaterThan(2.5e-3))


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
    min_noise: float = MIN_INFERRED_NOISE_LEVEL,
):
    # ----------------------------------------------------------------------
    # Device / dtype
    # ----------------------------------------------------------------------
    if device is None:
        device = torch.device("cpu") # "mps" if torch.backends.mps.is_available() else 

    train_X = train_X.to(device=device, dtype=dtype)
    train_Y = train_Y.to(device=device, dtype=dtype)

    # ----------------------------------------------------------------------
    # (Recommended) sort by task index for better cache locality
    # ----------------------------------------------------------------------
    task_col = train_X[:, task_feature].long()
    perm = torch.argsort(task_col)
    train_X = train_X[perm]
    train_Y = train_Y[perm]
    task_col = task_col[perm]

    num_tasks = len(torch.unique(task_col))

    # ----------------------------------------------------------------------
    # Likelihood
    # ----------------------------------------------------------------------
    likelihood = get_hadamard_gaussian_likelihood_with_lognormal_prior(
        num_tasks=num_tasks,
        task_feature_index=task_feature,
        min_noise=min_noise,
    ).to(device=device, dtype=dtype)

    # Mean & kernel to device/dtype BEFORE model construction
    mean_f = mean_f.to(device=device, dtype=dtype)
    kernel = kernel.to(device=device, dtype=dtype)

    # ----------------------------------------------------------------------
    # Model
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
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    model.likelihood.noise_covar.register_constraint(
        "raw_noise", GreaterThan(min_noise)
    )

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # ----------------------------------------------------------------------
    # Training loop (FAST COMPUTATIONS ENABLED)
    # ----------------------------------------------------------------------
    best_loss = float("inf")
    patience_counter = 0

    for it in range(training_iterations):
        optimizer.zero_grad()

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # FAST GP COMPUTATION CONTEXT (this is the key addition)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        with gpytorch.settings.fast_computations(
            log_prob=True,
            covar_root_decomposition=True,
            solves=True,
        ), gpytorch.settings.cholesky_jitter(1e-5):

            output = model(train_X)
            loss = -mll(output, train_Y, train_X)

        loss.backward()

        if visualize and (it + 1) % 25 == 0:
            print(f"Iter {it+1}/{training_iterations} - Loss: {loss.item():.3f}")

        optimizer.step()

        # Early stopping
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
    # Finalize
    # ----------------------------------------------------------------------
    model.eval()
    likelihood.eval()

    return model, likelihood
