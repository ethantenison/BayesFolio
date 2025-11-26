"""
Gaussian Process Kernels
"""

import torch
from gpytorch.kernels.kernel import Kernel
from torch import Tensor
from typing import Any, Dict, List
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import (
    MaternKernel,
    LinearKernel,
    RBFKernel,
    RQKernel,
    PeriodicKernel,
    SpectralMixtureKernel,
)
from gpytorch.priors import LogNormalPrior
from enum import StrEnum
from math import sqrt, log
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import IndexKernel
from gpytorch.priors import Prior
from gpytorch.means import ConstantMean, ZeroMean, LinearMean, MultitaskMean

torch.set_default_dtype(torch.float64)
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)


###### Length Scales ######
def adaptive_lengthscale_prior(num_dims:int):
    return LogNormalPrior(loc=SQRT2 + log(num_dims) * 0.5, scale=SQRT3)
def adaptive_lengthscale_constraint(num_dims:int):
    return GreaterThan(2.5e-2, initial_value=adaptive_lengthscale_prior(num_dims).mode)

def adaptive_lengthscale_prior_time(num_dims:int):
    return LogNormalPrior(loc=SQRT2 + log(num_dims) * 0.3, scale=SQRT3)
def adaptive_lengthscale_constraint_time(num_dims:int):
    return GreaterThan(2.5e-2, initial_value=adaptive_lengthscale_prior_time(num_dims).mode)


###### Means ######
class MeanF(StrEnum):
    """Supported Gaussian Process kernels."""

    LINEAR = "linear"
    CONSTANT = "constant"
    ZERO = "zero"
    MULTITASK_CONSTANT = "multitask_constant"
    
def initialize_mean(mean: MeanF, input_size: int| None = None, num_tasks: int | None = None):
    if mean == MeanF.CONSTANT:
        return ConstantMean()
    elif mean ==  MeanF.LINEAR:
        return LinearMean(input_size)
    elif mean ==  MeanF.ZERO:
        return ZeroMean()
    elif mean ==  MeanF.MULTITASK_CONSTANT:
        return MultitaskMean(
                ConstantMean(),
                num_tasks=num_tasks,
            )
    else:
        raise ValueError(f"Unknown mean function: {mean}")



class KernelType(StrEnum):
    """Supported Gaussian Process kernels."""

    MATERN = "matern"

    MATERN_LINEAR = "maternlinear"
    MATERN_RQ = "maternrq" 
    LINEAR = "linear"
    RQ = "rq"
    RBF = "rbf"

    RBF_LINEAR = "rbflinear"
    RBF_PERIODIC = "rbfperiodic"
    RQ_LINEAR = "rqlinear"

    PERIODIC = "periodic"
    PERIODIC_MATERN = "periodicmatern"
    MATERN_LINEAR_PERIODIC = "maternlinearperiodic"
    
    SPECTRAL_MIXTURE = "spectralmixture"


class CategoricalKernel(Kernel):
    r"""A Kernel for categorical features.

    Computes `exp(-dist(x1, x2) / lengthscale)`, where
    `dist(x1, x2)` is zero if `x1 == x2` and one if `x1 != x2`.
    If the last dimension is not a batch dimension, then the
    mean is considered.

    Note: This kernel is NOT differentiable w.r.t. the inputs.
    """

    has_lengthscale = True

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        delta = x1.unsqueeze(-2) != x2.unsqueeze(-3)
        dists = delta / self.lengthscale.unsqueeze(-2)
        if last_dim_is_batch:
            dists = dists.transpose(-3, -1)
        else:
            dists = dists.mean(-1)
        res = torch.exp(-dists)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res


class ContKernelFactory:
    """
    A base class for creating Gaussian Process kernels.

    This class encapsulates the shared logic for kernel creation and provides
    methods to initialize different kernel types.
    """

    def __init__(self, batch_shape: torch.Size, active_dims: List[int], smoothness: float = 2.5, period_length: float = 1.0, n_mixtures=2):
        """
        Initializes the ContKernelFactory with common parameters.

        Args:
            batch_shape (torch.Size): The batch shape for the kernel.
            gp_options (Dict[str, Any]): Options for configuring the kernel.
        """
        self.batch_shape = batch_shape
        self.active_dims = active_dims
        self.ard_num_dims = len(active_dims)
        self.smoothness = smoothness
        self.period_length = period_length
        self.n_mixtures = n_mixtures
        self.lengthscale_prior = adaptive_lengthscale_prior(self.ard_num_dims)
        self.lengthscale_constraint = adaptive_lengthscale_constraint(self.ard_num_dims)
        self.lengthscale_prior_time = adaptive_lengthscale_prior_time(self.ard_num_dims)
        self.lengthscale_constraint_time = adaptive_lengthscale_constraint_time(self.ard_num_dims)

    def create_matern(self) -> MaternKernel:
        return MaternKernel(
            nu=self.smoothness,
            ard_num_dims=self.ard_num_dims,
            active_dims=self.active_dims,
            batch_shape=self.batch_shape,
            lengthscale_prior=self.lengthscale_prior,
            lengthscale_constraint=self.lengthscale_constraint,
        )

    def create_rbf(self) -> RBFKernel:
        return RBFKernel(
            ard_num_dims=self.ard_num_dims,
            active_dims=self.active_dims,
            batch_shape=self.batch_shape,
            lengthscale_prior=self.lengthscale_prior,
            lengthscale_constraint=self.lengthscale_constraint,
        )

    def create_rq(self) -> RQKernel:
        return RQKernel(
            ard_num_dims=self.ard_num_dims,
            active_dims=self.active_dims,
            batch_shape=self.batch_shape,
            lengthscale_prior=self.lengthscale_prior,
            lengthscale_constraint=self.lengthscale_constraint,
        )

    def create_periodic(self) -> PeriodicKernel:
        return PeriodicKernel(
            ard_num_dims=self.ard_num_dims,
            active_dims=self.active_dims,
            batch_shape=self.batch_shape,
            period_length=self.period_length,
            period_length_prior=LogNormalPrior(
                loc=-2.54,    # ≈ math.log(period_length) - 0.5*sigma**2
                scale=0.2),
            lengthscale_prior=LogNormalPrior(
                loc=-4.02,
                scale=0.47),
            lengthscale_constraint=GreaterThan(
                2.5e-3,  # small but nonzero
                initial_value=0.02
            ),
        )
#         \sigma = \sqrt{\ln(1 + cv^2)},\quad
# \mu = \ln(m) - \frac{\sigma^2}{2} For lognormal setting

    def create_linear(self) -> LinearKernel:
        return LinearKernel(
            ard_num_dims=self.ard_num_dims,
            active_dims=self.active_dims,
            batch_shape=self.batch_shape,
        )
        
    def create_spectral_mixture(self):

        return SpectralMixtureKernel(
            num_mixtures=self.n_mixtures,
            ard_num_dims=self.ard_num_dims,
            active_dims=self.active_dims,
            batch_shape=self.batch_shape,
        )


def initialize_kernel(
    kernel: str, batch_shape: torch.Size, active_dims: List[int], smoothness: float = 2.5, period_length: float = 1.0, n_mixtures=2
):
    """
    Initializes the kernel for the Gaussian Process model based on the specified type.

    Args:
        kernel (str): The type of kernel to initialize.
        batch_shape (torch.Size): The batch shape for the kernel.
        gp_options (Dict[str, Any]): Options for configuring the kernel.

    Returns:
        ScaleKernel: The initialized kernel.
    """

    cont_kernel_factory = ContKernelFactory(batch_shape=batch_shape, active_dims=active_dims, smoothness=smoothness, period_length=period_length, n_mixtures=n_mixtures)

    # Define a factory function for each kernel type
    def create_kernel(kernel_type: str):
        if kernel_type == KernelType.MATERN:
            return cont_kernel_factory.create_matern()
        elif kernel_type == KernelType.MATERN_LINEAR:
            return cont_kernel_factory.create_matern() + cont_kernel_factory.create_linear()
        elif kernel_type == KernelType.RBF:
            return cont_kernel_factory.create_rbf()
        elif kernel_type == KernelType.RBF_LINEAR:
            return cont_kernel_factory.create_rbf() + cont_kernel_factory.create_linear()
        elif kernel_type == KernelType.RBF_PERIODIC:
            return cont_kernel_factory.create_rbf() + cont_kernel_factory.create_periodic()
        elif kernel_type == KernelType.RQ:
            return cont_kernel_factory.create_rq()
        elif kernel_type == KernelType.RQ_LINEAR:
            return cont_kernel_factory.create_rq() + cont_kernel_factory.create_linear()
        elif kernel_type == KernelType.PERIODIC:
            return cont_kernel_factory.create_periodic()
        elif kernel_type == KernelType.LINEAR:
            return cont_kernel_factory.create_linear()
        elif kernel_type == KernelType.MATERN_LINEAR_PERIODIC:
            return cont_kernel_factory.create_matern() + cont_kernel_factory.create_linear() + cont_kernel_factory.create_periodic()
        elif kernel_type == KernelType.PERIODIC_MATERN:
            return cont_kernel_factory.create_periodic() + cont_kernel_factory.create_matern()
        elif kernel_type == KernelType.SPECTRAL_MIXTURE:
            return cont_kernel_factory.create_spectral_mixture()
        elif kernel_type == KernelType.MATERN_RQ:
            return cont_kernel_factory.create_matern() + cont_kernel_factory.create_rq()
        else:
            raise ValueError(f"Unknown kernel function: {kernel_type}")

    # Lazily create the kernel using the factory function
    try:
        covar_module = create_kernel(kernel)
    except ValueError:
        raise ValueError(f"Unknown kernel function: {kernel}")

    return covar_module



class PositiveIndexKernel(IndexKernel):
    r"""
    A kernel for discrete indices with strictly positive correlations.
    This variant parameterizes correlations via a positive covar_factor and
    supports an LKJ prior on the full covariance matrix.

    k(i, j) = ((L L^T) / (L L^T)_{t,t})[i, j]
    """

    def __init__(
        self,
        num_tasks: int,
        rank: int = 1,
        task_prior: Prior | None = None,
        diag_prior: Prior | None = None,
        normalize_covar_matrix: bool = False,
        var_constraint: Interval | None = None,
        target_task_index: int = 0,
        unit_scale_for_target: bool = True,
        **kwargs,
    ):
        if rank > num_tasks:
            raise RuntimeError(
                "Cannot create a task covariance matrix larger than the number of tasks"
            )
        if not (0 <= target_task_index < num_tasks):
            raise ValueError(
                f"target_task_index must be between 0 and {num_tasks - 1}, "
                f"got {target_task_index}"
            )
        Kernel.__init__(self, **kwargs)

        if var_constraint is None:
            var_constraint = Positive()

        # Variance and covar factor parameters
        self.register_parameter(
            name="raw_var",
            parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, num_tasks)),
        )
        self.register_constraint("raw_var", var_constraint)

        self.register_parameter(
            name="raw_covar_factor",
            parameter=torch.nn.Parameter(
                torch.rand(*self.batch_shape, num_tasks, rank)
            ),
        )
        self.register_constraint("raw_covar_factor", GreaterThan(0.0))

        self.normalize_covar_matrix = normalize_covar_matrix
        self.num_tasks = num_tasks
        self.target_task_index = target_task_index
        self.unit_scale_for_target = unit_scale_for_target

        # ---- Priors -------------------------------------------------------
        if task_prior is not None:
            if not isinstance(task_prior, Prior):
                raise TypeError(
                    f"Expected gpytorch.priors.Prior but got {type(task_prior).__name__}"
                )
            # ✅ Register LKJ prior on the full covariance matrix
            self.register_prior(
                "IndexKernelPrior", task_prior, lambda m: m.covar_matrix
            )

        if diag_prior is not None:
            self.register_prior("ScalePrior", diag_prior, lambda m: m._diagonal)

    # ----------------------------------------------------------------------
    # Accessors
    # ----------------------------------------------------------------------
    @property
    def covar_factor(self):
        return self.raw_covar_factor_constraint.transform(self.raw_covar_factor)

    @covar_factor.setter
    def covar_factor(self, value):
        self._set_covar_factor(value)

    def _set_covar_factor(self, value):
        self.initialize(
            raw_covar_factor=self.raw_covar_factor_constraint.inverse_transform(value)
        )

    @property
    def _diagonal(self):
        return torch.diagonal(self.covar_matrix, dim1=-2, dim2=-1)

    @property
    def _corr_matrix(self):
        """Optional helper to access the correlation matrix."""
        covar = self.covar_matrix
        d = covar.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
        return covar / (d.unsqueeze(-1) * d.unsqueeze(-2))

    # ----------------------------------------------------------------------
    # Core covariance evaluation
    # ----------------------------------------------------------------------
    def _eval_covar_matrix(self):
        cf = self.covar_factor
        covar = cf @ cf.transpose(-1, -2) + self.var * torch.eye(
            self.num_tasks, dtype=cf.dtype, device=cf.device
        )
        if self.unit_scale_for_target:
            norm = covar[..., self.target_task_index, self.target_task_index]
            covar = covar / norm.unsqueeze(-1).unsqueeze(-1)
        return covar

    @property
    def covar_matrix(self):
        return self._eval_covar_matrix()