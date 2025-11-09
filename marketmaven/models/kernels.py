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
)
from gpytorch.priors import LogNormalPrior
from enum import StrEnum
from math import sqrt, log
from gpytorch.constraints import Interval, Positive
from gpytorch.kernels import IndexKernel
from gpytorch.priors import Prior

torch.set_default_dtype(torch.float64)
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)


class KernelType(StrEnum):
    """Supported Gaussian Process kernels."""

    MATERN = "matern"

    MATERN_LINEAR = "maternlinear"
    LINEAR = "linear"
    RQ = "rq"
    RBF = "rbf"

    RBF_LINEAR = "rbflinear"
    RBF_PERIODIC = "rbfperiodic"
    RQ_LINEAR = "rqlinear"

    PERIODIC = "periodic"
    PERIODIC_MATERN = "periodicmatern"
    MATERN_LINEAR_PERIODIC = "maternlinearperiodic"


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

    def __init__(self, batch_shape: torch.Size, active_dims: List[int], smoothness: float = 2.5, period_length: float = 1.0):
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
        self.lengthscale_prior = LogNormalPrior(loc=SQRT2 + log(self.ard_num_dims) * 0.5, scale=SQRT3)
        self.lengthscale_constraint = GreaterThan(2.5e-2, transform=None, initial_value=self.lengthscale_prior.mode)

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
            period_length_prior=LogNormalPrior(loc=0.0, scale=0.3),
            lengthscale_prior=self.lengthscale_prior,
            lengthscale_constraint=self.lengthscale_constraint,
        )

    def create_linear(self) -> LinearKernel:
        return LinearKernel(
            ard_num_dims=self.ard_num_dims,
            active_dims=self.active_dims,
            batch_shape=self.batch_shape,
        )


def initialize_kernel(
    kernel: str, batch_shape: torch.Size, active_dims: List[int], smoothness: float = 2.5, period_length: float =1.0
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

    cont_kernel_factory = ContKernelFactory(batch_shape=batch_shape, active_dims=active_dims, smoothness=smoothness, period_length=period_length)

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
    A kernel for discrete indices with strictly positive correlations. This is
    enforced by a positivity constraint on the decomposed covariance matrix.

    Similar to IndexKernel but ensures all off-diagonal correlations are positive
    by using a Cholesky-like parameterization with positive elements.

    .. math::
        k(i, j) = \frac{(LL^T)_{i,j}}{(LL^T)_{t,t}}

    where L is a lower triangular matrix with positive elements and t is the
    target_task_index.
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
        r"""A kernel for discrete indices with strictly positive correlations.

        Args:
            num_tasks (int): Total number of indices.
            rank (int): Rank of the covariance matrix parameterization.
            task_prior (Prior, optional): Prior for the covariance matrix.
            diag_prior (Prior, optional): Prior for the diagonal elements.
            normalize_covar_matrix (bool): Whether to normalize the covariance matrix.
            target_task_index (int): Index of the task whose diagonal element should be
                normalized to 1. Defaults to 0 (first task).
            unit_scale_for_target (bool): Whether to ensure the target task's has unit
                outputscale.
            **kwargs: Additional arguments passed to IndexKernel.
        """
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

        self.register_parameter(
            name="raw_var",
            parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, num_tasks)),
        )
        self.register_constraint("raw_var", var_constraint)
        # delete covar factor from parameters
        self.normalize_covar_matrix = normalize_covar_matrix
        self.num_tasks = num_tasks
        self.target_task_index = target_task_index
        self.register_parameter(
            name="raw_covar_factor",
            parameter=torch.nn.Parameter(
                torch.rand(*self.batch_shape, num_tasks, rank)
            ),
        )
        self.unit_scale_for_target = unit_scale_for_target
        if task_prior is not None:
            if not isinstance(task_prior, Prior):
                raise TypeError(
                    f"Expected gpytorch.priors.Prior but got "
                    f"{type(task_prior).__name__}"
                )
            self.register_prior(
                "IndexKernelPrior", task_prior, lambda m: m._lower_triangle_corr
            )
        if diag_prior is not None:
            self.register_prior("ScalePrior", diag_prior, lambda m: m._diagonal)

        self.register_constraint("raw_covar_factor", GreaterThan(0.0))

    def _covar_factor_params(self, m):
        return m.covar_factor

    def _covar_factor_closure(self, m, v):
        m._set_covar_factor(v)

    @property
    def covar_factor(self):
        return self.raw_covar_factor_constraint.transform(self.raw_covar_factor)

    @covar_factor.setter
    def covar_factor(self, value):
        self._set_covar_factor(value)

    def _set_covar_factor(self, value):
        # This must be a tensor
        self.initialize(
            raw_covar_factor=self.raw_covar_factor_constraint.inverse_transform(value)
        )

    @property
    def _lower_triangle_corr(self):
        lower_row, lower_col = torch.tril_indices(
            self.num_tasks, self.num_tasks, offset=-1
        )
        covar = self.covar_matrix
        norm_factor = covar.diagonal(dim1=-1, dim2=-2).sqrt()
        corr = covar / (norm_factor.unsqueeze(-1) * norm_factor.unsqueeze(-2))
        low_tri = corr[..., lower_row, lower_col]

        return low_tri

    @property
    def _diagonal(self):
        return torch.diagonal(self.covar_matrix, dim1=-2, dim2=-1)

    def _eval_covar_matrix(self):
        cf = self.covar_factor
        covar = cf @ cf.transpose(-1, -2) + self.var * torch.eye(
            self.num_tasks, dtype=cf.dtype, device=cf.device
        )
        # Normalize by the target task's diagonal element
        if self.unit_scale_for_target:
            norm_factor = covar[..., self.target_task_index, self.target_task_index]
            covar = covar / norm_factor.unsqueeze(-1).unsqueeze(-1)
        return covar

    @property
    def covar_matrix(self):
        return self._eval_covar_matrix()