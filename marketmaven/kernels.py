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
    ScaleKernel,
    ProductKernel,
    SpectralMixtureKernel,
)
from gpytorch.priors import LogNormalPrior
from math import sqrt, log

torch.set_default_dtype(torch.float32)
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)



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

    def __init__(self, batch_shape: torch.Size, active_dims: List[int], gp_options: Dict[str, Any]):
        """
        Initializes the ContKernelFactory with common parameters.

        Args:
            batch_shape (torch.Size): The batch shape for the kernel.
            gp_options (Dict[str, Any]): Options for configuring the kernel.
        """
        self.batch_shape = batch_shape
        self.gp_options = gp_options
        self.active_dims = active_dims
        self.ard_num_dims = len(active_dims)

        self.lengthscale_prior = LogNormalPrior(loc=SQRT2 + log(self.ard_num_dims) * 0.5, scale=SQRT3)
        self.lengthscale_constraint = GreaterThan(2.5e-2, transform=None, initial_value=self.lengthscale_prior.mode)

    def create_matern(self) -> MaternKernel:
        if self.gp_options.get("smoothness", 2.5) is None:
            smoothness = 2.5
        else:
            smoothness = self.gp_options.get("smoothness", 2.5)
        return MaternKernel(
            nu=smoothness,
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
    kernel: str, input_size: int, batch_shape: torch.Size, active_dims: List[int], gp_options: Dict[str, Any]
):
    """
    Initializes the kernel for the Gaussian Process model based on the specified type.

    Args:
        kernel (str): The type of kernel to initialize.
        input_size (int): The number of input dimensions.
        batch_shape (torch.Size): The batch shape for the kernel.
        gp_options (Dict[str, Any]): Options for configuring the kernel.

    Returns:
        ScaleKernel: The initialized kernel.
    """

    cont_kernel_factory = ContKernelFactory(batch_shape=batch_shape, active_dims=active_dims, gp_options=gp_options)

    # Define a factory function for each kernel type
    def create_kernel(kernel_type: str):
        if kernel_type == KernelType.MATERN:
            return ScaleKernel(cont_kernel_factory.create_matern(), batch_shape=batch_shape)
        elif kernel_type == KernelType.MATERN_LINEAR:
            return ScaleKernel(
                cont_kernel_factory.create_matern() + cont_kernel_factory.create_linear(), batch_shape=batch_shape
            )
        elif kernel_type == KernelType.RBF:
            return ScaleKernel(cont_kernel_factory.create_rbf(), batch_shape=batch_shape)
        elif kernel_type == KernelType.RBF_LINEAR:
            return ScaleKernel(
                cont_kernel_factory.create_rbf() + cont_kernel_factory.create_linear(), batch_shape=batch_shape
            )
        elif kernel_type == KernelType.RQ:
            return ScaleKernel(cont_kernel_factory.create_rq(), batch_shape=batch_shape)
        elif kernel_type == KernelType.RQ_LINEAR:
            return ScaleKernel(
                cont_kernel_factory.create_rq() + cont_kernel_factory.create_linear(), batch_shape=batch_shape
            )
        elif kernel_type == KernelType.PERIODIC:
            return ScaleKernel(cont_kernel_factory.create_periodic(), batch_shape=batch_shape)
        elif kernel_type == KernelType.LINEAR:
            return ScaleKernel(cont_kernel_factory.create_linear(), batch_shape=batch_shape)
        else:
            raise ValueError(f"Unknown kernel function: {kernel_type}")

    # Lazily create the kernel using the factory function
    try:
        covar_module = create_kernel(kernel)
    except ValueError:
        raise ValueError(f"Unknown kernel function: {kernel}")

    return covar_module


######## IMproved kernel factory

class ContKernelFactory:
    """
    A factory for continuous kernels that supports flexible active dimensions.
    """

    def __init__(self, batch_shape: torch.Size, gp_options: Dict[str, Any], active_dims_map: Dict[str, List[int]]):
        self.batch_shape = batch_shape
        self.gp_options = gp_options
        self.active_dims_map = active_dims_map

    def _get_active_dims(self, kernel_type: str, input_size: int) -> List[int]:
        """Lookup active dims for a kernel type, default = all dims."""
        return self.active_dims_map.get(kernel_type, list(range(input_size)))

    def _lengthscale_prior(self, ard_num_dims: int):
        return LogNormalPrior(loc=SQRT2 + log(ard_num_dims) * 0.5, scale=SQRT3)

    def _lengthscale_constraint(self, prior):
        return GreaterThan(2.5e-2, transform=None, initial_value=prior.mode)

    # ---- Kernel Creators ----
    def create_matern(self, input_size: int) -> MaternKernel:
        active_dims = self._get_active_dims("matern", input_size)
        prior = self._lengthscale_prior(len(active_dims))
        return MaternKernel(
            nu=self.gp_options.get("smoothness", 1.5),
            ard_num_dims=len(active_dims),
            active_dims=active_dims,
            batch_shape=self.batch_shape,
            lengthscale_prior=prior,
            lengthscale_constraint=self._lengthscale_constraint(prior),
        )

    def create_rbf(self, input_size: int) -> RBFKernel:
        active_dims = self._get_active_dims("rbf", input_size)
        prior = self._lengthscale_prior(len(active_dims))
        return RBFKernel(
            ard_num_dims=len(active_dims),
            active_dims=active_dims,
            batch_shape=self.batch_shape,
            lengthscale_prior=prior,
            lengthscale_constraint=self._lengthscale_constraint(prior),
        )

    def create_rq(self, input_size: int) -> RQKernel:
        active_dims = self._get_active_dims("rq", input_size)
        prior = self._lengthscale_prior(len(active_dims))
        return RQKernel(
            ard_num_dims=len(active_dims),
            active_dims=active_dims,
            batch_shape=self.batch_shape,
            lengthscale_prior=prior,
            lengthscale_constraint=self._lengthscale_constraint(prior),
        )

    def create_periodic(self, input_size: int) -> PeriodicKernel:
        active_dims = self._get_active_dims("periodic", input_size)
        prior = self._lengthscale_prior(len(active_dims))
        return PeriodicKernel(
            ard_num_dims=len(active_dims),
            active_dims=active_dims,
            batch_shape=self.batch_shape,
            period_length_prior=LogNormalPrior(loc=0.0, scale=0.3),
            lengthscale_prior=prior,
            lengthscale_constraint=self._lengthscale_constraint(prior),
        )

    def create_linear(self, input_size: int) -> LinearKernel:
        active_dims = self._get_active_dims("linear", input_size)
        return LinearKernel(
            ard_num_dims=len(active_dims),
            active_dims=active_dims,
            batch_shape=self.batch_shape,
        )
    def create_spectral_mixture(self, input_size: int) -> SpectralMixtureKernel:
        active_dims = self._get_active_dims("spectralmixture", input_size)
        num_mixtures = self.gp_options.get("num_mixtures", 4)
        return SpectralMixtureKernel(
            num_mixtures=num_mixtures,
            ard_num_dims=len(active_dims),
            active_dims=active_dims,
            batch_shape=self.batch_shape,
        )

# ---- Kernel Expression Parser ----
def build_kernel(expression: str, input_size: int, batch_shape: torch.Size, gp_options: Dict[str, Any],
                 active_dims_map: Dict[str, List[int]]):
    """
    Build a kernel from an expression string like "(rq*matern)+periodic".
    - * means ProductKernel
    - + means Sum
    - parentheses respected
    """

    factory = ContKernelFactory(batch_shape=batch_shape, gp_options=gp_options, active_dims_map=active_dims_map)

    def parse_token(token: str):
        token = token.strip().lower()
        if token == "matern":
            return factory.create_matern(input_size)
        elif token == "rbf":
            return factory.create_rbf(input_size)
        elif token == "rq":
            return factory.create_rq(input_size)
        elif token == "periodic":
            return factory.create_periodic(input_size)
        elif token == "linear":
            return factory.create_linear(input_size)
        elif token == "spectralmixture":
            return factory.create_spectral_mixture(input_size)
        else:
            raise ValueError(f"Unknown kernel type: {token}")

    def parse_expr(expr: str):
        expr = expr.strip()
        # Parentheses
        if expr.startswith("(") and expr.endswith(")"):
            return parse_expr(expr[1:-1])
        # Product
        if "*" in expr:
            parts = split_expr(expr, "*")
            return ProductKernel(*[parse_expr(p) for p in parts])
        # Sum
        if "+" in expr:
            parts = split_expr(expr, "+")
            return sum(parse_expr(p) for p in parts)
        # Base case
        return parse_token(expr)

    def split_expr(expr: str, op: str):
        """Split expr by op, respecting parentheses"""
        parts, depth, last = [], 0, 0
        for i, ch in enumerate(expr):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == op and depth == 0:
                parts.append(expr[last:i])
                last = i + 1
        parts.append(expr[last:])
        return parts

    kernel = parse_expr(expression)
    return ScaleKernel(kernel, batch_shape=batch_shape)



def create_kernel_initialization(kernel: KernelConfig, n_months: int):
    prior = adaptive_lengthscale_prior(num_dims=len(kernel.active_dims))
    
    
    # For periodic only basically 
    period_length = 12.0 / (n_months - 1)  # same as your code
    kernel_initialized = initialize_kernel(
        kernel.type,
        active_dims=kernel.active_dims,
        batch_shape=torch.Size(),
        smoothness=kernel.smoothness,
        q=kernel.q,
        prior=prior,
        period_length=period_length,
        n_mixtures=kernel.n_mixtures,
    )

    return kernel_initialized