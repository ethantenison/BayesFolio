"""
Means for GPs

This module provides functions to calculate various types of means used in Gaussian Processes.
"""

from gpytorch.priors import Prior
from gpytorch.constraints import Positive
from gpytorch.means import ConstantMean, ZeroMean, LinearMean, Mean
from enum import StrEnum
import torch


class GPMean(StrEnum):
    """Supported Gaussian Process kernels."""

    LINEAR = "linear"
    CONSTANT = "constant"
    EXPONENTIAL_SATURATION = "exponential_saturation"
    ZERO = "zero"

def compute_ewma2_means_torch(returns: torch.Tensor, task_ids: torch.Tensor, num_tasks: int, d: float = 0.94) -> torch.Tensor:
    """
    Compute per-task EWMA2 means (adjust=False) directly from tensors.

    Parameters
    ----------
    returns : torch.Tensor
        Tensor of returns, shape [N] or [N, 1].
    task_ids : torch.Tensor
        Tensor of integer task indices, shape [N] or [N, 1].
    num_tasks : int
        Number of unique tasks.
    d : float
        EWMA decay factor (Riskfolio default 0.94).

    Returns
    -------
    torch.Tensor
        Tensor of shape [num_tasks] containing the EWMA2 mean for each task.
    """
    # ensure correct shapes
    returns = returns.view(-1)
    task_ids = task_ids.view(-1).long()
    alpha = 1.0 - d

    means = torch.zeros(num_tasks, dtype=returns.dtype, device=returns.device)

    for t in range(num_tasks):
        mask = task_ids == t
        r_t = returns[mask]
        if len(r_t) == 0:
            means[t] = torch.tensor(0.0, device=returns.device)
            continue

        # compute EWMA (adjust=False)
        ewma = torch.zeros_like(r_t)
        ewma[0] = r_t[0]
        for i in range(1, len(r_t)):
            ewma[i] = alpha * r_t[i] + (1 - alpha) * ewma[i - 1]
        means[t] = ewma[-1]

    return means

class EWMATaskConstantMean(Mean):
    """
    Per-task constant mean initialized from EWMA2 means.
    Each task has its own constant expected return.
    Expects the task index as a column in X.
    """

    def __init__(
        self,
        task_means: torch.Tensor,
        task_feature: int = -1,
        learnable: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        super().__init__()
        device = device or torch.device("cpu")
        task_means = task_means.to(device=device, dtype=dtype).view(-1)

        if learnable:
            self.register_parameter("raw_task_means", torch.nn.Parameter(task_means.clone()))
            self._frozen = False
        else:
            self.register_buffer("task_means", task_means.clone())
            self._frozen = True

        self.task_feature = task_feature
        self.dtype = dtype

    @property
    def _means(self) -> torch.Tensor:
        return self.raw_task_means if not self._frozen else self.task_means

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.task_feature < 0:
            tcol = X.shape[-1] + self.task_feature
        else:
            tcol = self.task_feature

        task_ids = X[..., tcol].long().view(-1)
        vals = self._means[task_ids]
        return vals.view(*X.shape[:-1])




class ExponentialSaturationMean(Mean):
    """
    Exponential saturation mean function:

    .. math::
        \mu(t) = A \cdot (1 - \exp(-B t))

    where `A` and `B` are learnable parameters.

    :param A_prior: Prior for A.
    :type A_prior: ~gpytorch.priors.Prior, optional
    :param B_prior: Prior for B.
    :type B_prior: ~gpytorch.priors.Prior, optional
    :param A_constraint: Constraint for A (e.g. positive).
    :type A_constraint: ~gpytorch.constraints.Interval, optional
    :param B_constraint: Constraint for B (e.g. positive).
    :type B_constraint: ~gpytorch.constraints.Interval, optional
    :param batch_shape: The batch shape of the parameters.
    :type batch_shape: torch.Size, optional

    # TODO: Add support to select the 't' dimension in the input tensor.
    """

    def __init__(
        self,
        A_prior: Prior = None,
        B_prior: Prior = None,
        A_constraint=Positive(),
        B_constraint=Positive(),
        batch_shape: torch.Size = torch.Size(),
    ):
        super().__init__()

        self.batch_shape = batch_shape

        # Register raw parameters
        self.register_parameter(name="raw_A", parameter=torch.nn.Parameter(torch.zeros(batch_shape)))
        self.register_parameter(name="raw_B", parameter=torch.nn.Parameter(torch.zeros(batch_shape)))

        # Register constraints
        if A_constraint is not None:
            self.register_constraint("raw_A", A_constraint)
        if B_constraint is not None:
            self.register_constraint("raw_B", B_constraint)

        # Register priors
        if A_prior is not None:
            self.register_prior("A_prior", A_prior, self._A_param, self._A_closure)
        if B_prior is not None:
            self.register_prior("B_prior", B_prior, self._B_param, self._B_closure)

    ## A property + closure
    @property
    def A(self):
        return self.raw_A_constraint.transform(self.raw_A)

    @A.setter
    def A(self, value):
        self._A_closure(self, value)

    def _A_param(self, m):
        return m.A

    def _A_closure(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_A)
        m.initialize(raw_A=m.raw_A_constraint.inverse_transform(value))

    ## B property + closure
    @property
    def B(self):
        return self.raw_B_constraint.transform(self.raw_B)

    @B.setter
    def B(self, value):
        self._B_closure(self, value)

    def _B_param(self, m):
        return m.B

    def _B_closure(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_B)
        m.initialize(raw_B=m.raw_B_constraint.inverse_transform(value))

    def forward(self, x):
        t = x[..., 0]  # assume first dimension is time
        A = self.A.unsqueeze(-1)
        B = self.B.unsqueeze(-1)
        return A * (1 - torch.exp(-B * t))


def initialize_mean(mean: GPMean, input_size: int, batch_shape: torch.Size):
    if mean == GPMean.CONSTANT:
        return ConstantMean(batch_shape=batch_shape)
    elif mean == GPMean.LINEAR:
        return LinearMean(input_size, batch_shape)
    elif mean == GPMean.ZERO:
        return ZeroMean(batch_shape=batch_shape)
    elif mean == GPMean.EXPONENTIAL_SATURATION:
        return ExponentialSaturationMean(batch_shape=batch_shape)
    else:
        raise ValueError(f"Unknown mean function: {mean}")
