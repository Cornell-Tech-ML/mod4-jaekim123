from typing import Tuple
from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand
import numpy as np  # Ensure numpy is imported for numerical operations


def max_reduce(input: Tensor, dim: int) -> Tensor:
    """Use FastOps to compute the max reduction."""
    return FastOps.reduce(operators.max, -float("inf"))(input, dim).view()


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """
        Forward pass for the Max operation.

        Args:
            ctx (Context): Context to save information for backward pass.
            t1 (Tensor): Input tensor.
            dim (Tensor): Dimension along which to compute the max.

        Returns:
            Tensor: Result of the max operation.
        """
        ctx.save_for_backward(t1, dim)
        return max_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """
        Backward pass for the Max operation.

        Args:
            ctx (Context): Context containing saved tensors.
            grad_output (Tensor): Gradient of the loss w.r.t. the output.

        Returns:
            Tuple[Tensor, float]: Gradients w.r.t. the input tensor and the dimension (no gradient).
        """
        t1, dim = ctx.saved_values
        dim = int(dim.item())

        # Compute the max values along the specified dimension
        max_vals = max_reduce(t1, dim)

        # Create a boolean mask where the max values are located
        is_max = t1 == max_vals

        # Count the number of maxima along the specified dimension
        count_max = is_max.sum(dim=dim)
        count_max = count_max.view(
            *[1 if i == dim else s for i, s in enumerate(t1.shape)]
        )

        # Prevent division by zero in case there are no maxima (which shouldn't happen)
        count_max = count_max + 1e-12

        # Distribute the gradient equally among all maxima
        grad_input = (is_max * grad_output) / count_max

        return grad_input, 0.0  # No gradient for 'dim'


def max(input: Tensor, dim: int) -> Tensor:
    """
    Compute the maximum values of the input tensor along a specified dimension.

    Args:
        input (Tensor): The input tensor.
        dim (int): The dimension along which to compute the maximum.

    Returns:
        Tensor: The tensor containing the maximum values along the specified dimension.
    """
    return Max.apply(input, input._ensure_tensor(dim)) 