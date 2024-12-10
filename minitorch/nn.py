# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off

from typing import Tuple
from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    reshaped = input.view(batch, channel, new_height, kh, new_width, kw)

    tiled = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D average pooling on the input tensor.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width).
        kernel: Tuple (kernel_height, kernel_width).

    Returns:
    -------
        A tensor of shape (batch, channel, new_height, new_width) after pooling.

    """
    tiled, new_height, new_width = tile(input, kernel)

    pooled = tiled.mean(dim=-1)

    return pooled.view(input.shape[0], input.shape[1], new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -float("inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax of the input tensor along a specified dimension as a 1-hot tensor.

    Args:
    ----
        input: The input tensor.
        dim: The dimension along which to compute the argmax.

    Returns:
    -------
        A one-hot tensor of the same shape as `input`, with 1 at the argmax positions along the specified dimension, and 0 elsewhere.

    """
    max_vals = max_reduce(input, dim)
    return max_vals == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Computes the forward pass of the max operation.

        Args:
        ----
            ctx (Context): The context for the operation.
            t1 (Tensor): The input tensor.
            dim (int): The dimension along which to compute the maximum.

        Returns:
        -------
            Tensor: The result of the tensor operation.

        """
        ctx.save_for_backward(t1, dim)
        return max_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the backward pass for the tensor operation.

        Args:
        ----
            ctx (Context): The context object containing information from the forward pass.
            grad_output (Tensor): The gradient of the output tensor.

        Returns:
        -------
            Tensor: The gradient of the input tensor.

        """
        t1, dim = ctx.saved_values
        dim = int(dim.item())

        max_vals = max_reduce(t1, dim)

        is_max = t1 == max_vals

        count_max = is_max.sum(dim=dim)
        count_max = count_max.view(
            *[1 if i == dim else s for i, s in enumerate(t1.shape)]
        )

        grad_input = (is_max * grad_output) / count_max

        return grad_input, 0.0

        # t1, dim = ctx.saved_values
        # max_vals = max_reduce(t1, int(dim.item()))  # Get max values
        # is_max = t1 == max_vals  # Identify max positions (boolean mask)
        # count_max = is_max.sum(dim=int(dim.item()))  # Count the number of max occurrences along dim

        # # Manually reshape to broadcast the count
        # count_max = count_max.view(*[1 if i == int(dim.item()) else s for i, s in enumerate(t1.shape)])

        # # Divide the gradient among tied max values
        # return (is_max * grad_output) / count_max, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Compute the maximum value of the input tensor along a specified dimension.

    Args:
    ----
        input: The input tensor.
        dim: The dimension along which to compute the maximum.

    Returns:
    -------
        A Tensor containing the maximum values along the specified dimension.

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax of the input tensor along a specified dimension as a tensor.

    Args:
    ----
        input: The input tensor.
        dim: The dimension along which to compute the softmax.

    Returns:
    -------
        A tensor of the same shape as `input`, with the softmax operation applied to the tensor.

    """
    input_exp = input.exp()
    return input_exp / input_exp.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log softmax of the input tensor along a specified dimension as a tensor.

    Args:
    ----
        input: The input tensor.
        dim: The dimension along which to compute the log softmax.

    Returns:
    -------
        A tensor of the same shape as `input`, with the log softmax operation applied to the tensor.

    """
    t = max(input, dim)
    return input - (input - t).exp().sum(dim).log() - t


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D max pooling using the tile function.

    Args:
    ----
        input: Tensor of shape (batch, channel, height, width).
        kernel: Tuple (kernel_height, kernel_width) specifying the pooling size.

    Returns:
    -------
        A Tensor of shape (batch, channel, new_height, new_width) after max pooling.

    """
    batch, channel, _, _ = input.shape

    tiled, new_height, new_width = tile(input, kernel)

    max_pooled = max(tiled, dim=4)

    return max_pooled.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, prob: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor by zeroing out elements randomly.

    Args:
    ----
        input: The input tensor.
        prob: The probability of dropping each element (0 <= prob < 1).
        ignore: If True, apply dropout. If False, return the input unchanged.

    Returns:
    -------
        A tensor with elements randomly set to 0 with probability `prob`,
        scaled by 1/(1 - prob) to maintain the expected sum.

    """
    if ignore or prob <= 0.0:
        return input

    mask = rand(input.shape) > prob

    return input * mask


# TODO: Implement for Task 4.3.
