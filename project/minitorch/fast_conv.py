from typing import Tuple, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# JIT compile tensor_data functions
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation with Same Padding."""
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    # Cache strides
    s1_0, s1_1, s1_2 = input_strides
    s2_0, s2_1, s2_2 = weight_strides
    out_s0, out_s1, out_s2 = out_strides

    for b in range(batch_):
        for oc in range(out_channels):
            for w in range(out_width):
                acc = 0.0
                for ic in range(in_channels):
                    for k in range(kw):
                        # Handle reverse mode
                        w_p = w + k if not reverse else w + (kw - k - 1)
                        
                        if w_p < width:
                            i_pos = b * s1_0 + ic * s1_1 + w_p * s1_2
                            w_pos = oc * s2_0 + ic * s2_1 + k * s2_2
                            acc += input[i_pos] * weight[w_pos]
                
                out_pos = b * out_s0 + oc * out_s1 + w * out_s2
                out[out_pos] = acc


# Use regular njit without parallel for now
tensor_conv1d = njit(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution with Same Padding."""
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Calculate padding for 'same' convolution
        pad = kw // 2
        ctx.pad = pad  # Save padding for backward pass

        # Create padded input
        padded_width = w + 2 * pad
        padded = input.zeros((batch, in_channels, padded_width))

        # Assign input data to the center of the padded tensor
        for b in range(batch):
            for ic in range(in_channels):
                for i in range(w):
                    padded[b, ic, pad + i] = input[b, ic, i]

        # Output width remains the same as input width
        out_width = w
        output = input.zeros((batch, out_channels, out_width))
        tensor_conv1d(
            *output.tuple(), output.size, *padded.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        pad = ctx.pad
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape

        # Initialize gradient tensors
        grad_input_padded = input.zeros((batch, in_channels, w + 2 * pad))
        grad_weight = weight.zeros((out_channels, in_channels, kw))

        # Compute gradients w.r.t. input
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input_padded.tuple(),
            grad_input_padded.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True
        )

        # Extract the gradients w.r.t. input by removing padding
        grad_input = input.zeros((batch, in_channels, w))
        for b in range(batch):
            for ic in range(in_channels):
                for i in range(w):
                    grad_input[b, ic, i] = grad_input_padded[b, ic, pad + i]

        # Compute gradients w.r.t. weight
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False
        )

        return grad_input, grad_weight


conv1d = Conv1dFun.apply 