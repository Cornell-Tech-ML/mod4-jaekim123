from typing import Tuple, TypeVar, Any

from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
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
    """A wrapper for the Numba `njit` function that applies the inline optimization.

    Args:
    ----
        fn (Fn): The function to be JIT-compiled.
        **kwargs (Any): Additional keyword arguments to pass to the Numba `njit` function.

    Returns:
    -------
        Fn: The JIT-compiled version of the input function.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
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
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    for b in range(batch):
        for oc in range(out_channels):
            for ow in range(out_width):
                out_index = (
                    b * out_strides[0] + oc * out_strides[1] + ow * out_strides[2]
                )
                out[out_index] = 0.0

                for ic in range(in_channels):
                    for k in range(kw):
                        iw = ow + k if not reverse else ow - k

                        if 0 <= iw < width:
                            input_index = (
                                b * input_strides[0]
                                + ic * input_strides[1]
                                + iw * input_strides[2]
                            )
                            weight_index = (
                                oc * weight_strides[0]
                                + ic * weight_strides[1]
                                + (kw - 1 - k if reverse else k) * weight_strides[2]
                            )

                            out[out_index] += input[input_index] * weight[weight_index]


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradients of the input and weight tensors for a 1D convolution operation
        during the backward pass of the autograd system.

        Args:
        ----
            ctx (Context): The context object that stores information from the forward pass,
                including the saved tensors (input and weight) used for computing gradients.
            grad_output (Tensor): The gradient of the loss with respect to the output of the
                convolution operation.

        Returns:
        -------
            Tuple[Tensor, Tensor]: A tuple containing:
                - grad_input (Tensor): The gradient of the loss with respect to the input tensor.
                - grad_weight (Tensor): The gradient of the loss with respect to the weight tensor.

        Notes:
        -----
            - The method uses a permuted version of the input, weight, and gradient tensors
                to compute the necessary gradients.
            - Gradients are computed using `tensor_conv1d`, a lower-level function.
            - The computed gradients are returned in their original shapes by permuting
                them back to match the input format.

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
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
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    for out_index in range(out_size):
        b = (out_index // out_strides[0]) % out_shape[0]
        oc = (out_index // out_strides[1]) % out_shape[1]
        oh = (out_index // out_strides[2]) % out_shape[2]
        ow = (out_index // out_strides[3]) % out_shape[3]

        out[out_index] = 0.0

        for ic in range(in_channels):
            for kh_idx in range(kh):
                for kw_idx in range(kw):
                    ih = oh + kh_idx if not reverse else oh - kh_idx
                    iw = ow + kw_idx if not reverse else ow - kw_idx

                    if 0 <= ih < height and 0 <= iw < width:
                        input_index = b * s10 + ic * s11 + ih * s12 + iw * s13
                        weight_index = (
                            oc * s20
                            + ic * s21
                            + (kh - 1 - kh_idx if reverse else kh_idx) * s22
                            + (kw - 1 - kw_idx if reverse else kw_idx) * s23
                        )

                        out[out_index] += input[input_index] * weight[weight_index]


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradients of the input and weight tensors for a 2D convolution operation
        during the backward pass of the autograd system.

        Args:
        ----
            ctx (Context): The context object storing information from the forward pass,
                including saved tensors (`input` and `weight`) required for computing gradients.
            grad_output (Tensor): The gradient of the loss with respect to the output of the
                2D convolution operation.

        Returns:
        -------
            Tuple[Tensor, Tensor]: A tuple containing:
                - grad_input (Tensor): The gradient of the loss with respect to the input tensor.
                - grad_weight (Tensor): The gradient of the loss with respect to the weight tensor.

        Notes:
        -----
            - Gradients are computed using the lower-level `tensor_conv2d` function.
            - The input, weight, and gradient tensors are permuted to compute gradients efficiently
            and then permuted back to their original shape before returning.
            - The `grad_input` tensor has the same shape as the input tensor.
            - The `grad_weight` tensor has the same shape as the weight tensor.

        Example:
        -------
            If `input` has shape `(batch, in_channels, height, width)` and `weight` has shape
            `(out_channels, in_channels, kernel_height, kernel_width)`, the returned `grad_input`
            and `grad_weight` tensors will have the same shapes as `input` and `weight`, respectively.

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
