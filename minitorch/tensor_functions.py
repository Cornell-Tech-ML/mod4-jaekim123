"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend


if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass for the negation operation.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A tensor representing the element-wise negation of the input.

        """
        ctx.save_for_backward(t1)
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> tuple[Tensor]:
        """Computes the backward pass for the negation operation.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of the negation.

        Returns:
        -------
            tuple[Tensor]: The gradient of the loss with respect to the input of the negation.

        """
        (t1,) = ctx.saved_values
        grad_input = -grad_output
        return (grad_input,)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass for the inverse operation (1/t1).

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A tensor representing the element-wise inverse of the input.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass for the inverse operation.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of the inverse.

        Returns:
        -------
            Tensor: The gradient of the loss with respect to the input of the inverse.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the forward pass for element-wise multiplication.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A tensor representing the element-wise product of the inputs.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass for element-wise multiplication.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the loss with respect to the two inputs.

        """
        t1, t2 = ctx.saved_values  # Retrieve t1 and t2
        grad_t1 = grad_output.f.mul_zip(grad_output, t2)  # dL/dt1 = dL/dout * t2
        grad_t2 = grad_output.f.mul_zip(grad_output, t1)  # dL/dt2 = dL/dout * t1
        return grad_t1, grad_t2


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A tensor representing the element-wise sigmoid activation of the input.

        """
        result = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass for the sigmoid function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of the sigmoid.

        Returns:
        -------
            Tensor: The gradient of the loss with respect to the input of the sigmoid.

        """
        (sigmoid_output,) = ctx.saved_values

        sigmoid_derivative = sigmoid_output - sigmoid_output * sigmoid_output

        grad_input = grad_output * sigmoid_derivative

        return grad_input


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass for the ReLU activation function.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A tensor with element-wise ReLU activation applied.

        """
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> tuple[Tensor]:
        """Computes the backward pass for the ReLU activation function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of ReLU.

        Returns:
        -------
            tuple[Tensor]: The gradient of the loss with respect to the input of ReLU.

        """
        (t1,) = ctx.saved_values
        grad_mask = (t1 > 0) * 1.0
        grad_input = grad_output * grad_mask
        return (grad_input,)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass for the logarithm function.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A tensor representing the element-wise natural logarithm of the input.

        """
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass for the logarithm function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of the logarithm.

        Returns:
        -------
            Tensor: The gradient of the loss with respect to the input of the logarithm.

        """
        t1 = ctx.saved_values[0]
        inv_t1 = t1.f.inv_map(t1)
        return grad_output.f.mul_zip(grad_output, inv_t1)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Computes the forward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            t1 (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A tensor representing the element-wise exponential of the input.

        """
        result = t1.f.exp_map(t1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes the backward pass for the exponential function.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of the exponential.

        Returns:
        -------
            Tensor: The gradient of the loss with respect to the input of the exponential.

        """
        exp_output = ctx.saved_values[0]
        return grad_output.f.mul_zip(exp_output, grad_output)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim_tensor: Tensor) -> Tensor:
        """Computes the forward pass for the summation operation along a specific dimension.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            t1 (Tensor): The input tensor to be summed.
            dim_tensor (Tensor): A tensor specifying the dimension along which to sum.

        Returns:
        -------
            Tensor: A tensor with the sum computed along the specified dimension.

        """
        dim = int(dim_tensor.item())
        ctx.save_for_backward(t1, dim_tensor)
        return t1.f.add_reduce(t1, dim)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        """Computes the backward pass for the summation operation.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of the sum.

        Returns:
        -------
            tuple[Tensor, Tensor]: The gradient of the loss with respect to the input tensor and dimension.

        """
        t1, dim_tensor = ctx.saved_values
        dim = int(dim_tensor.item())
        shape = list(t1.shape)
        shape[dim] = 1

        grad_output_reshaped = grad_output.view(*shape)

        # Use the newly defined `ones` function
        ones_tensor = ones(t1.shape, backend=t1.backend)
        grad_input = grad_output_reshaped * ones_tensor
        zero_grad = zeros(dim_tensor.shape, backend=dim_tensor.backend)
        return grad_input, zero_grad


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the forward pass for the less-than comparison operation.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A tensor representing element-wise comparison (t1 < t2).

        """
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass for the less-than comparison operation.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of zero for both inputs since comparison has no gradients.

        """
        zero_grad = grad_output * 0
        return zero_grad, zero_grad


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the forward pass for the equality comparison operation.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A tensor representing element-wise comparison (t1 == t2).

        """
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass for the equality comparison operation.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of zero for both inputs since comparison has no gradients.

        """
        zero_grad = grad_output * 0
        return zero_grad, zero_grad


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the forward pass for the element-wise is-close operation.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A tensor indicating whether elements of t1 are close to corresponding elements in t2.

        """
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Computes the forward pass for the permutation operation, rearranging tensor dimensions.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            a (Tensor): The input tensor to be permuted.
            order (Tensor): A tensor specifying the new order of dimensions.

        Returns:
        -------
            Tensor: A tensor with permuted dimensions.

        """
        ctx.save_for_backward(order)

        order_list = [int(order[i]) for i in range(order.shape[0])]

        if len(order_list) != len(a.shape):
            raise ValueError(
                f"Permutation order length {len(order_list)} does not match tensor dimensions {len(a.shape)}."
            )

        if sorted(order_list) != list(range(len(a.shape))):
            raise ValueError(
                f"Invalid permutation order: {order_list}. Must be a permutation of {list(range(len(a.shape)))}."
            )

        permuted_shape = tuple(a.shape[i] for i in order_list)
        permuted_strides = tuple(a._tensor.strides[i] for i in order_list)

        return minitorch.Tensor.make(
            a._tensor._storage, permuted_shape, permuted_strides, backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass for the permutation operation.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of the permutation.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the loss with respect to the input tensor and order.

        """
        (order,) = ctx.saved_tensors

        inverse_order_storage = [0] * order._tensor.size

        for i in range(order._tensor.size):
            index = int(order._tensor._storage[i])
            inverse_order_storage[index] = i

        inverse_order = tensor(inverse_order_storage, backend=order.backend)

        grad_input = Permute.apply(grad_output, inverse_order)

        zero_grad = zeros(order.shape, backend=order.backend)

        return grad_input, zero_grad


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Computes the forward pass for element-wise addition.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A tensor representing the element-wise sum of the inputs.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the backward pass for element-wise addition.

        Args:
        ----
            ctx (Context): The context containing saved information from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of the addition.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients of the loss with respect to both input tensors.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim_tensor: Tensor) -> Tensor:
        """Computes the forward pass for the all operation, checking if all elements along a dimension meet a condition.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            a (Tensor): The input tensor.
            dim_tensor (Tensor): The dimension along which to check the condition.

        Returns:
        -------
            Tensor: A tensor indicating whether all elements meet the specified condition along the dimension.

        """
        dim = int(dim_tensor.item())
        ctx.save_for_backward(dim_tensor)
        return a.f.mul_reduce(a, dim)


# TODO: Implement for Task 2.3.


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Computes the forward pass for the view operation, reshaping the tensor to the specified shape.

        Args:
        ----
            ctx (Context): The context for storing information to be used in the backward pass.
            a (Tensor): The input tensor to be reshaped.
            shape (Tensor): The new shape for the tensor.

        Returns:
        -------
            Tensor: A tensor reshaped to the specified shape.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def ones(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a tensor of ones of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [1.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Computes the gradient of a function using the central difference method.

    Args:
    ----
        f (Any): The function for which the gradient is computed.
        *vals (Tensor): The input tensors to the function.
        arg (int, optional): The index of the argument with respect to which
                             the gradient is computed. Default is 0.
        epsilon (float, optional): The small increment used for finite differences. Default is 1e-6.
        ind (UserIndex): The index used to compute the gradient.

    Returns:
    -------
        float: The estimated gradient at the specified index.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
