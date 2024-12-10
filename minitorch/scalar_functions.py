from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch
import math

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the given values.

        This method takes any number of scalar-like values (which can be either
        Scalar objects or raw float/int values), applies the function to these
        values, and returns a new Scalar object with the result.

        Args:
        ----
            *vals (ScalarLike): Variable number of scalar-like inputs to the function.

        Returns:
        -------
            Scalar: A new Scalar object representing the result of applying the function.

        Note:
        ----
            This method handles the conversion of raw values to Scalar objects,
            creates a context for the computation, calls the forward method,
            and sets up the backward pass for automatic differentiation.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward pass of the addition operation.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of adding a and b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Perform the backward pass of the addition operation.

        Args:
        ----
            ctx (Context): The context object (unused in this case).
            d_output (float): The gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[float, ...]: A tuple containing the gradients with respect to each input.
                               In this case, it returns (d_output, d_output) since the
                               derivative of addition with respect to both inputs is 1.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the natural logarithm operation.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The natural logarithm of the input value.

        Note:
        ----
            This method saves the input value for use in the backward pass.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass of the natural logarithm operation.

        Args:
        ----
            ctx (Context): The context object containing saved values from the forward pass.
            d_output (float): The gradient of the loss with respect to the output of this function.

        Returns:
        -------
            float: The gradient of the loss with respect to the input of this function.

        Note:
        ----
            This method uses the chain rule to compute the gradient:
            d(log(x))/dx = 1/x, so we multiply d_output by 1/x.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    r"""Multiplication function \( f(x, y) = x \times y \)"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward pass of the multiplication operation.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The product of the two input values.

        Note:
        ----
            This method saves both input values for use in the backward pass.

        """
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform the backward pass of the multiplication operation.

        Args:
        ----
            ctx (Context): The context object containing saved values from the forward pass.
            d_output (float): The gradient of the loss with respect to the output of this function.

        Returns:
        -------
            Tuple[float, float]: A tuple containing the gradients of the loss with respect to
            the first and second inputs of this function, respectively.

        Note:
        ----
            This method uses the chain rule to compute the gradients:
            For f(a, b) = a * b, we have:
            df/da = b * d_outpu
            df/db = a * d_outpu

        """
        a, b = ctx.saved_values
        grad_a = b * d_output
        grad_b = a * d_output
        return grad_a, grad_b


class Inv(ScalarFunction):
    r"""Inverse function \( f(x) = 1 / x \)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the inverse operation.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The inverse of the input value (1 / a).

        Note:
        ----
            This method saves the input value for use in the backward pass.

        """
        ctx.save_for_backward(a)
        return 1 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass of the inverse operation.

        Args:
        ----
            ctx (Context): The context object containing saved values from the forward pass.
            d_output (float): The gradient of the loss with respect to the output of this function.

        Returns:
        -------
            float: The gradient of the loss with respect to the input of this function.

        Note:
        ----
            This method uses the chain rule to compute the gradient:
            For f(x) = 1/x, we have:
            df/dx = -1/x^2 * d_outpu

        """
        (a,) = ctx.saved_values
        grad_a = -1 / (a**2) * d_output
        return grad_a


class Neg(ScalarFunction):
    r"""Negation function \( f(x) = -x \)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the negation operation.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The negation of the input value (-a).

        Note:
        ----
            This method does not need to save any values for the backward pass
            as the gradient for negation is always -1.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass of the negation operation.

        Args:
        ----
            ctx (Context): The context object containing saved values from the forward pass.
                           In this case, it's not used as negation doesn't require saved values.
            d_output (float): The gradient of the loss with respect to the output of this function.

        Returns:
        -------
            float: The gradient of the loss with respect to the input of this function.

        Note:
        ----
            The gradient for negation is always -1 times the output gradient,
            as d(-x)/dx = -1 for all x.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    r"""Sigmoid function \( f(x) = \frac{1}{1 + e^{-x}} \)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the sigmoid function.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The result of the sigmoid function applied to the input.

        Note:
        ----
            This method saves the result for use in the backward pass.
            The sigmoid function is defined as sigmoid(x) = 1 / (1 + e^(-x)).

        """
        result = 1 / (1 + math.exp(-a))
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass of the sigmoid function.

        Args:
        ----
            ctx (Context): The context object containing saved values from the forward pass.
            d_output (float): The gradient of the loss with respect to the output of this function.

        Returns:
        -------
            float: The gradient of the loss with respect to the input of this function.

        Note:
        ----
            The gradient of the sigmoid function is computed as sigmoid(x) * (1 - sigmoid(x)) * d_output.

        """
        (sigmoid_a,) = ctx.saved_values
        grad_a = sigmoid_a * (1 - sigmoid_a) * d_output
        return grad_a


class ReLU(ScalarFunction):
    r"""ReLU function \( f(x) = \max(0, x) \)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the ReLU function.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The result of the ReLU function applied to the input.

        Note:
        ----
            This method saves the input for use in the backward pass.
            The ReLU function is defined as relu(x) = max(0, x).

        """
        ctx.save_for_backward(a)
        return max(0.0, a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass of the ReLU function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved values for backward pass.
            d_output (float): The gradient of the loss with respect to the output of this function.

        Returns:
        -------
            float: The gradient of the loss with respect to the input of this function.

        Note:
        ----
            The gradient of the ReLU function is computed as d_output if the input was positive, otherwise 0.

        """
        (a,) = ctx.saved_values
        grad_a = d_output if a > 0 else 0.0
        return grad_a


class Exp(ScalarFunction):
    r"""Exponential function \( f(x) = e^{x} \)"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Perform the forward pass of the Exponential function.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The input value.

        Returns:
        -------
            float: The result of the Exponential function applied to the input.

        Note:
        ----
            This method saves the result of the Exponential function for use in the backward pass.
            The Exponential function is defined as exp(x) = e^x.

        """
        result = math.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Perform the backward pass of the Exponential function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved values for backward pass.
            d_output (float): The derivative of the output with respect to the input.

        Returns:
        -------
            float: The derivative of the input with respect to the output.

        Note:
        ----
            This method computes the derivative of the Exponential function with respect to its input.
            The derivative of exp(x) is exp(x).

        """
        (exp_a,) = ctx.saved_values
        grad_a = exp_a * d_output
        return grad_a


class LT(ScalarFunction):
    r"""Less-than function \( f(x, y) = 1.0 \) if \( x < y \) else \( 0.0 \)"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward pass of the Less-than function.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the Less-than function applied to the inputs.

        Note:
        ----
            This method computes the Less-than function, which returns 1.0 if a is less than b, and 0.0 otherwise.

        """
        # No need to save anything for backward
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform the backward pass of the Less-than function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved values for backward pass.
            d_output (float): The derivative of the output with respect to the input.

        Returns:
        -------
            Tuple[float, float]: The derivatives of the input with respect to the output for both inputs.

        Note:
        ----
            This method computes the derivatives of the Less-than function with respect to its inputs.
            The derivative is zero almost everywhere, except at the point where the inputs are equal.

        """
        # Derivative is zero almost everywhere
        grad_a = 0.0
        grad_b = 0.0
        return grad_a, grad_b


class EQ(ScalarFunction):
    r"""Equal function \( f(x, y) = 1.0 \) if \( x == y \) else \( 0.0 \)"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward pass of the Equal function.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the Equal function applied to the inputs.

        Note:
        ----
            This method computes the Equal function, which returns 1.0 if a is equal to b, and 0.0 otherwise.

        """
        # Be cautious with floating-point equality
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform the backward pass of the Equal function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved values for backward pass.
            d_output (float): The derivative of the output with respect to the input.

        Returns:
        -------
            Tuple[float, float]: The derivatives of the input with respect to the output for both inputs.

        Note:
        ----
            This method computes the derivatives of the Equal function with respect to its inputs.
            The derivative is zero almost everywhere, except at the point where the inputs are equal.

        """
        # Derivative is zero almost everywhere
        grad_a = 0.0
        grad_b = 0.0
        return grad_a, grad_b


class Sub(ScalarFunction):
    r"""Subtraction function \( f(a, b) = a - b \)"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Perform the forward pass of the Subtraction function.

        Args:
        ----
            ctx (Context): The context object to save values for backward pass.
            a (float): The first input value.
            b (float): The second input value.

        Returns:
        -------
            float: The result of the Subtraction function applied to the inputs.

        Note:
        ----
            This method computes the Subtraction function, which returns the difference between a and b.

        """
        return a - b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Perform the backward pass of the Subtraction function.

        Args:
        ----
            ctx (Context): The context object to retrieve saved values for backward pass.
            d_output (float): The derivative of the output with respect to the input.

        Returns:
        -------
            Tuple[float, float]: The derivatives of the input with respect to the output for both inputs.

        Note:
        ----
            This method computes the derivatives of the Subtraction function with respect to its inputs.
            The derivative with respect to the first input is the derivative of the output, and the derivative
            with respect to the second input is the negated derivative of the output.

        """
        return d_output, -d_output


# TODO: Implement for Task 1.2.
