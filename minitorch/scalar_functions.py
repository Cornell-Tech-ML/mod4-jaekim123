from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

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
        """Apply the scalar function to the given values."""
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


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition.

        Args:
        ----
            ctx (Context): The context to save variables for backward.
            a (float): The first input scalar.
            b (float): The second input scalar.

        """
        return operators.add(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward pass for addition.

        Args:
        ----
            ctx (Context): The context with saved variables from forward.
            d_output (float): The derivative of the output.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for logarithm.

        Args:
        ----
            ctx (Context): The context to save variables for backward.
            a (float): The input scalar.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for logarithm.

        Args:
        ----
            ctx (Context): The context with saved variables from forward.
            d_output (float): The derivative of the output.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function for scalars."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication.

        Args:
        ----
            ctx (Context): The context to save variables for backward.
            a (float): The first input scalar.
            b (float): The second input scalar.

        Returns:
        -------
            float: The product of a and b.

        """
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication.

        Args:
        ----
            ctx (Context): The context with saved variables from forward.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to a and b.

        """
        a, b = ctx.saved_values
        return (b * d_output, a * d_output)


class Inv(ScalarFunction):
    """Inverse function for scalars."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse.

        Args:
        ----
            ctx (Context): The context to save variables for backward.
            a (float): The input scalar.

        Returns:
        -------
            float: The inverse of a.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for inverse.

        Args:
        ----
            ctx (Context): The context with saved variables from forward.
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float]: Gradient with respect to a.

        """
        (a,) = ctx.saved_values
        return ((-(operators.inv(a) ** 2)) * d_output,)


class Neg(ScalarFunction):
    """Negation function for scalars."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation.

        Args:
        ----
            ctx (Context): The context to save variables for backward.
            a (float): The input scalar.

        Returns:
        -------
            float: The negation of a.

        """
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation.

        Args:
        ----
            ctx (Context): The context from forward (no variables saved).
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float]: Gradient with respect to a.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function for scalars."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid.

        Args:
        ----
            ctx (Context): The context to save variables for backward.
            a (float): The input scalar.

        Returns:
        -------
            float: The sigmoid of a.

        """
        sig = operators.sigmoid(a)
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid.

        Args:
        ----
            ctx (Context): The context with saved variables from forward.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: Gradient with respect to a.

        """
        (sig,) = ctx.saved_values
        return sig * (1 - sig) * d_output


class ReLU(ScalarFunction):
    """ReLU function for scalars."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU.

        Args:
        ----
            ctx (Context): The context to save variables for backward.
            a (float): The input scalar.

        Returns:
        -------
            float: The ReLU of a.

        """
        relu_val = operators.relu(a)
        ctx.save_for_backward(a)
        return relu_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU.

        Args:
        ----
            ctx (Context): The context with saved variables from forward.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: Gradient with respect to a.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function for scalars."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential.

        Args:
        ----
            ctx (Context): The context to save variables for backward.
            a (float): The input scalar.

        Returns:
        -------
            float: The exponential of a.

        """
        exp_val = operators.exp(a)
        ctx.save_for_backward(exp_val)
        return exp_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponential.

        Args:
        ----
            ctx (Context): The context with saved variables from forward.
            d_output (float): The derivative of the output.

        Returns:
        -------
            float: Gradient with respect to a.

        """
        (exp_val,) = ctx.saved_values
        return exp_val * d_output


class LT(ScalarFunction):
    """Less than comparison for scalars."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less than comparison.

        Args:
        ----
            ctx (Context): The context (no variables saved).
            a (float): The first input scalar.
            b (float): The second input scalar.

        Returns:
        -------
            float: 1.0 if a < b else 0.0.

        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less than comparison.

        Args:
        ----
            ctx (Context): The context from forward (no variables saved).
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to a and b.

        """
        # Comparison operations are non-differentiable; gradients are zero.
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equality comparison for scalars."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality comparison.

        Args:
        ----
            ctx (Context): The context (no variables saved).
            a (float): The first input scalar.
            b (float): The second input scalar.

        Returns:
        -------
            float: 1.0 if a == b else 0.0.

        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equality comparison.

        Args:
        ----
            ctx (Context): The context from forward (no variables saved).
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to a and b.

        """
        # Comparison operations are non-differentiable; gradients are zero.
        return 0.0, 0.0


class Sub(ScalarFunction):
    """Subtraction function $f(x, y) = x - y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for subtraction.

        Args:
        ----
            ctx (Context): The context to save variables for backward.
            a (float): The first input scalar.
            b (float): The second input scalar.

        Returns:
        -------
            float: The difference of a and b.

        """
        return operators.sub(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for subtraction.

        Args:
        ----
            ctx (Context): The context from forward (no variables saved).
            d_output (float): The derivative of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to a and b.

        """
        return d_output, -d_output
