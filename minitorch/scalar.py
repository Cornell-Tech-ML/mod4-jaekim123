from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
    Sub,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __hash__(self):
        return hash(self.unique_id)

    # def __eq__(self, other: object) -> bool:
    #     if isinstance(other, Scalar):
    #         return self.unique_id == other.unique_id
    #     return False

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the Scalar is a constant.

        A Scalar is considered constant if it has no history, meaning it was no
        created as a result of an operation and has no gradient information.

        Returns
        -------
        bool
            True if the Scalar is a constant, False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns an iterable of the parent variables of this Scalar.

        This property retrieves the input variables that were used to create this Scalar
        through an operation. It assumes that the Scalar has a history (i.e., it's not a
        constant or leaf variable).

        Returns:
        -------
        Iterable[Variable]:
            An iterable containing the parent variables of this Scalar.

        Raises:
        ------
        AssertionError:
            If the Scalar doesn't have a history (i.e., it's a constant or leaf variable).

        Note:
        ----
        This property is used in backpropagation to traverse the computation graph.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for the Scalar's inputs.

        This method implements the chain rule of calculus for automatic differentiation.
        It computes the gradients of the Scalar with respect to its input variables,
        given the gradient of the output with respect to this Scalar.

        Args:
        ----
            d_output (Any): The gradient of the final output with respect to this Scalar.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple contains:
                - The input Variable
                - The gradient of the output with respect to that inpu

        Raises:
        ------
            AssertionError: If the Scalar doesn't have a history, last function, or context.

        Note:
        ----
            This method is a crucial part of the backpropagation algorithm in the autograd system.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        grads = h.last_fn._backward(h.ctx, d_output)

        if not isinstance(grads, tuple):
            grads = (grads,)

        result = []
        for input_var, grad in zip(h.inputs, grads):
            if not input_var.is_constant():
                result.append((input_var, grad))

        return result
        # TODO: Implement for Task 1.3.
        # raise NotImplementedError("Need to implement for Task 1.3")

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    # TODO: Implement for Task 1.2.

    def __add__(self, b: ScalarLike) -> Scalar:
        return Add.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        return Sub.apply(self, b)

    def __eq__(self, other: ScalarLike) -> Scalar:
        return EQ.apply(self, other)

    def __rsub__(self, b: ScalarLike) -> Scalar:
        return Sub.apply(b, self)

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

    def __lt__(self, b: ScalarLike) -> Scalar:
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        return LT.apply(b, self)

    def log(self) -> Scalar:
        """Compute the natural logarithm of this Scalar.

        Returns:
        -------
            Scalar: A new Scalar representing the natural logarithm of this Scalar.

        Note:
        ----
            This method applies the Log function to the current Scalar.

        """
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Compute the exponential of this Scalar.

        Returns:
        -------
            Scalar: A new Scalar representing e raised to the power of this Scalar.

        Note:
        ----
            This method applies the Exp function to the current Scalar.

        """
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Compute the sigmoid function of this Scalar.

        Returns:
        -------
            Scalar: A new Scalar representing the sigmoid of this Scalar.

        Note:
        ----
            This method applies the Sigmoid function to the current Scalar.
            The sigmoid function is defined as sigmoid(x) = 1 / (1 + exp(-x)).

        """
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Compute the Rectified Linear Unit (ReLU) function of this Scalar.

        Returns:
        -------
            Scalar: A new Scalar representing the ReLU of this Scalar.

        Note:
        ----
            This method applies the ReLU function to the current Scalar.
            The ReLU function is defined as relu(x) = max(0, x).

        """
        return ReLU.apply(self)

    # raise NotImplementedError("Need to implement for Task 1.2")


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a Python function by comparing
    the calculated derivatives with numerical approximations.

    Asserts False if the derivative is incorrect.

    Parameters
    ----------
    f : Callable
        A function that takes n Scalar arguments and returns a single Scalar value.
    *scalars : Scalar
        A variable-length argument list of Scalar objects to check the derivatives for.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
    Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
    but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
