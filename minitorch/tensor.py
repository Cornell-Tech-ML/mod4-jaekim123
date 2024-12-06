"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets whether gradients should be computed for this tensor.

        Args:
        ----
            x (bool): If True, gradients will be tracked for this tensor
                    during operations. If False, gradients will not be tracked.

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Checks if this tensor requires gradient computation.

        Returns
        -------
            bool: True if gradients are being tracked for this tensor,
                otherwise False.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor filled with zeros.

        Args:
        ----
            shape (Optional[UserShape]): The shape of the tensor to create.
                                        If None, a default shape will be used.

        Returns:
        -------
            Tensor: A tensor of the specified shape, filled with zeros.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the tensor is a constant (i.e., does not require gradients).

        Returns
        -------
            bool: True if the tensor does not track gradients, otherwise False.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent variables of this tensor in the computation graph.

        Returns
        -------
            Iterable[Variable]: The inputs to this tensor's operation, which are
                                considered its parents in the computation graph.

        Raises
        ------
            AssertionError: If the tensor does not have a computation history.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Computes the gradients of the inputs (parents) using the chain rule.

        Args:
        ----
            d_output (Any): The gradient of the output with respect to some
                            scalar value (typically the loss).

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: A collection of tuples, where each
                                            tuple contains a parent variable and
                                            its corresponding gradient contribution.

        Raises:
        ------
            AssertionError: If the tensor does not have a computation history.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Performs backpropagation to compute gradients for all variables in the computation graph.

        Args:
        ----
            grad_output (Optional[Tensor]): The gradient of the output with respect to
                                            the tensor. If None, the gradient is assumed
                                            to be 1 for scalar tensors. For non-scalar tensors,
                                            `grad_output` must be provided.

        Raises:
        ------
            AssertionError: If `grad_output` is not provided for a non-scalar tensor.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    def __add__(self, other: TensorLike) -> Tensor:
        """Element-wise addition of two tensors."""
        return Add.apply(self, self._ensure_tensor(other))

    def __sub__(self, other: TensorLike) -> Tensor:
        """Element-wise subtraction of two tensors."""
        return Add.apply(self, self._ensure_tensor(other) * -1)

    def __mul__(self, other: TensorLike) -> Tensor:
        """Element-wise multiplication of two tensors."""
        return Mul.apply(self, self._ensure_tensor(other))

    def __lt__(self, other: TensorLike) -> Tensor:
        """Element-wise less-than comparison."""
        return LT.apply(self, self._ensure_tensor(other))

    def __eq__(self, other: TensorLike) -> Tensor:
        """Element-wise equality comparison."""
        return EQ.apply(self, self._ensure_tensor(other))

    def __gt__(self, other: TensorLike) -> Tensor:
        """Element-wise greater-than comparison."""
        return LT.apply(self._ensure_tensor(other), self)

    def __neg__(self) -> Tensor:
        """Negates the tensor element-wise."""
        return Neg.apply(self)

    def __radd__(self, other: TensorLike) -> Tensor:
        """Right-hand addition for scalar + tensor."""
        return self + other

    def __rmul__(self, other: TensorLike) -> Tensor:
        """Right-hand multiplication for scalar * tensor."""
        return self * other

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Returns True if all elements are true along a dimension.

        Args:
        ----
            dim (Optional[int]): The dimension to reduce over. If None, reduces over all dimensions.

        Returns:
        -------
            Tensor: A tensor containing the result of the reduction.

        """
        if dim is None:
            # Reduce across all dimensions
            zero_tensor = Tensor.make(
                [0], (1,), backend=self.backend
            )  # Create a Tensor for 0
            return All.apply(
                self.contiguous().view(int(operators.prod(self.shape))), zero_tensor
            )
        else:
            # Reduce along the specified dimension
            dim_tensor = Tensor.make([dim], (1,), backend=self.backend)
            return All.apply(self, dim_tensor)

    def is_close(self, other: TensorLike) -> Tensor:
        """Element-wise check if two tensors are close within a tolerance."""
        return IsClose.apply(self, self._ensure_tensor(other))

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid function element-wise."""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies the ReLU function element-wise."""
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Applies the natural logarithm element-wise."""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Applies the exponential function element-wise."""
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Sums the tensor along a dimension.

        Args:
        ----
            dim (Optional[int]): The dimension to sum over. If None, sums over all elements.

        Returns:
        -------
            Tensor: A tensor containing the sum.

        """
        if dim is None:
            # If dim is None, we sum over all dimensions by reducing to a scalar
            return self.view(
                int(operators.prod(self.shape)),
            ).sum(0)

        # Pass dim as a Tensor for backward compatibility
        dim_tensor = Tensor.make([dim], (1,), backend=self.backend)
        return Sum.apply(self, dim_tensor)

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Computes the mean of the tensor along the given dimension (or of all elements if no dimension is provided).

        Args:
        ----
            dim (Optional[int]): The dimension to compute the mean along. If None, computes the mean of all elements.

        Returns:
        -------
            Tensor: A tensor with the mean.

        """
        if dim is None:
            total_elements = self.size
        else:
            total_elements = self.shape[dim]

        return self.sum(dim) * (1 / total_elements)

    def permute(self, *dims: int) -> Tensor:
        """Permute the dimensions of the tensor according to the specified order.

        Args:
        ----
            *dims (int): The desired ordering of dimensions.

        Returns:
        -------
            Tensor: A new tensor with permuted dimensions.

        """
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            # If a single list or tuple is provided, unpack it
            order = list(dims[0])
        else:
            order = list(dims)

        # Convert 'order' to a List[float]
        order_floats = [float(i) for i in order]

        # Create 'order_tensor' using the list of floats
        order_tensor = Tensor.make(order_floats, (len(order),), backend=self.backend)

        # Pass 'order_tensor' to apply
        return Permute.apply(self, order_tensor)

    def view(self, *shape: int, dim: Optional[int] = None) -> Tensor:
        """Reshapes the tensor to the specified shape.

        Args:
        ----
            *shape (int): The new shape(s) to reshape to.
            dim (Optional[int]): The dimension to reshape. If None, reshapes the entire tensor.

        Returns:
        -------
            Tensor: A reshaped tensor.

        """
        if dim is not None:
            # Reshape only the specified dimension
            assert len(shape) == 1, "When specifying `dim`, provide a single new size."
            new_shape = list(self.shape)
            new_shape[dim] = shape[0]
            return View.apply(self.contiguous(), tensor(new_shape))
        else:
            # Reshape the entire tensor
            return View.apply(self.contiguous(), tensor(shape))

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        return int(operators.prod(self.shape))

    @property
    def dims(self) -> int:
        """Returns the number of dimensions of the tensor."""
        return len(self.shape)

    def zero_grad_(self) -> None:
        """Reset the gradient to None."""
        self.grad = None

    # Functions
    # TODO: Implement for Task 2.3.
