from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type, List

from typing_extensions import Protocol

import minitorch

from . import operators
from .tensor_data import (
    shape_broadcast,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce placeholder"""
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Apply a reduction function to a tensor along a specific axis.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function to reduce elements.
            start (float, optional): The initial value for the reduction. Defaults to 0.0.

        Returns:
        -------
            Callable[["Tensor", int], "Tensor"]: A function that performs the reduction along the given axis.

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Perform matrix multiplication between two 2D tensors.

        Args:
        ----
            a (Tensor): The first matrix with shape (m, n).
            b (Tensor): The second matrix with shape (n, p).

        Returns:
        -------
            Tensor: The result of matrix multiplication with shape (m, p).

        """
        # Ensure the shapes are compatible for matrix multiplication
        assert (
            a.shape[-1] == b.shape[0]
        ), "Matrix shapes are not aligned for multiplication."

        # Get dimensions
        m, n = a.shape
        n, p = b.shape

        # Create an output tensor of shape (m, p)
        result = minitorch.zeros((m, p))

        # Perform matrix multiplication
        for i in range(m):
            for j in range(p):
                sum_value = 0.0
                for k in range(n):
                    sum_value += a[i, k] * b[k, j]
                result[i, j] = sum_value

        return result
        # raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
    ----
        fn: function from float-to-float to apply

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_size: int = 1
        for dim in out_shape:
            out_size *= dim

        def get_index(index: List[int], shape: Shape, strides: Strides) -> int:
            storage_index: int = 0
            for i, (idx, stride) in enumerate(zip(index, strides)):
                storage_index += idx * stride
            return storage_index

        def unravel_index(flat_index: int, shape: Shape) -> List[int]:
            idx: List[int] = []
            for dim in reversed(shape):
                idx.append(flat_index % dim)
                flat_index //= dim
            return list(reversed(idx))

        for i in range(out_size):
            out_idx: List[int] = unravel_index(i, out_shape)

            in_idx: List[int] = [
                0 if in_dim == 1 else out_dim
                for out_dim, in_dim in zip(
                    out_idx, [1] * (len(out_shape) - len(in_shape)) + list(in_shape)
                )
            ]

            in_storage_idx: int = get_index(in_idx, in_shape, in_strides)
            out_storage_idx: int = get_index(out_idx, out_shape, out_strides)

            out[out_storage_idx] = fn(in_storage[in_storage_idx])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
    ----
        fn: function mapping two floats to float to apply

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 2.3.

        out_size: int = 1
        for dim in out_shape:
            out_size *= dim

        def get_index(index: List[int], shape: Shape, strides: Strides) -> int:
            storage_index: int = 0
            for i, (idx, stride) in enumerate(zip(index, strides)):
                storage_index += idx * stride
            return storage_index

        def unravel_index(flat_index: int, shape: Shape) -> List[int]:
            idx: List[int] = []
            for dim in reversed(shape):
                idx.append(flat_index % dim)
                flat_index //= dim
            return list(reversed(idx))

        for i in range(out_size):
            out_idx: List[int] = unravel_index(i, out_shape)

            a_idx: List[int] = [
                0 if a_dim == 1 else out_dim
                for out_dim, a_dim in zip(
                    out_idx, [1] * (len(out_shape) - len(a_shape)) + list(a_shape)
                )
            ]

            b_idx: List[int] = [
                0 if b_dim == 1 else out_dim
                for out_dim, b_dim in zip(
                    out_idx, [1] * (len(out_shape) - len(b_shape)) + list(b_shape)
                )
            ]

            a_storage_idx: int = get_index(a_idx, a_shape, a_strides)
            b_storage_idx: int = get_index(b_idx, b_shape, b_strides)
            out_storage_idx: int = get_index(out_idx, out_shape, out_strides)

            out[out_storage_idx] = fn(
                a_storage[a_storage_idx], b_storage[b_storage_idx]
            )

    return _zip
    # raise NotImplementedError("Need to implement for Task 2.3")


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
    ----
        fn: reduction function mapping two floats to float

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 2.3.

        out_size: int = 1
        for dim in out_shape:
            out_size *= dim

        def get_index(index: List[int], shape: Shape, strides: Strides) -> int:
            storage_index: int = 0
            for i, (idx, stride) in enumerate(zip(index, strides)):
                storage_index += idx * stride
            return storage_index

        def unravel_index(flat_index: int, shape: Shape) -> List[int]:
            idx: List[int] = []
            for dim in reversed(shape):
                idx.append(flat_index % dim)
                flat_index //= dim
            return list(reversed(idx))

        for i in range(out_size):
            out_idx: List[int] = unravel_index(i, out_shape)

            a_idx: List[int] = out_idx.copy()
            a_idx[reduce_dim] = 0

            out_storage_idx: int = get_index(out_idx, out_shape, out_strides)
            a_storage_idx: int = get_index(a_idx, a_shape, a_strides)
            result: float = a_storage[a_storage_idx]

            for j in range(1, a_shape[reduce_dim]):
                a_idx[reduce_dim] = j
                a_storage_idx = get_index(a_idx, a_shape, a_strides)
                result = fn(result, a_storage[a_storage_idx])

            out[out_storage_idx] = result

    return _reduce

    # raise NotImplementedError("Need to implement for Task 2.3")


SimpleBackend = TensorBackend(SimpleOps)
