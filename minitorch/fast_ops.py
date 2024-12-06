from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Applies just-in-time (JIT) compilation to a function with specific optimizations.

    This function wraps another function `fn` and compiles it using `_njit`. The
    JIT compilation improves performance by converting the Python function into
    optimized machine code at runtime. The `inline="always"` argument ensures that
    the compiled function is inlined wherever possible for better efficiency.

    Args:
    ----
        fn (Fn): The function to be JIT-compiled.
        **kwargs (Any): Additional keyword arguments passed to `_njit`, allowing
            customization of the JIT compilation process.

    Returns:
    -------
        Fn: The JIT-compiled version of the input function.

    Notes:
    -----
        - This function is particularly useful for optimizing performance-intensive
          numerical operations or tensor computations.
        - The `inline="always"` setting enhances performance by reducing function
          call overhead during execution.
        - The `_njit` mechanism may require specific dependencies (e.g., CUDA, Numba).

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function."""

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        if (
            len(out_shape) == len(in_shape)
            and np.array_equal(out_shape, in_shape)
            and np.array_equal(out_strides, in_strides)
        ):
            for i in prange(int(np.prod(out_shape))):
                out[i] = fn(in_storage[i])
        else:
            for i in prange(int(np.prod(out_shape))):
                out_index = np.zeros(len(out_shape), dtype=np.int32)
                in_index = np.zeros(len(in_shape), dtype=np.int32)

                to_index(i, out_shape, out_index)

                broadcast_index(out_index, out_shape, in_shape, in_index)

                out_pos = index_to_position(out_index, out_strides)

                valid = True
                for d in range(len(in_shape)):
                    if not (0 <= in_index[d] < in_shape[d]):
                        valid = False
                        break

                if valid:
                    in_pos = index_to_position(in_index, in_strides)
                    out[out_pos] = fn(in_storage[in_pos])

    return njit(_map, parallel=True)


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function."""

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
        if (
            len(out_shape) == len(a_shape) == len(b_shape)
            and np.array_equal(out_shape, a_shape)
            and np.array_equal(a_shape, b_shape)
            and np.array_equal(out_strides, a_strides)
            and np.array_equal(a_strides, b_strides)
        ):
            for i in prange(int(np.prod(out_shape))):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            for i in prange(int(np.prod(out_shape))):
                out_index = np.zeros(len(out_shape), dtype=np.int32)
                a_index = np.zeros(len(a_shape), dtype=np.int32)
                b_index = np.zeros(len(b_shape), dtype=np.int32)

                to_index(i, out_shape, out_index)

                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)

                out_pos = index_to_position(out_index, out_strides)

                # Replace 'all' with explicit loop checks
                a_valid = True
                for d in range(len(a_shape)):
                    if not (0 <= a_index[d] < a_shape[d]):
                        a_valid = False
                        break

                b_valid = True
                for d in range(len(b_shape)):
                    if not (0 <= b_index[d] < b_shape[d]):
                        b_valid = False
                        break

                if a_valid and b_valid:
                    a_pos = index_to_position(a_index, a_strides)
                    b_pos = index_to_position(b_index, b_strides)
                    out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])
                    # just to make a new commit

    return njit(_zip, parallel=True)


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function."""

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        size = int(np.prod(out_shape))
        reduce_size = a_shape[reduce_dim]

        for i in prange(size):
            out_index = np.zeros(len(out_shape), dtype=np.int32)
            a_index = np.zeros(len(a_shape), dtype=np.int32)
            to_index(i, out_shape, out_index)

            for j in range(len(out_index)):
                a_index[j] = out_index[j]

            out_pos = index_to_position(out_index, out_strides)

            for j in range(reduce_size):
                a_index[reduce_dim] = j

                # Replace 'all' with explicit loop
                valid_index = True
                for d in range(len(a_shape)):
                    if not (0 <= a_index[d] < a_shape[d]):
                        valid_index = False
                        break

                if valid_index:
                    a_pos = index_to_position(a_index, a_strides)
                    out[out_pos] = fn(out[out_pos], a_storage[a_pos])

    return njit(_reduce, parallel=True)


def _tensor_matrix_multiply(
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
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    n_batches = max(
        a_shape[0] if len(a_shape) > 2 else 1, b_shape[0] if len(b_shape) > 2 else 1
    )
    row_size = a_shape[-2]
    inner_size = a_shape[-1]
    col_size = b_shape[-1]

    for batch in prange(n_batches):
        batch_offset_a = batch * a_batch_stride
        batch_offset_b = batch * b_batch_stride

        for i in range(row_size):
            for j in range(col_size):
                # Get output position
                out_pos = (
                    batch * out_strides[0]  # batch stride
                    + i * out_strides[-2]  # row stride
                    + j * out_strides[-1]  # col stride
                )

                acc = 0.0

                for k in range(inner_size):
                    a_pos = batch_offset_a + i * a_strides[-2] + k * a_strides[-1]
                    b_pos = batch_offset_b + k * b_strides[-2] + j * b_strides[-1]

                    acc += a_storage[a_pos] * b_storage[b_pos]

                out[out_pos] = acc


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
