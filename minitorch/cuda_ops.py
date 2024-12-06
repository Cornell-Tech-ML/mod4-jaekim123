# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """A decorator to enable just-in-time (JIT) compilation for device-specific operations.

    This function wraps another function and applies JIT compilation with device-specific
    optimizations (e.g., GPU acceleration). It uses `_jit` with the `device=True` flag to
    indicate that the function is intended to run on a device (e.g., CUDA-enabled GPU).

    Args:
    ----
        fn (Fn): The function to be JIT-compiled.
        **kwargs: Additional keyword arguments passed to the `_jit` function.

    Returns:
    -------
        Fn: The JIT-compiled version of the input function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Callable, **kwargs: Any) -> FakeCUDAKernel:
    """A decorator to enable just-in-time (JIT) compilation for general operations.

    This function wraps another function and applies JIT compilation using the `_jit`
    mechanism. The `kwargs` parameter allows customization of the JIT behavior, such
    as enabling specific optimizations or setting compilation parameters.

    Args:
    ----
        fn: The function to be JIT-compiled.
        **kwargs (Any): Additional keyword arguments passed to the `_jit` function
            to customize the JIT compilation process.

    Returns:
    -------
        FakeCUDAKernel: A JIT-compiled version of the input function, represented
        as a `FakeCUDAKernel` object.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Creates a function to apply an element-wise operation on two tensors.

        This method takes a binary function `fn` (operating on two floats) and
        returns a function that applies `fn` element-wise to two tensors, producing
        a new tensor as output. The operation is JIT-compiled for performance.

        Args:
        ----
        fn (Callable[[float, float], float]): A binary function that operates on
        two float inputs and returns a float output.

        Returns:
        -------
        Callable[[Tensor, Tensor], Tensor]: A function that applies the binary
        operation `fn` element-wise to two input tensors.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Creates a reduction function to apply an operation along a specified tensor dimension.

        This method takes a binary function `fn` (operating on two floats) and returns
        a function that reduces a tensor along a specified dimension using `fn`. The
        reduction is performed in parallel using CUDA for improved performance.

        Args:
        ----
        fn (Callable[[float, float], float]): A binary function that takes two floats
        and returns a float. This function defines the reduction operation (e.g.,
        summation, maximum, etc.).
        start (float, optional): The starting value for the reduction operation. Defaults to 0.0.

        Returns:
        -------
        Callable[[Tensor, int], Tensor]: A function that takes a tensor and a dimension
        index, reduces the tensor along the specified dimension using `fn`, and returns
        the resulting tensor.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs a batched matrix multiplication of two tensors.

        This method computes the matrix product of two tensors `a` and `b` with support for
        broadcasting and batched operations. The input tensors are internally reshaped to
        ensure they are 3-dimensional, where the first dimension represents the batch size.
        The multiplication is executed using a parallelized CUDA kernel for efficiency.

        Args:
        ----
            a (Tensor): The first input tensor. It must have a shape where the last dimension
                matches the second-to-last dimension of `b` for valid matrix multiplication.
            b (Tensor): The second input tensor. It must have a shape where the second-to-last
                dimension matches the last dimension of `a` for valid matrix multiplication.

        Returns:
        -------
            Tensor: The result of the matrix multiplication. The shape of the output tensor is
            determined by broadcasting the batch dimensions of `a` and `b`, followed by the
            last two dimensions representing the product of the matrices.

        Raises:
        ------
            AssertionError: If the inner dimensions of `a` and `b` are incompatible for matrix
            multiplication (i.e., `a.shape[-1] != b.shape[-2]`).

        Notes:
        -----
            - If both `a` and `b` are 2-dimensional tensors, they are temporarily reshaped into
            3-dimensional tensors with a batch size of 1 for consistent processing.
            - Broadcasting is applied to the batch dimensions of `a` and `b`.
            - The result is computed in parallel using a CUDA kernel (`tensor_matrix_multiply`).
            - If both `a` and `b` were originally 2-dimensional, the result is reshaped back
            into a 2-dimensional tensor before returning.

        """
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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:  # Boundary check to ensure we're within tensor bounds
            # Convert `i` to `out_index` in `out_shape`
            to_index(i, out_shape, out_index)

            # Broadcast `out_index` to `in_index` to get corresponding input position
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Convert multi-dimensional indices to flat positions
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)

            # Apply the function to the input element and store it in the output
            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Check boundary condition to ensure we're within bounds
        if i < out_size:
            # Convert flat index `i` to multi-dimensional `out_index` in `out_shape`
            to_index(i, out_shape, out_index)

            # Broadcast `out_index` to `a_index` and `b_index` based on shapes
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Convert multi-dimensional indices to flat positions
            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            # Apply the function to the corresponding elements in a and b and store in out
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    # Allocate shared memory for the block
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Compute the global index and the thread's position within the block
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Each thread loads one element from `a` into shared memory if within bounds
    if i < size:
        cache[pos] = a[i]
    else:
        cache[pos] = 0.0  # Fill out-of-bounds positions with 0
    cuda.syncthreads()

    # Perform parallel reduction within the block
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride and i + stride < size:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2

    # The first thread in each block writes the result to the output
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Computes a practice reduction sum of the input tensor using CUDA.

    This function performs a reduction sum of the elements in the input tensor `a`
    using a CUDA kernel. The result is stored in a new `TensorData` object, which
    is returned. The implementation is designed as a practice example for
    understanding GPU-based parallel reduction.

    Args:
    ----
        a (Tensor): The input tensor whose elements are to be summed. Must be a 1D tensor.

    Returns:
    -------
        TensorData: A tensor containing the result of the reduction sum. The output tensor
        has a shape of `(2,)` with the reduced values.

    Notes:
    -----
        - The input tensor `a` is transferred to the GPU for processing.
        - The reduction is parallelized using a CUDA kernel (`jit_sum_practice`) with
          `THREADS_PER_BLOCK` threads per block and enough blocks to cover the size of `a`.
        - The output tensor is pre-allocated on the GPU and returned with the sum result.
        - This function is intended for educational and practice purposes, demonstrating
          CUDA-based reduction operations.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # Set initial reduction value in shared memory
        cache[pos] = reduce_value

        # For each element in `out`, reduce along `reduce_dim` using `fn`
        to_index(out_pos, out_shape, out_index)
        for i in range(pos, a_shape[reduce_dim], BLOCK_DIM):
            out_index[reduce_dim] = i
            a_pos = index_to_position(out_index, a_strides)
            cache[pos] = fn(cache[pos], a_storage[a_pos])

        # Synchronize threads within the block
        cuda.syncthreads()

        # Perform parallel reduction in shared memory
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2

        # The first thread in each block writes the reduction result to `out`
        if pos == 0:
            out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    # Shared memory for a and b matrices
    shared_a = cuda.shared.array((MAX_DIMS, MAX_DIMS), numba.float32)
    shared_b = cuda.shared.array((MAX_DIMS, MAX_DIMS), numba.float32)

    # Load data into shared memory if within bounds
    if i < size and j < size:
        shared_a[i, j] = a[i * size + j]
        shared_b[i, j] = b[i * size + j]

    # Ensure all threads have loaded data
    cuda.syncthreads()

    # Only compute if within bounds
    if i < size and j < size:
        # Compute dot product for this element
        acc = 0.0
        for k in range(size):
            acc += shared_a[i, k] * shared_b[k, j]

        # Write result to global memory
        out[i * size + j] = acc


jit_mm_practice = cuda.jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Performs a practice matrix multiplication of two tensors using CUDA.

    This function computes the matrix product of two input tensors `a` and `b` using a
    CUDA kernel. The result is stored in a new `TensorData` object, which is returned.
    The implementation is designed as a practice example for understanding GPU-based
    parallel matrix multiplication.

    Args:
    ----
        a (Tensor): The first input tensor. It must have a shape `(size, size)`, where
            `size` is the number of rows and columns.
        b (Tensor): The second input tensor. It must have a shape `(size, size)`, where
            `size` is the number of rows and columns.

    Returns:
    -------
        TensorData: A tensor containing the result of the matrix multiplication. The
        output tensor has a shape `(size, size)`.

    Notes:
    -----
        - Both input tensors are expected to be square matrices of the same size.
        - The input tensors `a` and `b` are transferred to the GPU for processing.
        - The matrix multiplication is parallelized using a CUDA kernel (`jit_mm_practice`).
        - The CUDA kernel is configured with one block and `THREADS_PER_BLOCK` threads per block.
        - The output tensor is pre-allocated on the GPU and returned with the result.
        - This function is intended for educational and practice purposes, demonstrating
          CUDA-based matrix multiplication.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    batch = cuda.blockIdx.z

    # Get thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Get block indices
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # Calculate global row and column
    row = by * cuda.blockDim.y + ty
    col = bx * cuda.blockDim.x + tx

    # Define tile size
    TILE_SIZE = 32

    # Allocate shared memory for tiles
    tile_a = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)
    tile_b = cuda.shared.array((TILE_SIZE, TILE_SIZE), numba.float64)

    # Initialize accumulator
    acc = 0.0

    # Loop over tiles
    for t in range((a_shape[2] + TILE_SIZE - 1) // TILE_SIZE):
        # Load data into shared memory tiles
        if row < a_shape[1] and t * TILE_SIZE + tx < a_shape[2]:
            a_pos = (
                batch * a_batch_stride
                + row * a_strides[1]
                + (t * TILE_SIZE + tx) * a_strides[2]
            )
            tile_a[ty, tx] = a_storage[a_pos]
        else:
            tile_a[ty, tx] = 0.0

        if t * TILE_SIZE + ty < b_shape[1] and col < b_shape[2]:
            b_pos = (
                batch * b_batch_stride
                + (t * TILE_SIZE + ty) * b_strides[1]
                + col * b_strides[2]
            )
            tile_b[ty, tx] = b_storage[b_pos]
        else:
            tile_b[ty, tx] = 0.0

        # Synchronize threads
        cuda.syncthreads()

        # Compute partial dot product for this tile
        if row < out_shape[1] and col < out_shape[2]:
            for k in range(min(TILE_SIZE, a_shape[2] - t * TILE_SIZE)):
                acc += tile_a[ty, k] * tile_b[k, tx]

        # Synchronize before loading next tile
        cuda.syncthreads()

    # Write final result to global memory
    if row < out_shape[1] and col < out_shape[2]:
        out_pos = batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]
        out[out_pos] = acc


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
