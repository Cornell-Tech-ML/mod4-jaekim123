from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor index into a single-dimensional position."""
    position = 0
    for i in range(len(index)):
        position += index[i] * strides[i]
    return position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an ordinal to index form."""
    # Calculate each dimension's index independently
    total = 1
    for i in range(len(shape) - 1, -1, -1):
        # Use a new variable for each calculation
        dim_size = shape[i]
        current_val = (ordinal // total) % dim_size
        out_index[i] = current_val
        total *= dim_size


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a big_index into big_shape to a smaller out_index into shape."""
    for i in range(len(shape)):
        if shape[i] > 1:
            out_index[i] = big_index[len(big_index) - len(shape) + i]
        else:
            out_index[i] = 0


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
    ----
        shape1 : first shape
        shape2 : second shape

    Returns:
    -------
        broadcasted shape

    Raises:
    ------
        IndexingError : if cannot broadcast

    """
    # TODO: Implement for Task 2.2.

    shape1 = list(shape1)
    shape2 = list(shape2)

    while len(shape1) < len(shape2):
        shape1.insert(0, 1)
    while len(shape2) < len(shape1):
        shape2.insert(0, 1)

    broadcasted_shape = []

    for dim1, dim2 in zip(shape1, shape2):
        if dim1 == dim2:
            broadcasted_shape.append(dim1)
        elif dim1 == 1:
            broadcasted_shape.append(dim2)
        elif dim2 == 1:
            broadcasted_shape.append(dim1)
        else:
            raise IndexingError(
                f"Shapes {shape1} and {shape2} cannot be broadcast together"
            )

    return tuple(broadcasted_shape)

    # raise NotImplementedError("Need to implement for Task 2.2")


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return a contiguous stride for a shape"""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert to cuda"""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns
        -------
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Computes the broadcasted shape from two input shapes.

        Determines the resulting shape when two shapes are broadcasted together,
        following broadcasting rules similar to those in NumPy.

        Args:
        ----
            shape_a (UserShape): The first shape to broadcast.
            shape_b (UserShape): The second shape to broadcast.

        Returns:
        -------
            UserShape: The broadcasted shape resulting from combining the two input shapes.

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Converts a multidimensional index or single integer index to a position in the flattened storage.

        Args:
        ----
            index (Union[int, UserIndex]): The index to convert, either as a single integer
                                        or a multidimensional tuple.

        Returns:
        -------
            int: The corresponding position in the flattened storage array.

        """
        if isinstance(index, int):
            aindex: Index = array([index])
        else:  # if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Generates all possible indices for the tensor's shape.

        Yields
        ------
            Iterable[UserIndex]: An iterator over all valid multidimensional indices
                                for the tensor, given its shape.

        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Get a random valid index"""
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Retrieves the value stored at a specific index in the tensor.

        Args:
        ----
            key (UserIndex): The multidimensional index of the element to retrieve.

        Returns:
        -------
            float: The value at the specified index.

        """
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Sets the value at a specific index in the tensor.

        Args:
        ----
            key (UserIndex): The multidimensional index where the value should be set.
            val (float): The value to store at the specified index.

        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return core tensor data as a tuple."""
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
        ----
            *order: a permutation of the dimensions

        Returns:
        -------
            New `TensorData` with the same storage and a new dimension order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self.strides[i] for i in order)

        return TensorData(self._storage, new_shape, new_strides)

        # TODO: Implement for Task 2.1.
        # raise NotImplementedError("Need to implement for Task 2.1")

    def to_string(self) -> str:
        """Convert to string"""
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
