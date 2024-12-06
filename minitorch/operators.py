"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, TypeVar

# ## Task 0.1

T = TypeVar("T")

# Mathematical functions:


def mul(x: float, y: float) -> float:
    """Multiply two floating-point numbers."""
    return x * y


def id(x: float) -> float:
    """Return the input unchanged."""
    return x


def add(x: float, y: float) -> float:
    """Add two floating-point numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a floating-point number."""
    return -x


def lt(x: float, y: float) -> float:
    """Return 1.0 if x is less than y, else 0.0."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Return 1.0 if x is equal to y, else 0.0."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two floating-point numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two floating-point numbers are close to each other."""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Compute the sigmoid function for the input."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU (Rectified Linear Unit) function for the input."""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Compute the natural logarithm of the input."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Compute the exponential of the input."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the gradient of the natural logarithm."""
    return d / (x + EPS)


def inv(x: float) -> float:
    """Compute the inverse of the input."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the gradient of the inverse function."""
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Compute the gradient of the ReLU function."""
    return d if x > 0 else 0.0


# ## Task 0.2

# Higher-order functions:


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map function.

    Args:
    ----
        fn: Function from one float to one float.

    Returns:
    -------
        A function that takes a list of floats and returns an iterator of floats.

    """
    return lambda iterable: (fn(x) for x in iterable)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith function.

    Args:
    ----
        fn: Function from two floats to one float.

    Returns:
    -------
        A function that takes two lists of floats and returns an iterator of floats.

    """
    return lambda ls1, ls2: (fn(x, y) for x, y in zip(ls1, ls2))


def reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], float], float]:
    """Higher-order reduce function.

    Args:
    ----
        fn: Function from two floats to one float.

    Returns:
    -------
        A function that takes a list of floats and a starting float (usually 0.0)
        and returns a float.

    """

    def reducer(ls: Iterable[float], start: float) -> float:
        result = start
        for l in ls:
            result = fn(result, l)
        return result

    return reducer


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate each element in a list of floats.

    Args:
    ----
        ls: List of floats.

    Returns:
    -------
        Iterator of negated floats.

    """
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two lists of floats element-wise.

    Args:
    ----
        ls1: First list of floats.
        ls2: Second list of floats.

    Returns:
    -------
        Iterator of element-wise sums.

    """
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum a list of floats.

    Args:
    ----
        ls: List of floats.

    Returns:
    -------
        Sum of all elements in the list.

    """
    return reduce(add)(ls, 0.0)


def prod(ls: Iterable[float]) -> float:
    """Multiply all elements in a list of floats.

    Args:
    ----
        ls: List of floats.

    Returns:
    -------
        Product of all elements in the list.

    """
    return reduce(mul)(ls, 1.0)
