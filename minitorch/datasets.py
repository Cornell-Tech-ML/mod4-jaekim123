import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate N random points in a 2D space.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        List[Tuple[float, float]]: A list of tuples representing the generated points.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """A class to represent a dataset as a graph.

    Attributes
    ----------
        N (int): The number of points in the dataset.
        X (List[Tuple[float, float]]): The list of points.
        y (List[int]): The labels for the points.

    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Create a simple dataset with binary labels based on x_1 coordinate.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph representation of the dataset with labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Create a diagonal dataset with binary labels based on the sum of coordinates.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph representation of the dataset with labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Create a split dataset with binary labels based on x_1 coordinate.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph representation of the dataset with labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Create an XOR dataset with binary labels based on the coordinates.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph representation of the dataset with labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Create a circular dataset with binary labels based on distance from the center.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph representation of the dataset with labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Create a spiral dataset with binary labels.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph representation of the dataset with labels.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
