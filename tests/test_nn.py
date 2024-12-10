import pytest
from hypothesis import given, settings
import numpy as np

import minitorch
from minitorch import Tensor
from .tensor_strategies import tensors
from minitorch.nn import max  # Ensure that the max function is imported

from .strategies import assert_close


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
@settings(max_examples=50)
def test_max(t: Tensor) -> None:
    """
    Test the `minitorch.nn.max` function to ensure it correctly computes the maximum
    values along a specified dimension.

    Args:
    ----
        t (Tensor): The input tensor for the max operation.
    """
    # Choose a random dimension to test
    dim = np.random.randint(0, t.dim())

    # Compute expected max using NumPy
    t_np = t.to_numpy()
    expected = np.max(t_np, axis=dim)

    # Compute max using minitorch
    out = max(t, dim)

    # Ensure the output shape matches the expected shape
    assert (
        out.shape == expected.shape
    ), f"Output shape {out.shape} does not match expected shape {expected.shape}"

    # Compare the actual output with the expected output
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)


@pytest.mark.task4_4
def test_max_controlled() -> None:
    """
    Test the `minitorch.nn.max` function with a controlled input to ensure correctness.
    """
    t = minitorch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, 3.0, 3.0, 1.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    dim = 2  # Max along the last dimension

    # Expected result
    expected = np.array([[4.0, 4.0, 3.0], [0.0, 0.0, 0.0]])

    # Compute max using minitorch
    out = max(t, dim)

    # Compare the output
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-5, atol=1e-5)
