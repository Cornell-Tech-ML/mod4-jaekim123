"""
Minitorch Package

A minimalistic deep learning library inspired by PyTorch.

Modules:
- tensor
- tensor_data
- tensor_functions
- autodiff
- fast_ops
- fast_conv
- nn
"""

from .tensor import Tensor
from .operators import add, sub, mul, div, max, min  # Import operator functions

def is_close(a: Tensor, b: Tensor, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Element-wise comparison of two tensors to check if they are approximately equal.

    Args:
        a (Tensor): First tensor.
        b (Tensor): Second tensor.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.

    Returns:
        bool: True if all elements are approximately equal, False otherwise.
    """
    return (abs(a - b) <= (atol + rtol * abs(b))).all().item() == 1

from .testing import MathTest, MathTestVariable
from .datasets import load_dataset
from .optim import SGD, Adam  # Example optimizer imports
from .tensor_ops import apply_gradients
from .fast_conv import conv1d, conv2d
from .nn import Linear, Conv1d, CNNSentimentKim
from .tensor_data import shape, index_to_position, broadcast_index
from .tensor_functions import Function, rand
from .module import Module
from .autodiff import Context