"""Neural network models for the project."""

from .simple_cnn import SimpleCNN
from .simple_cnn_gap import SimpleCNNGAP
from .simple_cnn_gap_bn import SimpleCNNGAPBatchNorm
from .simple_cnn_gap_bn_dropout import SimpleCNNGAPBatchNormDropout
from .wide_cnn_gap_bn import WideCNNGAPBatchNorm

__all__ = [
    "SimpleCNN",
    "SimpleCNNGAP",
    "SimpleCNNGAPBatchNorm",
    "SimpleCNNGAPBatchNormDropout",
    "WideCNNGAPBatchNorm",
]
