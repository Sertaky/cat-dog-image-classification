"""Neural network models for the project."""

from .simple_cnn import SimpleCNN
from .simple_cnn_gap import SimpleCNNGAP
from .simple_cnn_gap_bn import SimpleCNNGAPBatchNorm
from .simple_cnn_gap_bn_dropout import SimpleCNNGAPBatchNormDropout

__all__ = [
    "SimpleCNN",
    "SimpleCNNGAP",
    "SimpleCNNGAPBatchNorm",
    "SimpleCNNGAPBatchNormDropout",
]
