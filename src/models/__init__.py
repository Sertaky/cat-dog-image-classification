"""Neural network models for the project."""

from .simple_cnn import SimpleCNN
from .simple_cnn_gap import SimpleCNNGAP
from .simple_cnn_gap_bn import SimpleCNNGAPBatchNorm

__all__ = ["SimpleCNN", "SimpleCNNGAP", "SimpleCNNGAPBatchNorm"]
