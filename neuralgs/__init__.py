"""Learned-profile Grad--Shafranov model for Challenge 2.

The package keeps machine geometry out of the learned decoder.  Coils and plasma
current are rendered with axisymmetric Green functions; the neural network only
predicts low-dimensional flux-surface source profiles and scalar calibrations.
"""

from .config import NeuralGSConfig

__all__ = ["NeuralGSConfig"]
