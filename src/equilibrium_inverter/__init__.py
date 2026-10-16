"""Geometry-conditioned Equilibrium Inverter inference.

This package is intentionally separate from the historical direct encoders.  A
tokamak row is first calibrated by :mod:`src.measurement`, then compiled into a
physical operator.  Learned parameters describe dimensionless plasma closures;
machine geometry never becomes a learned identifier.
"""

from typing import TYPE_CHECKING

from .config import EquilibriumInverterConfig

if TYPE_CHECKING:
    from .pipeline import EquilibriumInverter, EquilibriumPrediction

# Compatibility aliases for artifacts and callers created before the package
# was renamed. New code should use the Equilibrium Inverter names above.
BayesianEquilibriumConfig = EquilibriumInverterConfig

__all__ = [
    "EquilibriumInverterConfig",
    "EquilibriumInverter",
    "EquilibriumPrediction",
]


def __getattr__(name: str):
    """Keep lightweight submodules from paying the full inference import cost."""

    if name in {
        "EquilibriumInverter",
        "EquilibriumPrediction",
        "BayesianEquilibriumFramework",
    }:
        from .pipeline import EquilibriumInverter, EquilibriumPrediction

        return {
            "EquilibriumInverter": EquilibriumInverter,
            "EquilibriumPrediction": EquilibriumPrediction,
            "BayesianEquilibriumFramework": EquilibriumInverter,
        }[name]
    raise AttributeError(name)
