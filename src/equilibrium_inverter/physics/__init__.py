"""Differentiable Grad--Shafranov forward components.

The public objects are imported lazily because geometry compilation needs the
harmonic builder while the forward renderer, in turn, consumes the compiled
geometry.  Keeping package initialization side-effect free avoids turning that
valid dependency graph into a Python import cycle.
"""

from typing import Any

__all__ = ["EquilibriumForwardModel", "ForwardResult", "build_harmonic_basis"]


def __getattr__(name: str) -> Any:
    if name == "build_harmonic_basis":
        from .harmonics import build_harmonic_basis

        return build_harmonic_basis
    if name in {"EquilibriumForwardModel", "ForwardResult"}:
        from .forward import EquilibriumForwardModel, ForwardResult

        return {
            "EquilibriumForwardModel": EquilibriumForwardModel,
            "ForwardResult": ForwardResult,
        }[name]
    raise AttributeError(name)
