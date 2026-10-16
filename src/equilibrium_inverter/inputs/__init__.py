"""Adapters from calibrated challenge measurements into Bayesian model tensors."""

from .shot import PreparedShot, TorchShot, prepare_shot, to_torch_shot

__all__ = ["PreparedShot", "TorchShot", "prepare_shot", "to_torch_shot"]
