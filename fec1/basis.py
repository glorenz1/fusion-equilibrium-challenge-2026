"""Orthonormal DCT basis used by the retained correction model."""
from __future__ import annotations
import math
import torch


def dct_residual_basis(
    order: int,
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Orthonormal low-frequency 2-D DCT basis without the constant mode."""
    coordinate = torch.arange(size, device=device, dtype=dtype) + 0.5
    one_d = []
    for frequency in range(order):
        value = torch.cos(math.pi * frequency * coordinate / size)
        value = value / value.square().sum().sqrt().clamp_min(1.0e-8)
        one_d.append(value)
    modes = []
    for kz in range(order):
        for kr in range(order):
            if kz == 0 and kr == 0:
                continue
            modes.append(one_d[kz][:, None] * one_d[kr][None, :])
    return torch.stack(modes)
