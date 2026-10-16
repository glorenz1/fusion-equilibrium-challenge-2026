"""Measurement-to-state encoders for equilibrium inference.

The encoder package consumes the audited ``src.measurement`` interface and
predicts physical state for a separate equilibrium decoder.  Raw challenge
rows and equilibrium target columns do not belong in this package's public
inference interface.
"""

from .common_raster import (
    COMMON_RASTER_CHANNELS,
    COMMON_RASTER_VERSION,
    CommonRaster,
    build_common_raster,
)
from .common_inputs import (
    AUXILIARY_SCALAR_CHANNELS,
    D3D_PF_CURRENT_NAMES,
    D3D_PF_SCALAR_CHANNELS,
    D3D_MODEL_SPATIAL_CHANNELS,
    MODEL_SPATIAL_CHANNELS,
    SHARED_SCALAR_CHANNELS,
    SUPPORTED_ENCODER_SOURCE,
    THOMSON_TOKEN_FEATURES,
    CommonEncoderInputs,
    CommonEncoderFrameBatch,
    CommonScalarInputs,
    CommonThomsonTokens,
    build_common_encoder_inputs,
    build_common_thomson_tokens,
    stack_common_encoder_frames,
)
from .plotting import (
    DEFAULT_PLOT_CHANNELS,
    plot_common_raster,
    plot_common_raster_comparison,
)

__all__ = [
    "COMMON_RASTER_CHANNELS",
    "COMMON_RASTER_VERSION",
    "AUXILIARY_SCALAR_CHANNELS",
    "D3D_PF_CURRENT_NAMES",
    "D3D_PF_SCALAR_CHANNELS",
    "D3D_MODEL_SPATIAL_CHANNELS",
    "CommonRaster",
    "CommonEncoderInputs",
    "CommonEncoderFrameBatch",
    "CommonScalarInputs",
    "CommonThomsonTokens",
    "DEFAULT_PLOT_CHANNELS",
    "MODEL_SPATIAL_CHANNELS",
    "SHARED_SCALAR_CHANNELS",
    "SUPPORTED_ENCODER_SOURCE",
    "THOMSON_TOKEN_FEATURES",
    "build_common_encoder_inputs",
    "build_common_raster",
    "build_common_thomson_tokens",
    "stack_common_encoder_frames",
    "plot_common_raster",
    "plot_common_raster_comparison",
]
