"""Geometry-native physical equilibrium decoding.

The public inference decoder consumes physical state predicted by
``FEC1.encoder`` and deterministic active-coil fields derived from
``src.measurement``. Target-dependent fitting lives in :mod:`.oracle` and is
only for offline representation-ceiling audits.
"""

from .basis import (
    GreenModeBasis,
    HarmonicBasis,
    build_harmonic_basis,
    build_zero_net_green_basis,
)
from .interface import (
    DecoderFrameInputs,
    decoder_frame_from_encoder,
    spatial_decoder_state_from_encoder,
)
from .physical import (
    DecodedEquilibrium,
    DecoderState,
    PhysicalDecoderBasis,
    SpatialCurrentDecoderBasis,
    SpatialDecoderState,
    build_spatial_current_decoder_basis,
    decode_equilibrium,
    decode_spatial_current,
)

__all__ = [
    "DecodedEquilibrium",
    "DecoderFrameInputs",
    "DecoderState",
    "GreenModeBasis",
    "HarmonicBasis",
    "PhysicalDecoderBasis",
    "SpatialCurrentDecoderBasis",
    "SpatialDecoderState",
    "build_harmonic_basis",
    "build_spatial_current_decoder_basis",
    "build_zero_net_green_basis",
    "decode_equilibrium",
    "decode_spatial_current",
    "decoder_frame_from_encoder",
    "spatial_decoder_state_from_encoder",
]
