"""Machine-independent magnetic topology hypotheses."""

from __future__ import annotations

from enum import IntEnum


class Topology(IntEnum):
    """Connectivity class of the last closed flux surface."""

    LIMITED = 0
    LOWER_SINGLE_NULL = 1
    UPPER_SINGLE_NULL = 2
    DOUBLE_NULL = 3


TOPOLOGY_ORDER = (
    Topology.LIMITED,
    Topology.LOWER_SINGLE_NULL,
    Topology.UPPER_SINGLE_NULL,
    Topology.DOUBLE_NULL,
)
