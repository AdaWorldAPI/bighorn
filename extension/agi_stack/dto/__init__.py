"""
DTO Bridge — Wiring layer for 10K VSA
"""

from .affective import (
    AffectiveDTO,
    EroticaBridge,
    ArousalLevel,
    IntimacyDepth,
    BodyZone,
    RelationalMode,
    EroticFamily,
    create_affective_from_erotic_family,
)

from .location import (
    LocationDTO,
    MomentDTO,
    TrustDTO,
)

from .wire_10k import (
    Wire10K,
    DIMENSION_MAP,
    create_erotic_wire,
    wire_erotica_to_agi,
)

__all__ = [
    # Affective
    "AffectiveDTO",
    "EroticaBridge", 
    "ArousalLevel",
    "IntimacyDepth",
    "BodyZone",
    "RelationalMode",
    "EroticFamily",
    "create_affective_from_erotic_family",
    # Location
    "LocationDTO",
    "MomentDTO",
    "TrustDTO",
    # Wiring
    "Wire10K",
    "DIMENSION_MAP",
    "create_erotic_wire",
    "wire_erotica_to_agi",
]
