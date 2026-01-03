"""
DTO Bridge — Wiring layer for 10K VSA

Canonical dimension source: dimension_registry.py
"""

# Core
from .dimension_registry import (
    DIMENSION_REGISTRY,
    DimRange,
    get_range,
    get_slice,
    allocate,
    extract,
)

from .base_dto import (
    BaseDTO,
    DTORegistry,
    TOTAL_DIMENSIONS,
)

# Affective/Arousal
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

# Location/Moment
from .location import (
    LocationDTO,
    MomentDTO,
    TrustDTO,
)

# Wiring
from .wire_10k import (
    Wire10K,
    create_erotic_wire,
    wire_erotica_to_agi,
)

# Main DTOs
from .soul_dto import SoulDTO
from .felt_dto import FeltDTO
from .situation_dto import SituationDTO
from .volition_dto import VolitionDTO
from .vision_dto import VisionDTO
from .universal_dto import (
    UniversalThought,
    UniversalObserver,
    UniversalEpisode,
    UniversalTexture,
)
# Alias for backward compatibility
UniversalDTO = UniversalThought
from .moment_dto import MomentDTO as MomentDTOv2

# New DTOs (2026-01-03)
from .world_dto import WorldDTO
from .physics_dto import PhysicsDTO
from .qualia_edges_dto import QualiaEdgesDTO
from .friston_dto import FristonDTO
from .alternate_reality_dto import AlternateRealityDTO
from .media_dto import MediaDTO
from .synesthesia_dto import SynesthesiaDTO

# Thinking styles
from .thinking_style import ThinkingStyleDTO

__all__ = [
    # Registry (source of truth)
    "DIMENSION_REGISTRY",
    "DimRange",
    "get_range",
    "get_slice",
    "allocate",
    "extract",

    # Base
    "BaseDTO",
    "DTORegistry",
    "TOTAL_DIMENSIONS",

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
    "create_erotic_wire",
    "wire_erotica_to_agi",

    # Main DTOs
    "SoulDTO",
    "FeltDTO",
    "SituationDTO",
    "VolitionDTO",
    "VisionDTO",
    "UniversalDTO",

    # New DTOs
    "WorldDTO",
    "PhysicsDTO",
    "QualiaEdgesDTO",
    "FristonDTO",
    "AlternateRealityDTO",
    "MediaDTO",
    "SynesthesiaDTO",

    # Thinking
    "ThinkingStyleDTO",
]
