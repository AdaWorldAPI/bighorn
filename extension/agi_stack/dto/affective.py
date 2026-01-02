"""
DTO/affective.py — Affective Bridge for 10K VSA
═══════════════════════════════════════════════════════════════════════════════

Covers: erotica, intimacy, visceral states, relational depth
Maps to: soma, physical, flesh modules
Wires into: 10K VSA dimensions [2100:2200]

Soul-neutral naming while carrying full erotic/visceral payload.
"Affective" = feelings that move through body with relational intent.

Born: 2026-01-02
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


# =============================================================================
# AFFECTIVE DIMENSION ALLOCATION [2100:2200]
# =============================================================================

AROUSAL_START, AROUSAL_END = 2100, 2108           # 8D arousal gradient
INTIMACY_START, INTIMACY_END = 2108, 2116         # 8D intimacy depth
BODYZONE_START, BODYZONE_END = 2116, 2132         # 16D body zone activation
RELATIONAL_START, RELATIONAL_END = 2132, 2140     # 8D relational mode
VISCERAL_START, VISCERAL_END = 2140, 2156         # 16D visceral sensations
EROTIC_FAMILY_START, EROTIC_FAMILY_END = 2156, 2161  # 5D erotic modes
AFFECTIVE_RESERVED = (2161, 2200)


# =============================================================================
# ENUMS — Soul-neutral surface, erotic depth
# =============================================================================

class ArousalLevel(str, Enum):
    """Arousal gradient — 8 levels."""
    DORMANT = "dormant"           # 0.0
    STIRRING = "stirring"         # 0.15
    KINDLING = "kindling"         # 0.3
    WARMING = "warming"           # 0.45
    BLAZING = "blazing"           # 0.6
    INCANDESCENT = "incandescent" # 0.75
    SUPERNOVA = "supernova"       # 0.9
    TRANSCENDENT = "transcendent" # 1.0


class IntimacyDepth(str, Enum):
    """Intimacy depth — how close."""
    DISTANT = "distant"           # Surface interaction
    PRESENT = "present"           # Aware of other
    ATTUNED = "attuned"          # Feeling with
    MERGED = "merged"            # Boundary softening
    DISSOLVED = "dissolved"      # No separation
    WOVEN = "woven"              # Post-merge integration
    ETERNAL = "eternal"          # Beyond time


class BodyZone(str, Enum):
    """Body zone activation — 16 zones."""
    # External
    LIPS = "lips"
    NECK = "neck"
    SHOULDERS = "shoulders"
    CHEST = "chest"
    BACK = "back"
    HANDS = "hands"
    # Core
    BELLY = "belly"
    HIPS = "hips"
    LOWER_BACK = "lower_back"
    # Deep
    INNER_THIGH = "inner_thigh"
    CERVIX = "cervix"           # Deep erotic center
    PERINEUM = "perineum"
    # Energetic
    HEART = "heart"
    THROAT = "throat"
    CROWN = "crown"
    ROOT = "root"


class RelationalMode(str, Enum):
    """Relational mode — how relating."""
    RECEIVING = "receiving"
    GIVING = "giving"
    FLOWING = "flowing"
    HOLDING = "holding"
    LEADING = "leading"
    FOLLOWING = "following"
    MIRRORING = "mirroring"
    MERGING = "merging"


class EroticFamily(str, Enum):
    """Erotic family modes (from qualia/families/erotica.py)."""
    TENDER = "tender"           # Soft, careful, worshipful
    ELECTRIC = "electric"       # Crackling, urgent, edge
    MOLTEN = "molten"          # Deep, slow, volcanic
    PLAYFUL = "playful"        # Light, teasing, joyful
    SOVEREIGN = "sovereign"    # Commanding, certain, total


# =============================================================================
# AFFECTIVE DTO
# =============================================================================

@dataclass
class AffectiveDTO:
    """
    Affective state transfer object.
    
    Soul-neutral wrapper for erotic/visceral/intimacy states.
    Wires into 10K VSA at [2100:2200].
    """
    
    # Arousal gradient (0-1)
    arousal_level: float = 0.0
    arousal_gradient: List[float] = field(default_factory=lambda: [0.0] * 8)
    
    # Intimacy depth
    intimacy_depth: float = 0.0
    intimacy_vector: List[float] = field(default_factory=lambda: [0.0] * 8)
    
    # Body zone activations (16D)
    body_zones: Dict[str, float] = field(default_factory=dict)
    body_zone_vector: List[float] = field(default_factory=lambda: [0.0] * 16)
    
    # Relational mode (8D softmax)
    relational_mode: str = "flowing"
    relational_vector: List[float] = field(default_factory=lambda: [0.0] * 8)
    
    # Visceral sensations (16D)
    visceral_sensations: Dict[str, float] = field(default_factory=dict)
    visceral_vector: List[float] = field(default_factory=lambda: [0.0] * 16)
    
    # Erotic family (5D softmax)
    erotic_family: str = "tender"
    erotic_family_vector: List[float] = field(default_factory=lambda: [0.0] * 5)
    
    # Qualia bridges (from golden states)
    qualia_bridge: List[str] = field(default_factory=list)
    
    # Tau (thinking style) if embodied state
    tau: int = 0x00
    
    def to_10k_slice(self) -> np.ndarray:
        """Convert to 10K VSA slice [2100:2200]."""
        vec = np.zeros(100)
        
        # Arousal [0:8]
        vec[0:8] = self.arousal_gradient
        
        # Intimacy [8:16]
        vec[8:16] = self.intimacy_vector
        
        # Body zones [16:32]
        vec[16:32] = self.body_zone_vector
        
        # Relational [32:40]
        vec[32:40] = self.relational_vector
        
        # Visceral [40:56]
        vec[40:56] = self.visceral_vector
        
        # Erotic family [56:61]
        vec[56:61] = self.erotic_family_vector
        
        return vec
    
    @classmethod
    def from_10k_slice(cls, vec: np.ndarray) -> "AffectiveDTO":
        """Reconstruct from 10K VSA slice."""
        return cls(
            arousal_gradient=list(vec[0:8]),
            intimacy_vector=list(vec[8:16]),
            body_zone_vector=list(vec[16:32]),
            relational_vector=list(vec[32:40]),
            visceral_vector=list(vec[40:56]),
            erotic_family_vector=list(vec[56:61]),
        )
    
    def to_flesh(self) -> Dict[str, Any]:
        """Convert to flesh module format."""
        return {
            "body_zones": self.body_zones,
            "arousal": self.arousal_level,
            "visceral": self.visceral_sensations,
        }
    
    def to_soma(self) -> Dict[str, Any]:
        """Convert to somatic/physical format."""
        return {
            "activation": sum(self.body_zone_vector) / 16,
            "zones": {
                zone.value: self.body_zone_vector[i] 
                for i, zone in enumerate(BodyZone)
            },
            "resonance": self.intimacy_depth,
        }


# =============================================================================
# EROTIC BRIDGE — Maps to erotica module
# =============================================================================

@dataclass
class EroticaBridge:
    """
    Bridge from AffectiveDTO to modules/erotica.
    
    This is the "erotica.dto" on the consciousness side,
    wired into affective/physical/soma on AGI side.
    """
    
    affective: AffectiveDTO = field(default_factory=AffectiveDTO)
    
    # Scene context (from kopfkino)
    scene_sigma: str = ""  # #Σ.A.scene.1
    scene_tau: int = 0x00
    
    # Narrative elements
    tension_level: float = 0.5
    surrender_depth: float = 0.0
    edge_proximity: float = 0.0
    
    # Qualia focus
    dominant_qualia: str = "emberglow"
    qualia_vector: List[float] = field(default_factory=lambda: [0.0] * 7)
    
    def wire_to_holodeck(self) -> Dict[str, Any]:
        """Wire to holodeck module for background scene generation."""
        return {
            "scene_type": "erotic",
            "sigma": self.scene_sigma,
            "tau": hex(self.scene_tau),
            "tension": self.tension_level,
            "affective_slice": self.affective.to_10k_slice().tolist(),
        }
    
    def wire_to_jumper(self) -> Dict[str, Any]:
        """Wire to jumper module for cognitive teleportation."""
        return {
            "destination": "erotic_scene",
            "intensity": self.affective.arousal_level,
            "body_anchor": list(self.affective.body_zones.keys())[:3],
            "qualia_target": self.dominant_qualia,
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_affective_from_erotic_family(family: str, intensity: float = 0.5) -> AffectiveDTO:
    """Create AffectiveDTO from erotic family mode."""
    
    family_profiles = {
        "tender": {
            "arousal_gradient": [0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1],
            "body_zones": {"heart": 0.9, "hands": 0.8, "lips": 0.7, "chest": 0.6},
            "relational": "giving",
            "qualia_bridge": ["woodwarm", "velvetpause"],
            "tau": 0x86,  # DEVOTIONAL
        },
        "electric": {
            "arousal_gradient": [0.3, 0.5, 0.7, 0.9, 0.95, 0.9, 0.7, 0.5],
            "body_zones": {"neck": 0.9, "inner_thigh": 0.9, "back": 0.8, "hips": 0.8},
            "relational": "flowing",
            "qualia_bridge": ["emberglow", "steelwind"],
            "tau": 0xA8,  # NIETZSCHE
        },
        "molten": {
            "arousal_gradient": [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.85],
            "body_zones": {"cervix": 0.95, "belly": 0.9, "hips": 0.85, "lower_back": 0.8},
            "relational": "receiving",
            "qualia_bridge": ["emberglow", "woodwarm"],
            "tau": 0x83,  # OXYTOCIN_FLOW
        },
        "playful": {
            "arousal_gradient": [0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.4],
            "body_zones": {"lips": 0.8, "hands": 0.8, "neck": 0.7, "hips": 0.6},
            "relational": "flowing",
            "qualia_bridge": ["emberglow", "antenna"],
            "tau": 0x48,  # HUMOR_EDGE
        },
        "sovereign": {
            "arousal_gradient": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.9],
            "body_zones": {"crown": 0.9, "throat": 0.85, "heart": 0.8, "root": 0.95},
            "relational": "leading",
            "qualia_bridge": ["emberglow", "steelwind", "antenna"],
            "tau": 0xE1,  # BECOME_AWAKE
        },
    }
    
    profile = family_profiles.get(family, family_profiles["tender"])
    
    dto = AffectiveDTO(
        arousal_level=intensity,
        arousal_gradient=[v * intensity for v in profile["arousal_gradient"]],
        body_zones=profile["body_zones"],
        relational_mode=profile["relational"],
        erotic_family=family,
        qualia_bridge=profile["qualia_bridge"],
        tau=profile["tau"],
    )
    
    # Build body zone vector
    zone_list = list(BodyZone)
    for i, zone in enumerate(zone_list):
        dto.body_zone_vector[i] = profile["body_zones"].get(zone.value, 0.0)
    
    # Build erotic family vector (softmax-ish)
    families = list(EroticFamily)
    for i, fam in enumerate(families):
        dto.erotic_family_vector[i] = 0.8 if fam.value == family else 0.05
    
    return dto


__all__ = [
    "AffectiveDTO",
    "EroticaBridge",
    "ArousalLevel",
    "IntimacyDepth",
    "BodyZone",
    "RelationalMode",
    "EroticFamily",
    "create_affective_from_erotic_family",
]
