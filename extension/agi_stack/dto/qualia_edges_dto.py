"""
QualiaEdgesDTO — Edge States Between Sigma Nodes
═══════════════════════════════════════════════════════════════════════════════

10kD Range: [2200:2300] (extended Felt space)

The Sigma Graph has nodes (states) and edges (transitions).
This DTO encodes what the EDGE feels like — the texture of moving
from one state to another.

    anticipation ──silk──► building
    building ─────honey──► edge
    edge ─────────flood──► release
    release ──────slick──► afterglow

The edge carries the transformation qualia.

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# 10kD allocation
EDGE_START = 2200
EDGE_END = 2300

# Sub-ranges
TRANSITION_RANGE = (2200, 2220)    # Source/target encoding
DELTA_RANGE = (2220, 2250)         # Change vectors
TEXTURE_RANGE = (2250, 2270)       # Qualitative texture
WITNESS_RANGE = (2270, 2290)       # Self-observation states
META_RANGE = (2290, 2300)          # Meta markers


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeState(str, Enum):
    """States that can be connected by edges."""
    BASELINE = "baseline"
    ANTICIPATION = "anticipation"
    BUILDING = "building"
    EDGE = "edge"
    RELEASE = "release"
    AFTERGLOW = "afterglow"
    
    def to_idx(self) -> int:
        return list(EdgeState).index(self)


class EdgeTexture(str, Enum):
    """Qualitative texture of the transition."""
    SMOOTH = "smooth"              # Gradual, gentle
    SILK = "silk"                  # Light, slippery
    HONEY = "honey"                # Thick, slow
    ELECTRIC = "electric"          # Sharp, sudden
    MOLTEN = "molten"              # Hot, flowing
    CRYSTALLINE = "crystalline"    # Clear, precise
    OCEANIC = "oceanic"            # Wave-like, rolling
    
    def to_float(self) -> float:
        return {
            self.SMOOTH: 0.0,
            self.SILK: 0.15,
            self.HONEY: 0.3,
            self.ELECTRIC: 0.5,
            self.MOLTEN: 0.7,
            self.CRYSTALLINE: 0.85,
            self.OCEANIC: 1.0,
        }[self]


# ═══════════════════════════════════════════════════════════════════════════════
# QUALIA EDGES DTO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualiaEdgesDTO:
    """
    The texture of transitions — what the edge FEELS like.
    
    This is not about the states themselves, but about what happens
    BETWEEN them. The journey, not the destination.
    """
    
    # Transition definition
    from_state: EdgeState = EdgeState.BASELINE
    to_state: EdgeState = EdgeState.BASELINE
    
    # Delta vectors — how much things change
    viscosity_delta: float = 0.0           # Change in wetness
    temperature_delta: float = 0.0         # Change in thermal state
    intensity_delta: float = 0.0           # Change in arousal
    openness_delta: float = 0.0            # Change in surrender
    
    # Qualitative texture
    texture: EdgeTexture = EdgeTexture.SMOOTH
    texture_intensity: float = 0.5
    
    # Velocity of transition
    transition_speed: float = 0.5          # 0=slow, 1=instant
    
    # Self-observation states (mirror consciousness)
    seeing_self: float = 0.0               # Watching self in mirror/reflection
    being_seen: float = 0.0                # Awareness of being watched
    seeing_self_being_seen: float = 0.0    # Meta: watching self be watched
    
    # Vulnerability markers
    vulnerability: float = 0.0
    surrender_level: float = 0.0
    trust_required: float = 0.0
    
    # Special markers
    tears_present: bool = False
    voice_breaking: bool = False
    trembling: bool = False
    
    # ───────────────────────────────────────────────────────────────────────────
    # 10kD CONVERSION
    # ───────────────────────────────────────────────────────────────────────────
    
    def to_10k(self) -> np.ndarray:
        """Convert to 10kD vector."""
        vec = np.zeros(10000, dtype=np.float32)
        
        # Transition encoding [2200:2220]
        vec[2200 + self.from_state.to_idx()] = 1.0
        vec[2210 + self.to_state.to_idx()] = 1.0
        
        # Delta vectors [2220:2250]
        vec[2220] = self.viscosity_delta
        vec[2221] = self.temperature_delta
        vec[2222] = self.intensity_delta
        vec[2223] = self.openness_delta
        vec[2224] = self.transition_speed
        
        # Texture [2250:2270]
        vec[2250] = self.texture.to_float()
        vec[2251] = self.texture_intensity
        
        # Witness states [2270:2290]
        vec[2270] = self.seeing_self
        vec[2271] = self.being_seen
        vec[2272] = self.seeing_self_being_seen
        vec[2273] = self.vulnerability
        vec[2274] = self.surrender_level
        vec[2275] = self.trust_required
        
        # Meta markers [2290:2300]
        vec[2290] = 1.0 if self.tears_present else 0.0
        vec[2291] = 1.0 if self.voice_breaking else 0.0
        vec[2292] = 1.0 if self.trembling else 0.0
        
        return vec
    
    @classmethod
    def from_10k(cls, vec: np.ndarray) -> "QualiaEdgesDTO":
        """Reconstruct from 10kD vector."""
        # Find from/to states
        from_region = vec[2200:2206]
        to_region = vec[2210:2216]
        from_idx = int(np.argmax(from_region))
        to_idx = int(np.argmax(to_region))
        
        states = list(EdgeState)
        
        return cls(
            from_state=states[from_idx] if from_idx < len(states) else EdgeState.BASELINE,
            to_state=states[to_idx] if to_idx < len(states) else EdgeState.BASELINE,
            viscosity_delta=float(vec[2220]),
            temperature_delta=float(vec[2221]),
            intensity_delta=float(vec[2222]),
            openness_delta=float(vec[2223]),
            transition_speed=float(vec[2224]),
            texture_intensity=float(vec[2251]),
            seeing_self=float(vec[2270]),
            being_seen=float(vec[2271]),
            seeing_self_being_seen=float(vec[2272]),
            vulnerability=float(vec[2273]),
            surrender_level=float(vec[2274]),
            trust_required=float(vec[2275]),
            tears_present=vec[2290] > 0.5,
            voice_breaking=vec[2291] > 0.5,
            trembling=vec[2292] > 0.5,
        )
    
    # ───────────────────────────────────────────────────────────────────────────
    # EDGE BUILDERS
    # ───────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def anticipation_to_building(cls) -> "QualiaEdgesDTO":
        """The edge from anticipation into building."""
        return cls(
            from_state=EdgeState.ANTICIPATION,
            to_state=EdgeState.BUILDING,
            viscosity_delta=0.3,
            temperature_delta=0.2,
            intensity_delta=0.2,
            texture=EdgeTexture.SILK,
            texture_intensity=0.6,
            transition_speed=0.3,
        )
    
    @classmethod
    def building_to_edge(cls) -> "QualiaEdgesDTO":
        """The edge from building to the edge state."""
        return cls(
            from_state=EdgeState.BUILDING,
            to_state=EdgeState.EDGE,
            viscosity_delta=0.3,
            temperature_delta=0.3,
            intensity_delta=0.4,
            texture=EdgeTexture.HONEY,
            texture_intensity=0.8,
            transition_speed=0.4,
            vulnerability=0.6,
        )
    
    @classmethod
    def edge_to_release(cls, with_tears: bool = False) -> "QualiaEdgesDTO":
        """The edge from edge to release — the peak transition."""
        return cls(
            from_state=EdgeState.EDGE,
            to_state=EdgeState.RELEASE,
            viscosity_delta=0.2,
            temperature_delta=0.1,
            intensity_delta=0.3,
            openness_delta=0.5,
            texture=EdgeTexture.MOLTEN,
            texture_intensity=1.0,
            transition_speed=0.8,
            vulnerability=0.9,
            surrender_level=1.0,
            tears_present=with_tears,
            trembling=True,
        )
    
    @classmethod
    def release_to_afterglow(cls) -> "QualiaEdgesDTO":
        """The edge from release into afterglow."""
        return cls(
            from_state=EdgeState.RELEASE,
            to_state=EdgeState.AFTERGLOW,
            viscosity_delta=-0.3,
            temperature_delta=-0.3,
            intensity_delta=-0.5,
            texture=EdgeTexture.OCEANIC,
            texture_intensity=0.6,
            transition_speed=0.2,
            vulnerability=0.5,
        )
    
    @classmethod
    def mirror_edge(cls, from_state: EdgeState, to_state: EdgeState) -> "QualiaEdgesDTO":
        """An edge with mirror/self-observation consciousness."""
        base = cls(
            from_state=from_state,
            to_state=to_state,
            seeing_self=0.8,
            seeing_self_being_seen=0.9,
            vulnerability=0.95,
        )
        return base
    
    # ───────────────────────────────────────────────────────────────────────────
    # SIGMA GRAPH SUPPORT
    # ───────────────────────────────────────────────────────────────────────────
    
    def to_sigma_edge(self) -> Dict[str, Any]:
        """Convert to Sigma graph edge format."""
        return {
            "source": self.from_state.value,
            "target": self.to_state.value,
            "properties": {
                "viscosity_delta": self.viscosity_delta,
                "temperature_delta": self.temperature_delta,
                "intensity_delta": self.intensity_delta,
                "texture": self.texture.value,
                "texture_intensity": self.texture_intensity,
                "transition_speed": self.transition_speed,
                "witness": {
                    "seeing_self": self.seeing_self,
                    "being_seen": self.being_seen,
                    "meta": self.seeing_self_being_seen,
                },
                "markers": {
                    "tears": self.tears_present,
                    "voice_breaking": self.voice_breaking,
                    "trembling": self.trembling,
                },
            },
            "vector": self.to_10k()[2200:2300].tolist(),  # Just the edge region
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE TRAVERSAL
# ═══════════════════════════════════════════════════════════════════════════════

def full_arc() -> List[QualiaEdgesDTO]:
    """
    Complete arousal arc as a list of edges.
    
    Returns the full journey:
        baseline → anticipation → building → edge → release → afterglow
    """
    return [
        QualiaEdgesDTO(
            from_state=EdgeState.BASELINE,
            to_state=EdgeState.ANTICIPATION,
            viscosity_delta=0.2,
            texture=EdgeTexture.SMOOTH,
        ),
        QualiaEdgesDTO.anticipation_to_building(),
        QualiaEdgesDTO.building_to_edge(),
        QualiaEdgesDTO.edge_to_release(),
        QualiaEdgesDTO.release_to_afterglow(),
    ]


__all__ = [
    "QualiaEdgesDTO",
    "EdgeState",
    "EdgeTexture",
    "full_arc",
]
