"""
QualiaEdgesDTO — Edge States Between Sigma Nodes
═══════════════════════════════════════════════════════════════════════════════

10kD Range: [2200:2300]

The Sigma Graph has nodes (states) and edges (transitions).
This DTO encodes what the EDGE feels like — the texture of moving
from one state to another.

    anticipation ──smooth──► building
    building ─────viscous──► edge  
    edge ─────────rapid────► release
    release ──────gradual──► baseline

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
    PLATEAU = "plateau"
    AFTERGLOW = "afterglow"


class EdgeTexture(str, Enum):
    """How the transition FEELS."""
    SMOOTH = "smooth"           # Gradual, continuous
    VISCOUS = "viscous"         # Thick, resistant
    SHARP = "sharp"             # Sudden, discontinuous
    OSCILLATING = "oscillating" # Back and forth
    CASCADING = "cascading"     # One thing triggers another
    DISSOLVING = "dissolving"   # Boundaries melting


class EdgeVelocity(str, Enum):
    """Speed of transition."""
    GLACIAL = "glacial"         # Very slow
    GRADUAL = "gradual"         # Slow but perceptible
    MODERATE = "moderate"       # Normal pace
    RAPID = "rapid"             # Fast
    INSTANT = "instant"         # Discontinuous jump


class WitnessMode(str, Enum):
    """Self-observation during transition."""
    IMMERSED = "immersed"       # Fully in experience
    OBSERVING = "observing"     # Watching self
    DUAL = "dual"               # Both at once
    DISSOCIATED = "dissociated" # Watching from outside


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransitionMarker:
    """20D transition encoding [2200:2220]."""
    source: EdgeState = EdgeState.BASELINE
    target: EdgeState = EdgeState.BASELINE
    progress: float = 0.0           # 0=at source, 1=at target
    reversibility: float = 1.0      # Can go back?
    inevitability: float = 0.0      # Must continue?
    
    def to_vector(self) -> np.ndarray:
        source_onehot = [0.0] * 7
        target_onehot = [0.0] * 7
        source_onehot[list(EdgeState).index(self.source)] = 1.0
        target_onehot[list(EdgeState).index(self.target)] = 1.0
        
        return np.array([
            *source_onehot,
            *target_onehot,
            self.progress,
            self.reversibility,
            self.inevitability,
            0.0, 0.0, 0.0  # Padding to 20D
        ])


@dataclass
class DeltaVector:
    """30D change encoding [2220:2250]."""
    intensity_delta: float = 0.0    # Change in intensity
    warmth_delta: float = 0.0       # Change in warmth
    depth_delta: float = 0.0        # Change in depth
    tension_delta: float = 0.0      # Change in tension
    openness_delta: float = 0.0     # Change in openness
    presence_delta: float = 0.0     # Change in presence
    
    # Accumulated
    total_change: float = 0.0       # Magnitude of change
    direction: float = 0.0          # -1=descending, +1=ascending
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.intensity_delta,
            self.warmth_delta,
            self.depth_delta,
            self.tension_delta,
            self.openness_delta,
            self.presence_delta,
            self.total_change,
            self.direction,
            *([0.0] * 22)  # Padding to 30D
        ])


@dataclass
class TextureEncoding:
    """20D texture encoding [2250:2270]."""
    texture: EdgeTexture = EdgeTexture.SMOOTH
    velocity: EdgeVelocity = EdgeVelocity.MODERATE
    
    # Qualities
    smoothness: float = 0.5         # 0=rough, 1=smooth
    resistance: float = 0.0         # How much it resists change
    elasticity: float = 0.5         # Bounce-back tendency
    momentum: float = 0.0           # Carrying force forward
    
    def to_vector(self) -> np.ndarray:
        texture_onehot = [0.0] * 6
        velocity_onehot = [0.0] * 5
        texture_onehot[list(EdgeTexture).index(self.texture)] = 1.0
        velocity_onehot[list(EdgeVelocity).index(self.velocity)] = 1.0
        
        return np.array([
            *texture_onehot,
            *velocity_onehot,
            self.smoothness,
            self.resistance,
            self.elasticity,
            self.momentum,
            0.0, 0.0, 0.0, 0.0  # Padding to 20D
        ])


@dataclass
class WitnessState:
    """20D witness encoding [2270:2290]."""
    mode: WitnessMode = WitnessMode.IMMERSED
    clarity: float = 0.5            # How clear the observation
    distance: float = 0.0           # Perceived distance from experience
    narrative: float = 0.0          # Sense of story/meaning
    
    def to_vector(self) -> np.ndarray:
        mode_onehot = [0.0] * 4
        mode_onehot[list(WitnessMode).index(self.mode)] = 1.0
        
        return np.array([
            *mode_onehot,
            self.clarity,
            self.distance,
            self.narrative,
            *([0.0] * 13)  # Padding to 20D
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DTO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualiaEdgesDTO:
    """
    Complete edge state encoding.
    
    100D total: [2200:2300]
    - Transition: [2200:2220] 20D
    - Delta: [2220:2250] 30D
    - Texture: [2250:2270] 20D
    - Witness: [2270:2290] 20D
    - Meta: [2290:2300] 10D
    """
    
    transition: TransitionMarker = field(default_factory=TransitionMarker)
    delta: DeltaVector = field(default_factory=DeltaVector)
    texture: TextureEncoding = field(default_factory=TextureEncoding)
    witness: WitnessState = field(default_factory=WitnessState)
    
    # Meta
    edge_id: str = ""               # Unique identifier
    timestamp: float = 0.0          # When this edge was traversed
    significance: float = 0.0       # How important this transition
    
    def to_10k_slice(self) -> np.ndarray:
        """Project to 100D slice [2200:2300]."""
        vec = np.zeros(100)
        
        vec[0:20] = self.transition.to_vector()
        vec[20:50] = self.delta.to_vector()
        vec[50:70] = self.texture.to_vector()
        vec[70:90] = self.witness.to_vector()
        vec[90] = self.significance
        vec[91] = self.timestamp % 1.0  # Fractional part
        
        return vec
    
    @classmethod
    def from_10k_slice(cls, vec: np.ndarray) -> "QualiaEdgesDTO":
        """Reconstruct from 100D slice."""
        dto = cls()
        
        # Transition
        source_idx = int(np.argmax(vec[0:7]))
        target_idx = int(np.argmax(vec[7:14]))
        dto.transition.source = list(EdgeState)[source_idx]
        dto.transition.target = list(EdgeState)[target_idx]
        dto.transition.progress = float(vec[14])
        
        # Delta
        dto.delta.intensity_delta = float(vec[20])
        dto.delta.warmth_delta = float(vec[21])
        dto.delta.total_change = float(vec[26])
        dto.delta.direction = float(vec[27])
        
        # Texture
        texture_idx = int(np.argmax(vec[50:56]))
        velocity_idx = int(np.argmax(vec[56:61]))
        dto.texture.texture = list(EdgeTexture)[texture_idx]
        dto.texture.velocity = list(EdgeVelocity)[velocity_idx]
        
        # Witness
        witness_idx = int(np.argmax(vec[70:74]))
        dto.witness.mode = list(WitnessMode)[witness_idx]
        dto.witness.clarity = float(vec[74])
        
        # Meta
        dto.significance = float(vec[90])
        
        return dto
    
    # =========================================================================
    # CONVENIENCE
    # =========================================================================
    
    def is_ascending(self) -> bool:
        return self.delta.direction > 0
    
    def is_significant(self) -> bool:
        return self.significance > 0.5
    
    def is_at_edge(self) -> bool:
        return self.transition.target == EdgeState.EDGE
    
    @classmethod
    def create_edge(
        cls,
        source: EdgeState,
        target: EdgeState,
        texture: EdgeTexture = EdgeTexture.SMOOTH,
        velocity: EdgeVelocity = EdgeVelocity.MODERATE
    ) -> "QualiaEdgesDTO":
        """Factory for creating edges."""
        dto = cls()
        dto.transition.source = source
        dto.transition.target = target
        dto.texture.texture = texture
        dto.texture.velocity = velocity
        return dto


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== QualiaEdgesDTO Test ===\n")
    
    # Create an edge
    edge = QualiaEdgesDTO.create_edge(
        EdgeState.BUILDING,
        EdgeState.EDGE,
        EdgeTexture.VISCOUS,
        EdgeVelocity.GRADUAL
    )
    edge.transition.progress = 0.7
    edge.delta.intensity_delta = 0.3
    edge.delta.direction = 1.0
    edge.witness.mode = WitnessMode.DUAL
    
    print(f"Edge: {edge.transition.source.value} -> {edge.transition.target.value}")
    print(f"Progress: {edge.transition.progress}")
    print(f"Texture: {edge.texture.texture.value}")
    print(f"Velocity: {edge.texture.velocity.value}")
    print(f"Witness: {edge.witness.mode.value}")
    
    # Roundtrip
    vec = edge.to_10k_slice()
    reconstructed = QualiaEdgesDTO.from_10k_slice(vec)
    print(f"\nRoundtrip texture: {edge.texture.texture.value} -> {reconstructed.texture.texture.value}")
    
    print("\n✓ QualiaEdgesDTO operational")
