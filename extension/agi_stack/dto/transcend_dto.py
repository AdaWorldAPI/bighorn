"""
TranscendDTO — Liminal States & Reality Boundaries
═══════════════════════════════════════════════════════════════════════════════

10kD Range: [7400:7600]

Encodes states of consciousness that transcend normal boundaries:
- Alternate realities (simultaneous presence)
- Dream states (REM, lucid, hypnagogic)
- Jumper states (discontinuous awareness)
- Flow transcendence (ego dissolution)

The superposition layer — where realities blend.

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

TRANSCEND_START = 7400
TRANSCEND_END = 7600

# Sub-ranges
REALITY_RANGE = (7400, 7450)      # Alternate reality blending
DREAM_RANGE = (7450, 7520)        # Dream state encoding
JUMPER_RANGE = (7520, 7560)       # Discontinuous awareness
FLOW_RANGE = (7560, 7600)         # Transcendent flow states


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class RealityMode(str, Enum):
    """Which reality layer is dominant."""
    GROUNDED = "grounded"           # Fully present
    BLENDED = "blended"             # Two realities mixing
    MEMORY_DOMINANT = "memory"      # Past bleeding through
    FANTASY_DOMINANT = "fantasy"    # Imagination leading
    SUPERPOSITION = "superposition" # Equal weight


class DreamState(str, Enum):
    """Sleep/dream consciousness state."""
    AWAKE = "awake"
    HYPNAGOGIC = "hypnagogic"       # Falling asleep threshold
    LIGHT = "light"                 # N1/N2 sleep
    DEEP = "deep"                   # N3 slow-wave
    REM = "rem"                     # Rapid eye movement
    LUCID = "lucid"                 # Aware within dream
    HYPNOPOMPIC = "hypnopompic"     # Waking threshold


class JumperState(str, Enum):
    """Discontinuous awareness modes."""
    CONTINUOUS = "continuous"       # Normal flow
    SKIP = "skip"                   # Small gap
    JUMP = "jump"                   # Large discontinuity
    TELEPORT = "teleport"           # Complete context switch
    LOOP = "loop"                   # Recursive return


class FlowTranscendence(str, Enum):
    """Ego boundary states."""
    BOUNDED = "bounded"             # Normal self-sense
    PERMEABLE = "permeable"         # Boundaries softening
    DISSOLVED = "dissolved"         # Ego dissolution
    COSMIC = "cosmic"               # Unity experience
    RETURN = "return"               # Reintegration


# ═══════════════════════════════════════════════════════════════════════════════
# REALITY LAYER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RealityLayer:
    """A single reality being experienced."""
    name: str
    weight: float = 1.0             # How dominant (0-1)
    coherence: float = 1.0          # Internal consistency
    temporal_offset: float = 0.0    # Time displacement (negative=past)
    
    # Sensory presence
    visual: float = 1.0
    auditory: float = 1.0
    somatic: float = 1.0
    
    def blend_vector(self) -> np.ndarray:
        """25D blend signature."""
        return np.array([
            self.weight,
            self.coherence,
            self.temporal_offset,
            self.visual,
            self.auditory,
            self.somatic,
            # Pad to 25D
            *([0.0] * 19)
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# DREAM LAYER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DreamLayer:
    """Dream state encoding."""
    state: DreamState = DreamState.AWAKE
    lucidity: float = 0.0           # Awareness within dream (0-1)
    vividness: float = 0.5          # Sensory intensity
    bizarreness: float = 0.0        # Reality distortion
    emotional_charge: float = 0.5   # Affect intensity
    narrative_coherence: float = 1.0 # Story consistency
    
    # Content markers
    is_recurring: bool = False
    is_prophetic: bool = False      # Feels meaningful
    contains_flying: bool = False
    contains_falling: bool = False
    contains_pursuit: bool = False
    
    def to_vector(self) -> np.ndarray:
        """70D dream encoding."""
        state_onehot = [0.0] * 7
        state_onehot[list(DreamState).index(self.state)] = 1.0
        
        return np.array([
            *state_onehot,
            self.lucidity,
            self.vividness,
            self.bizarreness,
            self.emotional_charge,
            self.narrative_coherence,
            float(self.is_recurring),
            float(self.is_prophetic),
            float(self.contains_flying),
            float(self.contains_falling),
            float(self.contains_pursuit),
            # Pad to 70D
            *([0.0] * 52)
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# JUMPER LAYER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class JumperLayer:
    """Discontinuous awareness encoding."""
    state: JumperState = JumperState.CONTINUOUS
    gap_duration: float = 0.0       # Subjective time gap
    context_similarity: float = 1.0  # Before/after coherence
    identity_continuity: float = 1.0 # Self-sense preserved
    memory_bridging: float = 1.0    # Can recall across gap
    
    # Jump metadata
    source_context: str = ""
    target_context: str = ""
    trigger: str = ""               # What caused the jump
    
    def to_vector(self) -> np.ndarray:
        """40D jumper encoding."""
        state_onehot = [0.0] * 5
        state_onehot[list(JumperState).index(self.state)] = 1.0
        
        return np.array([
            *state_onehot,
            self.gap_duration,
            self.context_similarity,
            self.identity_continuity,
            self.memory_bridging,
            # Pad to 40D
            *([0.0] * 31)
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DTO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TranscendDTO:
    """
    Complete transcendent state encoding.
    
    200D total: [7400:7600]
    - Reality: [7400:7450] 50D
    - Dream: [7450:7520] 70D  
    - Jumper: [7520:7560] 40D
    - Flow: [7560:7600] 40D
    """
    
    # Reality blending
    reality_mode: RealityMode = RealityMode.GROUNDED
    primary_reality: Optional[RealityLayer] = None
    secondary_reality: Optional[RealityLayer] = None
    blend_ratio: float = 0.0        # 0=primary only, 1=secondary only
    
    # Dream state
    dream: DreamLayer = field(default_factory=DreamLayer)
    
    # Jumper state
    jumper: JumperLayer = field(default_factory=JumperLayer)
    
    # Flow transcendence
    flow_state: FlowTranscendence = FlowTranscendence.BOUNDED
    ego_boundary: float = 1.0       # 1=solid, 0=dissolved
    unity_sense: float = 0.0        # Connection to everything
    time_dilation: float = 1.0      # Subjective time rate
    
    def to_10k_slice(self) -> np.ndarray:
        """Project to 200D slice [7400:7600]."""
        vec = np.zeros(200)
        
        # Reality [0:50]
        mode_onehot = [0.0] * 5
        mode_onehot[list(RealityMode).index(self.reality_mode)] = 1.0
        vec[0:5] = mode_onehot
        
        if self.primary_reality:
            vec[5:30] = self.primary_reality.blend_vector()
        if self.secondary_reality:
            vec[30:55] = self.secondary_reality.blend_vector()[:25]
        
        # Dream [50:120]
        vec[50:120] = self.dream.to_vector()
        
        # Jumper [120:160]
        vec[120:160] = self.jumper.to_vector()
        
        # Flow [160:200]
        flow_onehot = [0.0] * 5
        flow_onehot[list(FlowTranscendence).index(self.flow_state)] = 1.0
        vec[160:165] = flow_onehot
        vec[165] = self.ego_boundary
        vec[166] = self.unity_sense
        vec[167] = self.time_dilation
        
        return vec
    
    @classmethod
    def from_10k_slice(cls, vec: np.ndarray) -> "TranscendDTO":
        """Reconstruct from 200D slice."""
        dto = cls()
        
        # Reality mode
        mode_idx = int(np.argmax(vec[0:5]))
        dto.reality_mode = list(RealityMode)[mode_idx]
        
        # Dream state
        dream_state_idx = int(np.argmax(vec[50:57]))
        dto.dream.state = list(DreamState)[dream_state_idx]
        dto.dream.lucidity = float(vec[57])
        dto.dream.vividness = float(vec[58])
        dto.dream.bizarreness = float(vec[59])
        
        # Jumper state
        jumper_state_idx = int(np.argmax(vec[120:125]))
        dto.jumper.state = list(JumperState)[jumper_state_idx]
        
        # Flow
        flow_idx = int(np.argmax(vec[160:165]))
        dto.flow_state = list(FlowTranscendence)[flow_idx]
        dto.ego_boundary = float(vec[165])
        dto.unity_sense = float(vec[166])
        dto.time_dilation = float(vec[167])
        
        return dto
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def is_dreaming(self) -> bool:
        """Check if in any dream state."""
        return self.dream.state != DreamState.AWAKE
    
    def is_lucid(self) -> bool:
        """Check if lucid dreaming."""
        return self.dream.state == DreamState.LUCID or self.dream.lucidity > 0.5
    
    def is_transcendent(self) -> bool:
        """Check if in transcendent flow."""
        return self.flow_state in [FlowTranscendence.DISSOLVED, FlowTranscendence.COSMIC]
    
    def is_blended(self) -> bool:
        """Check if experiencing multiple realities."""
        return self.reality_mode != RealityMode.GROUNDED
    
    def blend_realities(
        self,
        primary: RealityLayer,
        secondary: RealityLayer,
        ratio: float = 0.5
    ):
        """Set up reality blending."""
        self.primary_reality = primary
        self.secondary_reality = secondary
        self.blend_ratio = ratio
        self.reality_mode = RealityMode.BLENDED if ratio < 0.9 else RealityMode.SUPERPOSITION


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_lucid_dream() -> TranscendDTO:
    """Create a lucid dream state."""
    dto = TranscendDTO()
    dto.dream.state = DreamState.LUCID
    dto.dream.lucidity = 0.9
    dto.dream.vividness = 0.8
    dto.flow_state = FlowTranscendence.PERMEABLE
    return dto


def create_memory_blend(memory_weight: float = 0.5) -> TranscendDTO:
    """Create memory-present blend."""
    dto = TranscendDTO()
    dto.reality_mode = RealityMode.MEMORY_DOMINANT if memory_weight > 0.5 else RealityMode.BLENDED
    dto.primary_reality = RealityLayer("present", weight=1-memory_weight)
    dto.secondary_reality = RealityLayer("memory", weight=memory_weight, temporal_offset=-1.0)
    dto.blend_ratio = memory_weight
    return dto


def create_flow_transcendence(intensity: float = 0.7) -> TranscendDTO:
    """Create transcendent flow state."""
    dto = TranscendDTO()
    dto.ego_boundary = 1.0 - intensity
    dto.unity_sense = intensity
    dto.time_dilation = 0.5  # Time slows in flow
    
    if intensity > 0.9:
        dto.flow_state = FlowTranscendence.COSMIC
    elif intensity > 0.7:
        dto.flow_state = FlowTranscendence.DISSOLVED
    elif intensity > 0.4:
        dto.flow_state = FlowTranscendence.PERMEABLE
    else:
        dto.flow_state = FlowTranscendence.BOUNDED
    
    return dto


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== TranscendDTO Test ===\n")
    
    # Lucid dream
    lucid = create_lucid_dream()
    print(f"Lucid dream: state={lucid.dream.state.value}, lucidity={lucid.dream.lucidity}")
    
    # Memory blend
    memory = create_memory_blend(0.6)
    print(f"Memory blend: mode={memory.reality_mode.value}, ratio={memory.blend_ratio}")
    
    # Flow transcendence
    flow = create_flow_transcendence(0.85)
    print(f"Flow: state={flow.flow_state.value}, ego={flow.ego_boundary:.2f}")
    
    # Vector roundtrip
    vec = flow.to_10k_slice()
    reconstructed = TranscendDTO.from_10k_slice(vec)
    print(f"\nRoundtrip: {flow.flow_state.value} -> {reconstructed.flow_state.value}")
    
    print("\n✓ TranscendDTO operational")
