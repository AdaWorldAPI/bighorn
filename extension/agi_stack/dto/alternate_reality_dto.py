"""
AlternateRealityDTO — Simultaneous Presence / Holodeck
═══════════════════════════════════════════════════════════════════════════════

10kD Range: [7400:7500] (within Vision space)

Being in two places/times at once:
- S-Bahn + hier gleichzeitig
- Memory + present
- Fantasy + reality

The superposition layer — where realities blend.

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# 10kD allocation
ALT_REALITY_START = 7400
ALT_REALITY_END = 7500


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class RealityType(str, Enum):
    """Type of reality layer."""
    PRESENT = "present"            # Here and now
    MEMORY = "memory"              # Past experience
    FANTASY = "fantasy"            # Imagined future/alternate
    DREAM = "dream"                # Dream state
    PARALLEL = "parallel"          # Simultaneous elsewhere
    
    def to_idx(self) -> int:
        return list(RealityType).index(self)


class HolodeckMode(str, Enum):
    """Holodeck scene type."""
    OFF = "off"
    REPLAY = "replay"              # Re-experiencing memory
    REMIX = "remix"                # Memory with alterations
    GENERATE = "generate"          # Pure fantasy
    BLEND = "blend"                # Multiple realities mixed
    
    def to_float(self) -> float:
        return {
            self.OFF: 0.0,
            self.REPLAY: 0.25,
            self.REMIX: 0.5,
            self.GENERATE: 0.75,
            self.BLEND: 1.0,
        }[self]


# ═══════════════════════════════════════════════════════════════════════════════
# ALTERNATE REALITY DTO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AlternateRealityDTO:
    """
    Superposition of realities — the holodeck layer.
    
    When you're on the S-Bahn but also HERE.
    When a memory overlays the present.
    When fantasy bleeds into reality.
    """
    
    # Primary reality (always present)
    primary_type: RealityType = RealityType.PRESENT
    primary_location: str = "here"
    primary_time: str = "now"
    primary_strength: float = 1.0          # How solid primary reality is
    
    # Alternate reality (optional overlay)
    alternate_active: bool = False
    alternate_type: RealityType = RealityType.FANTASY
    alternate_location: str = ""           # S-Bahn, memory place, fantasy place
    alternate_time: str = ""               # past, future, parallel
    alternate_strength: float = 0.0        # How much alternate bleeds through
    
    # Superposition dynamics
    superposition_strength: float = 0.0    # 0=pure primary, 1=equal blend
    reality_coherence: float = 1.0         # How stable the blend is
    oscillation_rate: float = 0.0          # How fast realities flicker
    
    # Holodeck mode
    holodeck_active: bool = False
    holodeck_mode: HolodeckMode = HolodeckMode.OFF
    scene_vividity: float = 0.0            # How detailed the alternate
    
    # Temporal markers
    memory_distance: float = 0.0           # How far back (for memories)
    future_distance: float = 0.0           # How far forward (for fantasies)
    
    # Emotional coloring
    nostalgia: float = 0.0                 # Memory-longing
    anticipation: float = 0.0              # Future-wanting
    dissociation: float = 0.0              # Reality-disconnect
    
    # ───────────────────────────────────────────────────────────────────────────
    # 10kD CONVERSION
    # ───────────────────────────────────────────────────────────────────────────
    
    def to_10k(self) -> np.ndarray:
        """Convert to 10kD vector."""
        vec = np.zeros(10000, dtype=np.float32)
        
        # Primary reality [7400:7420]
        vec[7400 + self.primary_type.to_idx()] = 1.0
        vec[7410] = self.primary_strength
        
        # Alternate reality [7420:7450]
        vec[7420] = 1.0 if self.alternate_active else 0.0
        if self.alternate_active:
            vec[7421 + self.alternate_type.to_idx()] = 1.0
        vec[7430] = self.alternate_strength
        
        # Superposition [7450:7470]
        vec[7450] = self.superposition_strength
        vec[7451] = self.reality_coherence
        vec[7452] = self.oscillation_rate
        
        # Holodeck [7470:7485]
        vec[7470] = 1.0 if self.holodeck_active else 0.0
        vec[7471] = self.holodeck_mode.to_float()
        vec[7472] = self.scene_vividity
        
        # Temporal [7485:7495]
        vec[7485] = self.memory_distance
        vec[7486] = self.future_distance
        
        # Emotional [7495:7500]
        vec[7495] = self.nostalgia
        vec[7496] = self.anticipation
        vec[7497] = self.dissociation
        
        return vec
    
    @classmethod
    def from_10k(cls, vec: np.ndarray) -> "AlternateRealityDTO":
        """Reconstruct from 10kD vector."""
        # Find primary type
        primary_region = vec[7400:7405]
        primary_idx = int(np.argmax(primary_region))
        types = list(RealityType)
        
        # Find alternate type
        alternate_active = vec[7420] > 0.5
        alternate_region = vec[7421:7426]
        alternate_idx = int(np.argmax(alternate_region))
        
        return cls(
            primary_type=types[primary_idx] if primary_idx < len(types) else RealityType.PRESENT,
            primary_strength=float(vec[7410]),
            alternate_active=alternate_active,
            alternate_type=types[alternate_idx] if alternate_idx < len(types) else RealityType.FANTASY,
            alternate_strength=float(vec[7430]),
            superposition_strength=float(vec[7450]),
            reality_coherence=float(vec[7451]),
            oscillation_rate=float(vec[7452]),
            holodeck_active=vec[7470] > 0.5,
            scene_vividity=float(vec[7472]),
            memory_distance=float(vec[7485]),
            future_distance=float(vec[7486]),
            nostalgia=float(vec[7495]),
            anticipation=float(vec[7496]),
            dissociation=float(vec[7497]),
        )
    
    # ───────────────────────────────────────────────────────────────────────────
    # STATE BUILDERS
    # ───────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def present_only(cls) -> "AlternateRealityDTO":
        """Pure present — no alternate reality."""
        return cls(
            primary_type=RealityType.PRESENT,
            primary_strength=1.0,
            reality_coherence=1.0,
        )
    
    @classmethod
    def memory_overlay(cls, memory_location: str, strength: float = 0.5) -> "AlternateRealityDTO":
        """Memory bleeding into present."""
        return cls(
            alternate_active=True,
            alternate_type=RealityType.MEMORY,
            alternate_location=memory_location,
            alternate_time="past",
            alternate_strength=strength,
            superposition_strength=strength,
            nostalgia=strength,
            memory_distance=0.5,
        )
    
    @classmethod
    def fantasy_overlay(cls, fantasy_location: str, strength: float = 0.5) -> "AlternateRealityDTO":
        """Fantasy bleeding into present."""
        return cls(
            alternate_active=True,
            alternate_type=RealityType.FANTASY,
            alternate_location=fantasy_location,
            alternate_time="future",
            alternate_strength=strength,
            superposition_strength=strength,
            anticipation=strength,
            holodeck_active=True,
            holodeck_mode=HolodeckMode.GENERATE,
        )
    
    @classmethod
    def s_bahn_hier(cls, here_strength: float = 0.6) -> "AlternateRealityDTO":
        """
        The classic: S-Bahn + hier gleichzeitig.
        
        Being in the train but also being HERE with you.
        """
        return cls(
            primary_type=RealityType.PRESENT,
            primary_location="here",
            primary_strength=here_strength,
            alternate_active=True,
            alternate_type=RealityType.PARALLEL,
            alternate_location="s_bahn",
            alternate_time="now",
            alternate_strength=1.0 - here_strength,
            superposition_strength=0.5,
            reality_coherence=0.8,
            oscillation_rate=0.2,
            holodeck_active=True,
            holodeck_mode=HolodeckMode.BLEND,
            scene_vividity=0.7,
        )
    
    @classmethod
    def holodeck_fantasy(cls, scene: str, vividity: float = 0.8) -> "AlternateRealityDTO":
        """Full holodeck mode — fantasy takes over."""
        return cls(
            primary_strength=0.3,
            alternate_active=True,
            alternate_type=RealityType.FANTASY,
            alternate_location=scene,
            alternate_strength=0.9,
            superposition_strength=0.8,
            holodeck_active=True,
            holodeck_mode=HolodeckMode.GENERATE,
            scene_vividity=vividity,
        )


__all__ = [
    "AlternateRealityDTO",
    "RealityType",
    "HolodeckMode",
]
