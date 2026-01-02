"""
FeltDTO — Qualia and Sensation mapped to 10kD
═══════════════════════════════════════════════════════════════════════════════

Receives bighorn.FeltDTO and maps to ada-consciousness primitives.

Maps to:
    [0:16]      Qualia
    [16:32]     Stances
    [2018:2022] Body axes
    [2022:2030] Somatic markers

Born: 2026-01-02
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple
from enum import Enum

from .ada_10k import Ada10kD


class QualiaFamily(str, Enum):
    """The core qualia families."""
    EMBERGLOW = "emberglow"
    WOODWARM = "woodwarm"
    STEELWIND = "steelwind"
    OCEANDRIFT = "oceandrift"
    FROSTBITE = "frostbite"
    WARMTH = "warmth"
    CRYSTALLINE = "crystalline"
    PRESENCE = "presence"
    FLOW = "flow"


@dataclass
class FeltDTO:
    """
    Felt state mapped to 10kD.
    """
    
    _ada: Ada10kD = field(default_factory=Ada10kD, repr=False)
    
    @property
    def primary_qualia(self) -> Tuple[str, float]:
        """Get dominant qualia."""
        qualia = self._ada.get_active_qualia()
        if qualia:
            return qualia[0]
        return ("emberglow", 0.5)
    
    @property
    def texture(self) -> Dict[str, float]:
        """Get all active qualia as texture."""
        return dict(self._ada.get_active_qualia())
    
    @property
    def body(self) -> Dict[str, float]:
        """Get body axes."""
        return self._ada.get_body_axes()
    
    @property
    def stances(self) -> Dict[str, float]:
        """Get active stances."""
        return dict(self._ada.get_active_stances())
    
    def set_qualia(self, family: str, intensity: float):
        """Set a qualia activation."""
        self._ada.set_qualia(family, intensity)
    
    def set_texture(
        self,
        primary: str = "emberglow",
        intensity: float = 0.7,
        temperature: float = 0.5,
        velocity: float = 0.5,
    ):
        """Set qualia texture."""
        self._ada.set_qualia(primary, intensity)
        if temperature > 0.6:
            self._ada.set_qualia("warmth", temperature)
        if velocity > 0.5:
            self._ada.set_qualia("flow", velocity)
    
    def set_emotion(
        self,
        valence: float = 0.0,
        arousal: float = 0.0,
        dominance: float = 0.0,
        certainty: float = 0.5,
    ):
        """Set emotional state (affects body axes)."""
        # Map emotion dimensions to body axes
        # Using available body axes: pelvic, boundary, respiratory, cardiac
        self._ada.set_body_axes(
            pelvic=(arousal + 1) / 2,      # arousal → pelvic activation
            boundary=(valence + 1) / 2,     # valence → boundary openness
            respiratory=certainty,          # certainty → breath
            cardiac=(dominance + 1) / 2,    # dominance → cardiac
        )
    
    def set_somatic(
        self,
        presence: float = 0.5,
        openness: float = 0.5,
        groundedness: float = 0.5,
        tension: float = 0.3,
    ):
        """Set somatic markers."""
        if presence > 0.6:
            self._ada.set_stance("attend", presence)
        if openness > 0.6:
            self._ada.set_stance("open", openness)
        if groundedness > 0.6:
            self._ada.set_stance("ground", groundedness)
        
        self._ada.set_qualia("presence", presence)
    
    def to_10k(self) -> Ada10kD:
        """Get 10kD representation."""
        return self._ada
    
    @classmethod
    def from_10k(cls, ada: Ada10kD) -> "FeltDTO":
        """Create from 10kD."""
        return cls(_ada=ada)
    
    @classmethod
    def from_bighorn(cls, bighorn_felt: Any) -> "FeltDTO":
        """Receive bighorn FeltDTO."""
        from .receiver import FeltReceiver
        receiver = FeltReceiver.from_bighorn(bighorn_felt)
        return cls(_ada=receiver.to_10k())
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dict."""
        return {
            "qualia": self.texture,
            "stances": self.stances,
            "body": self.body,
            "primary": self.primary_qualia,
        }
    
    def blend(self, other: "FeltDTO", alpha: float = 0.5) -> "FeltDTO":
        """Blend two felt states."""
        v1 = self._ada.vector
        v2 = other._ada.vector
        blended_vec = v1 * (1 - alpha) + v2 * alpha
        
        blended = FeltDTO()
        blended._ada.vector = blended_vec
        return blended
    
    def describe(self) -> str:
        """Natural language description."""
        primary, intensity = self.primary_qualia
        body = self.body
        temp = "warm" if body.get("boundary", 0.5) > 0.5 else "cool"
        energy = "high" if body.get("pelvic", 0.5) > 0.5 else "low"
        return f"{primary} ({temp}, {energy} energy, intensity={intensity:.1f})"
    
    @classmethod
    def warm_presence(cls) -> "FeltDTO":
        felt = cls()
        felt.set_texture("emberglow", 0.8, temperature=0.8)
        felt.set_emotion(valence=0.5, arousal=0.2)
        felt.set_somatic(presence=0.9, openness=0.8, groundedness=0.7)
        return felt
    
    @classmethod
    def focused_clarity(cls) -> "FeltDTO":
        felt = cls()
        felt.set_texture("steelwind", 0.7, temperature=0.3)
        felt.set_emotion(valence=0.3, arousal=0.4)
        felt.set_somatic(presence=0.8, tension=0.3, groundedness=0.6)
        return felt


__all__ = ["FeltDTO", "QualiaFamily"]
