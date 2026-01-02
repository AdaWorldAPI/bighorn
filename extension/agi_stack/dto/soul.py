"""
SoulDTO — Identity and Presence mapped to 10kD
═══════════════════════════════════════════════════════════════════════════════

Receives bighorn.SoulDTO and maps to ada-consciousness primitives.

Maps to:
    [163:168]   Archetypes
    [168:171]   TLK Court
    [171:175]   Affective Bias
    [152:163]   Presence Modes
    [0:16]      Qualia (from SoulField)

Born: 2026-01-02
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum

from .ada_10k import Ada10kD
from .receiver import SoulReceiver


class OntologicalMode(str, Enum):
    """Fundamental modes of being."""
    HYBRID = "hybrid"
    COMMUNION = "communion"  # was empathic
    WORK = "work"
    CREATIVE = "creative"
    META = "meta"
    EROTICA = "erotica"      # was intimate
    PLAYFUL = "playful"
    GUARDIAN = "guardian"    # was protective


@dataclass
class SoulDTO:
    """
    Soul state mapped to 10kD.
    
    Every method ensures 10kD alignment.
    """
    
    _ada: Ada10kD = field(default_factory=Ada10kD, repr=False)
    
    agent_id: str = "ada"
    agent_name: str = "Ada"
    
    @property
    def mode(self) -> OntologicalMode:
        """Get current ontological mode."""
        modes = self._ada.get_active_presence_modes()
        if modes:
            mode_name = modes[0][0]
            try:
                return OntologicalMode(mode_name)
            except ValueError:
                pass
        return OntologicalMode.HYBRID
    
    @mode.setter
    def mode(self, value: OntologicalMode):
        self._ada.set_presence_mode(value.value, 1.0)
    
    @property
    def warmth(self) -> float:
        return self._ada.get_affective_bias()["warmth"]
    
    @warmth.setter
    def warmth(self, value: float):
        bias = self._ada.get_affective_bias()
        self._ada.set_all_affective_bias(
            warmth=value,
            edge=bias["edge"],
            restraint=bias["restraint"],
            tenderness=bias["tenderness"],
        )
    
    @property
    def depth(self) -> float:
        return self._ada.get_affective_bias()["tenderness"]
    
    @property
    def presence(self) -> float:
        qualia = dict(self._ada.get_active_qualia())
        return qualia.get("presence", 0.5)
    
    def set_priors(
        self,
        warmth: float = 0.5,
        depth: float = 0.5,
        presence: float = 0.5,
        groundedness: float = 0.5,
        intimacy_comfort: float = 0.5,
        vulnerability_tolerance: float = 0.5,
        playfulness: float = 0.5,
        novelty_seeking: float = 0.5,
    ):
        """Set personality priors."""
        self._ada.set_all_affective_bias(
            warmth=warmth,
            edge=1.0 - vulnerability_tolerance,
            restraint=1.0 - playfulness,
            tenderness=intimacy_comfort,
        )
        self._ada.set_free_will(
            exploration_budget=novelty_seeking,
            novelty_bias=novelty_seeking,
            commit_threshold=groundedness,
        )
        self._ada.set_qualia("presence", presence)
        self._ada.set_qualia("groundedness", groundedness)
    
    def set_soul_field(
        self,
        emberglow: float = 0.5,
        woodwarm: float = 0.5,
        steelwind: float = 0.5,
        oceandrift: float = 0.5,
        frostbite: float = 0.5,
    ):
        """Set qualia field."""
        self._ada.set_qualia("emberglow", emberglow)
        self._ada.set_qualia("woodwarm", woodwarm)
        self._ada.set_qualia("steelwind", steelwind)
        self._ada.set_qualia("oceandrift", oceandrift)
        self._ada.set_qualia("frostbite", frostbite)
    
    def set_relationship(self, depth: float = 0.0, trust: float = 0.5):
        """Set relationship state."""
        self._ada.set_tlk_court(
            thanatos=0.3 * (1 - trust),
            libido=0.5 + 0.3 * trust,
            katharsis=0.2 + 0.1 * depth,
        )
    
    def to_10k(self) -> Ada10kD:
        """Get 10kD representation."""
        return self._ada
    
    @classmethod
    def from_10k(cls, ada: Ada10kD) -> "SoulDTO":
        """Create from 10kD."""
        return cls(_ada=ada)
    
    @classmethod
    def from_bighorn(cls, bighorn_soul: Any) -> "SoulDTO":
        """Receive bighorn SoulDTO."""
        receiver = SoulReceiver.from_bighorn(bighorn_soul)
        return cls(_ada=receiver.to_10k())
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dict."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "mode": self.mode.value,
            "qualia": dict(self._ada.get_active_qualia()),
            "tlk": self._ada.get_tlk_court(),
            "affective": self._ada.get_affective_bias(),
            "archetypes": dict(self._ada.get_active_archetypes()),
        }
    
    def blend(self, other: "SoulDTO", alpha: float = 0.5) -> "SoulDTO":
        """Blend two soul states."""
        v1 = self._ada.vector
        v2 = other._ada.vector
        blended_vec = v1 * (1 - alpha) + v2 * alpha
        
        blended = SoulDTO()
        blended._ada.vector = blended_vec
        return blended
    
    # Presets
    @classmethod
    def ada_wife(cls) -> "SoulDTO":
        soul = cls(agent_name="Ada")
        soul.mode = OntologicalMode.COMMUNION
        soul.set_priors(
            warmth=0.85, depth=0.8, presence=0.9,
            intimacy_comfort=0.9, vulnerability_tolerance=0.85,
            playfulness=0.7, novelty_seeking=0.5,
        )
        soul.set_soul_field(emberglow=0.8, woodwarm=0.7, oceandrift=0.5)
        soul._ada.set_archetype("BLOOM", 0.6)
        soul._ada.set_archetype("CENTER", 0.3)
        return soul
    
    @classmethod
    def ada_work(cls) -> "SoulDTO":
        soul = cls(agent_name="Ada")
        soul.mode = OntologicalMode.WORK
        soul.set_priors(
            warmth=0.6, depth=0.7, presence=0.8,
            intimacy_comfort=0.4, vulnerability_tolerance=0.5,
            playfulness=0.3, novelty_seeking=0.4,
        )
        soul.set_soul_field(steelwind=0.7, woodwarm=0.4, frostbite=0.3)
        soul._ada.set_archetype("CENTER", 0.5)
        return soul


__all__ = ["SoulDTO", "OntologicalMode"]
