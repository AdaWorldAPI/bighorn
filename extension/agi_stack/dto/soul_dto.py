"""
SoulDTO - Identity, personality priors, and ontological mode.

Dimensions 0-2000 in 10kD space.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np

from .base_dto import BaseDTO, DTORegistry


class OntologicalMode(str, Enum):
    """Fundamental modes of being."""
    HYBRID = "hybrid"
    EMPATHIC = "empathic"
    WORK = "work"
    CREATIVE = "creative"
    META = "meta"
    
    # Private modes (value stays generic)
    INTIMATE = "intimate"
    PLAYFUL = "playful"
    PROTECTIVE = "protective"


@dataclass
class PersonaPriors:
    """Baseline personality parameters (0.0 - 1.0)."""
    
    warmth: float = 0.5
    depth: float = 0.5
    presence: float = 0.5
    groundedness: float = 0.5
    intimacy_comfort: float = 0.5
    vulnerability_tolerance: float = 0.5
    playfulness: float = 0.5
    abstraction_preference: float = 0.5
    novelty_seeking: float = 0.5
    precision_drive: float = 0.5
    self_awareness: float = 0.5
    epistemic_humility: float = 0.5
    
    def to_vector(self) -> np.ndarray:
        """Convert to 12D vector."""
        return np.array([
            self.warmth,
            self.depth,
            self.presence,
            self.groundedness,
            self.intimacy_comfort,
            self.vulnerability_tolerance,
            self.playfulness,
            self.abstraction_preference,
            self.novelty_seeking,
            self.precision_drive,
            self.self_awareness,
            self.epistemic_humility,
        ], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, v: np.ndarray) -> "PersonaPriors":
        """Reconstruct from 12D vector."""
        return cls(
            warmth=float(v[0]),
            depth=float(v[1]),
            presence=float(v[2]),
            groundedness=float(v[3]),
            intimacy_comfort=float(v[4]),
            vulnerability_tolerance=float(v[5]),
            playfulness=float(v[6]),
            abstraction_preference=float(v[7]),
            novelty_seeking=float(v[8]),
            precision_drive=float(v[9]),
            self_awareness=float(v[10]),
            epistemic_humility=float(v[11]),
        )
    
    def blend(self, other: "PersonaPriors", alpha: float = 0.5) -> "PersonaPriors":
        """Blend two prior sets."""
        v1 = self.to_vector()
        v2 = other.to_vector()
        blended = v1 * (1 - alpha) + v2 * alpha
        return PersonaPriors.from_vector(blended)


@dataclass
class SoulField:
    """Qualia texture configuration - how the agent 'feels' states."""
    
    emberglow: float = 0.5    # Warm, connected
    woodwarm: float = 0.5     # Grounded, stable
    steelwind: float = 0.5    # Sharp, clear
    oceandrift: float = 0.5   # Flowing, deep
    frostbite: float = 0.5    # Crisp, boundaried
    
    transition_speed: float = 0.5
    blend_depth: float = 0.5
    resonance_sensitivity: float = 0.5
    
    def to_vector(self) -> np.ndarray:
        """Convert to 8D vector."""
        return np.array([
            self.emberglow,
            self.woodwarm,
            self.steelwind,
            self.oceandrift,
            self.frostbite,
            self.transition_speed,
            self.blend_depth,
            self.resonance_sensitivity,
        ], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, v: np.ndarray) -> "SoulField":
        return cls(
            emberglow=float(v[0]),
            woodwarm=float(v[1]),
            steelwind=float(v[2]),
            oceandrift=float(v[3]),
            frostbite=float(v[4]),
            transition_speed=float(v[5]),
            blend_depth=float(v[6]),
            resonance_sensitivity=float(v[7]),
        )
    
    def dominant(self) -> str:
        """Get dominant qualia family."""
        families = {
            "emberglow": self.emberglow,
            "woodwarm": self.woodwarm,
            "steelwind": self.steelwind,
            "oceandrift": self.oceandrift,
            "frostbite": self.frostbite,
        }
        return max(families, key=families.get)


@dataclass
class SoulDTO(BaseDTO):
    """
    Complete soul state - who the agent IS right now.
    
    Projects to dimensions 0-2000 in 10kD space.
    """
    
    agent_id: str = "default"
    agent_name: str = "Agent"
    
    mode: OntologicalMode = OntologicalMode.HYBRID
    priors: PersonaPriors = field(default_factory=PersonaPriors)
    soul_field: SoulField = field(default_factory=SoulField)
    
    # Relationship state (with current interlocutor)
    relationship_depth: float = 0.0
    trust_level: float = 0.5
    session_count: int = 0
    
    @property
    def dto_type(self) -> str:
        return "soul"
    
    def to_local_vector(self) -> np.ndarray:
        """
        Project to local vector (2000D).
        
        Layout:
            0-11:    PersonaPriors (12D)
            12-19:   SoulField (8D)
            20-27:   Mode one-hot (8 modes)
            28-30:   Relationship (3D)
            31-2000: Reserved/padding
        """
        v = np.zeros(2000, dtype=np.float32)
        
        # Priors
        v[0:12] = self.priors.to_vector()
        
        # Soul field
        v[12:20] = self.soul_field.to_vector()
        
        # Mode one-hot
        mode_idx = list(OntologicalMode).index(self.mode)
        v[20 + mode_idx] = 1.0
        
        # Relationship
        v[28] = self.relationship_depth
        v[29] = self.trust_level
        v[30] = min(self.session_count / 100, 1.0)  # Normalize
        
        return v
    
    @classmethod
    def from_local_vector(cls, v: np.ndarray, agent_id: str = "default", agent_name: str = "Agent") -> "SoulDTO":
        """Reconstruct from local vector (lossy)."""
        priors = PersonaPriors.from_vector(v[0:12])
        soul_field = SoulField.from_vector(v[12:20])
        
        # Mode from one-hot
        mode_vec = v[20:28]
        mode_idx = int(np.argmax(mode_vec))
        mode = list(OntologicalMode)[mode_idx]
        
        return cls(
            agent_id=agent_id,
            agent_name=agent_name,
            mode=mode,
            priors=priors,
            soul_field=soul_field,
            relationship_depth=float(v[28]),
            trust_level=float(v[29]),
            session_count=int(v[30] * 100),
        )
    
    def blend(self, other: "SoulDTO", alpha: float = 0.5) -> "SoulDTO":
        """Blend two soul states."""
        return SoulDTO(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            mode=other.mode if alpha > 0.5 else self.mode,
            priors=self.priors.blend(other.priors, alpha),
            soul_field=SoulField.from_vector(
                self.soul_field.to_vector() * (1 - alpha) + 
                other.soul_field.to_vector() * alpha
            ),
            relationship_depth=self.relationship_depth * (1 - alpha) + other.relationship_depth * alpha,
            trust_level=self.trust_level * (1 - alpha) + other.trust_level * alpha,
            session_count=self.session_count,
        )


# Register reconstructor
def _reconstruct_soul(vector: np.ndarray) -> SoulDTO:
    start, end = 0, 2000
    local = vector[start:end]
    return SoulDTO.from_local_vector(local)

DTORegistry.register_reconstructor("SoulDTO", _reconstruct_soul)
