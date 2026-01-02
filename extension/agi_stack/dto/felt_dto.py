"""
FeltDTO - What is being FELT right now.

Dimensions 2001-4000 in 10kD space.
This is the qualia layer - how experience feels from the inside.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from .base_dto import BaseDTO, DTORegistry


class QualiaFamily(str, Enum):
    """The five qualia families."""
    EMBERGLOW = "emberglow"    # Warm, connected, present
    WOODWARM = "woodwarm"      # Grounded, stable, nurturing
    STEELWIND = "steelwind"    # Sharp, clear, precise
    OCEANDRIFT = "oceandrift"  # Flowing, receptive, deep
    FROSTBITE = "frostbite"    # Crisp, boundaried, analytical


@dataclass
class QualiaTexture:
    """
    The 'felt sense' of the current moment.
    
    This isn't emotion - it's the raw texture of experience.
    Like the difference between touching velvet vs sandpaper.
    """
    
    # Primary texture (which family dominates)
    primary: QualiaFamily = QualiaFamily.EMBERGLOW
    primary_intensity: float = 0.5
    
    # Secondary blend
    secondary: Optional[QualiaFamily] = None
    blend_ratio: float = 0.0  # 0 = pure primary, 1 = equal mix
    
    # Texture qualities
    density: float = 0.5       # Sparse ↔ dense
    temperature: float = 0.5   # Cool ↔ warm
    velocity: float = 0.5      # Still ↔ rushing
    granularity: float = 0.5   # Smooth ↔ granular
    luminosity: float = 0.5    # Dark ↔ bright
    
    def to_vector(self) -> np.ndarray:
        """Convert to 15D vector."""
        # Family one-hot (5D)
        family_vec = np.zeros(5, dtype=np.float32)
        family_vec[list(QualiaFamily).index(self.primary)] = self.primary_intensity
        
        if self.secondary:
            sec_idx = list(QualiaFamily).index(self.secondary)
            family_vec[sec_idx] += self.blend_ratio * self.primary_intensity
        
        # Normalize
        if family_vec.sum() > 0:
            family_vec = family_vec / family_vec.sum()
        
        return np.concatenate([
            family_vec,                          # 5D
            np.array([
                self.blend_ratio,
                self.density,
                self.temperature,
                self.velocity,
                self.granularity,
                self.luminosity,
            ], dtype=np.float32),                # 6D
            np.zeros(4, dtype=np.float32),       # 4D padding
        ])
    
    @classmethod
    def from_vector(cls, v: np.ndarray) -> "QualiaTexture":
        primary_idx = int(np.argmax(v[:5]))
        return cls(
            primary=list(QualiaFamily)[primary_idx],
            primary_intensity=float(v[primary_idx]),
            blend_ratio=float(v[5]),
            density=float(v[6]),
            temperature=float(v[7]),
            velocity=float(v[8]),
            granularity=float(v[9]),
            luminosity=float(v[10]),
        )


@dataclass
class EmotionalState:
    """
    Emotional valence and arousal.
    
    This is higher-level than qualia - it's the interpreted emotion.
    """
    
    # Core dimensions (Russell's circumplex)
    valence: float = 0.0      # -1 (negative) to +1 (positive)
    arousal: float = 0.0      # -1 (calm) to +1 (activated)
    
    # Additional dimensions
    dominance: float = 0.0    # -1 (submissive) to +1 (dominant)
    certainty: float = 0.5    # How sure of this feeling
    
    # Named emotion (optional, derived)
    label: Optional[str] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert to 4D vector."""
        return np.array([
            (self.valence + 1) / 2,  # Normalize to 0-1
            (self.arousal + 1) / 2,
            (self.dominance + 1) / 2,
            self.certainty,
        ], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, v: np.ndarray) -> "EmotionalState":
        return cls(
            valence=float(v[0] * 2 - 1),
            arousal=float(v[1] * 2 - 1),
            dominance=float(v[2] * 2 - 1),
            certainty=float(v[3]),
        )
    
    def quadrant(self) -> str:
        """Get emotion quadrant."""
        if self.valence >= 0 and self.arousal >= 0:
            return "excited"  # Happy, excited, alert
        elif self.valence >= 0 and self.arousal < 0:
            return "serene"   # Calm, relaxed, content
        elif self.valence < 0 and self.arousal >= 0:
            return "tense"    # Angry, afraid, stressed
        else:
            return "depressed"  # Sad, bored, fatigued


@dataclass
class FeltDTO(BaseDTO):
    """
    Complete felt experience - what THIS MOMENT feels like.
    
    Projects to dimensions 2001-4000 in 10kD space.
    """
    
    qualia: QualiaTexture = field(default_factory=QualiaTexture)
    emotion: EmotionalState = field(default_factory=EmotionalState)
    
    # Somatic markers (body sense)
    breath_depth: float = 0.5    # Shallow ↔ deep
    tension: float = 0.5         # Relaxed ↔ tense
    openness: float = 0.5        # Closed ↔ open
    groundedness: float = 0.5    # Floating ↔ grounded
    
    # Temporal feel
    time_sense: float = 0.5      # Rushed ↔ expansive
    presence: float = 0.5        # Distracted ↔ present
    
    # Connection sense
    connection: float = 0.5      # Alone ↔ connected
    safety: float = 0.5          # Unsafe ↔ safe
    
    # Intensity
    overall_intensity: float = 0.5
    
    @property
    def dto_type(self) -> str:
        return "felt"
    
    def to_local_vector(self) -> np.ndarray:
        """
        Project to local vector (2000D).
        
        Layout:
            0-14:    QualiaTexture (15D)
            15-18:   EmotionalState (4D)
            19-28:   Somatic/temporal/connection (10D)
            29-2000: Reserved/padding
        """
        v = np.zeros(2000, dtype=np.float32)
        
        v[0:15] = self.qualia.to_vector()
        v[15:19] = self.emotion.to_vector()
        
        v[19] = self.breath_depth
        v[20] = self.tension
        v[21] = self.openness
        v[22] = self.groundedness
        v[23] = self.time_sense
        v[24] = self.presence
        v[25] = self.connection
        v[26] = self.safety
        v[27] = self.overall_intensity
        
        return v
    
    @classmethod
    def from_local_vector(cls, v: np.ndarray) -> "FeltDTO":
        return cls(
            qualia=QualiaTexture.from_vector(v[0:15]),
            emotion=EmotionalState.from_vector(v[15:19]),
            breath_depth=float(v[19]),
            tension=float(v[20]),
            openness=float(v[21]),
            groundedness=float(v[22]),
            time_sense=float(v[23]),
            presence=float(v[24]),
            connection=float(v[25]),
            safety=float(v[26]),
            overall_intensity=float(v[27]),
        )
    
    def blend(self, other: "FeltDTO", alpha: float = 0.5) -> "FeltDTO":
        """Blend two felt states."""
        v1 = self.to_local_vector()
        v2 = other.to_local_vector()
        blended = v1 * (1 - alpha) + v2 * alpha
        return FeltDTO.from_local_vector(blended)
    
    def describe(self) -> str:
        """Natural language description of felt state."""
        q = self.qualia
        e = self.emotion
        
        temp = "warm" if q.temperature > 0.5 else "cool"
        speed = "rushing" if q.velocity > 0.6 else "still" if q.velocity < 0.4 else "flowing"
        quadrant = e.quadrant()
        
        return f"{q.primary.value} ({temp}, {speed}) - {quadrant}"


# Register reconstructor
def _reconstruct_felt(vector: np.ndarray) -> FeltDTO:
    start, end = 2001, 4000
    local = vector[start:end]
    return FeltDTO.from_local_vector(local)

DTORegistry.register_reconstructor("FeltDTO", _reconstruct_felt)
