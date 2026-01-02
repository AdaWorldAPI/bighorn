"""
MomentDTO - The unified 10kD snapshot of a complete experiential moment.

This is the master DTO that combines:
- Soul (who I am)
- Felt (what I feel)
- Situation (what's happening)
- Volition (what I want)
- Vision (what I imagine)

Into a single 10,000-dimensional vector that captures
the COMPLETE state of an experiential moment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime

from .base_dto import BaseDTO, DIMENSION_MAP, TOTAL_DIMENSIONS
from .soul_dto import SoulDTO
from .felt_dto import FeltDTO
from .situation_dto import SituationDTO
from .volition_dto import VolitionDTO
from .vision_dto import VisionDTO


@dataclass
class MomentDTO:
    """
    A complete experiential moment in 10kD space.
    
    This is the unified representation - one vector that captures
    the full texture of NOW.
    """
    
    # Component DTOs
    soul: SoulDTO = field(default_factory=SoulDTO)
    felt: FeltDTO = field(default_factory=FeltDTO)
    situation: SituationDTO = field(default_factory=SituationDTO)
    volition: VolitionDTO = field(default_factory=VolitionDTO)
    vision: VisionDTO = field(default_factory=VisionDTO)
    
    # Metadata
    moment_id: str = ""
    timestamp: Optional[str] = None
    session_id: Optional[str] = None
    
    # Coherence
    internal_coherence: float = 0.5  # How well the parts fit together
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if not self.moment_id:
            self.moment_id = f"moment_{int(datetime.utcnow().timestamp())}"
    
    def to_10kd(self) -> np.ndarray:
        """
        Combine all DTOs into unified 10kD vector.
        
        Dimension allocation:
            0-2000:     Soul
            2001-4000:  Felt
            4001-5500:  Situation
            5501-7000:  Volition
            7001-8500:  Vision
            8501-10000: Context/reserved
        """
        vector = np.zeros(TOTAL_DIMENSIONS, dtype=np.float32)
        
        # Each DTO projects to its allocated range
        vector[0:2000] = self.soul.to_local_vector()
        vector[2001:4001] = self.felt.to_local_vector()
        vector[4001:5501] = self.situation.to_local_vector()
        vector[5501:7001] = self.volition.to_local_vector()
        vector[7001:8501] = self.vision.to_local_vector()
        
        # Context dimensions (8501-10000) reserved for:
        # - Temporal markers
        # - Session continuity
        # - Cross-moment references
        
        return vector
    
    @classmethod
    def from_10kd(cls, vector: np.ndarray, metadata: Dict = None) -> "MomentDTO":
        """
        Reconstruct MomentDTO from 10kD vector.
        Note: This is lossy - detailed information is lost.
        """
        metadata = metadata or {}
        
        return cls(
            soul=SoulDTO.from_local_vector(vector[0:2000]),
            felt=FeltDTO.from_local_vector(vector[2001:4001]),
            situation=SituationDTO.from_local_vector(vector[4001:5501]),
            volition=VolitionDTO.from_local_vector(vector[5501:7001]),
            vision=VisionDTO.from_local_vector(vector[7001:8501]),
            moment_id=metadata.get("moment_id", ""),
            timestamp=metadata.get("timestamp"),
            session_id=metadata.get("session_id"),
        )
    
    def similarity(self, other: "MomentDTO") -> float:
        """Compute cosine similarity with another moment."""
        v1 = self.to_10kd()
        v2 = other.to_10kd()
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def component_similarities(self, other: "MomentDTO") -> Dict[str, float]:
        """Get similarity broken down by component."""
        return {
            "soul": self.soul.similarity(other.soul),
            "felt": self.felt.similarity(other.felt),
            "situation": self.situation.similarity(other.situation),
            "volition": self.volition.similarity(other.volition),
            "vision": self.vision.similarity(other.vision),
        }
    
    def blend(self, other: "MomentDTO", alpha: float = 0.5) -> "MomentDTO":
        """Blend two moments."""
        return MomentDTO(
            soul=self.soul.blend(other.soul, alpha),
            felt=self.felt.blend(other.felt, alpha),
            situation=SituationDTO.from_local_vector(
                self.situation.to_local_vector() * (1 - alpha) +
                other.situation.to_local_vector() * alpha
            ),
            volition=VolitionDTO.from_local_vector(
                self.volition.to_local_vector() * (1 - alpha) +
                other.volition.to_local_vector() * alpha
            ),
            vision=VisionDTO.from_local_vector(
                self.vision.to_local_vector() * (1 - alpha) +
                other.vision.to_local_vector() * alpha
            ),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "moment_id": self.moment_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "internal_coherence": self.internal_coherence,
            "soul": self.soul.to_dict(),
            "felt": self.felt.to_dict(),
            "situation": self.situation.to_dict(),
            "volition": self.volition.to_dict(),
            "vision": self.vision.to_dict(),
        }
    
    def describe(self) -> str:
        """Natural language description of the moment."""
        return f"""
Moment: {self.moment_id}
Soul: {self.soul.mode.value} mode, warmth={self.soul.priors.warmth:.1f}
Felt: {self.felt.describe()}
Situation: {self.situation.describe()}
Volition: {self.volition.describe()}
Vision: {self.vision.describe()}
"""
    
    def fingerprint(self) -> str:
        """Short fingerprint of this moment."""
        import hashlib
        v = self.to_10kd()
        return hashlib.md5(v.tobytes()).hexdigest()[:12]


# =============================================================================
# MOMENT STREAM
# =============================================================================

@dataclass
class MomentStream:
    """
    A stream of moments - for tracking experiential flow.
    """
    
    moments: List[MomentDTO] = field(default_factory=list)
    max_size: int = 100
    
    def add(self, moment: MomentDTO):
        """Add a moment to the stream."""
        self.moments.append(moment)
        if len(self.moments) > self.max_size:
            self.moments = self.moments[-self.max_size:]
    
    def find_similar(self, query: MomentDTO, top_k: int = 5) -> List[Tuple[MomentDTO, float]]:
        """Find moments similar to query."""
        scored = [(m, query.similarity(m)) for m in self.moments]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def trajectory(self, component: str = "felt") -> np.ndarray:
        """Get trajectory of a component over time."""
        if not self.moments:
            return np.array([])
        
        vectors = []
        for m in self.moments:
            dto = getattr(m, component, None)
            if dto:
                vectors.append(dto.to_local_vector())
        
        return np.array(vectors)
    
    def drift(self, component: str = "felt") -> float:
        """Compute drift (change) in a component over the stream."""
        traj = self.trajectory(component)
        if len(traj) < 2:
            return 0.0
        
        # Compute average step size
        steps = np.diff(traj, axis=0)
        step_sizes = np.linalg.norm(steps, axis=1)
        return float(np.mean(step_sizes))
