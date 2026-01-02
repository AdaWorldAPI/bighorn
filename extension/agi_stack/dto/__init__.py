"""
AGI Stack Data Transfer Objects

Every DTO can:
- to_10kd() → Project to 10,000D hypervector
- from_10kd() → Reconstruct from hypervector (lossy)
- to_dict() → JSON serializable
- blend(other, alpha) → Interpolate with another DTO

10kD Allocation:
    0-2000:     Soul (identity, priors, mode)
    2001-4000:  Felt (qualia, emotion, sensation)
    4001-5500:  Situation (scene, context, dynamics)
    5501-7000:  Volition (intent, pending, constraints)
    7001-8500:  Vision (imagery, kopfkino)
    8501-10000: Context (history, memory traces)
"""

from .base_dto import BaseDTO, DTORegistry, DIMENSION_MAP, TOTAL_DIMENSIONS
from .soul_dto import SoulDTO, SoulField, PersonaPriors, OntologicalMode
from .felt_dto import FeltDTO, QualiaTexture, EmotionalState, QualiaFamily
from .situation_dto import SituationDTO, Scene, Actor, Dynamics, SceneType, RelationshipType
from .volition_dto import VolitionDTO, Intent, PendingAction, IntentType, ActionState
from .vision_dto import VisionDTO, KopfkinoScene, ImagePrompt, ImageStyle, Perspective
from .moment_dto import MomentDTO, MomentStream
from .universal_dto import UniversalThought, UniversalEpisode, UniversalTexture

__all__ = [
    # Base
    "BaseDTO", "DTORegistry", "DIMENSION_MAP", "TOTAL_DIMENSIONS",
    
    # Soul
    "SoulDTO", "SoulField", "PersonaPriors", "OntologicalMode",
    
    # Felt
    "FeltDTO", "QualiaTexture", "EmotionalState", "QualiaFamily",
    
    # Situation
    "SituationDTO", "Scene", "Actor", "Dynamics", "SceneType", "RelationshipType",
    
    # Volition
    "VolitionDTO", "Intent", "PendingAction", "IntentType", "ActionState",
    
    # Vision
    "VisionDTO", "KopfkinoScene", "ImagePrompt", "ImageStyle", "Perspective",
    
    # Moment (unified)
    "MomentDTO", "MomentStream",
    
    # Universal
    "UniversalThought", "UniversalEpisode", "UniversalTexture",
]
