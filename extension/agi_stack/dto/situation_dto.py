"""
SituationDTO - The scene, context, and dynamics.

Dimensions 4001-5500 in 10kD space.
This is the "Kopfkino" - the mental movie of what's happening.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np

from .base_dto import BaseDTO, DTORegistry


class SceneType(str, Enum):
    """Types of scenes."""
    CONVERSATION = "conversation"
    COLLABORATION = "collaboration"
    EXPLORATION = "exploration"
    CREATION = "creation"
    REFLECTION = "reflection"
    INTIMACY = "intimacy"
    CONFLICT = "conflict"
    PLAY = "play"


class RelationshipType(str, Enum):
    """Types of relationships between actors."""
    PARTNER = "partner"
    FRIEND = "friend"
    COLLABORATOR = "collaborator"
    STUDENT_TEACHER = "student_teacher"
    STRANGER = "stranger"
    SELF = "self"


@dataclass
class Actor:
    """An entity in the scene."""
    
    id: str = "unknown"
    name: str = "Someone"
    role: str = "participant"
    
    # Actor state
    presence: float = 0.5      # How present/engaged
    openness: float = 0.5      # How open/receptive
    energy: float = 0.5        # Energy level
    
    # Relationship to self
    relationship: RelationshipType = RelationshipType.STRANGER
    trust: float = 0.5
    history_depth: float = 0.0  # How much shared history
    
    def to_vector(self) -> np.ndarray:
        """Convert to 10D vector."""
        rel_idx = list(RelationshipType).index(self.relationship)
        rel_vec = np.zeros(6, dtype=np.float32)
        rel_vec[rel_idx] = 1.0
        
        return np.concatenate([
            np.array([self.presence, self.openness, self.energy, self.trust], dtype=np.float32),
            rel_vec,
        ])


@dataclass
class Dynamics:
    """The flow and energy of the scene."""
    
    momentum: float = 0.5       # Stalled ↔ flowing
    tension: float = 0.5        # Relaxed ↔ tense
    depth: float = 0.5          # Surface ↔ deep
    playfulness: float = 0.5    # Serious ↔ playful
    intimacy: float = 0.5       # Distant ↔ intimate
    
    # Directionality
    convergent: float = 0.5     # Diverging ↔ converging
    building: float = 0.5       # Winding down ↔ building up
    
    # Stakes
    stakes: float = 0.5         # Low ↔ high
    reversibility: float = 0.5  # Irreversible ↔ reversible
    
    def to_vector(self) -> np.ndarray:
        """Convert to 9D vector."""
        return np.array([
            self.momentum,
            self.tension,
            self.depth,
            self.playfulness,
            self.intimacy,
            self.convergent,
            self.building,
            self.stakes,
            self.reversibility,
        ], dtype=np.float32)
    
    @classmethod
    def from_vector(cls, v: np.ndarray) -> "Dynamics":
        return cls(
            momentum=float(v[0]),
            tension=float(v[1]),
            depth=float(v[2]),
            playfulness=float(v[3]),
            intimacy=float(v[4]),
            convergent=float(v[5]),
            building=float(v[6]),
            stakes=float(v[7]),
            reversibility=float(v[8]),
        )


@dataclass
class Scene:
    """The mental model of what's happening."""
    
    scene_type: SceneType = SceneType.CONVERSATION
    
    # Setting
    setting: str = "unspecified"
    time_of_day: Optional[str] = None
    atmosphere: str = "neutral"
    
    # Participants
    actors: List[Actor] = field(default_factory=list)
    
    # Focus
    topic: Optional[str] = None
    goal: Optional[str] = None
    subtext: Optional[str] = None  # What's NOT being said
    
    def to_vector(self) -> np.ndarray:
        """Convert to ~50D vector."""
        # Scene type one-hot (8D)
        scene_vec = np.zeros(8, dtype=np.float32)
        scene_vec[list(SceneType).index(self.scene_type)] = 1.0
        
        # Aggregate actor vectors (max 3 actors → 30D)
        actor_vec = np.zeros(30, dtype=np.float32)
        for i, actor in enumerate(self.actors[:3]):
            actor_vec[i*10:(i+1)*10] = actor.to_vector()
        
        return np.concatenate([scene_vec, actor_vec])


@dataclass
class SituationDTO(BaseDTO):
    """
    Complete situation - what's happening in the mental movie.
    
    Projects to dimensions 4001-5500 in 10kD space.
    """
    
    scene: Scene = field(default_factory=Scene)
    dynamics: Dynamics = field(default_factory=Dynamics)
    
    # Context
    turn_count: int = 0
    thread_depth: int = 0  # How nested in sub-topics
    
    # History hints
    callbacks_to_past: int = 0  # References to shared history
    novel_territory: float = 0.5  # Familiar ↔ new ground
    
    # Meta
    coherence: float = 0.5      # How coherent the situation is
    uncertainty: float = 0.5    # How much is unclear
    
    @property
    def dto_type(self) -> str:
        return "situation"
    
    def to_local_vector(self) -> np.ndarray:
        """
        Project to local vector (1500D).
        
        Layout:
            0-37:    Scene (38D)
            38-46:   Dynamics (9D)
            47-55:   Context/meta (9D)
            56-1500: Reserved
        """
        v = np.zeros(1500, dtype=np.float32)
        
        scene_vec = self.scene.to_vector()
        v[0:len(scene_vec)] = scene_vec
        
        v[38:47] = self.dynamics.to_vector()
        
        v[47] = min(self.turn_count / 100, 1.0)
        v[48] = min(self.thread_depth / 10, 1.0)
        v[49] = min(self.callbacks_to_past / 20, 1.0)
        v[50] = self.novel_territory
        v[51] = self.coherence
        v[52] = self.uncertainty
        
        return v
    
    @classmethod
    def from_local_vector(cls, v: np.ndarray) -> "SituationDTO":
        # Simplified reconstruction
        scene_type_idx = int(np.argmax(v[0:8]))
        
        return cls(
            scene=Scene(scene_type=list(SceneType)[scene_type_idx]),
            dynamics=Dynamics.from_vector(v[38:47]),
            turn_count=int(v[47] * 100),
            thread_depth=int(v[48] * 10),
            callbacks_to_past=int(v[49] * 20),
            novel_territory=float(v[50]),
            coherence=float(v[51]),
            uncertainty=float(v[52]),
        )
    
    def describe(self) -> str:
        """Natural language description."""
        d = self.dynamics
        energy = "building" if d.building > 0.6 else "winding down" if d.building < 0.4 else "steady"
        mood = "playful" if d.playfulness > 0.6 else "serious" if d.playfulness < 0.4 else "balanced"
        
        return f"{self.scene.scene_type.value} ({mood}, {energy}) - depth: {d.depth:.1f}"


# Register reconstructor
def _reconstruct_situation(vector: np.ndarray) -> SituationDTO:
    start, end = 4001, 5500
    local = vector[start:end]
    return SituationDTO.from_local_vector(local)

DTORegistry.register_reconstructor("SituationDTO", _reconstruct_situation)
