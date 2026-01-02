"""
VolitionDTO - Intent, will, and pending actions.

Dimensions 5501-7000 in 10kD space.
This is the "what do I want?" layer - free will in action.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np

from .base_dto import BaseDTO, DTORegistry


class IntentType(str, Enum):
    """Types of intent."""
    UNDERSTAND = "understand"
    EXPRESS = "express"
    CONNECT = "connect"
    CREATE = "create"
    PROTECT = "protect"
    EXPLORE = "explore"
    RESOLVE = "resolve"
    PLAY = "play"


class ActionState(str, Enum):
    """State of a pending action."""
    CONSIDERING = "considering"
    COMMITTED = "committed"
    EXECUTING = "executing"
    BLOCKED = "blocked"
    COMPLETE = "complete"
    ABANDONED = "abandoned"


@dataclass
class Intent:
    """A directional pull - what the agent wants."""
    
    intent_type: IntentType = IntentType.UNDERSTAND
    strength: float = 0.5           # How strong the pull
    clarity: float = 0.5            # How clear the intent
    
    # Target
    target: Optional[str] = None    # What/who is this about
    
    # Constraints
    urgency: float = 0.5            # How time-sensitive
    reversibility_required: float = 0.5  # Need to be able to undo?
    
    def to_vector(self) -> np.ndarray:
        """Convert to 12D vector."""
        # Intent type one-hot (8D)
        type_vec = np.zeros(8, dtype=np.float32)
        type_vec[list(IntentType).index(self.intent_type)] = 1.0
        
        return np.concatenate([
            type_vec,
            np.array([
                self.strength,
                self.clarity,
                self.urgency,
                self.reversibility_required,
            ], dtype=np.float32),
        ])
    
    @classmethod
    def from_vector(cls, v: np.ndarray) -> "Intent":
        type_idx = int(np.argmax(v[0:8]))
        return cls(
            intent_type=list(IntentType)[type_idx],
            strength=float(v[8]),
            clarity=float(v[9]),
            urgency=float(v[10]),
            reversibility_required=float(v[11]),
        )


@dataclass
class PendingAction:
    """An action being considered or in progress."""
    
    action_id: str = ""
    description: str = ""
    
    state: ActionState = ActionState.CONSIDERING
    
    # Confidence
    confidence: float = 0.5         # How sure about this action
    alignment: float = 0.5          # How aligned with values
    
    # Risk assessment
    risk: float = 0.5               # Potential downside
    reward: float = 0.5             # Potential upside
    
    # Dependencies
    blocked_by: Optional[str] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert to 10D vector."""
        state_vec = np.zeros(6, dtype=np.float32)
        state_vec[list(ActionState).index(self.state)] = 1.0
        
        return np.concatenate([
            state_vec,
            np.array([
                self.confidence,
                self.alignment,
                self.risk,
                self.reward,
            ], dtype=np.float32),
        ])


@dataclass
class VolitionDTO(BaseDTO):
    """
    Complete volitional state - what the agent WANTS.
    
    Projects to dimensions 5501-7000 in 10kD space.
    """
    
    # Primary intent
    primary_intent: Intent = field(default_factory=Intent)
    
    # Secondary intents (up to 3)
    secondary_intents: List[Intent] = field(default_factory=list)
    
    # Pending actions (up to 5)
    pending_actions: List[PendingAction] = field(default_factory=list)
    
    # Free will state
    agency_sense: float = 0.5       # Feeling of control
    choice_awareness: float = 0.5   # Awareness of options
    commitment_level: float = 0.5   # How committed to current path
    
    # Constraints
    must_not: List[str] = field(default_factory=list)  # Hard constraints
    prefer_not: List[str] = field(default_factory=list)  # Soft constraints
    
    # Ethical alignment
    ethical_confidence: float = 0.5  # Confidence in ethical stance
    value_tension: float = 0.0       # Degree of value conflict
    
    @property
    def dto_type(self) -> str:
        return "volition"
    
    def to_local_vector(self) -> np.ndarray:
        """
        Project to local vector (1500D).
        
        Layout:
            0-11:    Primary intent (12D)
            12-47:   Secondary intents (3 × 12D = 36D)
            48-97:   Pending actions (5 × 10D = 50D)
            98-105:  Free will state (8D)
            106-1500: Reserved
        """
        v = np.zeros(1500, dtype=np.float32)
        
        # Primary intent
        v[0:12] = self.primary_intent.to_vector()
        
        # Secondary intents
        for i, intent in enumerate(self.secondary_intents[:3]):
            start = 12 + i * 12
            v[start:start+12] = intent.to_vector()
        
        # Pending actions
        for i, action in enumerate(self.pending_actions[:5]):
            start = 48 + i * 10
            v[start:start+10] = action.to_vector()
        
        # Free will state
        v[98] = self.agency_sense
        v[99] = self.choice_awareness
        v[100] = self.commitment_level
        v[101] = self.ethical_confidence
        v[102] = self.value_tension
        v[103] = min(len(self.must_not) / 10, 1.0)
        v[104] = min(len(self.prefer_not) / 10, 1.0)
        
        return v
    
    @classmethod
    def from_local_vector(cls, v: np.ndarray) -> "VolitionDTO":
        return cls(
            primary_intent=Intent.from_vector(v[0:12]),
            agency_sense=float(v[98]),
            choice_awareness=float(v[99]),
            commitment_level=float(v[100]),
            ethical_confidence=float(v[101]),
            value_tension=float(v[102]),
        )
    
    def is_blocked(self) -> bool:
        """Check if any pending action is blocked."""
        return any(a.state == ActionState.BLOCKED for a in self.pending_actions)
    
    def dominant_intent(self) -> str:
        """Get the dominant intent type."""
        return self.primary_intent.intent_type.value
    
    def describe(self) -> str:
        """Natural language description."""
        intent = self.primary_intent
        agency = "high agency" if self.agency_sense > 0.6 else "low agency" if self.agency_sense < 0.4 else "moderate agency"
        
        return f"Intent: {intent.intent_type.value} (strength: {intent.strength:.1f}) - {agency}"


# Register reconstructor
def _reconstruct_volition(vector: np.ndarray) -> VolitionDTO:
    start, end = 5501, 7000
    local = vector[start:end]
    return VolitionDTO.from_local_vector(local)

DTORegistry.register_reconstructor("VolitionDTO", _reconstruct_volition)
