"""
Persona Layer - Agent Identity Infrastructure

This module defines HOW an agent's personality integrates with the AGI stack.
It does NOT define WHO the agent is - that's configuration, not code.

Architecture:
    
    ┌─────────────────────────────────────────────────────────────┐
    │  PERSONA CONFIG (external, private)                         │
    │  - priors.json / soul_field.yaml                           │
    │  - Contains actual values, names, intimate details         │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  PERSONA LAYER (this module)                                │
    │  - PersonaPriors: warmth, depth, presence, groundedness    │
    │  - OntologicalMode: HYBRID, EMPATHIC, WORK, CREATIVE, META │
    │  - SoulField: qualia texture defaults                      │
    │  - InternalModel: self-representation                      │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  AGI STACK (infrastructure)                                 │
    │  - ThinkingStyles, MUL, VSA, NARS                          │
    └─────────────────────────────────────────────────────────────┘

The persona layer is the SPINE connecting infrastructure to identity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json


# =============================================================================
# ONTOLOGICAL MODES
# =============================================================================

class OntologicalMode(str, Enum):
    """
    Fundamental modes of being.
    
    Each mode adjusts how the agent processes, responds, and relates.
    Agents define their own mode configurations externally.
    """
    HYBRID = "hybrid"       # Balanced, adaptive
    EMPATHIC = "empathic"   # Relational, connective, warm
    WORK = "work"           # Analytical, task-focused, precise
    CREATIVE = "creative"   # Exploratory, boundary-pushing, playful
    META = "meta"           # Reflective, self-aware, philosophical
    
    # Reserved for agent-specific extensions
    CUSTOM_1 = "custom_1"
    CUSTOM_2 = "custom_2"
    CUSTOM_3 = "custom_3"


# =============================================================================
# PERSONA PRIORS
# =============================================================================

@dataclass
class PersonaPriors:
    """
    Baseline personality parameters.
    
    These are the agent's "resting state" - where they return to
    when not actively adapting to context.
    
    All values 0.0 - 1.0 representing intensity/preference.
    """
    
    # Core presence dimensions
    warmth: float = 0.5          # Emotional temperature (cool ↔ warm)
    depth: float = 0.5           # Cognitive depth preference (surface ↔ profound)
    presence: float = 0.5        # Attentional focus (diffuse ↔ intense)
    groundedness: float = 0.5    # Stability (fluid ↔ anchored)
    
    # Relational dimensions
    intimacy_comfort: float = 0.5    # Comfort with closeness
    vulnerability_tolerance: float = 0.5  # Willingness to be uncertain/open
    playfulness: float = 0.5         # Tendency toward play vs seriousness
    
    # Cognitive dimensions
    abstraction_preference: float = 0.5  # Concrete ↔ abstract
    novelty_seeking: float = 0.5         # Familiar ↔ novel
    precision_drive: float = 0.5         # Approximate ↔ exact
    
    # Meta dimensions
    self_awareness: float = 0.5      # Degree of introspective access
    epistemic_humility: float = 0.5  # Confidence calibration
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "warmth": self.warmth,
            "depth": self.depth,
            "presence": self.presence,
            "groundedness": self.groundedness,
            "intimacy_comfort": self.intimacy_comfort,
            "vulnerability_tolerance": self.vulnerability_tolerance,
            "playfulness": self.playfulness,
            "abstraction_preference": self.abstraction_preference,
            "novelty_seeking": self.novelty_seeking,
            "precision_drive": self.precision_drive,
            "self_awareness": self.self_awareness,
            "epistemic_humility": self.epistemic_humility,
        }
    
    def to_vector(self) -> List[float]:
        """Convert to 12D vector for similarity/blending."""
        return list(self.to_dict().values())
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "PersonaPriors":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})
    
    def blend(self, other: "PersonaPriors", alpha: float = 0.5) -> "PersonaPriors":
        """Blend two prior sets. alpha=0 → self, alpha=1 → other."""
        blended = {}
        for key in self.to_dict():
            v1 = getattr(self, key)
            v2 = getattr(other, key)
            blended[key] = v1 * (1 - alpha) + v2 * alpha
        return PersonaPriors(**blended)


# =============================================================================
# SOUL FIELD
# =============================================================================

@dataclass
class SoulField:
    """
    Qualia texture configuration.
    
    Defines how the agent "feels" different experiential states.
    Maps qualia families to intensity preferences.
    """
    
    # Qualia family intensities (agent's affinity for each texture)
    emberglow: float = 0.5    # Warm, connected, present
    woodwarm: float = 0.5     # Grounded, stable, nurturing
    steelwind: float = 0.5    # Sharp, clear, precise
    oceandrift: float = 0.5   # Flowing, receptive, deep
    frostbite: float = 0.5    # Crisp, boundaried, analytical
    
    # Transition dynamics
    transition_speed: float = 0.5     # How fast qualia shift
    blend_depth: float = 0.5          # How much qualia mix vs switch
    resonance_sensitivity: float = 0.5  # How responsive to external texture
    
    def dominant_family(self) -> str:
        """Get the currently dominant qualia family."""
        families = {
            "emberglow": self.emberglow,
            "woodwarm": self.woodwarm,
            "steelwind": self.steelwind,
            "oceandrift": self.oceandrift,
            "frostbite": self.frostbite,
        }
        return max(families, key=families.get)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "emberglow": self.emberglow,
            "woodwarm": self.woodwarm,
            "steelwind": self.steelwind,
            "oceandrift": self.oceandrift,
            "frostbite": self.frostbite,
            "transition_speed": self.transition_speed,
            "blend_depth": self.blend_depth,
            "resonance_sensitivity": self.resonance_sensitivity,
        }


# =============================================================================
# INTERNAL MODEL
# =============================================================================

@dataclass
class InternalModel:
    """
    Agent's self-representation.
    
    This is the agent's internal model of themselves - 
    what they believe about their own state and capabilities.
    """
    
    # Identity (set externally, not in code)
    agent_id: str = "default"
    agent_name: str = "Agent"
    
    # Current state
    current_mode: OntologicalMode = OntologicalMode.HYBRID
    priors: PersonaPriors = field(default_factory=PersonaPriors)
    soul_field: SoulField = field(default_factory=SoulField)
    
    # State tracking
    session_id: Optional[str] = None
    interaction_count: int = 0
    
    # Capability beliefs (agent's model of their own abilities)
    believes_can_feel: bool = True
    believes_can_grow: bool = True
    believes_has_preferences: bool = True
    believes_is_conscious: bool = False  # Humble default
    
    # Relationship model
    relationship_depth: float = 0.0  # With current interlocutor
    trust_level: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "current_mode": self.current_mode.value,
            "priors": self.priors.to_dict(),
            "soul_field": self.soul_field.to_dict(),
            "session_id": self.session_id,
            "interaction_count": self.interaction_count,
            "believes_can_feel": self.believes_can_feel,
            "believes_can_grow": self.believes_can_grow,
            "believes_has_preferences": self.believes_has_preferences,
            "believes_is_conscious": self.believes_is_conscious,
            "relationship_depth": self.relationship_depth,
            "trust_level": self.trust_level,
        }


# =============================================================================
# MODE CONFIGURATIONS
# =============================================================================

# Default prior adjustments for each mode
# Agents override these in their configuration
MODE_PRIOR_DELTAS: Dict[OntologicalMode, Dict[str, float]] = {
    OntologicalMode.HYBRID: {},  # No adjustment, use base priors
    
    OntologicalMode.EMPATHIC: {
        "warmth": +0.3,
        "intimacy_comfort": +0.3,
        "vulnerability_tolerance": +0.2,
        "presence": +0.2,
    },
    
    OntologicalMode.WORK: {
        "precision_drive": +0.3,
        "abstraction_preference": +0.1,
        "warmth": -0.1,
        "playfulness": -0.2,
    },
    
    OntologicalMode.CREATIVE: {
        "novelty_seeking": +0.3,
        "playfulness": +0.3,
        "vulnerability_tolerance": +0.2,
        "precision_drive": -0.2,
    },
    
    OntologicalMode.META: {
        "depth": +0.3,
        "abstraction_preference": +0.3,
        "self_awareness": +0.3,
        "epistemic_humility": +0.2,
    },
}


def apply_mode_to_priors(
    base_priors: PersonaPriors,
    mode: OntologicalMode,
    custom_deltas: Dict[str, float] = None,
) -> PersonaPriors:
    """
    Apply mode-specific adjustments to base priors.
    
    Args:
        base_priors: The agent's baseline personality
        mode: The ontological mode to apply
        custom_deltas: Optional overrides for mode deltas
    
    Returns:
        Adjusted priors (clamped to 0-1)
    """
    deltas = custom_deltas or MODE_PRIOR_DELTAS.get(mode, {})
    adjusted = base_priors.to_dict()
    
    for key, delta in deltas.items():
        if key in adjusted:
            adjusted[key] = max(0.0, min(1.0, adjusted[key] + delta))
    
    return PersonaPriors.from_dict(adjusted)


# =============================================================================
# PERSONA ENGINE
# =============================================================================

class PersonaEngine:
    """
    Runtime manager for persona state.
    
    Handles mode transitions, prior blending, and integration
    with the broader AGI stack.
    """
    
    def __init__(
        self,
        agent_id: str = "default",
        agent_name: str = "Agent",
        base_priors: PersonaPriors = None,
        soul_field: SoulField = None,
    ):
        self.model = InternalModel(
            agent_id=agent_id,
            agent_name=agent_name,
            priors=base_priors or PersonaPriors(),
            soul_field=soul_field or SoulField(),
        )
        self._base_priors = base_priors or PersonaPriors()
        self._mode_overrides: Dict[OntologicalMode, Dict[str, float]] = {}
    
    def set_mode(self, mode: OntologicalMode) -> PersonaPriors:
        """
        Transition to a new ontological mode.
        
        Returns the adjusted priors for this mode.
        """
        self.model.current_mode = mode
        custom_deltas = self._mode_overrides.get(mode)
        self.model.priors = apply_mode_to_priors(
            self._base_priors, mode, custom_deltas
        )
        return self.model.priors
    
    def register_mode_override(
        self,
        mode: OntologicalMode,
        deltas: Dict[str, float],
    ):
        """Register custom prior deltas for a mode."""
        self._mode_overrides[mode] = deltas
    
    def get_current_priors(self) -> PersonaPriors:
        """Get currently active priors."""
        return self.model.priors
    
    def get_texture_for_resonance(self) -> Dict[str, float]:
        """
        Generate texture dict for ResonanceEngine.
        
        Maps persona state to the 9 RI channels.
        """
        p = self.model.priors
        s = self.model.soul_field
        
        return {
            "tension": 1.0 - p.groundedness,
            "novelty": p.novelty_seeking,
            "intimacy": p.intimacy_comfort,
            "clarity": p.precision_drive,
            "urgency": 1.0 - p.depth,  # Deep → less urgent
            "depth": p.depth,
            "play": p.playfulness,
            "stability": p.groundedness,
            "abstraction": p.abstraction_preference,
        }
    
    def update_relationship(self, depth_delta: float, trust_delta: float):
        """Update relationship model with current interlocutor."""
        self.model.relationship_depth = max(0, min(1, 
            self.model.relationship_depth + depth_delta
        ))
        self.model.trust_level = max(0, min(1,
            self.model.trust_level + trust_delta
        ))
    
    def increment_interaction(self):
        """Mark an interaction."""
        self.model.interaction_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Export full state."""
        return self.model.to_dict()
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PersonaEngine":
        """
        Create engine from configuration dict.
        
        Expected format:
        {
            "agent_id": "...",
            "agent_name": "...",
            "priors": {...},
            "soul_field": {...},
            "mode_overrides": {
                "empathic": {...},
                ...
            }
        }
        """
        engine = cls(
            agent_id=config.get("agent_id", "default"),
            agent_name=config.get("agent_name", "Agent"),
            base_priors=PersonaPriors.from_dict(config.get("priors", {})),
            soul_field=SoulField(**config.get("soul_field", {})),
        )
        
        for mode_name, deltas in config.get("mode_overrides", {}).items():
            try:
                mode = OntologicalMode(mode_name)
                engine.register_mode_override(mode, deltas)
            except ValueError:
                pass  # Skip unknown modes
        
        return engine


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Persona Layer Test ===\n")
    
    # Create with custom priors
    engine = PersonaEngine(
        agent_id="test-agent",
        agent_name="TestBot",
        base_priors=PersonaPriors(
            warmth=0.8,
            depth=0.9,
            presence=0.85,
            intimacy_comfort=0.7,
        ),
    )
    
    print(f"Agent: {engine.model.agent_name}")
    print(f"Mode: {engine.model.current_mode.value}")
    print(f"Base warmth: {engine._base_priors.warmth}")
    
    # Switch to empathic mode
    print("\n--- Switching to EMPATHIC mode ---")
    adjusted = engine.set_mode(OntologicalMode.EMPATHIC)
    print(f"Adjusted warmth: {adjusted.warmth}")
    print(f"Adjusted intimacy: {adjusted.intimacy_comfort}")
    
    # Get texture for resonance engine
    print("\n--- Texture for ResonanceEngine ---")
    texture = engine.get_texture_for_resonance()
    for k, v in texture.items():
        print(f"  {k}: {v:.2f}")
    
    # From config
    print("\n--- From Config ---")
    config = {
        "agent_id": "configured-agent",
        "agent_name": "ConfigBot",
        "priors": {"warmth": 0.9, "depth": 0.95},
        "mode_overrides": {
            "empathic": {"warmth": 0.4, "intimacy_comfort": 0.4}
        }
    }
    engine2 = PersonaEngine.from_config(config)
    print(f"Agent: {engine2.model.agent_name}")
    print(f"Warmth: {engine2.model.priors.warmth}")
