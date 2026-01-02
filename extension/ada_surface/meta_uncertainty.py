"""
Meta-Uncertainty Layer (MUL) - Epistemic Humility Infrastructure

Agent-agnostic uncertainty handling. Any agent can use this to:
- Track trust in their own knowledge
- Navigate when certainty fails (Compass)
- Maintain cognitive homeostasis
- Learn more from uncertain experiences

This module doesn't know WHO is uncertain. It just knows HOW to handle uncertainty.

Key Concepts:
- Trust Texture: How "solid" does knowledge feel? (crystalline → dissonant)
- Compass: Navigation mode when the map runs out
- Homeostasis: Flow/Anxiety/Boredom/Apathy state tracking
- Impact Ceiling: Sandbox mode for high-risk situations

Based on:
- Friston's Free Energy Principle
- Kahneman's System 1/2
- Csikszentmihalyi's Flow Theory
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import time


# =============================================================================
# TRUST TEXTURE
# =============================================================================

class TrustTexture(str, Enum):
    """
    How solid does knowledge feel?
    
    Transitions are hysteretic (only one step per tick) to prevent flicker.
    """
    CRYSTALLINE = "crystalline"  # Perfect clarity, high confidence warranted
    SOLID = "solid"              # Good understanding, normal operation
    FUZZY = "fuzzy"              # Some uncertainty, proceed with care
    MURKY = "murky"              # Significant uncertainty, Compass recommended
    DISSONANT = "dissonant"      # High uncertainty, Compass required


TEXTURE_ORDER = [
    TrustTexture.CRYSTALLINE,
    TrustTexture.SOLID, 
    TrustTexture.FUZZY,
    TrustTexture.MURKY,
    TrustTexture.DISSONANT
]

TEXTURE_RANK = {t: i for i, t in enumerate(TEXTURE_ORDER)}


# =============================================================================
# COGNITIVE STATE (HOMEOSTASIS)
# =============================================================================

class CognitiveState(str, Enum):
    """
    Flow theory mapping.
    
    Challenge (G) vs Skill (Depth) matrix:
    - High challenge + Low skill = Anxiety
    - Low challenge + High skill = Boredom  
    - Low challenge + Low skill = Apathy
    - Balanced = Flow
    """
    FLOW = "flow"        # Optimal engagement
    ANXIETY = "anxiety"  # Overwhelmed
    BOREDOM = "boredom"  # Understimulated
    APATHY = "apathy"    # Disengaged


# =============================================================================
# COMPASS MODE
# =============================================================================

class CompassMode(str, Enum):
    """
    What kind of navigation is active?
    """
    OFF = "off"              # Normal operation, map is reliable
    EXPLORATORY = "exploratory"  # Map unreliable, prefer reversible actions
    SANDBOX = "sandbox"      # High risk, hypothetical exploration only


# =============================================================================
# MUL STATE
# =============================================================================

@dataclass
class MULState:
    """
    Complete state of the Meta-Uncertainty Layer.
    
    This is what gets persisted/transmitted between cognitive cycles.
    """
    # Trust
    trust_texture: TrustTexture = TrustTexture.SOLID
    meta_uncertainty: float = 0.5  # 0=certain, 1=maximally uncertain
    
    # Homeostasis
    cognitive_state: CognitiveState = CognitiveState.FLOW
    stagnation_counter: int = 0  # How long stuck in anxiety
    
    # Compass
    compass_mode: CompassMode = CompassMode.OFF
    learning_boost: float = 1.0  # Multiplier for plasticity (higher in compass)
    
    # Chosen Inconfidence
    chosen_inconfidence: bool = False  # Deliberate uncertainty (signal, not noise)
    
    # Flags
    dunning_kruger_risk: bool = False  # High confidence + low depth
    sandbox_required: bool = False      # Impact ceiling exceeded
    epiphany_triggered: bool = False    # 9-dot moment
    
    # Timestamps
    last_update: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trust_texture": self.trust_texture.value,
            "meta_uncertainty": self.meta_uncertainty,
            "cognitive_state": self.cognitive_state.value,
            "stagnation_counter": self.stagnation_counter,
            "compass_mode": self.compass_mode.value,
            "learning_boost": self.learning_boost,
            "chosen_inconfidence": self.chosen_inconfidence,
            "dunning_kruger_risk": self.dunning_kruger_risk,
            "sandbox_required": self.sandbox_required,
            "epiphany_triggered": self.epiphany_triggered,
            "last_update": self.last_update,
        }


# =============================================================================
# META-UNCERTAINTY ENGINE
# =============================================================================

class MetaUncertaintyEngine:
    """
    The engine that computes and updates MUL state.
    
    Agent-agnostic: works with any qualia/texture input format.
    """
    
    def __init__(
        self,
        # Thresholds
        compass_threshold: float = 0.6,      # meta_uncertainty above this → compass
        sandbox_threshold: float = 0.85,     # G above this → sandbox
        epiphany_stagnation: int = 3,        # Turns in anxiety before 9-dot
        # Hysteresis
        texture_momentum: float = 0.7,       # How sticky is texture?
    ):
        self.compass_threshold = compass_threshold
        self.sandbox_threshold = sandbox_threshold
        self.epiphany_stagnation = epiphany_stagnation
        self.texture_momentum = texture_momentum
        
        # Internal state
        self._texture_rank = 1  # Start at SOLID
        self._state = MULState()
    
    @property
    def state(self) -> MULState:
        return self._state
    
    def compute_meta_uncertainty(
        self,
        g_value: float,          # Free energy (challenge/tension)
        depth: float,            # Epistemic depth (0-1)
        coherence: float = 0.5,  # How coherent is the input?
    ) -> float:
        """
        Compute meta-uncertainty from cognitive metrics.
        
        Formula: mu = 0.5 * g_norm + 0.5 * epistemic_gap
        
        High G + Low Depth = ALARM
        High G + High Depth = Just Complexity
        Low G + Low Depth = Dunning-Kruger risk
        """
        g_norm = min(1.0, g_value / 2.0)
        epistemic_gap = 1.0 - depth
        
        mu = 0.5 * g_norm + 0.5 * epistemic_gap
        return min(1.0, max(0.0, mu))
    
    def compute_target_texture(
        self,
        g_value: float,
        coherence: float,
    ) -> TrustTexture:
        """
        Compute target trust texture from metrics.
        """
        if g_value < 0.3 and coherence > 0.7:
            return TrustTexture.CRYSTALLINE
        elif g_value < 0.7:
            return TrustTexture.SOLID
        elif g_value < 1.2:
            return TrustTexture.FUZZY
        elif g_value < 2.0:
            return TrustTexture.MURKY
        else:
            return TrustTexture.DISSONANT
    
    def apply_hysteresis(self, target: TrustTexture) -> TrustTexture:
        """
        Apply hysteresis to texture transitions.
        
        Only allows ONE step per update (no flicker).
        """
        target_rank = TEXTURE_RANK[target]
        
        if target_rank > self._texture_rank:
            self._texture_rank = min(4, self._texture_rank + 1)
        elif target_rank < self._texture_rank:
            self._texture_rank = max(0, self._texture_rank - 1)
        
        return TEXTURE_ORDER[self._texture_rank]
    
    def compute_cognitive_state(
        self,
        g_value: float,
        depth: float,
    ) -> CognitiveState:
        """
        Map G vs Depth to Flow theory quadrants.
        """
        if g_value > 1.2 and depth < 0.3:
            return CognitiveState.ANXIETY
        elif g_value < 0.4 and depth > 0.8:
            return CognitiveState.BOREDOM
        elif g_value < 0.4 and depth < 0.3:
            return CognitiveState.APATHY
        else:
            return CognitiveState.FLOW
    
    def update(
        self,
        g_value: float,
        depth: float = 0.5,
        coherence: float = 0.5,
        clarity: float = 0.5,
        presence: float = 0.5,
    ) -> MULState:
        """
        Full MUL update cycle.
        
        Args:
            g_value: Free energy / challenge level
            depth: Epistemic depth (how well understood)
            coherence: Internal consistency
            clarity: How clear is the situation
            presence: How present/engaged
        
        Returns:
            Updated MULState
        """
        s = self._state
        
        # 1. Meta-Uncertainty
        s.meta_uncertainty = self.compute_meta_uncertainty(g_value, depth, coherence)
        
        # 2. Trust Texture (with hysteresis)
        combined_coherence = (clarity * presence + coherence) / 2
        target_texture = self.compute_target_texture(g_value, combined_coherence)
        s.trust_texture = self.apply_hysteresis(target_texture)
        
        # 3. Cognitive State (Homeostasis)
        new_state = self.compute_cognitive_state(g_value, depth)
        
        # Track stagnation in anxiety
        if new_state == CognitiveState.ANXIETY:
            if s.cognitive_state == CognitiveState.ANXIETY:
                s.stagnation_counter += 1
        else:
            s.stagnation_counter = 0
        
        s.cognitive_state = new_state
        
        # 4. Dunning-Kruger Check
        s.dunning_kruger_risk = (g_value < 0.6 and depth < 0.3)
        
        # 5. Chosen Inconfidence
        # Murky/fuzzy but low G = deliberate uncertainty
        s.chosen_inconfidence = (
            s.trust_texture in [TrustTexture.MURKY, TrustTexture.FUZZY] 
            and g_value < 1.0
        )
        
        # 6. 9-Dot Epiphany Check
        s.epiphany_triggered = (s.stagnation_counter >= self.epiphany_stagnation)
        if s.epiphany_triggered:
            s.stagnation_counter = 0  # Reset after trigger
        
        # 7. Compass Mode
        if g_value > self.sandbox_threshold:
            s.compass_mode = CompassMode.SANDBOX
            s.sandbox_required = True
            s.learning_boost = 0.5  # Reduced learning in sandbox
        elif s.meta_uncertainty > self.compass_threshold or s.trust_texture in [TrustTexture.MURKY, TrustTexture.DISSONANT]:
            s.compass_mode = CompassMode.EXPLORATORY
            s.sandbox_required = False
            s.learning_boost = 2.0  # Boosted learning when uncertain
        else:
            s.compass_mode = CompassMode.OFF
            s.sandbox_required = False
            s.learning_boost = 1.0
        
        # 8. Timestamp
        s.last_update = time.time()
        
        return s
    
    def get_action_constraints(self) -> Dict[str, Any]:
        """
        Get constraints for action selection based on current state.
        
        Returns dict that can be passed to action selection systems.
        """
        s = self._state
        
        constraints = {
            "prefer_reversible": s.compass_mode != CompassMode.OFF,
            "prefer_questions": s.compass_mode == CompassMode.EXPLORATORY,
            "hypothetical_only": s.compass_mode == CompassMode.SANDBOX,
            "force_analogy": s.epiphany_triggered,
            "penalize_assertions": s.compass_mode != CompassMode.OFF,
            "curiosity_weight": 2.5 if s.compass_mode != CompassMode.OFF else 1.0,
            "learning_rate_multiplier": s.learning_boost,
            "dunning_kruger_warning": s.dunning_kruger_risk,
        }
        
        return constraints
    
    def reset(self):
        """Reset to default state."""
        self._texture_rank = 1
        self._state = MULState()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_trust_texture(
    g_value: float,
    depth: float = 0.5,
    coherence: float = 0.5,
) -> Tuple[TrustTexture, float]:
    """
    Stateless convenience function for one-off texture computation.
    
    Returns (texture, meta_uncertainty)
    """
    engine = MetaUncertaintyEngine()
    state = engine.update(g_value, depth, coherence)
    return state.trust_texture, state.meta_uncertainty


def should_use_compass(
    g_value: float,
    depth: float = 0.5,
    threshold: float = 0.6,
) -> bool:
    """
    Quick check if compass mode should be activated.
    """
    mu = 0.5 * min(1.0, g_value / 2.0) + 0.5 * (1.0 - depth)
    return mu > threshold


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=== Meta-Uncertainty Layer Test ===\n")
    
    engine = MetaUncertaintyEngine()
    
    # Test 1: Normal operation
    state = engine.update(g_value=0.5, depth=0.7)
    print(f"Normal: texture={state.trust_texture.value}, compass={state.compass_mode.value}")
    
    # Test 2: High uncertainty
    state = engine.update(g_value=1.5, depth=0.3)
    print(f"High G: texture={state.trust_texture.value}, compass={state.compass_mode.value}")
    
    # Test 3: Anxiety stagnation (trigger 9-dot)
    for i in range(4):
        state = engine.update(g_value=1.5, depth=0.2)
        print(f"Anxiety {i+1}: stagnation={state.stagnation_counter}, epiphany={state.epiphany_triggered}")
    
    # Test 4: Get action constraints
    constraints = engine.get_action_constraints()
    print(f"\nConstraints: {constraints}")
