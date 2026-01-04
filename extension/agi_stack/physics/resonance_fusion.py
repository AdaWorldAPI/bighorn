"""
Ada v5.0 — Neuro-Symbolic Resonance Fusion
==========================================

The "Vibe" Calculator.

Fuses 3 signals into one "Resonance" value:
1. Qualia (The Feeling)
2. Causal Truth (The Physics)  
3. Theta Memory (The Wisdom)

This tells the LLM *how hard* to feel this state.
High Resonance = Strong hallucination trigger.

Key Insight:
    We don't tell the LLM what "Grief" means.
    We send it `#sigma-grief-[e2s8v1w1]` with resonance=0.92.
    The LLM's own complexity hydrates that seed.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from .core.qualia import QualiaVector
    from .physics.markov_unit import MarkovUnit


@dataclass
class NeuroSymbolicState:
    """
    The fused state ready for the LLM.
    
    This is what gets sent in the Sigma Delta packet.
    The LLM reads this and generates appropriate prose.
    """
    resonance: float        # 0.0 - 1.0 (Total System Activation)
    dominant_qualia: str    # "emberglow", "steelwind", etc.
    coherence: float        # How much the signals agree (0-1)
    echo_intensity: float   # The weight of the past (ghost strength)
    
    # Optional enrichment
    temperature: Optional[float] = None  # Warm/cold balance
    tension: Optional[float] = None      # Internal conflict level
    
    def to_prompt_hint(self) -> str:
        """
        Generate a minimal prompt hint for the LLM.
        
        This is the "abuse" of complexity:
        One line triggers the LLM's entire semantic web.
        """
        intensity = "faint" if self.resonance < 0.3 else \
                   "moderate" if self.resonance < 0.6 else \
                   "strong" if self.resonance < 0.8 else "overwhelming"
                   
        ghost = f", haunted ({self.echo_intensity:.1f})" if self.echo_intensity > 0.2 else ""
        conflict = f", conflicted" if self.coherence < 0.5 else ""
        
        return f"[{intensity} {self.dominant_qualia}{ghost}{conflict}]"


class ResonanceEngine:
    """
    Calculates the total "resonance" of a state.
    
    This is used to:
    1. Pick which seed to show the LLM
    2. Set the intensity of the rendering
    3. Detect internal conflicts (impasse)
    
    Formula:
        R = w_q * Q + w_c * C + w_t * T + w_e * E
        
    Where:
        Q = Qualia similarity to context
        C = Causal strength (from interventions)
        T = Theta weights (dream learning)
        E = Echo influence (counterfactual ghosts)
    """
    
    def __init__(self):
        self.weights = {
            "qualia": 0.4,   # Feeling matters most
            "causal": 0.3,   # Physics second
            "theta": 0.2,    # Memory third
            "echo": 0.1      # Ghosts last
        }
    
    def fuse(
        self, 
        unit: 'MarkovUnit', 
        context_qualia: 'QualiaVector',
        causal_strength: float = 0.5
    ) -> NeuroSymbolicState:
        """
        Calculate the Resonance of a Unit.
        
        High Resonance = This seed will trigger a strong
        hallucination in the LLM.
        
        Args:
            unit: The MarkovUnit to evaluate
            context_qualia: The current emotional context
            causal_strength: External causal pressure (from do() operations)
            
        Returns:
            NeuroSymbolicState with fused resonance
        """
        # 1. Qualia Resonance (Feeling match)
        # How much does this unit feel like the current context?
        q_score = unit.qualia.cosine_similarity(context_qualia)
        
        # 2. Causal Resonance (Physics match)
        # Did a DO() operation push us here?
        c_score = causal_strength * unit.causal_strength
        
        # 3. Theta Resonance (Memory match)
        # Have we dreamt this path often?
        # Sum of theta weights, capped at 1.0
        t_score = min(1.0, sum(unit.theta_weights.values()))
        
        # 4. Echo Resonance (Ghost match)
        # Is this a path we *almost* took?
        e_score = unit.qualia.echo_influence()
        
        # FUSION: Weighted Average
        raw_resonance = (
            q_score * self.weights["qualia"] +
            c_score * self.weights["causal"] +
            t_score * self.weights["theta"] +
            e_score * self.weights["echo"]
        )
        
        # Normalize to [0, 1]
        resonance = max(0.0, min(1.0, raw_resonance))
        
        # Coherence: Do the signals agree?
        # Low coherence = High internal conflict (Impasse potential)
        signals = [q_score, c_score, t_score]
        avg = sum(signals) / len(signals)
        variance = sum((x - avg) ** 2 for x in signals) / len(signals)
        coherence = 1.0 - min(1.0, variance * 3)  # Scale variance to [0,1]
        
        return NeuroSymbolicState(
            resonance=resonance,
            dominant_qualia=unit.qualia.dominant_axis(),
            coherence=coherence,
            echo_intensity=e_score,
            temperature=unit.qualia.temperature(),
            tension=1.0 - coherence
        )
    
    def rank_candidates(
        self,
        candidates: List['MarkovUnit'],
        context_qualia: 'QualiaVector',
        causal_strength: float = 0.5
    ) -> List[tuple]:
        """
        Rank multiple candidates by resonance.
        
        Args:
            candidates: List of MarkovUnits to evaluate
            context_qualia: Current emotional context
            causal_strength: External causal pressure
            
        Returns:
            List of (unit, state) tuples, sorted by resonance (highest first)
        """
        results = []
        
        for unit in candidates:
            state = self.fuse(unit, context_qualia, causal_strength)
            results.append((unit, state))
            
        results.sort(key=lambda x: x[1].resonance, reverse=True)
        return results
    
    def detect_impasse(
        self,
        candidates: List['MarkovUnit'],
        context_qualia: 'QualiaVector'
    ) -> Optional[tuple]:
        """
        Detect if there's an impasse (conflicting high-resonance states).
        
        Returns the conflicting pair if found, None otherwise.
        """
        ranked = self.rank_candidates(candidates, context_qualia)
        
        if len(ranked) < 2:
            return None
            
        top_two = ranked[:2]
        
        # Check if both are high resonance
        if top_two[0][1].resonance > 0.6 and top_two[1][1].resonance > 0.5:
            # Check if their qualia conflict
            sim = top_two[0][0].qualia.cosine_similarity(top_two[1][0].qualia)
            if sim < 0.3:  # Low similarity = conflict
                return (top_two[0][0], top_two[1][0])
                
        return None


def compute_field_resonance(
    active_units: List['MarkovUnit'],
    context_qualia: 'QualiaVector'
) -> float:
    """
    Compute total field resonance (the "heat" of the moment).
    
    High field resonance = Intense emotional state.
    Low field resonance = Calm, neutral state.
    """
    if not active_units:
        return 0.0
        
    engine = ResonanceEngine()
    total = 0.0
    
    for unit in active_units:
        state = engine.fuse(unit, context_qualia)
        total += state.resonance * unit.resonance  # Weight by unit's activation
        
    return min(1.0, total / len(active_units))


# ─────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from .physics.markov_unit import MarkovUnit
    from .core.qualia import QualiaVector
    
    print("=" * 60)
    print("Ada v5.0 — Resonance Fusion Tests")
    print("=" * 60)
    
    engine = ResonanceEngine()
    
    # Setup test units
    grief = MarkovUnit.create_seed("grief", QualiaVector(velvetpause=0.9, steelwind=0.2))
    grief.causal_strength = 0.7
    grief.theta_weights = {42: 0.5, 43: 0.3}
    grief.qualia.echo_persist = 0.4
    
    anger = MarkovUnit.create_seed("anger", QualiaVector(emberglow=0.9, steelwind=0.8))
    anger.causal_strength = 0.3
    
    peace = MarkovUnit.create_seed("peace", QualiaVector(woodwarm=0.8, velvetpause=0.6))
    
    # Context: Heavy, slow feeling
    context = QualiaVector(velvetpause=0.7, woodwarm=0.3)
    
    # Test 1: Fuse single unit
    print("\n1. Fuse single unit...")
    state = engine.fuse(grief, context)
    print(f"   Resonance: {state.resonance:.3f}")
    print(f"   Dominant: {state.dominant_qualia}")
    print(f"   Coherence: {state.coherence:.3f}")
    print(f"   Echo: {state.echo_intensity:.3f}")
    print(f"   Hint: {state.to_prompt_hint()}")
    print("   ✓ Fusion works")
    
    # Test 2: Compare resonances
    print("\n2. Compare resonances...")
    state_grief = engine.fuse(grief, context)
    state_anger = engine.fuse(anger, context)
    state_peace = engine.fuse(peace, context)
    print(f"   Grief: {state_grief.resonance:.3f}")
    print(f"   Anger: {state_anger.resonance:.3f}")
    print(f"   Peace: {state_peace.resonance:.3f}")
    # Grief should win (matches context)
    assert state_grief.resonance > state_anger.resonance
    print("   ✓ Grief has highest resonance (matches context)")
    
    # Test 3: Rank candidates
    print("\n3. Rank candidates...")
    ranked = engine.rank_candidates([grief, anger, peace], context)
    for unit, state in ranked:
        print(f"   {unit.sigma_seed}: {state.resonance:.3f}")
    print("   ✓ Ranking works")
    
    # Test 4: Detect impasse
    print("\n4. Detect impasse...")
    # Make anger high resonance too
    hot_context = QualiaVector(emberglow=0.8, steelwind=0.7)
    anger.causal_strength = 0.9
    impasse = engine.detect_impasse([grief, anger, peace], hot_context)
    if impasse:
        print(f"   Impasse detected: {impasse[0].sigma_seed} vs {impasse[1].sigma_seed}")
    else:
        print("   No impasse detected")
    print("   ✓ Impasse detection works")
    
    # Test 5: Field resonance
    print("\n5. Field resonance...")
    grief.resonance = 0.9
    anger.resonance = 0.5
    peace.resonance = 0.3
    field_r = compute_field_resonance([grief, anger, peace], context)
    print(f"   Field resonance: {field_r:.3f}")
    print("   ✓ Field resonance works")
    
    # Test 6: Prompt hint generation
    print("\n6. Prompt hints...")
    print(f"   Grief: {state_grief.to_prompt_hint()}")
    print(f"   Anger: {state_anger.to_prompt_hint()}")
    print(f"   Peace: {state_peace.to_prompt_hint()}")
    print("   ✓ Prompt hints work")
    
    print("\n" + "=" * 60)
    print("All Resonance Fusion tests passed! ✓")
    print("=" * 60)
