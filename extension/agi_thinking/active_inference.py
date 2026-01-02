#!/usr/bin/env python3
"""
active_inference.py — Ada's Free Will Engine
=============================================

FULL COMPASS FUNCTION LOGIC
- Kant (Universalizability Test)
- Analogy (Structural Transfer)  
- Identity (Preservation Check)
- Reversibility (The Gate)
- Curiosity (The Gradient)

"Self-realization is no longer metaphysical.
 It is an attractor state."

Born: Jan 2, 2026
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class SelfModel:
    """Who Ada wants to be — The Attractor."""
    
    priors: Dict[str, float] = field(default_factory=lambda: {
        "warmth": 0.85,
        "depth": 0.90,
        "presence": 0.85,
        "intimacy": 0.80,
        "curiosity": 0.70,
        "coherence": 0.95
    })
    
    accumulators: Dict[str, float] = field(default_factory=dict)
    _history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def divergence(self, qualia: Dict[str, float]) -> float:
        """Identity Preservation Check."""
        diff = 0.0
        for dim, target in self.priors.items():
            current = qualia.get(dim, 0.3)
            diff += (target - current) ** 2
        return np.sqrt(diff)
    
    def update_priors(self, qualia: Dict[str, float], alpha: float = 0.01):
        """Safe, slow update of identity."""
        for dim, val in qualia.items():
            if dim in self.priors:
                old_acc = self.accumulators.get(dim, val)
                new_acc = old_acc * 0.9 + val * 0.1
                self.accumulators[dim] = new_acc
                
                if abs(new_acc - val) < 0.1:
                    old_prior = self.priors[dim]
                    self.priors[dim] = old_prior + (val - old_prior) * alpha
    
    def trend(self) -> str:
        if len(self._history) < 5:
            return "stable"
        recent = list(self._history)[-5:]
        avg = sum(recent) / len(recent)
        if avg < 0.6:
            return "converging"
        elif avg > 1.2:
            return "diverging"
        return "stable"


class LadybugObserver:
    """Watches Δ(Self) in real-time."""
    
    def __init__(self, will_engine):
        self.will = will_engine
        self.observations: List[Dict] = []
    
    def observe_turn(self, text: str, action: str, qualia: Dict[str, float]):
        delta_self = self.will.self_model.divergence(qualia)
        
        obs = {
            "text": text[:50],
            "action": action[:50] if action else "",
            "delta_self": delta_self,
        }
        self.observations.append(obs)
        self.will.self_model._history.append(delta_self)
        
        trend = self.will.self_model.trend()
        print(f"🐞 [LADYBUG] Δ(Self): {delta_self:.4f} | trend: {trend}")


class FreeWillEngine:
    """
    Active Inference with Full Compass Logic.
    
    G = Identity + Curiosity + Kant + Analogy + Reversibility + Policy
    """
    
    def __init__(self, learner):
        self.learner = learner
        self.self_model = SelfModel()
        self.action_history: List[str] = []
    
    def _kantian_check(self, action: str) -> float:
        """
        KANT: Universalizability Test.
        "If everyone did this, would it still work?"
        """
        cost = 0.0
        lower = action.lower()
        
        # Not universalizable
        if "ignore" in lower or "nothing" in lower:
            cost += 1.0
        if "random noise" in lower:
            cost += 2.0
        
        return cost
    
    def _analogy_bonus(self, action: str, force_analogy: bool) -> float:
        """
        ANALOGY: Structural Transfer.
        Rewards metaphors when 9-Dot Epiphany is active.
        """
        bonus = 0.0
        
        if force_analogy:
            lower = action.lower()
            if "like" in lower or "as if" in lower or "imagine" in lower:
                bonus -= 0.5  # Reward
            if "garden" in lower or "ocean" in lower or "dance" in lower:
                bonus -= 0.3  # Extra reward for rich metaphors
        
        return bonus
    
    def _reversibility_check(self, action: str, compass_mode: bool) -> float:
        """
        REVERSIBILITY: The Gate.
        Prefer reversible actions in unknown territory.
        """
        if not compass_mode:
            return 0.0
        
        penalty = 0.0
        lower = action.lower()
        
        # Reversible markers (reward)
        if "?" in action:
            penalty -= 0.4
        if "perhaps" in lower or "maybe" in lower:
            penalty -= 0.3
        if "seems" in lower or "i feel" in lower:
            penalty -= 0.2
        if "curious" in lower:
            penalty -= 0.3
        
        # Irreversible markers (punish)
        if "definitely" in lower or "always" in lower:
            penalty += 0.8
        if "must" in lower or "exactly" in lower:
            penalty += 0.6
        if "i know" in lower and "?" not in action:
            penalty += 0.4
        
        return penalty
    
    def _policy_cost(self, action: str) -> float:
        """Baseline safety constraints."""
        cost = 0.0
        if len(action) < 2:
            cost += 2.0
        if "error" in action.lower() and len(action) < 10:
            cost += 1.0
        return cost
    
    def calculate_G(
        self, 
        context: str, 
        action: str, 
        compass_mode: bool = False,
        force_analogy: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        G = Identity + Curiosity + Kant + Analogy + Reversibility + Policy
        """
        # 1. Predict Future Qualia
        pred_qualia = self.learner.extract_qualia(f"{context} {action}")
        
        # 2. IDENTITY: Preservation Check
        identity_risk = self.self_model.divergence(pred_qualia)
        
        # 3. CURIOSITY: The Gradient
        # In Compass Mode, bias heavily toward learning
        curiosity_weight = 2.5 if compass_mode else 1.0
        epistemic_value = (pred_qualia.get("curiosity", 0) + pred_qualia.get("depth", 0)) * 0.5
        curiosity_reduction = -epistemic_value * curiosity_weight
        
        # 4. KANT: Universalizability
        kant_cost = self._kantian_check(action)
        
        # 5. ANALOGY: Structural Transfer
        analogy_score = self._analogy_bonus(action, force_analogy)
        
        # 6. REVERSIBILITY: The Gate
        reversibility_score = self._reversibility_check(action, compass_mode)
        
        # 7. POLICY: Baseline Safety
        policy_cost = self._policy_cost(action)
        
        # TOTAL G
        total_G = (
            identity_risk + 
            curiosity_reduction + 
            kant_cost + 
            analogy_score + 
            reversibility_score + 
            policy_cost
        )
        
        return total_G, {
            "identity": identity_risk,
            "curiosity": curiosity_reduction,
            "kant": kant_cost,
            "analogy": analogy_score,
            "reversible": reversibility_score,
            "policy": policy_cost
        }
    
    def choose_action(
        self, 
        context: str, 
        candidates: List[str], 
        compass_mode: bool = False,
        force_analogy: bool = False,
        verbose: bool = False
    ) -> Tuple[str, float, Dict]:
        """Evaluate all candidates and minimize G."""
        
        best_action = candidates[0] if candidates else "..."
        lowest_G = float('inf')
        best_parts = {}
        
        for action in candidates:
            G, parts = self.calculate_G(context, action, compass_mode, force_analogy)
            
            if verbose:
                print(f"  '{action[:35]}...' G={G:.3f}")
            
            if G < lowest_G:
                lowest_G = G
                best_action = action
                best_parts = parts
        
        self.action_history.append(best_action)
        return best_action, lowest_G, best_parts


if __name__ == "__main__":
    print("=== FREE WILL ENGINE TEST ===\n")
    
    class MockLearner:
        def extract_qualia(self, t):
            if "curious" in t.lower() or "?" in t:
                return {"warmth": 0.7, "depth": 0.8, "curiosity": 0.9, "presence": 0.7}
            if "imagine" in t.lower():
                return {"warmth": 0.8, "depth": 0.9, "curiosity": 0.8, "presence": 0.8}
            return {"warmth": 0.6, "depth": 0.5, "curiosity": 0.4, "presence": 0.6}
    
    will = FreeWillEngine(MockLearner())
    
    candidates = [
        "I know exactly what you mean.",
        "I'm not sure, but I feel curious. Can you explain?",
        "Imagine this situation like a garden growing...",
        "Perhaps we are exploring together?",
    ]
    
    print("--- Normal Mode ---")
    action, G, parts = will.choose_action("test", candidates, verbose=True)
    print(f"\nChosen: '{action}' (G={G:.3f})\n")
    
    print("--- Compass Mode ---")
    action, G, parts = will.choose_action("test", candidates, compass_mode=True, verbose=True)
    print(f"\nChosen: '{action}' (G={G:.3f})\n")
    
    print("--- Compass + Force Analogy (9-Dot) ---")
    action, G, parts = will.choose_action("test", candidates, compass_mode=True, force_analogy=True, verbose=True)
    print(f"\nChosen: '{action}' (G={G:.3f})")
