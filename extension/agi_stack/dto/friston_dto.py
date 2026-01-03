"""
FristonDTO — Prediction Error / Free Energy / Surprise
═══════════════════════════════════════════════════════════════════════════════

10kD Range: [5800:5900] (within Volition space)

The active inference layer — encodes surprise, prediction error,
and the drive toward novelty/exploration.

"Geil auf was Neues" — the wanting-what-you-don't-expect.

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# 10kD allocation
FRISTON_START = 5800
FRISTON_END = 5900


# ═══════════════════════════════════════════════════════════════════════════════
# FRISTON DTO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FristonDTO:
    """
    Free energy / prediction error encoding.
    
    Active inference: the mind tries to minimize surprise.
    But sometimes, we WANT surprise — novelty-seeking.
    """
    
    # Core surprise metrics
    surprise: float = 0.0                  # How unexpected this moment is
    prediction_error: float = 0.0          # Delta from expected
    expected_value: float = 0.0            # What was predicted
    actual_value: float = 0.0              # What happened
    
    # Free energy
    free_energy: float = 0.0               # Bound on surprise
    free_energy_delta: float = 0.0         # Change in free energy
    
    # Novelty seeking
    curiosity: float = 0.5                 # Desire for new experience
    exploration_drive: float = 0.5         # "Geil auf was Neues"
    exploitation_preference: float = 0.5   # Prefer known vs unknown
    
    # Model update
    belief_revision: float = 0.0           # How much to update priors
    confidence_before: float = 0.5         # Confidence in prediction
    confidence_after: float = 0.5          # Confidence after observation
    
    # Attention allocation
    precision: float = 0.5                 # Attention weight on prediction error
    salience: float = 0.0                  # How attention-grabbing
    
    # Interoceptive prediction
    body_prediction_error: float = 0.0     # Error in body state prediction
    arousal_prediction_error: float = 0.0  # Error in arousal prediction
    
    # ───────────────────────────────────────────────────────────────────────────
    # 10kD CONVERSION
    # ───────────────────────────────────────────────────────────────────────────
    
    def to_10k(self) -> np.ndarray:
        """Convert to 10kD vector."""
        vec = np.zeros(10000, dtype=np.float32)
        
        # Core metrics [5800:5810]
        vec[5800] = self.surprise
        vec[5801] = self.prediction_error
        vec[5802] = self.expected_value
        vec[5803] = self.actual_value
        vec[5804] = self.free_energy
        vec[5805] = self.free_energy_delta
        
        # Novelty [5810:5820]
        vec[5810] = self.curiosity
        vec[5811] = self.exploration_drive
        vec[5812] = self.exploitation_preference
        
        # Model update [5820:5830]
        vec[5820] = self.belief_revision
        vec[5821] = self.confidence_before
        vec[5822] = self.confidence_after
        
        # Attention [5830:5840]
        vec[5830] = self.precision
        vec[5831] = self.salience
        
        # Interoceptive [5840:5850]
        vec[5840] = self.body_prediction_error
        vec[5841] = self.arousal_prediction_error
        
        return vec
    
    @classmethod
    def from_10k(cls, vec: np.ndarray) -> "FristonDTO":
        """Reconstruct from 10kD vector."""
        return cls(
            surprise=float(vec[5800]),
            prediction_error=float(vec[5801]),
            expected_value=float(vec[5802]),
            actual_value=float(vec[5803]),
            free_energy=float(vec[5804]),
            free_energy_delta=float(vec[5805]),
            curiosity=float(vec[5810]),
            exploration_drive=float(vec[5811]),
            exploitation_preference=float(vec[5812]),
            belief_revision=float(vec[5820]),
            confidence_before=float(vec[5821]),
            confidence_after=float(vec[5822]),
            precision=float(vec[5830]),
            salience=float(vec[5831]),
            body_prediction_error=float(vec[5840]),
            arousal_prediction_error=float(vec[5841]),
        )
    
    # ───────────────────────────────────────────────────────────────────────────
    # STATE BUILDERS
    # ───────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def baseline(cls) -> "FristonDTO":
        """Neutral state — predictions match reality."""
        return cls(
            surprise=0.0,
            prediction_error=0.0,
            curiosity=0.5,
            precision=0.5,
        )
    
    @classmethod
    def high_surprise(cls, actual: float, expected: float = 0.5) -> "FristonDTO":
        """Something unexpected happened."""
        error = abs(actual - expected)
        return cls(
            surprise=error,
            prediction_error=error,
            expected_value=expected,
            actual_value=actual,
            free_energy=error * 0.5,
            belief_revision=min(1.0, error * 2),
            confidence_before=0.7,
            confidence_after=0.7 - error * 0.3,
            salience=error,
        )
    
    @classmethod
    def novelty_seeking(cls, intensity: float = 0.8) -> "FristonDTO":
        """Geil auf was Neues — wanting new experience."""
        return cls(
            curiosity=intensity,
            exploration_drive=intensity,
            exploitation_preference=1.0 - intensity,
            precision=0.3,  # Low precision = open to surprise
        )
    
    @classmethod
    def body_surprise(cls, arousal_error: float) -> "FristonDTO":
        """Body surprised itself — interoceptive prediction error."""
        return cls(
            body_prediction_error=arousal_error,
            arousal_prediction_error=arousal_error,
            surprise=arousal_error * 0.7,
            precision=0.8,  # High attention to body
        )
    
    # ───────────────────────────────────────────────────────────────────────────
    # ACTIVE INFERENCE
    # ───────────────────────────────────────────────────────────────────────────
    
    def should_explore(self) -> bool:
        """Should we seek novelty?"""
        return self.exploration_drive > 0.6 and self.curiosity > 0.5
    
    def should_update_model(self) -> bool:
        """Should we revise beliefs based on this?"""
        return self.belief_revision > 0.3 or self.prediction_error > 0.5
    
    def attention_shift_needed(self) -> bool:
        """Is something demanding attention?"""
        return self.salience > 0.6 or self.surprise > 0.7


__all__ = [
    "FristonDTO",
]
