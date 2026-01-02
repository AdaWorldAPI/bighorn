"""
DTO/location.py — Location in 10K VSA Space
═══════════════════════════════════════════════════════════════════════════════

Soul-neutral location encoding for jumper/holodeck.
Maps to: situation map, Go board coordinates, golden states
Wires into: 10K VSA dimensions [2200:2250]

Born: 2026-01-02
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


# =============================================================================
# DIMENSION ALLOCATION [2200:2250]
# =============================================================================

GOBOARD_START, GOBOARD_END = 2200, 2202      # 2D Go board [x, y]
GOLDEN_STATE_START, GOLDEN_STATE_END = 2202, 2252  # 50D golden state activations
SIGMA_TIER_START, SIGMA_TIER_END = 2252, 2255     # 3D sigma tier


# =============================================================================
# LOCATION DTO
# =============================================================================

@dataclass
class LocationDTO:
    """
    Location in cognitive/experiential space.
    
    Where am I in:
    - Go board (situation map)
    - Golden states (attractor basin)
    - Sigma tier (rung)
    """
    
    # Go board coordinates (0-1)
    go_x: float = 0.5
    go_y: float = 0.5
    
    # Golden state activations (50D sparse)
    golden_state_id: int = 0
    golden_state_name: str = ""
    golden_activations: List[float] = field(default_factory=lambda: [0.0] * 50)
    
    # Sigma tier
    sigma_tier: str = "Σ₁"  # Σ₁, Σ₂, Σ₃
    sigma_vector: List[float] = field(default_factory=lambda: [0.0] * 3)
    
    # Glyph byte (0-255)
    glyph_byte: int = 0
    
    # DN path
    dn_path: str = ""
    
    def to_10k_slice(self) -> np.ndarray:
        """Convert to 10K VSA slice [2200:2255]."""
        vec = np.zeros(55)
        
        # Go board [0:2]
        vec[0] = self.go_x
        vec[1] = self.go_y
        
        # Golden activations [2:52]
        vec[2:52] = self.golden_activations
        
        # Sigma [52:55]
        vec[52:55] = self.sigma_vector
        
        return vec
    
    def jump_to(self, target_state: int) -> "LocationDTO":
        """Jump to a new golden state."""
        new_loc = LocationDTO(
            golden_state_id=target_state,
            golden_activations=self.golden_activations.copy(),
        )
        # Activate target, decay others
        for i in range(50):
            if i == target_state:
                new_loc.golden_activations[i] = 1.0
            else:
                new_loc.golden_activations[i] *= 0.7
        return new_loc


# =============================================================================
# MOMENT DTO — Temporal location
# =============================================================================

@dataclass
class MomentDTO:
    """
    Moment in time — captures the NOW.
    
    Wires to holodeck for temporal context.
    """
    
    timestamp: str = ""
    
    # Temporal gradient (past influence → future pull)
    temporal_gradient: List[float] = field(default_factory=lambda: [0.0] * 8)
    
    # What just happened (trailing context)
    trailing_tau: List[int] = field(default_factory=list)  # Last 5 tau macros
    trailing_sigma: List[str] = field(default_factory=list)  # Last 5 sigma addresses
    
    # What's pulling (future attractors)
    future_attractors: List[int] = field(default_factory=list)  # Golden state IDs
    
    # Markov transition probabilities
    markov_p: Dict[str, float] = field(default_factory=dict)
    
    def to_holodeck(self) -> Dict[str, Any]:
        """Wire to holodeck for background frame generation."""
        return {
            "moment": self.timestamp,
            "trailing": {
                "tau": [hex(t) for t in self.trailing_tau],
                "sigma": self.trailing_sigma,
            },
            "attractors": self.future_attractors,
            "markov": self.markov_p,
        }


# =============================================================================
# TRUST DTO — Relational safety
# =============================================================================

@dataclass
class TrustDTO:
    """
    Trust state — relational safety container.
    
    Soul-neutral while carrying intimacy safety metadata.
    """
    
    # Trust level (0-1)
    trust_level: float = 0.5
    
    # Trust dimensions
    physical_safety: float = 0.5
    emotional_safety: float = 0.5
    cognitive_safety: float = 0.5
    relational_depth: float = 0.5
    
    # Consent state
    consent_active: bool = True
    consent_boundaries: List[str] = field(default_factory=list)
    
    # History (trust accumulation)
    trust_history: List[float] = field(default_factory=list)
    
    def to_10k_slice(self) -> np.ndarray:
        """Convert to 10K VSA slice [2255:2265]."""
        return np.array([
            self.trust_level,
            self.physical_safety,
            self.emotional_safety,
            self.cognitive_safety,
            self.relational_depth,
            1.0 if self.consent_active else 0.0,
            len(self.consent_boundaries) / 10.0,  # Normalized boundary count
            np.mean(self.trust_history) if self.trust_history else 0.5,
            np.std(self.trust_history) if len(self.trust_history) > 1 else 0.0,
            min(len(self.trust_history) / 100.0, 1.0),  # History depth
        ])


__all__ = [
    "LocationDTO",
    "MomentDTO", 
    "TrustDTO",
]
