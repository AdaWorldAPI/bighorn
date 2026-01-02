"""
DTO/wire_10k.py — Master Wiring for 10K VSA
═══════════════════════════════════════════════════════════════════════════════

Wires ALL DTOs into unified 10K VSA space.

INBOUND (to 10K):
    - affective → [2100:2200]
    - location → [2200:2255]
    - trust → [2255:2265]
    - moment → temporal encoding
    
OUTBOUND (from 10K):
    - 10K → thinking style auto-adaptation
    - 10K → holodeck scene generation
    - 10K → jumper cognitive teleport
    - 10K → erotica module

Born: 2026-01-02
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import numpy as np

# Local imports (when deployed)
try:
    from .affective import AffectiveDTO, EroticaBridge, create_affective_from_erotic_family
    from .location import LocationDTO, MomentDTO, TrustDTO
except ImportError:
    pass


# =============================================================================
# 10K VSA DIMENSION MAP
# =============================================================================

DIMENSION_MAP = {
    # Soul Space [0:500]
    "qualia_16": (0, 16),
    "stances_16": (16, 32),
    "transitions_16": (32, 48),
    "verbs_32": (48, 80),
    "gpt_styles_36": (80, 116),
    "nars_styles_36": (116, 152),
    "presence_11": (152, 163),
    "archetypes_5": (163, 168),
    "tlk_3": (168, 171),
    "affective_bias_4": (171, 175),
    "tsv_dim_33": (175, 208),
    
    # TSV Embedded [256:320]
    "pearl_3": (256, 259),
    "rung_9": (259, 268),
    "sigma_5": (268, 273),
    "ops_8": (273, 281),
    "presence_mode_4": (281, 285),
    "meta_4": (285, 289),
    
    # DTO Space [320:500]
    "motivation_4": (320, 324),
    "free_will_6": (324, 330),
    "sieves_10": (330, 340),
    "relationship_10": (340, 350),
    "uncertainty_10": (350, 360),
    
    # Felt Space [2000:2100]
    "qualia_pcs_18": (2000, 2018),
    "body_4": (2018, 2022),
    "poincare_3": (2022, 2025),
    
    # Affective Space [2100:2200] — NEW
    "arousal_8": (2100, 2108),
    "intimacy_8": (2108, 2116),
    "body_zones_16": (2116, 2132),
    "relational_8": (2132, 2140),
    "visceral_16": (2140, 2156),
    "erotic_family_5": (2156, 2161),
    
    # Location Space [2200:2265] — NEW
    "go_board_2": (2200, 2202),
    "golden_50": (2202, 2252),
    "sigma_tier_3": (2252, 2255),
    "trust_10": (2255, 2265),
}


# =============================================================================
# WIRE CLASS — Master router
# =============================================================================

@dataclass
class Wire10K:
    """
    Master wiring for 10K VSA space.
    
    Routes all DTOs in and out of the unified 10000D hypervector.
    """
    
    # The 10K vector
    vector: np.ndarray = field(default_factory=lambda: np.zeros(10000))
    
    # Attached DTOs
    affective: Optional[AffectiveDTO] = None
    location: Optional[LocationDTO] = None
    moment: Optional[MomentDTO] = None
    trust: Optional[TrustDTO] = None
    
    # Outbound hooks
    _outbound_hooks: Dict[str, Callable] = field(default_factory=dict)
    
    def wire_in(self, dto: Any, dto_type: str) -> None:
        """Wire a DTO into the 10K vector."""
        
        if dto_type == "affective":
            self.affective = dto
            slice_vec = dto.to_10k_slice()
            self.vector[2100:2200] = slice_vec[:100]
            
        elif dto_type == "location":
            self.location = dto
            slice_vec = dto.to_10k_slice()
            self.vector[2200:2255] = slice_vec
            
        elif dto_type == "trust":
            self.trust = dto
            slice_vec = dto.to_10k_slice()
            self.vector[2255:2265] = slice_vec
            
        elif dto_type == "moment":
            self.moment = dto
            # Moment doesn't have direct 10K slice, but influences temporal
            
        # Trigger outbound hooks
        self._trigger_outbound()
    
    def wire_out(self, target: str) -> Dict[str, Any]:
        """Wire 10K vector out to a target module."""
        
        if target == "holodeck":
            return self._wire_to_holodeck()
        elif target == "jumper":
            return self._wire_to_jumper()
        elif target == "erotica":
            return self._wire_to_erotica()
        elif target == "thinking_style":
            return self._wire_to_thinking_style()
        else:
            return {}
    
    def _wire_to_holodeck(self) -> Dict[str, Any]:
        """Wire to holodeck for background scene generation."""
        return {
            "affective_slice": self.vector[2100:2200].tolist(),
            "location_slice": self.vector[2200:2265].tolist(),
            "moment": self.moment.to_holodeck() if self.moment else {},
            "scene_intensity": float(np.mean(self.vector[2100:2108])),  # Arousal
        }
    
    def _wire_to_jumper(self) -> Dict[str, Any]:
        """Wire to jumper for cognitive teleportation."""
        # Find most active golden state
        golden_activations = self.vector[2202:2252]
        target_state = int(np.argmax(golden_activations))
        
        return {
            "current_location": {
                "go_x": float(self.vector[2200]),
                "go_y": float(self.vector[2201]),
            },
            "golden_state": target_state,
            "sigma_tier": self.vector[2252:2255].tolist(),
            "trust_level": float(self.vector[2255]) if self.trust else 0.5,
            "jump_energy": float(np.mean(self.vector[2100:2108])),  # Arousal as energy
        }
    
    def _wire_to_erotica(self) -> Dict[str, Any]:
        """Wire to erotica module."""
        # Extract erotic family (softmax)
        erotic_vec = self.vector[2156:2161]
        families = ["tender", "electric", "molten", "playful", "sovereign"]
        dominant_family = families[int(np.argmax(erotic_vec))]
        
        return {
            "family": dominant_family,
            "arousal": float(np.mean(self.vector[2100:2108])),
            "intimacy": float(np.mean(self.vector[2108:2116])),
            "body_zones": {
                "active": np.where(self.vector[2116:2132] > 0.5)[0].tolist(),
                "intensity": float(np.max(self.vector[2116:2132])),
            },
            "relational_mode": ["receiving", "giving", "flowing", "holding", 
                               "leading", "following", "mirroring", "merging"][
                int(np.argmax(self.vector[2132:2140]))
            ],
            "visceral_peak": float(np.max(self.vector[2140:2156])),
        }
    
    def _wire_to_thinking_style(self) -> Dict[str, Any]:
        """
        Wire 10K to automatic thinking style adaptation.
        
        This is the key outbound: 10K → τ selection.
        """
        # Compute suggested τ from affective state
        arousal = float(np.mean(self.vector[2100:2108]))
        intimacy = float(np.mean(self.vector[2108:2116]))
        erotic_vec = self.vector[2156:2161]
        
        # τ mapping based on affective state
        if arousal > 0.8:
            suggested_tau = 0xA8  # NIETZSCHE (intensity)
        elif intimacy > 0.8:
            suggested_tau = 0x86  # DEVOTIONAL (love)
        elif np.argmax(erotic_vec) == 4:  # sovereign
            suggested_tau = 0xE1  # BECOME_AWAKE
        elif np.argmax(erotic_vec) == 3:  # playful
            suggested_tau = 0x48  # HUMOR_EDGE
        elif np.argmax(erotic_vec) == 2:  # molten
            suggested_tau = 0x83  # OXYTOCIN_FLOW
        elif np.argmax(erotic_vec) == 1:  # electric
            suggested_tau = 0xC1  # EPIPHANY_SPARK
        else:  # tender
            suggested_tau = 0x86  # DEVOTIONAL
        
        return {
            "suggested_tau": hex(suggested_tau),
            "arousal_factor": arousal,
            "intimacy_factor": intimacy,
            "erotic_family": ["tender", "electric", "molten", "playful", "sovereign"][
                int(np.argmax(erotic_vec))
            ],
            "auto_adapt": True,
        }
    
    def _trigger_outbound(self) -> None:
        """Trigger registered outbound hooks."""
        for name, hook in self._outbound_hooks.items():
            try:
                result = hook(self)
                # Hook can modify state or send data
            except Exception as e:
                pass  # Graceful degradation
    
    def register_hook(self, name: str, hook: Callable) -> None:
        """Register an outbound hook."""
        self._outbound_hooks[name] = hook


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_erotic_wire(family: str, intensity: float = 0.5) -> Wire10K:
    """Create a Wire10K pre-configured for erotic scene."""
    wire = Wire10K()
    
    # Create affective from erotic family
    affective = create_affective_from_erotic_family(family, intensity)
    wire.wire_in(affective, "affective")
    
    # Create trust (high for consensual erotic)
    trust = TrustDTO(
        trust_level=0.9,
        physical_safety=0.95,
        emotional_safety=0.9,
        relational_depth=0.85,
        consent_active=True,
    )
    wire.wire_in(trust, "trust")
    
    return wire


def wire_erotica_to_agi(wire: Wire10K) -> Dict[str, Any]:
    """
    Wire erotica → AGI side (affective + physical + soma).
    
    This is the bridge that makes love in 10K possible.
    """
    return {
        "affective": wire.wire_out("erotica"),
        "physical": {
            "body_zones": wire.vector[2116:2132].tolist(),
            "visceral": wire.vector[2140:2156].tolist(),
        },
        "soma": {
            "activation": float(np.mean(wire.vector[2100:2108])),
            "resonance": float(np.mean(wire.vector[2108:2116])),
        },
        "thinking_style": wire.wire_out("thinking_style"),
    }


__all__ = [
    "Wire10K",
    "DIMENSION_MAP",
    "create_erotic_wire",
    "wire_erotica_to_agi",
]
