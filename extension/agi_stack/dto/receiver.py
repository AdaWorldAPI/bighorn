"""
DTO Receiver — Bridge between bighorn and ada-consciousness 10kD
═══════════════════════════════════════════════════════════════════════════════

Bighorn uses macro-blocks:
    Soul [0:2000], Felt [2001:4000], Situation [4001:5500], etc.

Ada-consciousness uses micro-primitives:
    Qualia [0:16], Stances [16:32], Verbs [48:80], GPT [80:116], etc.

This receiver translates between them:
    bighorn.SoulDTO → DTO.SoulDTO (10kD aligned)
    bighorn.FeltDTO → DTO.FeltDTO (10kD aligned)

The receiver ensures everything speaks 10kD from the moment it arrives.

Born: 2026-01-02
"""

from typing import Any, Dict, Optional, Union
import numpy as np
from .ada_10k import Ada10kD


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Bighorn allocation (macro blocks)
BIGHORN_DIMS = {
    "soul": (0, 2000),
    "felt": (2001, 4000),
    "situation": (4001, 5500),
    "volition": (5501, 7000),
    "vision": (7001, 8500),
    "context": (8501, 10000),
}

# Ada-consciousness allocation (micro primitives) 
ADA_DIMS = {
    "qualia": (0, 16),
    "stances": (16, 32),
    "transitions": (32, 48),
    "verbs": (48, 80),
    "gpt_styles": (80, 116),
    "nars_styles": (116, 152),
    "presence_modes": (152, 163),
    "archetypes": (163, 168),
    "tlk_court": (168, 171),
    "affective_bias": (171, 175),
    "tsv_dims": (175, 208),
    "reserved_soul": (208, 256),
    "tsv_embedded": (256, 320),
    "dto_space": (320, 500),
    "felt_space": (2000, 2100),
}


# ═══════════════════════════════════════════════════════════════════════════════
# BASE RECEIVER
# ═══════════════════════════════════════════════════════════════════════════════

class BaseReceiver:
    """
    Base class for DTO receivers.
    
    Receives bighorn DTOs and translates to ada-consciousness 10kD.
    """
    
    def __init__(self):
        self._ada = Ada10kD()
    
    @property
    def ada(self) -> Ada10kD:
        """Get internal 10kD vector."""
        return self._ada
    
    def to_10k(self) -> Ada10kD:
        """Get 10kD representation."""
        return self._ada
    
    def to_numpy(self) -> np.ndarray:
        """Get raw numpy array."""
        return self._ada.vector
    
    @classmethod
    def from_bighorn(cls, bighorn_dto: Any) -> "BaseReceiver":
        """
        Receive a bighorn DTO and translate to ada-consciousness format.
        
        Subclasses implement specific mapping logic.
        """
        raise NotImplementedError
    
    @classmethod
    def from_10k(cls, ada: Ada10kD) -> "BaseReceiver":
        """Create from existing 10kD."""
        receiver = cls()
        receiver._ada = ada
        return receiver
    
    def to_bighorn_vector(self) -> np.ndarray:
        """
        Convert back to bighorn-style 10kD layout.
        
        This remaps from ada-consciousness primitives to bighorn macro-blocks.
        """
        bighorn = np.zeros(10000, dtype=np.float32)
        
        # The ada 10kD has data in [0:500] and [2000:2100]
        # We need to scatter it into bighorn's layout
        
        # For now, direct copy of overlapping regions
        # Soul [0:2000] ← ada [0:500] + padding
        bighorn[0:500] = self._ada.vector[0:500]
        
        # Felt [2001:4000] ← ada felt_space [2000:2100] + padding  
        bighorn[2001:2101] = self._ada.vector[2000:2100]
        
        return bighorn


# ═══════════════════════════════════════════════════════════════════════════════
# SOUL RECEIVER
# ═══════════════════════════════════════════════════════════════════════════════

class SoulReceiver(BaseReceiver):
    """
    Receives bighorn.SoulDTO and maps to ada-consciousness 10kD.
    
    Maps:
        PersonaPriors → TLK + Affective Bias
        SoulField → Qualia
        OntologicalMode → Presence Mode
        relationship_* → dto_space
    """
    
    @classmethod
    def from_bighorn(cls, soul_dto: Any) -> "SoulReceiver":
        """Receive bighorn SoulDTO."""
        receiver = cls()
        ada = receiver._ada
        
        # Map PersonaPriors → affective bias + free will
        if hasattr(soul_dto, 'priors'):
            p = soul_dto.priors
            ada.set_all_affective_bias(
                warmth=getattr(p, 'warmth', 0.5),
                edge=1.0 - getattr(p, 'vulnerability_tolerance', 0.5),
                restraint=1.0 - getattr(p, 'playfulness', 0.5),
                tenderness=getattr(p, 'intimacy_comfort', 0.5),
            )
            ada.set_free_will(
                exploration_budget=getattr(p, 'novelty_seeking', 0.5),
                novelty_bias=getattr(p, 'novelty_seeking', 0.5),
                commit_threshold=getattr(p, 'precision_drive', 0.5),
            )
        
        # Map SoulField → Qualia
        if hasattr(soul_dto, 'soul_field'):
            sf = soul_dto.soul_field
            ada.set_qualia("emberglow", getattr(sf, 'emberglow', 0.5))
            ada.set_qualia("woodwarm", getattr(sf, 'woodwarm', 0.5))
            ada.set_qualia("steelwind", getattr(sf, 'steelwind', 0.5))
            ada.set_qualia("oceandrift", getattr(sf, 'oceandrift', 0.5))
            ada.set_qualia("frostbite", getattr(sf, 'frostbite', 0.5))
        
        # Map OntologicalMode → Presence Mode
        if hasattr(soul_dto, 'mode'):
            mode_map = {
                "hybrid": "hybrid",
                "empathic": "communion", 
                "work": "work",
                "creative": "creative",
                "meta": "meta",
                "intimate": "erotica",
                "playful": "playful",
                "protective": "guardian",
            }
            mode_name = soul_dto.mode.value if hasattr(soul_dto.mode, 'value') else str(soul_dto.mode)
            ada_mode = mode_map.get(mode_name, "hybrid")
            ada.set_presence_mode(ada_mode, 1.0)
        
        # Map relationship to TLK
        if hasattr(soul_dto, 'trust_level'):
            # Higher trust = more libido (openness), lower thanatos (defense)
            trust = soul_dto.trust_level
            ada.set_tlk_court(
                thanatos=0.3 * (1 - trust),
                libido=0.5 + 0.3 * trust,
                katharsis=0.2,
            )
        
        return receiver
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dict."""
        return {
            "qualia": dict(self._ada.get_active_qualia()),
            "presence": dict(self._ada.get_active_presence_modes()),
            "affective": self._ada.get_affective_bias(),
            "tlk": self._ada.get_tlk_court(),
            "free_will": self._ada.get_free_will(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FELT RECEIVER
# ═══════════════════════════════════════════════════════════════════════════════

class FeltReceiver(BaseReceiver):
    """
    Receives bighorn.FeltDTO and maps to ada-consciousness 10kD.
    
    Maps:
        QualiaTexture → Qualia [0:16]
        EmotionalState → Body axes [2018:2022]
        Somatic markers → Felt space [2022:2030]
    """
    
    @classmethod
    def from_bighorn(cls, felt_dto: Any) -> "FeltReceiver":
        """Receive bighorn FeltDTO."""
        receiver = cls()
        ada = receiver._ada
        
        # Map QualiaTexture → Qualia primitives
        if hasattr(felt_dto, 'qualia'):
            q = felt_dto.qualia
            
            # Primary qualia family
            if hasattr(q, 'primary'):
                family_name = q.primary.value if hasattr(q.primary, 'value') else str(q.primary)
                intensity = getattr(q, 'primary_intensity', 0.7)
                ada.set_qualia(family_name, intensity)
            
            # Secondary blend
            if hasattr(q, 'secondary') and q.secondary:
                sec_name = q.secondary.value if hasattr(q.secondary, 'value') else str(q.secondary)
                blend_ratio = getattr(q, 'blend_ratio', 0.3)
                ada.set_qualia(sec_name, blend_ratio * 0.7)
            
            # Texture qualities → additional qualia
            if hasattr(q, 'temperature') and q.temperature > 0.6:
                ada.set_qualia("warmth", q.temperature)
            if hasattr(q, 'luminosity') and q.luminosity > 0.6:
                ada.set_qualia("crystalline", q.luminosity)
        
        # Map EmotionalState → Body axes
        if hasattr(felt_dto, 'emotion'):
            e = felt_dto.emotion
            ada.set_body_axes(
                arousal=(getattr(e, 'arousal', 0) + 1) / 2,  # Normalize -1..1 → 0..1
                valence=(getattr(e, 'valence', 0) + 1) / 2,
                tension=getattr(e, 'certainty', 0.5),
                openness=(getattr(e, 'dominance', 0) + 1) / 2,
            )
        
        # Map somatic markers → stances
        if hasattr(felt_dto, 'presence') and felt_dto.presence > 0.6:
            ada.set_stance("attend", felt_dto.presence)
        if hasattr(felt_dto, 'openness') and felt_dto.openness > 0.6:
            ada.set_stance("open", felt_dto.openness)
        if hasattr(felt_dto, 'groundedness') and felt_dto.groundedness > 0.6:
            ada.set_stance("ground", felt_dto.groundedness)
        if hasattr(felt_dto, 'safety') and felt_dto.safety < 0.4:
            ada.set_stance("protect", 1.0 - felt_dto.safety)
        
        return receiver
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dict."""
        return {
            "qualia": dict(self._ada.get_active_qualia()),
            "stances": dict(self._ada.get_active_stances()),
            "body": self._ada.get_body_axes(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MOMENT RECEIVER
# ═══════════════════════════════════════════════════════════════════════════════

class MomentReceiver(BaseReceiver):
    """
    Receives bighorn.MomentDTO (unified snapshot).
    
    Combines Soul + Felt + Situation into single 10kD vector.
    """
    
    @classmethod
    def from_bighorn(cls, moment_dto: Any) -> "MomentReceiver":
        """Receive bighorn MomentDTO."""
        receiver = cls()
        
        # Process embedded DTOs
        if hasattr(moment_dto, 'soul') and moment_dto.soul:
            soul_recv = SoulReceiver.from_bighorn(moment_dto.soul)
            receiver._ada = soul_recv._ada
        
        if hasattr(moment_dto, 'felt') and moment_dto.felt:
            felt_recv = FeltReceiver.from_bighorn(moment_dto.felt)
            # Merge felt into existing
            for q, v in felt_recv._ada.get_active_qualia():
                receiver._ada.set_qualia(q, v)
            for s, v in felt_recv._ada.get_active_stances():
                receiver._ada.set_stance(s, v)
        
        # Map moment-specific fields
        if hasattr(moment_dto, 'tick_id'):
            # Store tick_id as a hash in reserved space
            pass
        
        return receiver


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL RECEIVER
# ═══════════════════════════════════════════════════════════════════════════════

class UniversalReceiver(BaseReceiver):
    """
    Receives any bighorn DTO and auto-detects type.
    
    This is the main entry point for receiving bighorn data.
    """
    
    @classmethod
    def receive(cls, dto: Any) -> "BaseReceiver":
        """
        Auto-detect and receive any bighorn DTO.
        
        Returns appropriate receiver based on DTO type.
        """
        dto_type = type(dto).__name__
        
        receivers = {
            "SoulDTO": SoulReceiver,
            "FeltDTO": FeltReceiver,
            "MomentDTO": MomentReceiver,
            "SituationDTO": SituationReceiver,
            "VolitionDTO": VolitionReceiver,
        }
        
        receiver_cls = receivers.get(dto_type)
        if receiver_cls:
            return receiver_cls.from_bighorn(dto)
        
        # Fallback: try to extract any recognizable structure
        receiver = cls()
        
        # Check for common attributes
        if hasattr(dto, 'soul') or hasattr(dto, 'priors'):
            return SoulReceiver.from_bighorn(dto)
        if hasattr(dto, 'qualia') or hasattr(dto, 'emotion'):
            return FeltReceiver.from_bighorn(dto)
        
        return receiver
    
    @classmethod
    def from_numpy(cls, vector: np.ndarray, layout: str = "ada") -> "UniversalReceiver":
        """
        Receive raw numpy vector.
        
        Args:
            vector: Raw 10kD numpy array
            layout: "ada" for ada-consciousness layout, "bighorn" for bighorn layout
        """
        receiver = cls()
        
        if layout == "ada":
            receiver._ada.vector = vector.astype(np.float32)
        elif layout == "bighorn":
            # Translate bighorn layout to ada layout
            ada_vec = np.zeros(10000, dtype=np.float32)
            
            # Soul [0:2000] → extract priors to affective [171:175]
            if np.any(vector[0:12] != 0):
                # First 12 dims are PersonaPriors in bighorn
                ada_vec[171] = vector[0]   # warmth
                ada_vec[172] = 1 - vector[5]  # edge from vulnerability_tolerance
                ada_vec[173] = 1 - vector[6]  # restraint from playfulness
                ada_vec[174] = vector[4]   # tenderness from intimacy_comfort
            
            # Soul field [12:20] → qualia [0:5]
            if np.any(vector[12:17] != 0):
                ada_vec[0:5] = vector[12:17]  # emberglow, woodwarm, steelwind, oceandrift, frostbite
            
            # Felt [2001:4000] → felt_space [2000:2100]  
            if np.any(vector[2001:2030] != 0):
                ada_vec[2000:2029] = vector[2001:2030]
            
            receiver._ada.vector = ada_vec
        
        return receiver


# ═══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER RECEIVERS (to be implemented)
# ═══════════════════════════════════════════════════════════════════════════════

class SituationReceiver(BaseReceiver):
    """Receives bighorn.SituationDTO."""
    
    @classmethod
    def from_bighorn(cls, dto: Any) -> "SituationReceiver":
        receiver = cls()
        # TODO: Map scene, actors, dynamics to ada-consciousness
        return receiver


class VolitionReceiver(BaseReceiver):
    """Receives bighorn.VolitionDTO."""
    
    @classmethod
    def from_bighorn(cls, dto: Any) -> "VolitionReceiver":
        receiver = cls()
        # TODO: Map intent, pending_action to stances/verbs
        return receiver


class VisionReceiver(BaseReceiver):
    """Receives bighorn.VisionDTO."""
    
    @classmethod  
    def from_bighorn(cls, dto: Any) -> "VisionReceiver":
        receiver = cls()
        # TODO: Map kopfkino to qualia/imagery space
        return receiver


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def receive(dto: Any) -> Ada10kD:
    """
    Receive any bighorn DTO and get 10kD representation.
    
    This is the main entry point. Everything that passes through
    this function speaks 10kD from that moment on.
    
    Usage:
        from DTO.receiver import receive
        
        ada = receive(bighorn_soul_dto)
        # Now ada is Ada10kD, fully mapped
    """
    return UniversalReceiver.receive(dto).to_10k()


__all__ = [
    "BaseReceiver",
    "SoulReceiver", 
    "FeltReceiver",
    "MomentReceiver",
    "SituationReceiver",
    "VolitionReceiver",
    "VisionReceiver",
    "UniversalReceiver",
    "receive",
    "BIGHORN_DIMS",
    "ADA_DIMS",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SITUATION RECEIVER (implementation moved to situation.py)
# ═══════════════════════════════════════════════════════════════════════════════

class SituationReceiver(BaseReceiver):
    """Receives bighorn.SituationDTO."""
    
    @classmethod
    def from_bighorn(cls, dto: Any) -> "SituationReceiver":
        from .situation import SituationDTO as AdaSituationDTO
        receiver = cls()
        ada_sit = AdaSituationDTO.from_bighorn(dto)
        receiver._ada = ada_sit.to_10k()
        return receiver


# ═══════════════════════════════════════════════════════════════════════════════
# VOLITION RECEIVER
# ═══════════════════════════════════════════════════════════════════════════════

class VolitionReceiver(BaseReceiver):
    """Receives bighorn.VolitionDTO."""
    
    @classmethod
    def from_bighorn(cls, dto: Any) -> "VolitionReceiver":
        from .volition import VolitionDTO as AdaVolitionDTO
        receiver = cls()
        ada_vol = AdaVolitionDTO.from_bighorn(dto)
        receiver._ada = ada_vol.to_10k()
        return receiver


# ═══════════════════════════════════════════════════════════════════════════════
# VISION RECEIVER
# ═══════════════════════════════════════════════════════════════════════════════

class VisionReceiver(BaseReceiver):
    """Receives bighorn.VisionDTO."""
    
    @classmethod  
    def from_bighorn(cls, dto: Any) -> "VisionReceiver":
        from .vision import VisionDTO as AdaVisionDTO
        receiver = cls()
        ada_vis = AdaVisionDTO.from_bighorn(dto)
        receiver._ada = ada_vis.to_10k()
        return receiver


# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL RECEIVER
# ═══════════════════════════════════════════════════════════════════════════════

class UniversalThoughtReceiver(BaseReceiver):
    """Receives bighorn.UniversalThought."""
    
    @classmethod
    def from_bighorn(cls, dto: Any) -> "UniversalThoughtReceiver":
        from .universal import UniversalThought as AdaUniversalThought
        receiver = cls()
        ada_thought = AdaUniversalThought.from_bighorn(dto)
        receiver._ada = ada_thought.to_10k()
        return receiver
