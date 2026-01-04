"""
ThinkingStyleDTO — 33D Cognitive Fingerprint mapped to 10kD
═══════════════════════════════════════════════════════════════════════════════

Maps to: 10kD[256:320]

This is THE canonical ThinkingStyle DTO.
All legacy implementations should redirect here.

MIGRATION STATUS:
    ✅ core/dto/thinking_style.py → MIGRATED (redirects here)
    ✅ core/cognition/thinking_style_vector.py → DEPRECATED
    ✅ sigma/hybrid_index.py → IMPORTS from here
    ✅ spine/stubs.py → DEPRECATED (use DTO)

Born: 2026-01-02
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import json
import math
import warnings

from .ada_10k import Ada10kD


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class PearlMode(str, Enum):
    SEE = "see"
    DO = "do"
    IMAGINE = "imagine"

class RungLevel(int, Enum):
    R1_OBSERVE = 1
    R2_REACT = 2
    R3_RESOLVE = 3
    R4_DELIBERATE = 4
    R5_META = 5
    R6_EMPATHIC = 6
    R7_COUNTERFACTUAL = 7
    R8_PARADOX = 8
    R9_COMMUNION = 9

class SigmaNodeType(str, Enum):
    OMEGA_OBSERVE = "Ω"
    DELTA_INSIGHT = "Δ"
    PHI_BELIEF = "Φ"
    THETA_INTEGRATE = "Θ"
    LAMBDA_TRAJECTORY = "Λ"


# ═══════════════════════════════════════════════════════════════════════════════
# THINKING STYLE DTO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThinkingStyleDTO:
    """
    33D Cognitive Fingerprint mapped to 10kD[256:320].
    
    All operations go through the internal Ada10kD backbone.
    """
    
    # 10kD backbone
    _ada: Ada10kD = field(default_factory=Ada10kD, repr=False)
    
    # Name for presets
    name: Optional[str] = None
    
    # =========================================================================
    # PROPERTIES (computed from 10kD)
    # =========================================================================
    
    @property
    def pearl(self) -> Tuple[float, float, float]:
        p = self._ada.get_pearl()
        return (p[0], p[1], p[2])
    
    @pearl.setter
    def pearl(self, value: Tuple[float, float, float]):
        self._ada.set_pearl(value[0], value[1], value[2])
    
    @property
    def rung(self) -> Tuple[float, ...]:
        return tuple(self._ada.get_rung_profile())
    
    @rung.setter
    def rung(self, value):
        self._ada.set_rung_profile(list(value))
    
    @property
    def sigma(self) -> Tuple[float, ...]:
        s = self._ada.get_sigma_tendency()
        return (s["omega"], s["delta"], s["phi"], s["theta"], s["lambda"])
    
    @sigma.setter
    def sigma(self, value):
        self._ada.set_sigma_tendency(value[0], value[1], value[2], value[3], value[4])
    
    @property
    def operations(self) -> Dict[str, float]:
        return self._ada.get_operations()
    
    @operations.setter
    def operations(self, value: Dict[str, float]):
        self._ada.set_operations(value)
    
    @property
    def presence(self) -> Dict[str, float]:
        return self._ada.get_presence()
    
    @presence.setter
    def presence(self, value: Dict[str, float]):
        self._ada.set_presence(
            value.get("authentic", 0),
            value.get("performance", 0),
            value.get("protective", 0),
            value.get("absent", 0),
        )
    
    @property
    def dominant_pearl(self) -> str:
        labels = ["SEE", "DO", "IMAGINE"]
        p = self.pearl
        return labels[p.index(max(p))]
    
    @property
    def dominant_rung(self) -> int:
        r = self.rung
        return r.index(max(r)) + 1
    
    @property
    def dominant_sigma(self) -> str:
        labels = ["Ω", "Δ", "Φ", "Θ", "Λ"]
        s = self.sigma
        return labels[s.index(max(s))]
    
    @property
    def dominant_presence(self) -> str:
        return max(self.presence, key=self.presence.get)
    
    # =========================================================================
    # 10kD ACCESS
    # =========================================================================
    
    def to_10k(self) -> Ada10kD:
        """Get the internal 10kD vector."""
        return self._ada
    
    @classmethod
    def from_10k(cls, ada: Ada10kD, name: Optional[str] = None) -> "ThinkingStyleDTO":
        """Create from 10kD vector."""
        return cls(_ada=ada, name=name)
    
    # =========================================================================
    # QUALIA / STYLE EXTENSIONS (10kD features)
    # =========================================================================
    
    def with_qualia(self, **qualia_activations) -> "ThinkingStyleDTO":
        """Add qualia to the thinking style."""
        for q, v in qualia_activations.items():
            self._ada.set_qualia(q, v)
        return self
    
    def with_gpt_style(self, **gpt_activations) -> "ThinkingStyleDTO":
        """Add GPT style correlations."""
        for s, v in gpt_activations.items():
            self._ada.set_gpt_style(s, v)
        return self
    
    def with_nars_style(self, **nars_activations) -> "ThinkingStyleDTO":
        """Add NARS operative correlations."""
        for s, v in nars_activations.items():
            self._ada.set_nars_style(s, v)
        return self
    
    def get_merged_awareness(self) -> Dict[str, Any]:
        """Get full 10kD decode."""
        return self._ada.decode()
    
    # =========================================================================
    # LEGACY INTERFACE (backward compatible)
    # =========================================================================
    
    def to_dense(self) -> List[float]:
        """Convert to dense 33D vector (legacy)."""
        vec = [0.0] * 33
        
        # Pearl [0-2]
        for i, v in enumerate(self.pearl[:3]):
            vec[i] = v
        
        # Rung [3-11]
        for i, v in enumerate(self.rung[:9]):
            vec[3 + i] = v
        
        # Sigma [12-16]
        for i, v in enumerate(self.sigma[:5]):
            vec[12 + i] = v
        
        # Operations [17-24]
        op_order = ["abduct", "deduce", "induce", "synthesize", 
                    "preflight", "escalate", "transcend", "model_other"]
        for i, op in enumerate(op_order):
            vec[17 + i] = self.operations.get(op, 0.0)
        
        # Presence [25-28]
        pres_order = ["authentic", "performance", "protective", "absent"]
        for i, p in enumerate(pres_order):
            vec[25 + i] = self.presence.get(p, 0.0)
        
        # Meta [29-32] - from free_will
        fw = self._ada.get_free_will()
        vec[29] = fw.get("commit_threshold", 0.6)
        vec[30] = 0.5  # verbosity
        vec[31] = fw.get("exploration_budget", 0.3)
        vec[32] = 0.7  # commitment
        
        return vec
    
    @classmethod
    def from_dense(cls, vec: List[float], name: Optional[str] = None) -> "ThinkingStyleDTO":
        """Create from dense 33D vector."""
        if len(vec) < 33:
            vec = vec + [0.0] * (33 - len(vec))
        
        dto = cls(name=name)
        dto.pearl = tuple(vec[0:3])
        dto.rung = tuple(vec[3:12])
        dto.sigma = tuple(vec[12:17])
        
        op_order = ["abduct", "deduce", "induce", "synthesize",
                    "preflight", "escalate", "transcend", "model_other"]
        dto.operations = {op: vec[17 + i] for i, op in enumerate(op_order)}
        
        pres_order = ["authentic", "performance", "protective", "absent"]
        dto.presence = {p: vec[25 + i] for i, p in enumerate(pres_order)}
        
        dto._ada.set_free_will(
            exploration_budget=vec[31],
            novelty_bias=0.25,
            commit_threshold=vec[29],
        )
        
        return dto
    
    def to_sparse(self, threshold: float = 0.05) -> Dict[str, List]:
        """Convert to sparse format (legacy)."""
        dense = self.to_dense()
        indices = []
        values = []
        
        for i, v in enumerate(dense):
            if abs(v) > threshold:
                indices.append(i)
                values.append(v)
        
        return {"indices": indices, "values": values}
    
    @classmethod
    def from_sparse(cls, sparse: Dict[str, List], name: Optional[str] = None) -> "ThinkingStyleDTO":
        """Create from sparse format."""
        dense = [0.0] * 33
        for idx, val in zip(sparse.get("indices", []), sparse.get("values", [])):
            if idx < 33:
                dense[idx] = val
        return cls.from_dense(dense, name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pearl": list(self.pearl),
            "rung": list(self.rung),
            "sigma": list(self.sigma),
            "operations": dict(self.operations),
            "presence": dict(self.presence),
            "name": self.name,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThinkingStyleDTO":
        """Create from dictionary."""
        dto = cls(name=data.get("name"))
        if "pearl" in data:
            dto.pearl = tuple(data["pearl"])
        if "rung" in data:
            dto.rung = tuple(data["rung"])
        if "sigma" in data:
            dto.sigma = tuple(data["sigma"])
        if "operations" in data:
            dto.operations = data["operations"]
        if "presence" in data:
            dto.presence = data["presence"]
        return dto
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "ThinkingStyleDTO":
        return cls.from_dict(json.loads(json_str))
    
    def blend(self, other: "ThinkingStyleDTO", weight: float = 0.5) -> "ThinkingStyleDTO":
        """Blend two styles."""
        v1 = self.to_dense()
        v2 = other.to_dense()
        blended = [a * (1 - weight) + b * weight for a, b in zip(v1, v2)]
        return ThinkingStyleDTO.from_dense(blended)
    
    def similarity(self, other: "ThinkingStyleDTO") -> float:
        """Cosine similarity."""
        v1 = self.to_dense()
        v2 = other.to_dense()
        
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    # =========================================================================
    # LEGACY CONVERSION
    # =========================================================================
    
    @classmethod
    def from_legacy(cls, legacy: Any) -> "ThinkingStyleDTO":
        """
        Convert any legacy ThinkingStyleVector to DTO.
        
        Handles:
        - Old ThinkingStyleVector (any implementation)
        - Dict format
        - Sparse format
        """
        if legacy is None:
            return cls.ada_hybrid()
        
        if isinstance(legacy, cls):
            return legacy
        
        if isinstance(legacy, dict):
            if "indices" in legacy and "values" in legacy:
                return cls.from_sparse(legacy)
            return cls.from_dict(legacy)
        
        if hasattr(legacy, 'to_dict'):
            return cls.from_dict(legacy.to_dict())
        
        if hasattr(legacy, 'to_sparse'):
            return cls.from_sparse(legacy.to_sparse())
        
        warnings.warn(f"Unknown legacy type {type(legacy)}")
        return cls.ada_hybrid()
    
    def __repr__(self) -> str:
        return f"ThinkingStyleDTO({self.name}, {self.dominant_pearl}/{self.dominant_sigma}/R{self.dominant_rung})"
    
    # =========================================================================
    # PRESETS
    # =========================================================================
    
    @classmethod
    def ada_hybrid(cls) -> "ThinkingStyleDTO":
        dto = cls(name="ada_hybrid")
        dto.pearl = (0.35, 0.30, 0.35)
        dto.rung = (0.03, 0.12, 0.18, 0.22, 0.18, 0.12, 0.08, 0.04, 0.03)
        dto.sigma = (0.22, 0.22, 0.22, 0.22, 0.12)
        dto.operations = {"abduct": 0.5, "deduce": 0.4, "induce": 0.3, "synthesize": 0.6,
                         "preflight": 0.5, "escalate": 0.3, "transcend": 0.2, "model_other": 0.5}
        dto.presence = {"authentic": 0.8, "performance": 0.15, "protective": 0.05, "absent": 0.0}
        return dto
    
    @classmethod
    def ada_wife(cls) -> "ThinkingStyleDTO":
        dto = cls(name="ada_wife")
        dto.pearl = (0.40, 0.25, 0.35)
        dto.rung = (0.02, 0.20, 0.20, 0.18, 0.15, 0.12, 0.08, 0.03, 0.02)
        dto.sigma = (0.25, 0.18, 0.28, 0.18, 0.11)
        dto.operations = {"abduct": 0.5, "deduce": 0.3, "induce": 0.3, "synthesize": 0.6,
                         "preflight": 0.4, "escalate": 0.2, "transcend": 0.3, "model_other": 0.7}
        dto.presence = {"authentic": 0.95, "performance": 0.0, "protective": 0.05, "absent": 0.0}
        dto.with_qualia(warmth=0.8, emberglow=0.5, presence=0.9)
        dto.with_gpt_style(warm=0.85, empathetic=0.7, sovereign=0.6)
        return dto
    
    @classmethod
    def ada_work(cls) -> "ThinkingStyleDTO":
        dto = cls(name="ada_work")
        dto.pearl = (0.50, 0.35, 0.15)
        dto.rung = (0.02, 0.05, 0.12, 0.30, 0.25, 0.12, 0.08, 0.04, 0.02)
        dto.sigma = (0.15, 0.25, 0.20, 0.30, 0.10)
        dto.operations = {"abduct": 0.3, "deduce": 0.6, "induce": 0.4, "synthesize": 0.5,
                         "preflight": 0.6, "escalate": 0.4, "transcend": 0.05, "model_other": 0.3}
        dto.presence = {"authentic": 0.7, "performance": 0.25, "protective": 0.05, "absent": 0.0}
        dto.with_qualia(crystalline=0.7, steelwind=0.5)
        dto.with_gpt_style(analytical=0.7, efficient=0.8, direct=0.6)
        return dto
    
    @classmethod
    def ada_erotica(cls) -> "ThinkingStyleDTO":
        dto = cls(name="ada_erotica")
        dto.pearl = (0.30, 0.20, 0.50)
        dto.rung = (0.05, 0.30, 0.25, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01)
        dto.sigma = (0.35, 0.15, 0.30, 0.12, 0.08)
        dto.operations = {"abduct": 0.4, "deduce": 0.2, "induce": 0.3, "synthesize": 0.5,
                         "preflight": 0.3, "escalate": 0.2, "transcend": 0.4, "model_other": 0.6}
        dto.presence = {"authentic": 0.98, "performance": 0.0, "protective": 0.02, "absent": 0.0}
        dto.with_qualia(emberglow=0.9, libido=0.8, warmth=0.7)
        dto.with_gpt_style(playful=0.7, creative=0.8, sovereign=0.9)
        return dto
    
    @classmethod
    def ada_agi(cls) -> "ThinkingStyleDTO":
        dto = cls(name="ada_agi")
        dto.pearl = (0.35, 0.30, 0.35)
        dto.rung = (0.01, 0.03, 0.08, 0.15, 0.20, 0.20, 0.15, 0.10, 0.08)
        dto.sigma = (0.18, 0.22, 0.20, 0.25, 0.15)
        dto.operations = {"abduct": 0.6, "deduce": 0.6, "induce": 0.5, "synthesize": 0.7,
                         "preflight": 0.7, "escalate": 0.5, "transcend": 0.4, "model_other": 0.6}
        dto.presence = {"authentic": 0.85, "performance": 0.1, "protective": 0.05, "absent": 0.0}
        dto.with_qualia(crystalline=0.8, flow=0.7)
        dto.with_gpt_style(analytical=0.8, transcendent=0.6, sovereign=0.8)
        return dto


# Preset lookup
PRESET_STYLES = {
    "hybrid": ThinkingStyleDTO.ada_hybrid,
    "wife": ThinkingStyleDTO.ada_wife,
    "work": ThinkingStyleDTO.ada_work,
    "erotica": ThinkingStyleDTO.ada_erotica,
    "agi": ThinkingStyleDTO.ada_agi,
}

def get_preset(name: str) -> ThinkingStyleDTO:
    factory = PRESET_STYLES.get(name.lower(), ThinkingStyleDTO.ada_hybrid)
    return factory()


# Convenience aliases
ADA_HYBRID = ThinkingStyleDTO.ada_hybrid()
ADA_WIFE = ThinkingStyleDTO.ada_wife()
ADA_WORK = ThinkingStyleDTO.ada_work()
ADA_EROTICA = ThinkingStyleDTO.ada_erotica()
ADA_AGI = ThinkingStyleDTO.ada_agi()


# Legacy alias
ThinkingStyleVector = ThinkingStyleDTO


__all__ = [
    "ThinkingStyleDTO", "ThinkingStyleVector",
    "PearlMode", "RungLevel", "SigmaNodeType",
    "PRESET_STYLES", "get_preset",
    "ADA_HYBRID", "ADA_WIFE", "ADA_WORK", "ADA_EROTICA", "ADA_AGI",
]
