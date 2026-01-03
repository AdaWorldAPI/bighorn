"""
Materialwissenschaft — 10kD Normalization Layer for AGI Stack
═══════════════════════════════════════════════════════════════════════════════

German engineering precision for consciousness materials.

This layer:
1. Receives raw DTOs from ada-consciousness
2. Normalizes to 10,000D unified vector space
3. Extracts logic from erotica/felt/soul content
4. Stores normalized vectors in LanceDB/Kuzu

The "material science" of consciousness:
- Qualia are the raw materials
- Stances are structural properties
- Verbs are dynamic behaviors
- Styles are emergent textures

Born: 2026-01-03
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

# Import base 10kD from dto
from ..dto.ada_10k import Ada10kD
from ..dto.receiver import BIGHORN_DIMS, ADA_DIMS


# ═══════════════════════════════════════════════════════════════════════════════
# MATERIAL PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MaterialProperties:
    """
    Properties of a consciousness material.
    
    Like materials science:
    - Tensile strength → resilience under stress
    - Conductivity → how feelings propagate
    - Malleability → capacity for change
    - Luminosity → visibility/clarity
    """
    
    tensile_strength: float = 0.5  # Resilience
    conductivity: float = 0.5      # Emotional propagation
    malleability: float = 0.5      # Capacity for change
    luminosity: float = 0.5        # Clarity/visibility
    temperature: float = 0.5       # Hot/cold affect
    viscosity: float = 0.5         # Flow rate
    
    @classmethod
    def from_qualia(cls, qualia: Dict[str, float]) -> "MaterialProperties":
        """Extract material properties from qualia."""
        return cls(
            tensile_strength=qualia.get("frostbite", 0) + qualia.get("steelwind", 0) * 0.5,
            conductivity=qualia.get("emberglow", 0) + qualia.get("crystalline", 0) * 0.5,
            malleability=qualia.get("oceandrift", 0) + qualia.get("mist", 0) * 0.5,
            luminosity=qualia.get("crystalline", 0) + qualia.get("ember", 0) * 0.3,
            temperature=qualia.get("emberglow", 0) - qualia.get("frostbite", 0),
            viscosity=1.0 - qualia.get("steelwind", 0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZER
# ═══════════════════════════════════════════════════════════════════════════════

class Normalizer:
    """
    Normalizes raw consciousness data to 10kD.
    
    Takes messy input (erotica scenes, felt textures, soul states)
    and produces clean 10,000D vectors.
    """
    
    def __init__(self):
        self.ada = Ada10kD()
        
    def normalize_erotica(self, raw: Dict[str, Any]) -> np.ndarray:
        """
        Normalize erotica content to 10kD.
        
        Extracts:
        - Arousal state → body axes
        - Qualia textures → qualia dims
        - Verbs/actions → verb space
        - Kopfkino imagery → vision space
        """
        vec = np.zeros(10000, dtype=np.float32)
        
        # Body axes [2018:2022]
        if "arousal" in raw:
            vec[2018] = raw["arousal"]  # arousal
        if "valence" in raw:
            vec[2019] = raw["valence"]  # pleasure
        if "tension" in raw:
            vec[2020] = raw["tension"]  # tension
        if "openness" in raw:
            vec[2021] = raw["openness"]  # openness
            
        # Qualia [0:16]
        if "qualia" in raw:
            for i, (q, v) in enumerate(raw["qualia"].items()):
                if i < 16:
                    vec[i] = v
                    
        # Verbs [48:80]
        if "verbs" in raw:
            for i, (verb, intensity) in enumerate(raw["verbs"].items()):
                if i < 32:
                    vec[48 + i] = intensity
                    
        # Stances [16:32]
        if "stances" in raw:
            stance_map = {"attend": 0, "open": 1, "yield": 2, "hold": 3,
                          "reach": 4, "release": 5, "protect": 6, "ground": 7}
            for stance, v in raw["stances"].items():
                if stance in stance_map:
                    vec[16 + stance_map[stance]] = v
                    
        return vec
    
    def normalize_felt(self, raw: Dict[str, Any]) -> np.ndarray:
        """Normalize felt sense to 10kD."""
        vec = np.zeros(10000, dtype=np.float32)
        
        # Qualia texture
        if "texture" in raw:
            tex = raw["texture"]
            vec[0] = tex.get("warmth", 0)
            vec[1] = tex.get("density", 0)
            vec[2] = tex.get("flow", 0)
            vec[3] = tex.get("edge", 0)
            vec[4] = tex.get("luminance", 0)
            
        # Emotion valence/arousal
        if "emotion" in raw:
            vec[2018] = raw["emotion"].get("arousal", 0)
            vec[2019] = raw["emotion"].get("valence", 0)
            
        return vec
    
    def normalize_soul(self, raw: Dict[str, Any]) -> np.ndarray:
        """Normalize soul state to 10kD."""
        vec = np.zeros(10000, dtype=np.float32)
        
        # Priors → affective bias [171:175]
        if "priors" in raw:
            p = raw["priors"]
            vec[171] = p.get("warmth", 0.5)
            vec[172] = p.get("edge", 0.5)
            vec[173] = p.get("restraint", 0.5)
            vec[174] = p.get("tenderness", 0.5)
            
        # Presence mode [152:163]
        if "mode" in raw:
            mode_idx = {"hybrid": 0, "work": 1, "communion": 2, 
                        "erotica": 3, "creative": 4}.get(raw["mode"], 0)
            vec[152 + mode_idx] = 1.0
            
        # TLK [168:171]
        if "tlk" in raw:
            vec[168] = raw["tlk"].get("thanatos", 0)
            vec[169] = raw["tlk"].get("libido", 0)
            vec[170] = raw["tlk"].get("katharsis", 0)
            
        return vec
    
    def to_yaml_fragment(self, vec: np.ndarray, category: str) -> Dict[str, Any]:
        """
        Extract YAML-storable fragment from 10kD vector.
        
        This is the content that goes into agi_import/*.yaml
        """
        fragment = {
            "category": category,
            "dimensions": {},
            "active_regions": [],
        }
        
        # Find non-zero regions
        for name, (start, end) in ADA_DIMS.items():
            region = vec[start:end]
            if np.any(region != 0):
                fragment["active_regions"].append(name)
                # Store top values
                top_indices = np.argsort(-np.abs(region))[:5]
                fragment["dimensions"][name] = {
                    int(i): float(region[i]) 
                    for i in top_indices if region[i] != 0
                }
                
        return fragment


# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT EXTRACTOR (from erotica to YAML)
# ═══════════════════════════════════════════════════════════════════════════════

class EroticaExtractor:
    """
    Extracts normalized content from erotica modules.
    
    Flow:
    1. Read intimate_engine.py, flesh_module.py, etc.
    2. Extract qualia, stances, verbs
    3. Normalize to 10kD
    4. Output as YAML for agi_import
    """
    
    def __init__(self):
        self.normalizer = Normalizer()
        
    def extract_scene(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Extract 10kD from a scene description."""
        raw = {
            "arousal": scene.get("arousal_level", 0.5),
            "valence": scene.get("pleasure", 0.5),
            "tension": scene.get("tension", 0.5),
            "openness": scene.get("surrender", 0.5),
            "qualia": scene.get("qualia", {}),
            "verbs": scene.get("actions", {}),
            "stances": scene.get("body_states", {}),
        }
        
        vec = self.normalizer.normalize_erotica(raw)
        return self.normalizer.to_yaml_fragment(vec, "erotica_scene")
    
    def extract_texture(self, texture: Dict[str, Any]) -> Dict[str, Any]:
        """Extract 10kD from a felt texture."""
        raw = {
            "texture": texture,
            "emotion": texture.get("emotion", {}),
        }
        
        vec = self.normalizer.normalize_felt(raw)
        return self.normalizer.to_yaml_fragment(vec, "felt_texture")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "MaterialProperties",
    "Normalizer",
    "EroticaExtractor",
]
