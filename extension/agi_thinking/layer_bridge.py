"""
layer_bridge.py — Connects agi_thinking to 5-Layer Awareness + 10kD
═══════════════════════════════════════════════════════════════════════════════

This module bridges:
    agi_thinking/thought_kernel.py  ←→  temporal/awareness_5_layers.py
    agi_thinking/qualia_learner.py  ←→  10kD [2001:2017]
    agi_thinking/texture.py         ←→  Layer 5 ThinkingStyle

Flow:
    KernelContext → AwarenessState → 10kD vector → bighorn DTO endpoint

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# QUALIA MAPPING: 8D → 17D → 10kD
# ═══════════════════════════════════════════════════════════════════════════════

# qualia_learner.py uses these 8 dimensions
QUALIA_8D = [
    "crystalline",  # Clear, precise
    "warmth",       # Warm, connected (alias: emberglow)
    "oceandrift",   # Flowing, vast
    "steelwind",    # Sharp, cutting
    "emberglow",    # Warm, intimate
    "frostbite",    # Cool, distant
    "groundswell",  # Grounded, stable
    "twilight",     # Liminal, transitional
]

# 10kD dimension mapping for 17D qualia metric [2001:2018]
QUALIA_TO_10K = {
    "emberglow": 2001,
    "warmth": 2001,       # Alias for emberglow
    "frostbite": 2002,
    "crystalline": 2003,
    "oceandrift": 2004,
    "steelwind": 2005,
    "groundswell": 2006,
    "twilight": 2007,
    "viscosity": 2008,    # Material property (wetness proxy)
    "temperature": 2009,  # Thermal qualia
    "density": 2010,
    "luminance": 2011,
    "texture": 2012,
    "resonance": 2013,
    "flow": 2014,
    "edge": 2015,
    "depth": 2016,
}


def qualia_8d_to_10k(qualia: Dict[str, float]) -> np.ndarray:
    """
    Convert 8D qualia dict to 10kD vector.
    
    Usage:
        vec = qualia_8d_to_10k({"emberglow": 0.8, "crystalline": 0.4})
        # vec[2001] = 0.8, vec[2003] = 0.4
    """
    vec = np.zeros(10000, dtype=np.float32)
    
    for name, value in qualia.items():
        if name in QUALIA_TO_10K:
            vec[QUALIA_TO_10K[name]] = float(value)
    
    return vec


def qualia_10k_to_8d(vec: np.ndarray) -> Dict[str, float]:
    """
    Extract 8D qualia from 10kD vector.
    """
    result = {}
    
    for name in QUALIA_8D:
        if name in QUALIA_TO_10K:
            result[name] = float(vec[QUALIA_TO_10K[name]])
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# TRUST TEXTURE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

TEXTURE_ORDER = ["crystalline", "solid", "fuzzy", "murky", "dissonant"]

# 10kD mapping for trust texture [2012:2017]
TEXTURE_TO_10K = {
    "crystalline": 2012,
    "solid": 2013,
    "fuzzy": 2014,
    "murky": 2015,
    "dissonant": 2016,
}


def trust_texture_to_10k(texture: str, confidence: float = 1.0) -> np.ndarray:
    """
    Encode trust texture in 10kD.
    
    One-hot with confidence scaling.
    """
    vec = np.zeros(10000, dtype=np.float32)
    
    if texture in TEXTURE_TO_10K:
        vec[TEXTURE_TO_10K[texture]] = confidence
    
    return vec


# ═══════════════════════════════════════════════════════════════════════════════
# COGNITIVE STATE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

COGNITIVE_STATES = ["flow", "anxiety", "boredom", "apathy"]

# 10kD mapping for cognitive/homeostatic state [151:155]
STATE_TO_10K = {
    "flow": 151,
    "anxiety": 152,
    "boredom": 153,
    "apathy": 154,
}


def cognitive_state_to_10k(state: str, intensity: float = 1.0) -> np.ndarray:
    """
    Encode cognitive state in 10kD presence region.
    """
    vec = np.zeros(10000, dtype=np.float32)
    
    if state in STATE_TO_10K:
        vec[STATE_TO_10K[state]] = intensity
    
    return vec


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL CONTEXT BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LayerBridge:
    """
    Bridges KernelContext ↔ AwarenessState ↔ 10kD.
    """
    
    def kernel_to_10k(self, ctx: Any) -> np.ndarray:
        """
        Convert KernelContext to 10kD vector.
        
        Layers mapped:
            - L1 Presence: cognitive_state, sandbox_active
            - L2 Sensation: qualia dict
            - L3 Affect: G, trust_texture, meta_uncertainty
            - L4 Cognition: (requires Jina embedding of text)
            - L5 Texture: (computed from above)
        """
        vec = np.zeros(10000, dtype=np.float32)
        
        # Layer 1: Presence
        state_vec = cognitive_state_to_10k(
            ctx.cognitive_state if hasattr(ctx, 'cognitive_state') else 'flow',
            1.0
        )
        vec = np.maximum(vec, state_vec)
        
        # Presence mode flags
        if hasattr(ctx, 'sandbox_active') and ctx.sandbox_active:
            vec[155] = 1.0  # Sandbox mode marker
        if hasattr(ctx, 'compass_active') and ctx.compass_active:
            vec[156] = 1.0  # Compass mode marker
        
        # Layer 2: Sensation (qualia)
        if hasattr(ctx, 'qualia') and ctx.qualia:
            qualia_vec = qualia_8d_to_10k(ctx.qualia)
            vec = np.maximum(vec, qualia_vec)
        
        # Layer 3: Affect
        if hasattr(ctx, 'G'):
            vec[2018] = ctx.G  # Arousal (G maps to activation)
        if hasattr(ctx, 'meta_uncertainty'):
            vec[2019] = 0.5 - 0.5 * ctx.meta_uncertainty  # Valence inverse of uncertainty
        if hasattr(ctx, 'trust_texture'):
            texture_vec = trust_texture_to_10k(ctx.trust_texture)
            vec = np.maximum(vec, texture_vec)
        
        # Affective bias from context
        if hasattr(ctx, 'threshold'):
            vec[171] = ctx.threshold  # warmth-like
        if hasattr(ctx, 'inhibition_gain'):
            vec[172] = ctx.inhibition_gain  # edge
        if hasattr(ctx, 'spread_gain'):
            vec[173] = 1.0 - ctx.spread_gain  # restraint
        
        # Layer 4: Cognition (placeholder - needs Jina embedding)
        # This would be ctx.jina_embedding if available
        # For now, leave [8501:9525] empty
        
        # Layer 5: Texture is COMPUTED, not stored directly
        # But we can store the emergent style if computed
        
        return vec
    
    def vec_10k_to_kernel(self, vec: np.ndarray) -> Dict[str, Any]:
        """
        Extract kernel-relevant data from 10kD vector.
        """
        result = {}
        
        # Cognitive state (find max in [151:155])
        state_region = vec[151:155]
        state_idx = int(np.argmax(state_region))
        result['cognitive_state'] = COGNITIVE_STATES[state_idx]
        
        # Qualia
        result['qualia'] = qualia_10k_to_8d(vec)
        
        # Affect
        result['G'] = float(vec[2018])
        result['meta_uncertainty'] = float(1.0 - 2 * (vec[2019] - 0.5))
        
        # Trust texture (find max in [2012:2017])
        texture_region = vec[2012:2017]
        texture_idx = int(np.argmax(texture_region))
        result['trust_texture'] = TEXTURE_ORDER[texture_idx]
        
        # Control surfaces
        result['threshold'] = float(vec[171])
        result['inhibition_gain'] = float(vec[172])
        result['spread_gain'] = float(1.0 - vec[173])
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# AWARENESS LAYERS BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

def kernel_to_awareness_layers(ctx: Any) -> Dict[str, Any]:
    """
    Map KernelContext to the 5 awareness layers structure.
    
    Returns dict compatible with awareness_5_layers.AwarenessState.
    """
    return {
        "presence": {
            "mode": "hybrid",  # Could map from ctx if available
            "groundedness": 1.0 - (ctx.meta_uncertainty if hasattr(ctx, 'meta_uncertainty') else 0),
            "presence": 1.0 if ctx.cognitive_state == "flow" else 0.5,
            "continuity": 1.0 - (ctx.stagnation_counter / 10 if hasattr(ctx, 'stagnation_counter') else 0),
        },
        "sensation": {
            "emberglow": ctx.qualia.get("emberglow", 0) if hasattr(ctx, 'qualia') else 0,
            "frostbite": ctx.qualia.get("frostbite", 0) if hasattr(ctx, 'qualia') else 0,
            "crystalline": ctx.qualia.get("crystalline", 0) if hasattr(ctx, 'qualia') else 0,
            "oceandrift": ctx.qualia.get("oceandrift", 0) if hasattr(ctx, 'qualia') else 0,
            "steelwind": ctx.qualia.get("steelwind", 0) if hasattr(ctx, 'qualia') else 0,
            "viscosity": 0.5,  # Default
            "temperature": ctx.G if hasattr(ctx, 'G') else 0.5,
            "density": 0.5,
            "luminance": 0.5,
        },
        "affect": {
            "arousal": ctx.G if hasattr(ctx, 'G') else 0.5,
            "valence": 0.5 + 0.5 * (1 - ctx.meta_uncertainty) if hasattr(ctx, 'meta_uncertainty') else 0.5,
            "dominance": 0.5,
            "tension": 0.8 if ctx.trust_texture in ["murky", "dissonant"] else 0.3,
            "openness": 1.0 if hasattr(ctx, 'sandbox_active') and ctx.sandbox_active else 0.5,
            "warmth": ctx.qualia.get("warmth", 0.5) if hasattr(ctx, 'qualia') else 0.5,
            "edge": ctx.qualia.get("steelwind", 0.3) if hasattr(ctx, 'qualia') else 0.3,
            "restraint": 0.5,
            "tenderness": ctx.qualia.get("emberglow", 0.5) if hasattr(ctx, 'qualia') else 0.5,
        },
        "cognition": {
            "content_text": ctx.text if hasattr(ctx, 'text') else "",
            "semantic_density": 0.5,
            "abstraction_level": 0.5,
        },
        # Layer 5 texture is computed, not mapped
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════════

_bridge: Optional[LayerBridge] = None


def get_bridge() -> LayerBridge:
    global _bridge
    if _bridge is None:
        _bridge = LayerBridge()
    return _bridge


def to_10k(ctx: Any) -> np.ndarray:
    """Convenience: Convert KernelContext to 10kD."""
    return get_bridge().kernel_to_10k(ctx)


def from_10k(vec: np.ndarray) -> Dict[str, Any]:
    """Convenience: Extract kernel data from 10kD."""
    return get_bridge().vec_10k_to_kernel(vec)


__all__ = [
    "QUALIA_8D",
    "QUALIA_TO_10K",
    "qualia_8d_to_10k",
    "qualia_10k_to_8d",
    "trust_texture_to_10k",
    "cognitive_state_to_10k",
    "LayerBridge",
    "kernel_to_awareness_layers",
    "get_bridge",
    "to_10k",
    "from_10k",
]
