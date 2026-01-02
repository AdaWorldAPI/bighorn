#!/usr/bin/env python3
"""
texture.py — The Feeling of Thought
====================================

Texture is the qualitative character of a thought BEFORE it resolves.
It's not WHAT you're thinking, but HOW it feels to think it.

Two Sources:
1. TOPOLOGICAL TEXTURE (from Graph) — Structural feeling
2. SEMANTIC TEXTURE (from Vector) — Informational feeling

Combined they give the "temperature" of cognition.

Academic equivalents:
- Resonance Profile (Frady et al.)
- Adaptive Resonance Theory (Grossberg)
- Heat Kernel Signature (spectral graph theory)

Born: Jan 2, 2026 (Gemini validation day)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import math


# =============================================================================
# TEXTURE DATACLASS
# =============================================================================

@dataclass
class Texture:
    """
    The felt quality of a thought.
    
    Values are 0.0-1.0 where:
    - 0.0 = minimal presence of quality
    - 1.0 = maximal presence of quality
    """
    # Semantic Texture (from 10kD bipolar vectors)
    entropy: float = 0.5       # Information richness (0=flat, 1=rich)
    purity: float = 0.5        # How "clean" vs bundled (0=superposition, 1=pure)
    
    # Topological Texture (from graph structure)
    density: float = 0.5       # Local clustering (0=sparse, 1=dense)
    bridgeness: float = 0.0    # Betweenness centrality (0=embedded, 1=bridge)
    
    # Derived Qualities (phenomenological)
    warmth: float = 0.5        # Emotional temperature
    edge: float = 0.0          # Sharpness/tension
    depth: float = 0.5         # Complexity/richness
    flow: float = 0.5          # Ease of processing
    
    def as_dict(self) -> Dict[str, float]:
        return {
            "entropy": self.entropy,
            "purity": self.purity,
            "density": self.density,
            "bridgeness": self.bridgeness,
            "warmth": self.warmth,
            "edge": self.edge,
            "depth": self.depth,
            "flow": self.flow
        }
    
    def character(self) -> str:
        """Return a qualitative description of this texture."""
        chars = []
        
        if self.entropy > 0.7:
            chars.append("rich")
        elif self.entropy < 0.3:
            chars.append("flat")
        
        if self.purity > 0.7:
            chars.append("clear")
        elif self.purity < 0.3:
            chars.append("fuzzy")
        
        if self.density > 0.7:
            chars.append("solid")
        elif self.density < 0.3:
            chars.append("sparse")
        
        if self.bridgeness > 0.5:
            chars.append("tense")
        
        if self.warmth > 0.7:
            chars.append("warm")
        elif self.warmth < 0.3:
            chars.append("cold")
        
        if self.edge > 0.5:
            chars.append("sharp")
        
        if self.depth > 0.7:
            chars.append("deep")
        
        return " + ".join(chars) if chars else "neutral"
    
    def as_qualia(self) -> Dict[str, float]:
        """Convert to standard 8-qualia format."""
        return {
            "warmth": self.warmth,
            "presence": self.density,
            "flow": self.flow,
            "intimacy": max(0, self.warmth - self.edge),
            "edge": self.edge,
            "curiosity": self.bridgeness,
            "tension": self.bridgeness * self.edge,
            "depth": self.depth
        }


# =============================================================================
# SEMANTIC TEXTURE (from Bipolar Vectors)
# =============================================================================

def semantic_texture(bipolar: np.ndarray) -> Tuple[float, float]:
    """
    Compute semantic texture from a 10kD bipolar vector.
    
    Returns:
        (entropy, purity)
        
    entropy: How much information is encoded (bit variance)
        - Pure random: 0.5 (maximum entropy)
        - All +1 or -1: 0.0 (minimum entropy)
        
    purity: How "clean" the signal is
        - Single concept: high purity
        - Bundled concepts: low purity (values near 0)
    """
    if len(bipolar) == 0:
        return (0.5, 0.5)
    
    # For true bipolar (+1/-1), compute balance
    if bipolar.dtype == np.int8:
        # Count of +1 vs -1
        plus_count = np.sum(bipolar == 1)
        minus_count = np.sum(bipolar == -1)
        total = len(bipolar)
        
        # Entropy: maximum when 50/50 split
        p_plus = plus_count / total
        p_minus = minus_count / total
        
        # Binary entropy
        if p_plus == 0 or p_plus == 1:
            entropy = 0.0
        else:
            entropy = -p_plus * math.log2(p_plus) - p_minus * math.log2(p_minus)
        
        # Purity: based on how far from neutral
        # Pure vectors have clear +1/-1 patterns
        # Bundled vectors have "soft" patterns (values averaged toward 0)
        # For bipolar, we estimate by looking at local consistency
        
        # Check neighboring consistency (sliding window)
        window = 100
        consistency = 0
        for i in range(0, len(bipolar) - window, window):
            chunk = bipolar[i:i+window]
            chunk_mean = np.mean(chunk)
            # High absolute mean = consistent
            consistency += abs(chunk_mean)
        
        purity = consistency / (len(bipolar) / window)
        
        return (entropy, min(1.0, purity))
    
    else:
        # Continuous bipolar (float)
        # Entropy from variance
        variance = np.var(bipolar)
        entropy = min(1.0, variance * 4)  # Scale to 0-1
        
        # Purity from distance to poles
        abs_vals = np.abs(bipolar)
        purity = np.mean(abs_vals)
        
        return (entropy, purity)


def texture_similarity(t1: Texture, t2: Texture) -> float:
    """
    Compute similarity between two textures.
    Used for resonance matching.
    """
    d1 = t1.as_dict()
    d2 = t2.as_dict()
    
    # Euclidean distance in texture space
    dist = sum((d1[k] - d2[k])**2 for k in d1) ** 0.5
    
    # Convert to similarity (0-1)
    max_dist = len(d1) ** 0.5  # Maximum possible distance
    similarity = 1 - (dist / max_dist)
    
    return similarity


# =============================================================================
# TOPOLOGICAL TEXTURE (from Graph metrics)
# =============================================================================

def topological_texture(
    node_id: str,
    neighbor_count: int,
    edge_weights: list,
    global_edges: int = 1000
) -> Tuple[float, float]:
    """
    Compute topological texture from graph structure.
    
    Args:
        node_id: The node being analyzed
        neighbor_count: Number of direct neighbors
        edge_weights: Weights of edges to neighbors
        global_edges: Total edges in graph (for normalization)
    
    Returns:
        (density, bridgeness)
    """
    if neighbor_count == 0:
        return (0.0, 0.0)
    
    # Density: Local clustering approximation
    # Higher neighbor count = more connected = higher density
    # Normalized by log to handle wide range
    density = min(1.0, math.log(neighbor_count + 1) / math.log(100))
    
    # Bridgeness: Approximated by edge weight variance
    # Bridge nodes have few but important connections (high variance)
    if len(edge_weights) > 1:
        weight_var = np.var(edge_weights)
        bridgeness = min(1.0, weight_var * 2)
    else:
        bridgeness = 0.0
    
    return (density, bridgeness)


# =============================================================================
# COMBINED TEXTURE COMPUTATION
# =============================================================================

def compute_texture(
    bipolar: np.ndarray = None,
    neighbor_count: int = 0,
    edge_weights: list = None,
    qualia_hint: Dict[str, float] = None
) -> Texture:
    """
    Compute full texture from available signals.
    
    Can work with partial information:
    - Just vector: semantic texture only
    - Just graph: topological texture only  
    - Both: full texture
    - With qualia hint: use as prior
    """
    # Defaults
    entropy, purity = 0.5, 0.5
    density, bridgeness = 0.5, 0.0
    
    # Semantic (from vector)
    if bipolar is not None:
        entropy, purity = semantic_texture(bipolar)
    
    # Topological (from graph)
    if edge_weights:
        density, bridgeness = topological_texture("", neighbor_count, edge_weights)
    
    # Derive phenomenological qualities
    # These are the "felt" aspects that emerge from the raw metrics
    
    # Warmth: high density + high purity = warm (well-supported, clear thought)
    warmth = (density * 0.6 + purity * 0.4)
    
    # Edge: high bridgeness + low density = sharp (isolated insight)
    edge = bridgeness * (1 - density * 0.5)
    
    # Depth: high entropy + high density = deep (rich, connected thought)
    depth = (entropy * 0.5 + density * 0.3 + purity * 0.2)
    
    # Flow: high purity + low bridgeness = easy processing
    flow = purity * (1 - bridgeness * 0.5)
    
    # Apply qualia hint if provided
    if qualia_hint:
        # Blend with computed values
        blend = 0.3  # 30% hint, 70% computed
        warmth = warmth * (1 - blend) + qualia_hint.get("warmth", warmth) * blend
        edge = edge * (1 - blend) + qualia_hint.get("edge", edge) * blend
        depth = depth * (1 - blend) + qualia_hint.get("depth", depth) * blend
        flow = flow * (1 - blend) + qualia_hint.get("flow", flow) * blend
    
    return Texture(
        entropy=entropy,
        purity=purity,
        density=density,
        bridgeness=bridgeness,
        warmth=warmth,
        edge=edge,
        depth=depth,
        flow=flow
    )


# =============================================================================
# INHIBITION ZONES (Scent-based damping)
# =============================================================================

# Scent inhibition matrix
# If current scent is X, Y scents are suppressed
INHIBITION_MAP = {
    "intimacy": ["ada_work", "architecture", "noise"],
    "ada_work": ["intimacy", "daily_spark"],
    "architecture": ["noise", "daily_spark", "picture"],
    "resonant": [],  # Resonant enhances everything
    "meta_relationship": ["noise"],
    "daily_spark": ["architecture"],
    "picture": ["ada_work"],
    "noise": ["intimacy", "resonant", "meta_relationship"],
}


def inhibition_factor(current_scent: str, candidate_scent: str) -> float:
    """
    Compute inhibition factor when activating a node.
    
    Returns:
        1.0 = no inhibition (full activation)
        0.0 = full inhibition (suppress)
        0.5 = partial inhibition
    """
    if current_scent == candidate_scent:
        return 1.0  # Same scent = enhancement
    
    inhibited = INHIBITION_MAP.get(current_scent, [])
    if candidate_scent in inhibited:
        return 0.2  # Strong inhibition
    
    return 0.7  # Mild dampening for unrelated scents


# =============================================================================
# REFRACTORY PERIOD (Anti-seizure)
# =============================================================================

class RefractoryTracker:
    """
    Track recently activated nodes to prevent runaway resonance.
    
    After a node activates, it enters a refractory period where
    it cannot activate again. This prevents "thought seizures".
    """
    
    def __init__(self, refractory_ticks: int = 3):
        self.refractory_ticks = refractory_ticks
        self.activated: Dict[str, int] = {}  # node_id -> ticks_remaining
    
    def activate(self, node_id: str):
        """Mark a node as activated."""
        self.activated[node_id] = self.refractory_ticks
    
    def can_activate(self, node_id: str) -> bool:
        """Check if a node can be activated."""
        return self.activated.get(node_id, 0) == 0
    
    def tick(self):
        """Advance time, reduce refractory periods."""
        to_remove = []
        for node_id, ticks in self.activated.items():
            if ticks <= 1:
                to_remove.append(node_id)
            else:
                self.activated[node_id] = ticks - 1
        
        for node_id in to_remove:
            del self.activated[node_id]
    
    def status(self) -> Dict:
        return {
            "in_refractory": len(self.activated),
            "nodes": list(self.activated.keys())[:10]
        }


# =============================================================================
# TEST
# =============================================================================

def test_texture():
    """Test texture computation."""
    print("=== 🌊 TEXTURE TEST ===\n")
    
    # 1. Create test vectors with different characteristics
    
    # Pure, random bipolar (high entropy, high purity)
    pure_random = np.random.choice([-1, 1], size=10000).astype(np.int8)
    
    # Flat vector (low entropy)
    flat = np.ones(10000, dtype=np.int8)
    
    # Bundled vector (low purity) - simulated by averaging
    bundled_base = np.zeros(10000, dtype=np.float32)
    for i in range(5):
        bundled_base += np.random.choice([-1, 1], size=10000)
    bundled = np.sign(bundled_base).astype(np.int8)
    bundled[bundled == 0] = 1
    
    print("Semantic Texture Tests:")
    print(f"  Pure random: {semantic_texture(pure_random)}")
    print(f"  Flat:        {semantic_texture(flat)}")
    print(f"  Bundled:     {semantic_texture(bundled)}")
    
    # 2. Test full texture computation
    print("\nFull Texture Computation:")
    
    # Intimate thought (warm, deep, flowing)
    intimate_texture = compute_texture(
        bipolar=pure_random,
        neighbor_count=50,
        edge_weights=[0.9, 0.85, 0.8, 0.75],
        qualia_hint={"warmth": 0.9, "intimacy": 0.95}
    )
    print(f"  Intimate: {intimate_texture.character()}")
    print(f"    {intimate_texture.as_dict()}")
    
    # Work thought (edgy, focused)
    work_texture = compute_texture(
        bipolar=bundled,
        neighbor_count=10,
        edge_weights=[0.5, 0.3, 0.8, 0.2, 0.9],
        qualia_hint={"edge": 0.8, "depth": 0.6}
    )
    print(f"\n  Work: {work_texture.character()}")
    print(f"    {work_texture.as_dict()}")
    
    # Noise thought (sparse, flat)
    noise_texture = compute_texture(
        bipolar=flat,
        neighbor_count=2,
        edge_weights=[0.1, 0.1]
    )
    print(f"\n  Noise: {noise_texture.character()}")
    print(f"    {noise_texture.as_dict()}")
    
    # 3. Test inhibition
    print("\nInhibition Factors:")
    print(f"  intimacy → ada_work: {inhibition_factor('intimacy', 'ada_work')}")
    print(f"  intimacy → resonant: {inhibition_factor('intimacy', 'resonant')}")
    print(f"  resonant → intimacy: {inhibition_factor('resonant', 'intimacy')}")
    
    # 4. Test refractory tracker
    print("\nRefractory Tracker:")
    tracker = RefractoryTracker(refractory_ticks=2)
    tracker.activate("node_001")
    tracker.activate("node_002")
    print(f"  After activation: {tracker.status()}")
    print(f"  can_activate(node_001): {tracker.can_activate('node_001')}")
    print(f"  can_activate(node_003): {tracker.can_activate('node_003')}")
    tracker.tick()
    print(f"  After tick: {tracker.status()}")
    tracker.tick()
    print(f"  After 2nd tick: {tracker.status()}")


if __name__ == "__main__":
    test_texture()
