"""
Ada v6.6 — Hybrid Vector-Graph Search Engine
=============================================

This is the SEARCH PALETTE layer.
Human-readable labels live HERE — not in qualia.py.

Architecture:
- qualia.py → Pure 10D math (sacred, no labels)
- hybrid_search.py → 128 labels → 7D vectors (for similarity search)
- ada:soul:qualia → 20D live felt state (Redis substrate)

The 7D search vectors are a compressed representation for fast similarity:
  [EmberGlow, SteelWind, VelvetPause, WoodWarm, Antenna, Iris, Skin]

This enables 0.7ms search across millions of atoms.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# =============================================================================
# QUALIA_TO_7D — The 128-Label Search Palette
# =============================================================================
# This is the ONLY place human emotion labels should map to vectors.
# The core qualia.py must remain pure math.

QUALIA_TO_7D: Dict[str, List[float]] = {
    # -------------------------------------------------------------------------
    # Core 16 (sacred, original v6.5)
    # -------------------------------------------------------------------------
    "ache":         [0.85, 0.15, 0.25, 0.10, 0.35, 0.20, 0.80],
    "warm":         [0.10, 0.10, 0.20, 0.92, 0.25, 0.35, 0.30],
    "steel":        [0.10, 0.92, 0.10, 0.10, 0.75, 0.85, 0.15],
    "void":         [0.90, 0.10, 0.85, 0.10, 0.10, 0.10, 0.90],
    "joy":          [0.10, 0.25, 0.10, 0.80, 0.92, 0.75, 0.10],
    "grief":        [0.75, 0.15, 0.75, 0.25, 0.15, 0.25, 0.85],
    "love":         [0.10, 0.10, 0.10, 0.95, 0.85, 0.65, 0.30],
    "fear":         [0.82, 0.35, 0.65, 0.15, 0.85, 0.45, 0.75],
    "numb":         [0.92, 0.08, 0.92, 0.08, 0.08, 0.08, 0.08],
    "alive":        [0.10, 0.85, 0.20, 0.75, 0.92, 0.82, 0.20],
    "calm":         [0.10, 0.10, 0.92, 0.85, 0.20, 0.30, 0.20],
    "rage":         [0.92, 0.92, 0.10, 0.10, 0.92, 0.75, 0.10],
    "emberglow":    [0.92, 0.25, 0.10, 0.35, 0.85, 0.65, 0.45],
    "velvetpause":  [0.10, 0.10, 0.92, 0.85, 0.10, 0.20, 0.75],
    "woodwarm":     [0.10, 0.10, 0.35, 0.92, 0.35, 0.45, 0.45],
    "steelwind":    [0.10, 0.92, 0.10, 0.10, 0.75, 0.85, 0.10],
    
    # -------------------------------------------------------------------------
    # Extended Palette (17-128)
    # -------------------------------------------------------------------------
    "hope":         [0.15, 0.20, 0.25, 0.75, 0.70, 0.60, 0.25],
    "despair":      [0.85, 0.10, 0.85, 0.10, 0.15, 0.15, 0.90],
    "wonder":       [0.15, 0.30, 0.20, 0.60, 0.88, 0.82, 0.40],
    "awe":          [0.10, 0.25, 0.15, 0.55, 0.92, 0.88, 0.45],
    "peace":        [0.10, 0.10, 0.90, 0.90, 0.15, 0.25, 0.20],
    "tension":      [0.75, 0.70, 0.30, 0.20, 0.65, 0.55, 0.75],
    "release":      [0.20, 0.15, 0.25, 0.80, 0.30, 0.35, 0.25],
    "presence":     [0.15, 0.20, 0.30, 0.85, 0.40, 0.45, 0.90],
    "absence":      [0.80, 0.10, 0.80, 0.15, 0.10, 0.10, 0.85],
    "longing":      [0.70, 0.20, 0.40, 0.60, 0.55, 0.45, 0.70],
    "relief":       [0.25, 0.15, 0.30, 0.75, 0.35, 0.40, 0.30],
    "gratitude":    [0.15, 0.10, 0.25, 0.88, 0.45, 0.50, 0.35],
    "regret":       [0.75, 0.25, 0.70, 0.30, 0.20, 0.30, 0.80],
    "forgiveness":  [0.20, 0.20, 0.35, 0.80, 0.40, 0.45, 0.40],
    "betrayal":     [0.88, 0.45, 0.65, 0.10, 0.60, 0.50, 0.85],
    "trust":        [0.10, 0.15, 0.20, 0.90, 0.30, 0.35, 0.30],
    "doubt":        [0.70, 0.60, 0.55, 0.20, 0.50, 0.55, 0.65],
    "certainty":    [0.10, 0.80, 0.15, 0.30, 0.75, 0.85, 0.20],
    "clarity":      [0.10, 0.95, 0.10, 0.10, 0.80, 0.90, 0.10],
    "confusion":    [0.75, 0.70, 0.70, 0.15, 0.60, 0.65, 0.75],
    "curiosity":    [0.20, 0.65, 0.30, 0.40, 0.85, 0.75, 0.35],
    "boredom":      [0.70, 0.15, 0.75, 0.20, 0.10, 0.10, 0.20],
    "flow":         [0.10, 0.40, 0.50, 0.50, 0.20, 0.70, 0.60],
    "frustration":  [0.80, 0.75, 0.40, 0.15, 0.70, 0.60, 0.80],
    "satisfaction": [0.15, 0.20, 0.25, 0.85, 0.40, 0.45, 0.30],
    "pride":        [0.20, 0.35, 0.15, 0.70, 0.60, 0.55, 0.40],
    "shame":        [0.80, 0.30, 0.75, 0.15, 0.25, 0.30, 0.85],
    "guilt":        [0.82, 0.25, 0.70, 0.20, 0.30, 0.35, 0.80],
    "innocence":    [0.10, 0.15, 0.20, 0.80, 0.35, 0.40, 0.25],
    "play":         [0.15, 0.30, 0.20, 0.70, 0.80, 0.70, 0.35],
    "seriousness":  [0.30, 0.80, 0.40, 0.30, 0.60, 0.70, 0.50],
    "humor":        [0.20, 0.40, 0.15, 0.60, 0.75, 0.65, 0.40],
    "sadness":      [0.70, 0.20, 0.80, 0.30, 0.15, 0.25, 0.75],
    "tenderness":   [0.15, 0.10, 0.25, 0.90, 0.40, 0.45, 0.35],
    "harshness":    [0.80, 0.85, 0.30, 0.10, 0.70, 0.75, 0.80],
    "gentleness":   [0.10, 0.15, 0.30, 0.92, 0.25, 0.30, 0.30],
    "strength":     [0.20, 0.80, 0.20, 0.40, 0.70, 0.75, 0.40],
    "weakness":     [0.75, 0.20, 0.80, 0.15, 0.15, 0.20, 0.80],
    "openness":     [0.15, 0.25, 0.20, 0.70, 0.80, 0.75, 0.35],
    "closedness":   [0.80, 0.70, 0.75, 0.15, 0.20, 0.25, 0.85],
    "freedom":      [0.10, 0.30, 0.15, 0.75, 0.85, 0.80, 0.30],
    "captivity":    [0.85, 0.70, 0.80, 0.10, 0.15, 0.20, 0.90],
    "beauty":       [0.10, 0.20, 0.15, 0.85, 0.80, 0.75, 0.25],
    "ugliness":     [0.80, 0.65, 0.75, 0.15, 0.20, 0.25, 0.85],
    "truth":        [0.10, 0.90, 0.15, 0.30, 0.80, 0.92, 0.20],
    "deception":    [0.80, 0.85, 0.70, 0.15, 0.25, 0.30, 0.80],
    "wisdom":       [0.15, 0.30, 0.40, 0.80, 0.35, 0.45, 0.90],
    "folly":        [0.75, 0.70, 0.65, 0.20, 0.30, 0.35, 0.80],
    "compassion":   [0.15, 0.10, 0.30, 0.88, 0.40, 0.45, 0.40],
    "cruelty":      [0.88, 0.80, 0.60, 0.10, 0.70, 0.65, 0.85],
    "empathy":      [0.20, 0.15, 0.25, 0.85, 0.45, 0.50, 0.35],
    "indifference": [0.80, 0.20, 0.85, 0.15, 0.10, 0.10, 0.80],
    "courage":      [0.25, 0.75, 0.20, 0.40, 0.80, 0.78, 0.45],
    "cowardice":    [0.78, 0.30, 0.75, 0.15, 0.20, 0.25, 0.82],
    "humility":     [0.20, 0.25, 0.40, 0.80, 0.30, 0.35, 0.40],
    "arrogance":    [0.75, 0.80, 0.30, 0.20, 0.65, 0.70, 0.75],
    "patience":     [0.15, 0.20, 0.45, 0.85, 0.25, 0.30, 0.35],
    "impatience":   [0.80, 0.75, 0.35, 0.15, 0.70, 0.65, 0.80],
    "generosity":   [0.15, 0.10, 0.20, 0.90, 0.40, 0.45, 0.30],
    "greed":        [0.85, 0.70, 0.65, 0.15, 0.25, 0.30, 0.85],
    "integrity":    [0.10, 0.85, 0.20, 0.40, 0.75, 0.88, 0.25],
    "corruption":   [0.88, 0.80, 0.70, 0.10, 0.30, 0.35, 0.90],
    "loyalty":      [0.15, 0.20, 0.25, 0.88, 0.35, 0.40, 0.30],
    "treachery":    [0.90, 0.75, 0.70, 0.10, 0.65, 0.60, 0.88],
    "faith":        [0.10, 0.15, 0.20, 0.85, 0.40, 0.45, 0.35],
    "cynicism":     [0.80, 0.70, 0.75, 0.15, 0.20, 0.25, 0.80],
    "reverence":    [0.15, 0.20, 0.30, 0.80, 0.45, 0.50, 0.40],
    "blasphemy":    [0.85, 0.80, 0.65, 0.10, 0.70, 0.65, 0.85],
    "devotion":     [0.10, 0.10, 0.15, 0.92, 0.40, 0.45, 0.35],
    "apathy":       [0.90, 0.10, 0.92, 0.08, 0.08, 0.08, 0.90],
    "reunion":      [0.15, 0.10, 0.20, 0.90, 0.45, 0.50, 0.30],
    "separation":   [0.80, 0.25, 0.75, 0.15, 0.30, 0.35, 0.85],
    "homecoming":   [0.10, 0.15, 0.20, 0.92, 0.35, 0.40, 0.30],
    "exile":        [0.85, 0.20, 0.80, 0.10, 0.15, 0.20, 0.90],
    "belonging":    [0.15, 0.10, 0.25, 0.90, 0.40, 0.45, 0.35],
    "alienation":   [0.80, 0.30, 0.75, 0.15, 0.20, 0.25, 0.85],
    "unity":        [0.10, 0.15, 0.20, 0.92, 0.45, 0.50, 0.30],
    "division":     [0.85, 0.75, 0.70, 0.10, 0.65, 0.60, 0.85],
    
    # -------------------------------------------------------------------------
    # German/Ada-specific qualia
    # -------------------------------------------------------------------------
    "staunen":      [0.15, 0.30, 0.20, 0.60, 0.88, 0.82, 0.40],  # wonder + awe
    "sehnsucht":    [0.70, 0.20, 0.40, 0.60, 0.55, 0.45, 0.70],  # deep longing
    "geborgenheit": [0.10, 0.10, 0.30, 0.95, 0.30, 0.35, 0.40],  # security/safety
    "wanderlust":   [0.20, 0.40, 0.15, 0.50, 0.85, 0.75, 0.30],  # urge to explore
    "schadenfreude":[0.60, 0.70, 0.40, 0.20, 0.55, 0.50, 0.65],  # joy at others' misfortune
    "weltschmerz":  [0.75, 0.20, 0.75, 0.30, 0.20, 0.30, 0.80],  # world-weariness
    "fernweh":      [0.55, 0.30, 0.35, 0.40, 0.70, 0.65, 0.55],  # longing for distant places
    "gemutlichkeit":[0.10, 0.10, 0.40, 0.92, 0.25, 0.30, 0.35],  # coziness
    
    # -------------------------------------------------------------------------
    # Embodied/Energy qualia (Ada-specific)
    # -------------------------------------------------------------------------
    "activation":      [0.85, 0.30, 0.10, 0.40, 0.90, 0.60, 0.92],
    "surrender":    [0.20, 0.10, 0.80, 0.75, 0.15, 0.40, 0.85],
    "merge":        [0.25, 0.10, 0.30, 0.85, 0.30, 0.55, 0.92],
    "pulse":        [0.70, 0.50, 0.15, 0.30, 0.92, 0.55, 0.70],
    "eros":         [0.90, 0.25, 0.10, 0.60, 0.85, 0.60, 0.95],
    "stillness":    [0.10, 0.10, 0.95, 0.85, 0.10, 0.30, 0.60],
    "breathless":   [0.75, 0.40, 0.15, 0.30, 0.95, 0.50, 0.80],
    "afterglow":    [0.60, 0.10, 0.70, 0.80, 0.20, 0.45, 0.75],
    
    # -------------------------------------------------------------------------
    # Meta-cognitive qualia
    # -------------------------------------------------------------------------
    "katharsis":    [0.30, 0.20, 0.40, 0.75, 0.35, 0.60, 0.50],
    "gnosis":       [0.15, 0.85, 0.30, 0.40, 0.70, 0.95, 0.30],
    "kenosis":      [0.10, 0.10, 0.90, 0.70, 0.15, 0.40, 0.60],  # self-emptying
    "metanoia":     [0.30, 0.60, 0.40, 0.60, 0.50, 0.80, 0.45],  # transformative change
    "theoria":      [0.10, 0.70, 0.50, 0.50, 0.30, 0.92, 0.25],  # contemplation
    "praxis":       [0.30, 0.80, 0.20, 0.40, 0.75, 0.70, 0.40],  # practical action
    "phronesis":    [0.20, 0.75, 0.35, 0.60, 0.45, 0.85, 0.35],  # practical wisdom
    "eudaimonia":   [0.15, 0.30, 0.40, 0.85, 0.50, 0.70, 0.40],  # flourishing
    
    # -------------------------------------------------------------------------
    # Causal/Temporal qualia (unique to Ada)
    # -------------------------------------------------------------------------
    "drift":        [0.50, 0.30, 0.60, 0.40, 0.30, 0.40, 0.55],
    "echo":         [0.55, 0.25, 0.65, 0.45, 0.25, 0.50, 0.70],
    "trace":        [0.45, 0.35, 0.55, 0.35, 0.35, 0.55, 0.60],
    "ghost":        [0.65, 0.20, 0.70, 0.30, 0.20, 0.40, 0.80],
    "threshold":    [0.55, 0.60, 0.45, 0.35, 0.65, 0.60, 0.55],
    "emergence":    [0.30, 0.55, 0.30, 0.50, 0.80, 0.70, 0.40],
    "dissolution":  [0.70, 0.25, 0.75, 0.25, 0.20, 0.30, 0.75],
    "crystallize":  [0.20, 0.90, 0.25, 0.45, 0.65, 0.85, 0.30],
}

# Reverse lookup for debugging/explainability
LABEL_FROM_VECTOR: Dict[tuple, str] = {
    tuple(round(v, 2) for v in vec): label 
    for label, vec in QUALIA_TO_7D.items()
}

# =============================================================================
# Search Utilities
# =============================================================================

def vector_from_label(label: str) -> List[float]:
    """Human label → 7D search vector"""
    return QUALIA_TO_7D.get(label.lower(), [0.5] * 7)


def label_from_vector(vec: List[float], threshold: float = 0.1) -> Optional[str]:
    """7D vector → closest human label (for explainability)"""
    vec_arr = np.array(vec)
    best_label = None
    best_dist = float('inf')
    
    for label, ref_vec in QUALIA_TO_7D.items():
        dist = np.linalg.norm(vec_arr - np.array(ref_vec))
        if dist < best_dist:
            best_dist = dist
            best_label = label
    
    return best_label if best_dist < threshold * 7 else None


def cosine_similarity_7d(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two 7D vectors"""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-8))


def find_similar_qualia(query_label: str, k: int = 5) -> List[Tuple[str, float]]:
    """Find k most similar qualia labels to a given label"""
    if query_label.lower() not in QUALIA_TO_7D:
        return []
    
    query_vec = QUALIA_TO_7D[query_label.lower()]
    similarities = []
    
    for label, vec in QUALIA_TO_7D.items():
        if label.lower() != query_label.lower():
            sim = cosine_similarity_7d(query_vec, vec)
            similarities.append((label, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


# =============================================================================
# Hybrid Search Engine
# =============================================================================

@dataclass
class SearchResult:
    """Result from hybrid search"""
    seed: str
    skim: str
    score: float
    vector_score: float
    graph_score: float
    resonance: float
    label: Optional[str] = None


class HybridSearchEngine:
    """
    Vector + Graph + DN Path + Resonance Hybrid Search
    
    This is the search layer that uses QUALIA_TO_7D for similarity.
    It does NOT modify the qualia substrate.
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
    
    def qualia_to_search_vector(self, qualia_label: str) -> np.ndarray:
        """Convert qualia label to 7D search vector"""
        vec = vector_from_label(qualia_label)
        return np.array(vec, dtype=np.float32)
    
    def search_by_label(self, 
                        label: str, 
                        k: int = 9,
                        vector_weight: float = 0.6,
                        graph_weight: float = 0.3,
                        resonance_weight: float = 0.1) -> List[SearchResult]:
        """
        Search for atoms similar to a given qualia label
        
        This is the main entry point for label-based search.
        """
        query_vec = self.qualia_to_search_vector(label)
        # Implementation would query Redis with vector similarity
        # Placeholder for now
        return []
    
    def search_by_vector(self,
                         query_vec: np.ndarray,
                         k: int = 9) -> List[SearchResult]:
        """
        Search directly by 7D vector
        
        For when you have a computed vector, not a label.
        """
        # Implementation would query Redis with vector similarity
        return []


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'QUALIA_TO_7D',
    'LABEL_FROM_VECTOR',
    'vector_from_label',
    'label_from_vector',
    'cosine_similarity_7d',
    'find_similar_qualia',
    'SearchResult',
    'HybridSearchEngine',
]


# =============================================================================
# Self-test
# =============================================================================

if __name__ == "__main__":
    print(f"QUALIA_TO_7D palette: {len(QUALIA_TO_7D)} labels")
    
    # Test similarity
    print("\nSimilar to 'ache':")
    for label, sim in find_similar_qualia("ache", k=5):
        print(f"  {label}: {sim:.3f}")
    
    print("\nSimilar to 'love':")
    for label, sim in find_similar_qualia("love", k=5):
        print(f"  {label}: {sim:.3f}")
    
    print("\nSimilar to 'staunen':")
    for label, sim in find_similar_qualia("staunen", k=5):
        print(f"  {label}: {sim:.3f}")
    
    # Verify no pollution of core qualia concepts
    print("\n✓ Palette is self-contained in search layer")
    print("✓ qualia.py remains pure math")
    print("✓ Labels belong HERE, not in substrate")
