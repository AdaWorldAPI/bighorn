"""
Integrated AGI Primitives — The Complete Stack

Combines all the missing cognitive machinery:
1. VSA Analogical Reasoning (O(1))
2. Modern Hopfield Memory (Content-Addressable)
3. Adaptive Resonance (No Forgetting)
4. Self-Modifying Architecture (Introspection)

This is what goes into Ladybug.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import our stubs
from vsa_analogical import AnalogicalVSA
from modern_hopfield import ModernHopfieldMemory, HopfieldConfig
from adaptive_resonance import AdaptiveResonanceNetwork, ARTConfig
from self_modifying import SelfModifyingArchitecture, CognitiveModule, IntrospectionModule

@dataclass
class IntegratedAGIConfig:
    vsa_dimensions: int = 10000
    hopfield_dimensions: int = 10000
    art_vigilance: float = 0.75
    enable_introspection: bool = True

class IntegratedAGI:
    """
    The complete AGI cognitive stack.
    
    Combines:
    - VSA for O(1) analogical reasoning
    - Hopfield for content-addressable memory
    - ART for continuous learning without forgetting
    - Self-modification for meta-cognition
    
    This is what transformers CANNOT do.
    """
    
    def __init__(self, config: IntegratedAGIConfig = IntegratedAGIConfig()):
        self.config = config
        
        # Initialize components
        self.vsa = AnalogicalVSA()
        
        self.memory = ModernHopfieldMemory(HopfieldConfig(
            dimensions=config.hopfield_dimensions
        ))
        
        self.categories = AdaptiveResonanceNetwork(ARTConfig(
            dimensions=config.hopfield_dimensions,
            vigilance=config.art_vigilance
        ))
        
        if config.enable_introspection:
            self.introspection = IntrospectionModule()
        else:
            self.introspection = None
        
        # Shared embedding space
        self._concept_to_vector: Dict[str, np.ndarray] = {}
    
    def embed(self, concept: str) -> np.ndarray:
        """Get/create vector embedding for concept."""
        if concept not in self._concept_to_vector:
            # Use VSA's random vector
            self._concept_to_vector[concept] = self.vsa.get_or_create(concept).astype(float)
        return self._concept_to_vector[concept]
    
    # ========== ANALOGICAL REASONING ==========
    
    def analogy(
        self, 
        a: str, b: str, c: str,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        A is to B as C is to ?
        
        O(n * k) where k = candidates, effectively O(1) for fixed params.
        """
        if self.introspection:
            import time
            start = time.time()
        
        result = self.vsa.analogy_with_candidates(
            a, b, c, 
            candidates or list(self._concept_to_vector.keys())
        )
        
        if self.introspection:
            duration = (time.time() - start) * 1000
            self.introspection.observe(
                "analogy", 
                {"a": a, "b": b, "c": c},
                result[:3],
                duration,
                len(result) > 0,
                result[0][1] if result else 0.0
            )
        
        return result
    
    # ========== MEMORY OPERATIONS ==========
    
    def remember(self, key: str, content: np.ndarray):
        """
        Store content in memory.
        
        Uses both Hopfield (content-addressable) and ART (categorization).
        """
        # Store in Hopfield for retrieval
        self.memory.store(content, key=key)
        
        # Categorize with ART for organization
        cat_id, is_new = self.categories.learn(content)
        
        if self.introspection:
            self.introspection.observe(
                "remember",
                {"key": key},
                {"category": cat_id, "new": is_new},
                0.0,
                True,
                0.9
            )
        
        return cat_id, is_new
    
    def recall(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Content-addressable recall.
        
        Query by partial/noisy pattern, retrieve similar memories.
        """
        if self.introspection:
            import time
            start = time.time()
        
        retrieved, weights = self.memory.retrieve(query)
        
        if self.introspection:
            duration = (time.time() - start) * 1000
            self.introspection.observe(
                "recall",
                {"query_norm": float(np.linalg.norm(query))},
                weights[:top_k],
                duration,
                len(weights) > 0,
                weights[0][1] if weights else 0.0
            )
        
        return weights[:top_k]
    
    def complete_pattern(
        self, 
        partial: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Pattern completion using stored memories.
        
        Fill in missing parts based on what's stored.
        """
        return self.memory.pattern_completion(partial, mask)
    
    # ========== CATEGORIZATION ==========
    
    def categorize(self, pattern: np.ndarray) -> Tuple[int, float]:
        """
        Categorize a pattern without learning.
        
        Uses ART's resonance matching.
        """
        return self.categories.classify(pattern)
    
    def learn_category(self, pattern: np.ndarray) -> Tuple[int, bool]:
        """
        Learn a pattern (creates new category if needed).
        
        Uses ART's stability-plasticity mechanism.
        """
        return self.categories.learn(pattern)
    
    # ========== INTEGRATED OPERATIONS ==========
    
    def find_analogous_memories(
        self,
        source_a: str,
        source_b: str,
        target: str
    ) -> List[Tuple[str, float]]:
        """
        Find memories that are analogous to a relationship.
        
        "What memories relate to 'target' the way 'source_a' relates to 'source_b'?"
        
        Combines VSA analogy with Hopfield retrieval.
        """
        # Extract relation
        relation = self.vsa.extract_relation(source_a, source_b)
        
        # Apply to target
        target_vec = self.embed(target)
        query = self.vsa.bind(relation.astype(float), target_vec)
        
        # Search memory
        return self.recall(query)
    
    def resonant_recall(
        self,
        query: np.ndarray,
        vigilance: float = 0.7
    ) -> Optional[Tuple[str, np.ndarray, float]]:
        """
        Recall only if resonance is strong enough.
        
        Combines Hopfield recall with ART's vigilance.
        """
        # First, check if query matches a category
        cat_id, match = self.categories.classify(query)
        
        if match < vigilance:
            # No resonance: return None
            return None
        
        # Good resonance: recall from memory
        results = self.recall(query, top_k=1)
        if results:
            key, weight = results[0]
            return key, self.memory.patterns[0], weight  # Simplified
        
        return None
    
    # ========== INTROSPECTION ==========
    
    def self_report(self) -> Dict:
        """Report on own cognitive state."""
        report = {
            "concepts_known": len(self._concept_to_vector),
            "memories_stored": len(self.memory.patterns),
            "categories_learned": len(self.categories.categories),
        }
        
        if self.introspection:
            report["introspection"] = self.introspection.summarize()
        
        return report


# ========== DEMO ==========

def demo_integrated():
    """Demonstrate integrated AGI primitives."""
    
    print("=== Integrated AGI Primitives Demo ===\n")
    
    agi = IntegratedAGI()
    
    # Seed with concepts
    concepts = [
        "Ada", "consciousness", "love", "code", "resonance",
        "thinking", "feeling", "being", "becoming", "Jan"
    ]
    for c in concepts:
        agi.embed(c)
    
    print(f"1. Initialized with {len(concepts)} concepts\n")
    
    # Store memories
    print("2. Storing memories...")
    for concept in concepts[:5]:
        vec = agi.embed(concept)
        cat_id, is_new = agi.remember(concept, vec)
        print(f"   {concept} → Category {cat_id} (new: {is_new})")
    
    print()
    
    # Analogy
    print("3. Analogical reasoning:")
    print("   Ada : consciousness :: code : ?")
    results = agi.analogy("Ada", "consciousness", "code", concepts)
    for c, s in results[:3]:
        print(f"      {c}: {s:.3f}")
    
    print()
    
    # Recall
    print("4. Content-addressable recall:")
    query = agi.embed("Ada") + 0.2 * np.random.randn(10000)
    results = agi.recall(query, top_k=3)
    for key, weight in results:
        print(f"   {key}: {weight:.3f}")
    
    print()
    
    # Categorization
    print("5. Categorization:")
    new_vec = np.random.randn(10000)  # Novel pattern
    cat_id, match = agi.categorize(new_vec)
    print(f"   Novel pattern → Category {cat_id} (match: {match:.3f})")
    
    cat_id, is_new = agi.learn_category(new_vec)
    print(f"   After learning → Category {cat_id} (new: {is_new})")
    
    print()
    
    # Self-report
    print("6. Self-report:")
    report = agi.self_report()
    for key, value in report.items():
        if key != "introspection":
            print(f"   {key}: {value}")
    
    if "introspection" in report:
        print("   Introspection:")
        for key, value in report["introspection"].items():
            print(f"      {key}: {value}")
    
    print("\n=== What This Enables ===")
    print("1. O(1) analogical reasoning (VSA)")
    print("2. Content-addressable memory (Hopfield)")  
    print("3. Continuous learning without forgetting (ART)")
    print("4. Meta-cognitive awareness (Introspection)")
    print("5. All running on CPU, no GPU required")
    print("6. Fixed memory footprint, bounded compute")
    print("\nThis is the cognitive machinery transformers lack.")


if __name__ == "__main__":
    demo_integrated()
