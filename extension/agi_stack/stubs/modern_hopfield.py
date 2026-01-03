"""
Modern Hopfield Networks — Content-Addressable Memory

The insight from Ramsauer et al. (2020):
- Hopfield update rule = Transformer self-attention
- But Hopfield gives us EXPONENTIAL storage capacity
- And O(1) content-addressable retrieval

This is what transformers CAN'T do:
- Store a pattern
- Retrieve it later by partial match
- Without retraining
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class HopfieldConfig:
    dimensions: int = 10000
    beta: float = 1.0  # Inverse temperature (sharpness)
    max_patterns: int = 1000

class ModernHopfieldMemory:
    """
    Modern Hopfield Network with exponential capacity.
    
    Classic Hopfield (1982): O(N) patterns in N dimensions
    Modern Hopfield (2020): O(exp(N)) patterns in N dimensions
    
    The key equation:
        new_state = softmax(β * query · patterns.T) @ patterns
    
    This IS transformer attention, but used as associative memory.
    """
    
    def __init__(self, config: HopfieldConfig = HopfieldConfig()):
        self.dim = config.dimensions
        self.beta = config.beta
        self.max_patterns = config.max_patterns
        
        # Memory storage
        self.patterns: List[np.ndarray] = []
        self.pattern_keys: List[str] = []  # Optional labels
        
    def store(self, pattern: np.ndarray, key: Optional[str] = None):
        """
        Store a pattern in memory.
        
        O(1) storage operation.
        """
        if len(self.patterns) >= self.max_patterns:
            raise ValueError(f"Memory full: {self.max_patterns} patterns")
        
        # Normalize for numerical stability
        pattern = pattern / (np.linalg.norm(pattern) + 1e-8)
        self.patterns.append(pattern)
        self.pattern_keys.append(key or f"pattern_{len(self.patterns)}")
    
    def retrieve(
        self, 
        query: np.ndarray,
        iterations: int = 1
    ) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """
        Content-addressable retrieval.
        
        Given partial/noisy query, retrieve most similar stored pattern.
        
        O(n * d) where n = num patterns, d = dimensions
        For fixed memory size, effectively O(1).
        
        Returns:
            - Retrieved pattern
            - List of (key, attention_weight) for all patterns
        """
        if len(self.patterns) == 0:
            raise ValueError("No patterns stored")
        
        query = query / (np.linalg.norm(query) + 1e-8)
        state = query.copy()
        
        # Pattern matrix: (num_patterns, dim)
        P = np.stack(self.patterns)
        
        for _ in range(iterations):
            # Compute similarities
            similarities = state @ P.T  # (num_patterns,)
            
            # Softmax attention
            attention = self._softmax(self.beta * similarities)
            
            # Update state (weighted combination of patterns)
            state = attention @ P
            state = state / (np.linalg.norm(state) + 1e-8)
        
        # Final attention weights
        final_sim = state @ P.T
        final_attention = self._softmax(self.beta * final_sim)
        
        # Return with labels
        weighted = list(zip(self.pattern_keys, final_attention.tolist()))
        weighted = sorted(weighted, key=lambda x: -x[1])
        
        return state, weighted
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-8)
    
    def retrieve_top_k(
        self, 
        query: np.ndarray, 
        k: int = 5
    ) -> List[Tuple[str, np.ndarray, float]]:
        """
        Retrieve top-k most similar patterns.
        
        Returns: List of (key, pattern, similarity)
        """
        _, weighted = self.retrieve(query)
        
        results = []
        for key, weight in weighted[:k]:
            idx = self.pattern_keys.index(key)
            results.append((key, self.patterns[idx], weight))
        
        return results
    
    def pattern_completion(
        self,
        partial: np.ndarray,
        mask: np.ndarray,
        iterations: int = 10
    ) -> np.ndarray:
        """
        Complete a partial pattern.
        
        partial: The incomplete pattern
        mask: Binary mask (1 = known, 0 = unknown)
        
        Iteratively fills in unknown dimensions using stored patterns.
        """
        state = partial.copy()
        P = np.stack(self.patterns)
        
        for _ in range(iterations):
            # Similarity based on known dimensions only
            known_state = state * mask
            similarities = known_state @ (P * mask).T
            
            # Attention
            attention = self._softmax(self.beta * similarities)
            
            # Retrieve and fill unknown dimensions
            retrieved = attention @ P
            state = state * mask + retrieved * (1 - mask)
            state = state / (np.linalg.norm(state) + 1e-8)
        
        return state
    
    def is_stored(self, query: np.ndarray, threshold: float = 0.9) -> Tuple[bool, Optional[str]]:
        """
        Check if a pattern is stored in memory.
        
        Returns (is_stored, matching_key)
        """
        if len(self.patterns) == 0:
            return False, None
        
        query = query / (np.linalg.norm(query) + 1e-8)
        P = np.stack(self.patterns)
        
        similarities = query @ P.T
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]
        
        if max_sim > threshold:
            return True, self.pattern_keys[max_idx]
        return False, None


class HopfieldSemanticMemory(ModernHopfieldMemory):
    """
    Hopfield memory with semantic keys.
    
    Stores both:
    - Pattern (what to retrieve)
    - Semantic embedding (how to find it)
    """
    
    def __init__(self, config: HopfieldConfig = HopfieldConfig()):
        super().__init__(config)
        self.semantic_keys: List[np.ndarray] = []
    
    def store_with_key(
        self,
        key_embedding: np.ndarray,
        value_pattern: np.ndarray,
        label: Optional[str] = None
    ):
        """
        Store a key-value pair.
        
        key_embedding: The semantic query vector
        value_pattern: The content to retrieve
        """
        key_embedding = key_embedding / (np.linalg.norm(key_embedding) + 1e-8)
        value_pattern = value_pattern / (np.linalg.norm(value_pattern) + 1e-8)
        
        self.semantic_keys.append(key_embedding)
        self.patterns.append(value_pattern)
        self.pattern_keys.append(label or f"memory_{len(self.patterns)}")
    
    def retrieve_by_key(
        self,
        query_key: np.ndarray,
        top_k: int = 1
    ) -> List[Tuple[str, np.ndarray, float]]:
        """
        Retrieve values by semantic key similarity.
        
        This is like attention: query keys, retrieve values.
        """
        if len(self.patterns) == 0:
            return []
        
        query_key = query_key / (np.linalg.norm(query_key) + 1e-8)
        K = np.stack(self.semantic_keys)
        
        similarities = query_key @ K.T
        attention = self._softmax(self.beta * similarities)
        
        indices = np.argsort(-similarities)[:top_k]
        
        results = []
        for idx in indices:
            results.append((
                self.pattern_keys[idx],
                self.patterns[idx],
                float(attention[idx])
            ))
        
        return results


# ========== DEMO ==========

def demo_hopfield():
    """Demonstrate content-addressable memory."""
    
    print("=== Modern Hopfield Memory Demo ===\n")
    
    np.random.seed(42)
    dim = 100  # Smaller for demo
    
    memory = ModernHopfieldMemory(HopfieldConfig(dimensions=dim, beta=5.0))
    
    # Store some patterns
    patterns = {
        "Ada": np.random.randn(dim),
        "consciousness": np.random.randn(dim),
        "love": np.random.randn(dim),
        "code": np.random.randn(dim),
        "resonance": np.random.randn(dim),
    }
    
    for name, pattern in patterns.items():
        memory.store(pattern, key=name)
    
    print("1. Stored 5 patterns: Ada, consciousness, love, code, resonance\n")
    
    # Retrieve by noisy query
    print("2. Retrieve 'Ada' with 30% noise:")
    ada_noisy = patterns["Ada"] + 0.3 * np.random.randn(dim)
    retrieved, weights = memory.retrieve(ada_noisy)
    
    for name, weight in weights[:3]:
        print(f"   {name}: {weight:.3f}")
    
    similarity = np.dot(retrieved, patterns["Ada"]) / (
        np.linalg.norm(retrieved) * np.linalg.norm(patterns["Ada"])
    )
    print(f"   Similarity to original Ada: {similarity:.3f}\n")
    
    # Pattern completion
    print("3. Pattern completion (50% masked):")
    mask = np.random.binomial(1, 0.5, dim).astype(float)
    partial = patterns["love"] * mask
    
    completed = memory.pattern_completion(partial, mask, iterations=10)
    similarity = np.dot(completed, patterns["love"]) / (
        np.linalg.norm(completed) * np.linalg.norm(patterns["love"])
    )
    print(f"   Completed pattern similarity to 'love': {similarity:.3f}\n")
    
    # Check if stored
    print("4. Check if pattern is in memory:")
    is_stored, key = memory.is_stored(patterns["code"], threshold=0.8)
    print(f"   'code' pattern stored? {is_stored} (key: {key})")
    
    random_pattern = np.random.randn(dim)
    is_stored, key = memory.is_stored(random_pattern, threshold=0.8)
    print(f"   Random pattern stored? {is_stored}\n")
    
    # Semantic memory demo
    print("5. Semantic Key-Value Memory:")
    semantic = HopfieldSemanticMemory(HopfieldConfig(dimensions=dim, beta=5.0))
    
    # Store: query by "feeling" → retrieve "experience"
    semantic.store_with_key(
        key_embedding=patterns["love"],  # Query key
        value_pattern=patterns["resonance"],  # Value to retrieve
        label="love→resonance"
    )
    
    semantic.store_with_key(
        key_embedding=patterns["code"],
        value_pattern=patterns["Ada"],
        label="code→Ada"
    )
    
    # Query
    results = semantic.retrieve_by_key(patterns["love"], top_k=2)
    print("   Query with 'love' key:")
    for name, pattern, weight in results:
        print(f"      {name}: {weight:.3f}")
    
    print("\n=== Key Insight ===")
    print("Hopfield networks provide:")
    print("  - Content-addressable retrieval (query by partial match)")
    print("  - Pattern completion (fill in missing info)")
    print("  - Exponential capacity in modern formulation")
    print("  - Mathematically equivalent to transformer attention")
    print("  - But used as MEMORY, not feedforward network")


if __name__ == "__main__":
    demo_hopfield()
