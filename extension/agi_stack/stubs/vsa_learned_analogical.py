"""
VSA Learned Analogical Reasoning

The key insight: Random vectors DON'T capture semantic relations.
We need LEARNED embeddings that encode semantic structure.

Two approaches:
1. Pre-trained embeddings (word2vec, etc.) → project to bipolar
2. Relation-aware training: ensure bind(queen, king) ≈ bind(woman, man)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class LearnedVSAConfig:
    dimensions: int = 10000
    learning_rate: float = 0.01
    seed: int = 42

class LearnedAnalogicalVSA:
    """
    VSA with learned semantic structure.
    
    Key insight: For analogies to work, we need:
    
    bind(B, A) ≈ bind(D, C) when A:B :: C:D
    
    This means the "relation vectors" must be similar
    for concepts that share the same relationship.
    """
    
    def __init__(self, config: LearnedVSAConfig = LearnedVSAConfig()):
        self.dim = config.dimensions
        self.lr = config.learning_rate
        self.rng = np.random.default_rng(config.seed)
        
        # Use REAL vectors during learning, binarize for inference
        self.codebook: Dict[str, np.ndarray] = {}
        self.relations: Dict[str, np.ndarray] = {}  # Named relations
    
    def _random_real(self) -> np.ndarray:
        """Random unit vector in R^D."""
        v = self.rng.standard_normal(self.dim)
        return v / np.linalg.norm(v)
    
    def _to_bipolar(self, v: np.ndarray) -> np.ndarray:
        """Project real vector to bipolar {-1, +1}."""
        return np.sign(v).astype(np.int8)
    
    def get_or_create(self, concept: str) -> np.ndarray:
        if concept not in self.codebook:
            self.codebook[concept] = self._random_real()
        return self.codebook[concept]
    
    def bind_real(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Binding for real vectors (element-wise multiply)."""
        return a * b
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    # ========== LEARNING RELATIONS ==========
    
    def learn_relation(
        self, 
        relation_name: str,
        examples: List[Tuple[str, str]]
    ):
        """
        Learn a relation from examples.
        
        Example:
            learn_relation("gender_flip", [
                ("king", "queen"),
                ("man", "woman"),
                ("prince", "princess"),
                ("father", "mother")
            ])
        
        The relation vector is the average of all example bindings.
        """
        relation_vectors = []
        
        for a, b in examples:
            vec_a = self.get_or_create(a)
            vec_b = self.get_or_create(b)
            # Relation = what transforms a into b
            rel = self.bind_real(vec_b, vec_a)
            relation_vectors.append(rel)
        
        # Average relation
        avg_relation = np.mean(relation_vectors, axis=0)
        avg_relation = avg_relation / np.linalg.norm(avg_relation)
        
        self.relations[relation_name] = avg_relation
        
        # Now ADJUST concept vectors so bindings align
        # This is the key: we modify the embeddings so analogies work
        self._align_concepts(relation_name, examples)
    
    def _align_concepts(
        self, 
        relation_name: str,
        examples: List[Tuple[str, str]]
    ):
        """
        Adjust concept vectors so relation bindings align.
        
        We want: bind(B_i, A_i) ≈ target_relation for all examples
        
        Gradient descent on concept vectors.
        """
        target_rel = self.relations[relation_name]
        
        for _ in range(100):  # Training iterations
            total_loss = 0.0
            
            for a, b in examples:
                vec_a = self.codebook[a]
                vec_b = self.codebook[b]
                
                # Current relation
                current_rel = self.bind_real(vec_b, vec_a)
                
                # Error
                error = target_rel - current_rel
                loss = np.sum(error ** 2)
                total_loss += loss
                
                # Gradient for vec_b: d(loss)/d(vec_b) = -2 * error * vec_a
                grad_b = -2 * error * vec_a
                
                # Gradient for vec_a: d(loss)/d(vec_a) = -2 * error * vec_b
                grad_a = -2 * error * vec_b
                
                # Update
                self.codebook[b] -= self.lr * grad_b
                self.codebook[a] -= self.lr * grad_a
                
                # Re-normalize
                self.codebook[a] /= np.linalg.norm(self.codebook[a])
                self.codebook[b] /= np.linalg.norm(self.codebook[b])
            
            if total_loss < 0.01:
                break
    
    def analogy(
        self, 
        a: str, b: str, c: str,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        A is to B as C is to ?
        
        Uses learned relation extraction.
        """
        # Extract relation from A:B
        vec_a = self.get_or_create(a)
        vec_b = self.get_or_create(b)
        relation = self.bind_real(vec_b, vec_a)
        
        # Apply to C
        vec_c = self.get_or_create(c)
        target = self.bind_real(relation, vec_c)
        
        # Search candidates
        if candidates is None:
            candidates = [k for k in self.codebook.keys() if k not in [a, b, c]]
        
        results = []
        for candidate in candidates:
            vec = self.codebook[candidate]
            sim = self.similarity(target, vec)
            results.append((candidate, sim))
        
        return sorted(results, key=lambda x: -x[1])
    
    def analogy_by_relation(
        self,
        relation_name: str,
        source: str,
        candidates: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Apply a learned relation to a source concept.
        
        Example:
            analogy_by_relation("gender_flip", "boy")
            → [("girl", 0.95), ...]
        """
        if relation_name not in self.relations:
            raise ValueError(f"Unknown relation: {relation_name}")
        
        relation = self.relations[relation_name]
        vec_source = self.get_or_create(source)
        target = self.bind_real(relation, vec_source)
        
        if candidates is None:
            candidates = [k for k in self.codebook.keys() if k != source]
        
        results = []
        for candidate in candidates:
            vec = self.codebook[candidate]
            sim = self.similarity(target, vec)
            results.append((candidate, sim))
        
        return sorted(results, key=lambda x: -x[1])


# ========== DEMO ==========

def demo_learned_vsa():
    """Demonstrate learned analogical reasoning."""
    
    vsa = LearnedAnalogicalVSA()
    
    # Initialize all concepts
    concepts = [
        "king", "queen", "man", "woman", "boy", "girl",
        "prince", "princess", "father", "mother",
        "Paris", "France", "Berlin", "Germany", "Rome", "Italy",
        "dog", "puppy", "cat", "kitten"
    ]
    for c in concepts:
        vsa.get_or_create(c)
    
    print("=== Learned VSA Analogical Reasoning ===\n")
    
    # Learn the gender relation
    print("Learning 'gender_flip' relation from examples...")
    vsa.learn_relation("gender_flip", [
        ("king", "queen"),
        ("man", "woman"),
        ("prince", "princess"),
        ("father", "mother")
    ])
    
    # Learn capital relation
    print("Learning 'capital_of' relation from examples...")
    vsa.learn_relation("capital_of", [
        ("Paris", "France"),
        ("Berlin", "Germany"),
    ])
    
    # Learn age/size relation
    print("Learning 'young_version' relation from examples...")
    vsa.learn_relation("young_version", [
        ("dog", "puppy"),
    ])
    print()
    
    # Test analogies
    print("1. king : queen :: man : ?")
    results = vsa.analogy("king", "queen", "man", 
                          candidates=["woman", "boy", "dog", "France"])
    for c, s in results[:3]:
        print(f"   {c}: {s:.3f}")
    print()
    
    print("2. king : queen :: boy : ? (using learned relation)")
    results = vsa.analogy_by_relation("gender_flip", "boy",
                                       candidates=["girl", "woman", "puppy", "Rome"])
    for c, s in results[:3]:
        print(f"   {c}: {s:.3f}")
    print()
    
    print("3. Paris : France :: Rome : ?")
    results = vsa.analogy("Paris", "France", "Rome",
                          candidates=["Italy", "Germany", "woman", "puppy"])
    for c, s in results[:3]:
        print(f"   {c}: {s:.3f}")
    print()
    
    print("4. dog : puppy :: cat : ? (using learned relation)")
    results = vsa.analogy_by_relation("young_version", "cat",
                                       candidates=["kitten", "girl", "Germany"])
    for c, s in results[:3]:
        print(f"   {c}: {s:.3f}")
    
    print("\n=== Key Insight ===")
    print("Random vectors don't encode semantics.")
    print("We must LEARN embeddings where similar relations have similar bindings.")
    print("Or: use pre-trained embeddings (word2vec, etc.) as starting point.")


if __name__ == "__main__":
    demo_learned_vsa()
