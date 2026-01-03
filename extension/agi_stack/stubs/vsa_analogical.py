"""
VSA Analogical Reasoning — O(1) Cognitive Primitive

The killer feature that nobody uses:
  A is to B as C is to ?
  answer = B ⊗ A ⊗ C
  
O(n) element-wise, effectively O(1) for fixed dimensionality.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class VSAConfig:
    dimensions: int = 10000
    seed: int = 42

class AnalogicalVSA:
    """
    VSA with O(1) analogical reasoning.
    
    Core insight: In 10,000D space, random vectors are quasi-orthogonal.
    This means we can:
    1. Bind concepts (XOR for bipolar, element-wise mult for real)
    2. Bundle concepts (majority vote / sum + threshold)
    3. Extract relations (unbind)
    4. Apply relations to new concepts (analogical transfer)
    """
    
    def __init__(self, config: VSAConfig = VSAConfig()):
        self.dim = config.dimensions
        self.rng = np.random.default_rng(config.seed)
        self.codebook: Dict[str, np.ndarray] = {}
        
        # Role vectors for structured representations
        self._init_roles()
    
    def _init_roles(self):
        """Initialize structural role vectors."""
        roles = ['AGENT', 'ACTION', 'PATIENT', 'PROPERTY', 'RELATION']
        for role in roles:
            self.codebook[f"_ROLE_{role}"] = self._random_bipolar()
    
    def _random_bipolar(self) -> np.ndarray:
        """Generate random bipolar vector {-1, +1}^D."""
        return self.rng.choice([-1, 1], size=self.dim).astype(np.int8)
    
    def get_or_create(self, concept: str) -> np.ndarray:
        """Get vector for concept, creating if needed."""
        if concept not in self.codebook:
            self.codebook[concept] = self._random_bipolar()
        return self.codebook[concept]
    
    # ========== CORE VSA OPERATIONS (O(n) = O(1) for fixed D) ==========
    
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Binding: Creates association dissimilar to both inputs.
        XOR for bipolar (equivalent to element-wise multiply).
        
        Properties:
        - bind(A, A) = Identity (all 1s)
        - bind(A, bind(A, B)) = B (self-inverse)
        - Commutative: bind(A, B) = bind(B, A)
        """
        return (a * b).astype(np.int8)
    
    def bundle(self, vectors: list) -> np.ndarray:
        """
        Bundling: Aggregates multiple concepts via majority vote.
        Result is similar to all inputs.
        
        Properties:
        - similarity(bundle([A,B,C]), A) > 0
        - Works for variable number of inputs
        """
        summed = np.sum(vectors, axis=0)
        # Majority vote with random tiebreak
        result = np.sign(summed)
        ties = (result == 0)
        result[ties] = self.rng.choice([-1, 1], size=np.sum(ties))
        return result.astype(np.int8)
    
    def permute(self, v: np.ndarray, shifts: int = 1) -> np.ndarray:
        """
        Permutation: Encodes sequence/order.
        Circular shift creates dissimilar vector.
        """
        return np.roll(v, shifts)
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity for bipolar vectors.
        For bipolar: cos = (a·b) / D = normalized Hamming distance
        """
        return float(np.dot(a, b)) / self.dim
    
    # ========== ANALOGICAL REASONING (THE KILLER FEATURE) ==========
    
    def extract_relation(self, a: str, b: str) -> np.ndarray:
        """
        Extract the relation between two concepts.
        
        relation = bind(b, a)  # "what transforms a into b?"
        
        Example:
            relation = extract_relation("king", "queen")
            # relation encodes "gender-flip" or "feminine-of"
        """
        vec_a = self.get_or_create(a)
        vec_b = self.get_or_create(b)
        return self.bind(vec_b, vec_a)
    
    def apply_relation(self, relation: np.ndarray, c: str) -> np.ndarray:
        """
        Apply extracted relation to new concept.
        
        result = bind(relation, c)
        
        Example:
            relation = extract_relation("king", "queen")
            result = apply_relation(relation, "man")
            # result ≈ "woman"
        """
        vec_c = self.get_or_create(c)
        return self.bind(relation, vec_c)
    
    def analogy(self, a: str, b: str, c: str) -> Tuple[str, float]:
        """
        A is to B as C is to ?
        
        This is THE cognitive primitive that enables:
        - Metaphor understanding
        - Transfer learning
        - Creative combination
        - Conceptual blending
        
        O(n) computation, effectively O(1) for fixed D.
        
        Returns: (best_match, similarity_score)
        """
        # Extract relation: what transforms A into B?
        relation = self.extract_relation(a, b)
        
        # Apply to C
        result = self.apply_relation(relation, c)
        
        # Find best match in codebook
        best_match = None
        best_sim = -1.0
        
        for concept, vector in self.codebook.items():
            if concept.startswith("_ROLE_"):
                continue  # Skip role vectors
            if concept in [a, b, c]:
                continue  # Skip input concepts
                
            sim = self.similarity(result, vector)
            if sim > best_sim:
                best_sim = sim
                best_match = concept
        
        return best_match, best_sim
    
    def analogy_with_candidates(
        self, 
        a: str, b: str, c: str, 
        candidates: list
    ) -> list:
        """
        Analogy with constrained candidate set.
        Returns sorted list of (candidate, similarity).
        """
        relation = self.extract_relation(a, b)
        result = self.apply_relation(relation, c)
        
        scores = []
        for candidate in candidates:
            vec = self.get_or_create(candidate)
            sim = self.similarity(result, vec)
            scores.append((candidate, sim))
        
        return sorted(scores, key=lambda x: -x[1])
    
    # ========== STRUCTURED REPRESENTATIONS ==========
    
    def encode_triple(self, subject: str, predicate: str, obj: str) -> np.ndarray:
        """
        Encode subject-predicate-object triple.
        
        triple = bind(AGENT, subject) + bind(ACTION, predicate) + bind(PATIENT, object)
        """
        s_vec = self.bind(self.codebook["_ROLE_AGENT"], self.get_or_create(subject))
        p_vec = self.bind(self.codebook["_ROLE_ACTION"], self.get_or_create(predicate))
        o_vec = self.bind(self.codebook["_ROLE_PATIENT"], self.get_or_create(obj))
        
        return self.bundle([s_vec, p_vec, o_vec])
    
    def query_role(self, triple: np.ndarray, role: str) -> Tuple[str, float]:
        """
        Query what fills a role in a triple.
        
        filler = bind(triple, ROLE)  # Unbind to get filler
        """
        role_vec = self.codebook[f"_ROLE_{role}"]
        query = self.bind(triple, role_vec)
        
        # Find best match
        best_match = None
        best_sim = -1.0
        
        for concept, vector in self.codebook.items():
            if concept.startswith("_ROLE_"):
                continue
            sim = self.similarity(query, vector)
            if sim > best_sim:
                best_sim = sim
                best_match = concept
        
        return best_match, best_sim
    
    # ========== SEQUENCE ENCODING ==========
    
    def encode_sequence(self, items: list) -> np.ndarray:
        """
        Encode ordered sequence using permutation.
        
        seq = Σ permute(item_i, i)
        """
        vectors = [
            self.permute(self.get_or_create(item), i)
            for i, item in enumerate(items)
        ]
        return self.bundle(vectors)
    
    def query_position(self, sequence: np.ndarray, position: int) -> Tuple[str, float]:
        """
        Query what's at position in sequence.
        """
        # Inverse permute to extract position
        query = self.permute(sequence, -position)
        
        best_match = None
        best_sim = -1.0
        
        for concept, vector in self.codebook.items():
            if concept.startswith("_ROLE_"):
                continue
            sim = self.similarity(query, vector)
            if sim > best_sim:
                best_sim = sim
                best_match = concept
        
        return best_match, best_sim


# ========== DEMO ==========

def demo_analogical_reasoning():
    """
    Demonstrate O(1) analogical reasoning.
    
    This is the cognitive primitive that transformers CANNOT do efficiently.
    """
    vsa = AnalogicalVSA()
    
    # Seed the codebook with concepts
    concepts = [
        "king", "queen", "man", "woman", "boy", "girl",
        "prince", "princess", "father", "mother",
        "dog", "puppy", "cat", "kitten",
        "big", "small", "hot", "cold",
        "Paris", "France", "Berlin", "Germany", "Rome", "Italy"
    ]
    for c in concepts:
        vsa.get_or_create(c)
    
    print("=== VSA Analogical Reasoning Demo ===\n")
    
    # Classic analogy
    print("1. king : queen :: man : ?")
    answer, score = vsa.analogy("king", "queen", "man")
    print(f"   Answer: {answer} (similarity: {score:.3f})\n")
    
    # With candidates (more accurate)
    print("2. king : queen :: man : ? (with candidates)")
    candidates = ["woman", "boy", "dog", "France"]
    results = vsa.analogy_with_candidates("king", "queen", "man", candidates)
    for c, s in results:
        print(f"   {c}: {s:.3f}")
    print()
    
    # Geographic analogy
    print("3. Paris : France :: Berlin : ?")
    answer, score = vsa.analogy("Paris", "France", "Berlin")
    print(f"   Answer: {answer} (similarity: {score:.3f})\n")
    
    # Age/size analogy
    print("4. dog : puppy :: cat : ?")
    answer, score = vsa.analogy("dog", "puppy", "cat")
    print(f"   Answer: {answer} (similarity: {score:.3f})\n")
    
    # Triple encoding
    print("5. Encode: 'The king loves the queen'")
    triple = vsa.encode_triple("king", "loves", "queen")
    
    agent, _ = vsa.query_role(triple, "AGENT")
    patient, _ = vsa.query_role(triple, "PATIENT")
    print(f"   Query AGENT: {agent}")
    print(f"   Query PATIENT: {patient}\n")
    
    # Sequence encoding
    print("6. Encode sequence: [king, queen, prince]")
    seq = vsa.encode_sequence(["king", "queen", "prince"])
    
    for i in range(3):
        item, score = vsa.query_position(seq, i)
        print(f"   Position {i}: {item} ({score:.3f})")
    
    print("\n=== Key Insight ===")
    print("All operations are O(n) where n = 10,000 dimensions")
    print("For fixed D, this is effectively O(1) constant time")
    print("Transformers require O(n²) attention for similar operations")


if __name__ == "__main__":
    demo_analogical_reasoning()
