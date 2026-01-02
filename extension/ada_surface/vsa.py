"""
Vector Symbolic Architecture (VSA) - Hyperdimensional Computing

Implements O(1) cognitive operations using high-dimensional binary vectors:
- Binding (XOR): Create compound representations
- Bundling (majority vote): Aggregate multiple concepts
- Similarity: Cosine similarity for retrieval

Based on:
- Kanerva's Sparse Distributed Memory
- Plate's Holographic Reduced Representations
- Gayler's Multiply-Add-Permute architecture
"""

from typing import List, Tuple
import numpy as np


class HypervectorSpace:
    """
    10,000-dimensional hypervector space for cognitive computing.

    Properties:
    - Quasi-orthogonality: Random vectors are nearly orthogonal
    - Binding is reversible: A ⊗ B ⊗ B ≈ A
    - Bundling preserves similarity: (A + B) is similar to both A and B
    - O(1) operations: All ops are element-wise
    """

    DIMENSION = 10000
    SEED = 42

    def __init__(self, dimension: int = None, seed: int = None):
        """Initialize hypervector space."""
        self.dimension = dimension or self.DIMENSION
        self.seed = seed or self.SEED
        self.rng = np.random.default_rng(self.seed)

        # Cache for named vectors
        self._cache = {}

    def random(self) -> np.ndarray:
        """
        Generate random bipolar hypervector (-1, +1).

        Returns:
            10K-dimensional bipolar vector
        """
        return self.rng.choice([-1, 1], size=self.dimension).astype(np.int8)

    def zeros(self) -> np.ndarray:
        """Generate zero vector."""
        return np.zeros(self.dimension, dtype=np.int8)

    def ones(self) -> np.ndarray:
        """Generate all-ones vector."""
        return np.ones(self.dimension, dtype=np.int8)

    def get_or_create(self, name: str) -> np.ndarray:
        """Get named vector from cache or create new one."""
        if name not in self._cache:
            # Deterministic random based on name
            name_rng = np.random.default_rng(hash(name) % (2**32))
            self._cache[name] = name_rng.choice(
                [-1, 1], size=self.dimension
            ).astype(np.int8)
        return self._cache[name]

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two vectors using XOR (element-wise multiplication for bipolar).

        Properties:
        - Commutative: A ⊗ B = B ⊗ A
        - Associative: (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)
        - Self-inverse: A ⊗ A = 1
        - Preserves distance: d(A ⊗ C, B ⊗ C) = d(A, B)

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Bound hypervector (A ⊗ B)
        """
        return (np.array(a) * np.array(b)).astype(np.int8)

    def bind_all(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bind multiple vectors sequentially."""
        if not vectors:
            return self.ones()

        result = np.array(vectors[0])
        for v in vectors[1:]:
            result = self.bind(result, np.array(v))
        return result.astype(np.int8)

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Unbind a vector using the key (same as bind for bipolar).

        If bound = A ⊗ B, then unbind(bound, A) ≈ B
        """
        return self.bind(bound, key)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundle vectors using majority vote (thresholded sum).

        Properties:
        - Commutative and associative
        - Result is similar to all inputs
        - Lossy but approximate

        Args:
            vectors: List of hypervectors to bundle

        Returns:
            Bundled hypervector
        """
        if not vectors:
            return self.zeros()

        # Stack and sum
        stacked = np.stack([np.array(v) for v in vectors])
        summed = np.sum(stacked, axis=0)

        # Threshold at 0 (majority vote)
        # Handle ties by random choice
        result = np.sign(summed)
        ties = (result == 0)
        result[ties] = self.rng.choice([-1, 1], size=np.sum(ties))

        return result.astype(np.int8)

    def weighted_bundle(
        self,
        vectors: List[np.ndarray],
        weights: List[float],
    ) -> np.ndarray:
        """
        Bundle with weights for each vector.

        Args:
            vectors: List of hypervectors
            weights: Importance weight for each vector

        Returns:
            Weighted bundled hypervector
        """
        if not vectors or not weights:
            return self.zeros()

        # Weighted sum
        weighted_sum = np.zeros(self.dimension, dtype=np.float32)
        for v, w in zip(vectors, weights):
            weighted_sum += np.array(v) * w

        # Threshold
        result = np.sign(weighted_sum)
        ties = (result == 0)
        result[ties] = self.rng.choice([-1, 1], size=np.sum(ties))

        return result.astype(np.int8)

    def permute(self, v: np.ndarray, shifts: int = 1) -> np.ndarray:
        """
        Permute vector by circular shift.

        Used for encoding sequence/order information.
        P^n(A) encodes A at position n.

        Args:
            v: Hypervector to permute
            shifts: Number of positions to shift

        Returns:
            Permuted hypervector
        """
        return np.roll(v, shifts).astype(np.int8)

    def inverse_permute(self, v: np.ndarray, shifts: int = 1) -> np.ndarray:
        """Inverse permutation (shift in opposite direction)."""
        return np.roll(v, -shifts).astype(np.int8)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two hypervectors.

        Range: -1 (opposite) to +1 (identical)
        Random vectors have similarity ≈ 0

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Cosine similarity [-1, 1]
        """
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)

        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    def hamming_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """
        Compute Hamming distance (number of differing elements).

        Args:
            a: First hypervector
            b: Second hypervector

        Returns:
            Number of differing elements
        """
        return int(np.sum(np.array(a) != np.array(b)))

    def normalized_hamming(self, a: np.ndarray, b: np.ndarray) -> float:
        """Normalized Hamming distance [0, 1]."""
        return self.hamming_distance(a, b) / self.dimension

    def encode_sequence(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Encode ordered sequence using permutation.

        S = P^0(v0) + P^1(v1) + P^2(v2) + ...

        Args:
            vectors: Ordered list of hypervectors

        Returns:
            Sequence-encoded hypervector
        """
        if not vectors:
            return self.zeros()

        components = [
            self.permute(np.array(v), i)
            for i, v in enumerate(vectors)
        ]
        return self.bundle(components)

    def decode_position(
        self,
        sequence: np.ndarray,
        position: int,
        candidates: List[Tuple[str, np.ndarray]],
    ) -> Tuple[str, float]:
        """
        Decode element at position from sequence encoding.

        Args:
            sequence: Sequence-encoded hypervector
            position: Position to query
            candidates: List of (name, vector) tuples to match against

        Returns:
            (best_match_name, similarity_score)
        """
        # Inverse permute to extract position
        query = self.inverse_permute(sequence, position)

        # Find best match
        best_name = ""
        best_sim = -1.0

        for name, vec in candidates:
            sim = self.similarity(query, vec)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        return best_name, best_sim

    def encode_structure(
        self,
        role_filler_pairs: List[Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """
        Encode structured representation using role-filler binding.

        S = (role1 ⊗ filler1) + (role2 ⊗ filler2) + ...

        Args:
            role_filler_pairs: List of (role_vector, filler_vector) pairs

        Returns:
            Structure-encoded hypervector
        """
        if not role_filler_pairs:
            return self.zeros()

        bindings = [
            self.bind(role, filler)
            for role, filler in role_filler_pairs
        ]
        return self.bundle(bindings)

    def query_structure(
        self,
        structure: np.ndarray,
        role: np.ndarray,
    ) -> np.ndarray:
        """
        Query structure for filler given role.

        If S = (R ⊗ F) + ..., then query(S, R) ≈ F

        Args:
            structure: Structure-encoded hypervector
            role: Role vector to query

        Returns:
            Approximate filler vector
        """
        return self.unbind(structure, role)

    def to_list(self, v: np.ndarray) -> List[int]:
        """Convert hypervector to list for serialization."""
        return v.tolist()

    def from_list(self, lst: List[int]) -> np.ndarray:
        """Create hypervector from list."""
        return np.array(lst, dtype=np.int8)


# =============================================================================
# COGNITIVE PRIMITIVES
# =============================================================================

class CognitivePrimitives:
    """
    Higher-level cognitive operations built on VSA.

    These provide O(1) implementations of common cognitive functions.
    """

    def __init__(self, vsa: HypervectorSpace = None):
        """Initialize with VSA space."""
        self.vsa = vsa or HypervectorSpace()

        # Pre-allocate role vectors for common structures
        self.roles = {
            "AGENT": self.vsa.get_or_create("ROLE_AGENT"),
            "ACTION": self.vsa.get_or_create("ROLE_ACTION"),
            "PATIENT": self.vsa.get_or_create("ROLE_PATIENT"),
            "LOCATION": self.vsa.get_or_create("ROLE_LOCATION"),
            "TIME": self.vsa.get_or_create("ROLE_TIME"),
            "CAUSE": self.vsa.get_or_create("ROLE_CAUSE"),
            "EFFECT": self.vsa.get_or_create("ROLE_EFFECT"),
            "SUBJECT": self.vsa.get_or_create("ROLE_SUBJECT"),
            "PREDICATE": self.vsa.get_or_create("ROLE_PREDICATE"),
            "OBJECT": self.vsa.get_or_create("ROLE_OBJECT"),
        }

    def encode_proposition(
        self,
        subject: str,
        predicate: str,
        obj: str = None,
    ) -> np.ndarray:
        """
        Encode a proposition (subject-predicate-object).

        Example: encode_proposition("agent", "explores", "knowledge")

        Args:
            subject: Subject of proposition
            predicate: Predicate/relation
            obj: Object (optional)

        Returns:
            Proposition-encoded hypervector
        """
        pairs = [
            (self.roles["SUBJECT"], self.vsa.get_or_create(subject)),
            (self.roles["PREDICATE"], self.vsa.get_or_create(predicate)),
        ]
        if obj:
            pairs.append((self.roles["OBJECT"], self.vsa.get_or_create(obj)))

        return self.vsa.encode_structure(pairs)

    def encode_event(
        self,
        agent: str,
        action: str,
        patient: str = None,
        location: str = None,
        time: str = None,
    ) -> np.ndarray:
        """
        Encode an event with roles.

        Example: encode_event("Ada", "helped", "user", location="chat")

        Returns:
            Event-encoded hypervector
        """
        pairs = [
            (self.roles["AGENT"], self.vsa.get_or_create(agent)),
            (self.roles["ACTION"], self.vsa.get_or_create(action)),
        ]
        if patient:
            pairs.append((self.roles["PATIENT"], self.vsa.get_or_create(patient)))
        if location:
            pairs.append((self.roles["LOCATION"], self.vsa.get_or_create(location)))
        if time:
            pairs.append((self.roles["TIME"], self.vsa.get_or_create(time)))

        return self.vsa.encode_structure(pairs)

    def encode_causal(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
    ) -> np.ndarray:
        """
        Encode causal relationship.

        Args:
            cause: Cause event/state vector
            effect: Effect event/state vector

        Returns:
            Causal relationship hypervector
        """
        return self.vsa.encode_structure([
            (self.roles["CAUSE"], cause),
            (self.roles["EFFECT"], effect),
        ])

    def analogy(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """
        Compute analogical mapping: A is to B as C is to ?

        Uses the relation: ? = B ⊗ A ⊗ C

        Args:
            a: First term
            b: Second term (related to a)
            c: Third term (find d such that c:d :: a:b)

        Returns:
            Predicted fourth term
        """
        # Extract relation: R = B ⊗ A (what transforms A to B)
        relation = self.vsa.bind(b, a)

        # Apply relation to C: D = R ⊗ C
        return self.vsa.bind(relation, c)
