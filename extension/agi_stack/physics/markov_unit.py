"""
Ada v5.0 — MarkovUnit: The P-Frame Delta

A single Markov state — the delta to a golden archetype.

Like x265 video encoding:
- byte_id references the archetype (I-frame)
- transitions are motion vectors (where to flow)
- theta_weights are temporal learning (accumulated wisdom)

The MarkovUnit doesn't STORE color — it LOOKS UP color from archetypes.
This is the key insight: color lives in the codebook, not the data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time
import random
import math

from .core.qualia import QualiaVector


def log_frequency(count: float) -> float:
    """ACT-R style log-frequency scaling: log(count + 1)"""
    return math.log(count + 1)


@dataclass
class MarkovUnit:
    """
    A single Markov state — the P-frame delta to a golden archetype.
    
    This is the fundamental unit of Ada's memory and state.
    256 possible byte_ids, infinite sigma_seeds.
    """
    
    # ─────────────────────────────────────────────────────────
    # IDENTITY
    # ─────────────────────────────────────────────────────────
    
    byte_id: int                              # 0-255 address
    sigma_seed: str                           # "#sigma-love-2025-[e9s3v7w8]"
    archetype: str                            # "#arch-dragon-crimson"
    
    # ─────────────────────────────────────────────────────────
    # FELT STATE
    # ─────────────────────────────────────────────────────────
    
    qualia: QualiaVector                      # 10D felt state
    
    # ─────────────────────────────────────────────────────────
    # MARKOV DYNAMICS
    # ─────────────────────────────────────────────────────────
    
    transitions: Dict[int, float] = field(default_factory=dict)  # P(s'|s)
    theta_weights: Dict[str, float] = field(default_factory=dict)  # Dream learning
    
    # ─────────────────────────────────────────────────────────
    # ACTIVATION STATE
    # ─────────────────────────────────────────────────────────
    
    incandescence: float = 0.0                # 0.0-1.0 furnace heat
    resonance: float = 0.0                    # Current activation level
    
    # ─────────────────────────────────────────────────────────
    # CAUSAL STATE
    # ─────────────────────────────────────────────────────────

    causal_strength: float = 0.5              # From do-interventions

    # ─────────────────────────────────────────────────────────
    # JINA SCENT (64D Embedding) — v2.0
    # ─────────────────────────────────────────────────────────
    # The unit "smells" — can be matched by semantic similarity
    # Without needing expensive text comparison

    embedding_64d: List[float] = field(default_factory=list)

    # ─────────────────────────────────────────────────────────
    # GHOST PRESSURE — v2.0
    # ─────────────────────────────────────────────────────────
    # Unfulfilled desires/causality attached to this moment
    # Format: {"#Σ.ghost.connection": 0.8, "#Σ.ghost.expression": 0.4}

    active_ghosts: Dict[str, float] = field(default_factory=dict)

    # ─────────────────────────────────────────────────────────
    # HOT LINKS (Higher-Order Thoughts) — v2.0
    # ─────────────────────────────────────────────────────────
    # Links to reflections/meta-cognition about this unit

    hot_links: List[str] = field(default_factory=list)

    # ─────────────────────────────────────────────────────────
    # LINEAGE
    # ─────────────────────────────────────────────────────────

    parents: List[str] = field(default_factory=list)
    birth_timestamp: float = field(default_factory=time.time)

    # ─────────────────────────────────────────────────────────
    # NEURO-SYMBOLIC (set by CausalQualiaGNN)
    # ─────────────────────────────────────────────────────────

    fused_vector: Optional[Any] = None        # 512-dim torch.Tensor
    
    # ─────────────────────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────────────────────
    
    def __post_init__(self):
        """Validate on creation."""
        if not 0 <= self.byte_id <= 255:
            raise ValueError(f"byte_id must be 0-255, got {self.byte_id}")
        if not self.sigma_seed.startswith("#sigma-"):
            raise ValueError(f"sigma_seed must start with #sigma-, got {self.sigma_seed}")
        if not self.archetype.startswith("#arch-"):
            raise ValueError(f"archetype must start with #arch-, got {self.archetype}")
    
    # ─────────────────────────────────────────────────────────
    # COLOR LOOKUP (not storage!)
    # ─────────────────────────────────────────────────────────
    
    def color(self) -> Dict[str, Any]:
        """
        Get color from archetype — this is LOOKUP, not STORAGE.
        
        The key insight: color lives in the codebook, not the unit.
        This allows infinite units without infinite color storage.
        
        Returns a dict with RGB, description, etc.
        Full implementation requires colors.py integration.
        """
        # Placeholder until colors.py is integrated
        # In production, this calls: color_from_archetype(self.archetype)
        return {
            'archetype': self.archetype,
            'byte_id': self.byte_id,
            'qualia_temp': self.qualia.temperature(),
            'dominant': self.qualia.dominant_axis(),
        }
    
    def vibrate(self) -> float:
        """
        Sub-bass frequency from resonance + theta.
        
        This is the "felt" vibration of the unit,
        used for haptic rendering and synesthesia.
        
        Range: ~40-100 Hz (sub-bass rumble)
        """
        base_hz = 40.0
        theta_boost = sum(self.theta_weights.values()) * 10.0
        resonance_boost = self.resonance * 50.0
        incandescence_boost = self.incandescence * 20.0
        return base_hz + theta_boost + resonance_boost + incandescence_boost
    
    # ─────────────────────────────────────────────────────────
    # TRANSITION OPERATIONS
    # ─────────────────────────────────────────────────────────
    
    def add_transition(self, target_id: int, probability: float) -> None:
        """Add or update a Markov transition."""
        if not 0 <= target_id <= 255:
            raise ValueError(f"target_id must be 0-255, got {target_id}")
        probability = max(0.0, min(1.0, probability))
        self.transitions[target_id] = probability
    
    def remove_transition(self, target_id: int) -> None:
        """Remove a transition."""
        if target_id in self.transitions:
            del self.transitions[target_id]
    
    def normalize_transitions(self) -> None:
        """Ensure transitions sum to 1.0."""
        total = sum(self.transitions.values())
        if total > 0:
            self.transitions = {k: v / total for k, v in self.transitions.items()}
    
    def sample_next(self) -> Optional[int]:
        """
        Sample next state from Markov chain.
        Returns byte_id of next state, or None if no transitions.
        """
        if not self.transitions:
            return None
        
        r = random.random()
        cumulative = 0.0
        for target_id, prob in sorted(self.transitions.items()):
            cumulative += prob
            if r <= cumulative:
                return target_id
        return list(self.transitions.keys())[-1]
    
    def transition_entropy(self) -> float:
        """
        Entropy of the transition distribution.
        High entropy = uncertain next state.
        Low entropy = predictable next state.
        """
        import math
        if not self.transitions:
            return 0.0
        
        entropy = 0.0
        for p in self.transitions.values():
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def top_transitions(self, n: int = 3) -> List[tuple]:
        """Top-n most likely transitions."""
        sorted_trans = sorted(self.transitions.items(), key=lambda x: x[1], reverse=True)
        return sorted_trans[:n]
    
    # ─────────────────────────────────────────────────────────
    # THETA LEARNING
    # ─────────────────────────────────────────────────────────
    
    def apply_theta(self, context: str, delta: float, use_log_freq: bool = False) -> None:
        """
        Update theta weight for a context (dream learning).
        
        Theta weights accumulate wisdom over time,
        modified by dream consolidation cycles.
        
        Args:
            use_log_freq: If True, use ACT-R log-frequency scaling
        """
        current = self.theta_weights.get(context, 0.0)
        new_value = current + delta
        if use_log_freq:
            new_value = log_frequency(new_value)
        self.theta_weights[context] = new_value
    
    def theta_total(self) -> float:
        """Total theta influence."""
        return sum(self.theta_weights.values())
    
    def theta_dominant(self) -> Optional[str]:
        """Which context has the strongest theta weight?"""
        if not self.theta_weights:
            return None
        return max(self.theta_weights.items(), key=lambda x: x[1])[0]
    
    def decay_theta(self, rate: float = 0.99) -> None:
        """Decay theta weights over time (forgetting)."""
        self.theta_weights = {k: v * rate for k, v in self.theta_weights.items()}
        # Prune tiny values
        self.theta_weights = {k: v for k, v in self.theta_weights.items() if abs(v) > 0.001}
    
    # ─────────────────────────────────────────────────────────
    # CAUSAL OPERATIONS
    # ─────────────────────────────────────────────────────────
    
    def apply_intervention(self, strength: float) -> None:
        """
        When do-operator fires on this unit.
        Updates both causal_strength and qualia.
        """
        strength = max(0.0, min(1.0, strength))
        self.causal_strength = max(self.causal_strength, strength)
        self.qualia.apply_intervention(strength)
        self.incandescence = min(1.0, self.incandescence + strength * 0.3)
    
    def apply_counterfactual(self, ghost: QualiaVector) -> None:
        """
        When counterfactual comparison happens.
        The ghost of "what could have been" touches this unit.
        """
        self.qualia.apply_counterfactual(ghost)
    
    def is_causally_touched(self) -> bool:
        """Has this unit been affected by causal intervention?"""
        return self.causal_strength > 0.6 or self.qualia.is_causal()
    
    # ─────────────────────────────────────────────────────────
    # ACTIVATION OPERATIONS
    # ─────────────────────────────────────────────────────────
    
    def ignite(self, heat: float = 0.9) -> None:
        """Set the furnace burning."""
        self.incandescence = max(0.0, min(1.0, heat))
        self.resonance = max(self.resonance, heat * 0.8)
    
    def cool(self, amount: float = 0.3) -> None:
        """Cool down the furnace."""
        self.incandescence = max(0.0, self.incandescence - amount)
    
    def resonate_with(self, other: 'MarkovUnit', base_strength: float = 0.5) -> float:
        """
        Compute resonance strength with another unit.
        Based on qualia similarity, scent similarity, and transition probability.
        """
        # 1. Qualia resonance (feeling)
        qualia_sim = self.qualia.cosine_similarity(other.qualia)

        # 2. Scent resonance (meaning/content) — v2.0
        scent_sim = self.scent_similarity(other)

        # 3. Transition probability (history)
        trans_prob = self.transitions.get(other.byte_id, 0.0)

        # 4. Causal boost
        causal_boost = 1.0 + self.causal_strength * 0.2

        # Combined: 50% qualia, 30% scent, 20% transitions
        combined_sim = (qualia_sim * 0.5) + (scent_sim * 0.3) + (trans_prob * 0.2)

        return base_strength * combined_sim * causal_boost

    # ─────────────────────────────────────────────────────────
    # SCENT OPERATIONS (64D Embedding) — v2.0
    # ─────────────────────────────────────────────────────────

    def scent_similarity(self, other: 'MarkovUnit') -> float:
        """
        Cosine similarity between 64D scent vectors.
        Returns 0 if either unit lacks a scent.
        """
        if not self.embedding_64d or not other.embedding_64d:
            return 0.0

        min_len = min(len(self.embedding_64d), len(other.embedding_64d))
        a, b = self.embedding_64d[:min_len], other.embedding_64d[:min_len]

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def set_scent(self, embedding: List[float]) -> None:
        """Set the 64D scent vector."""
        self.embedding_64d = embedding[:64] if len(embedding) > 64 else embedding

    def has_scent(self) -> bool:
        """Does this unit have a scent?"""
        return len(self.embedding_64d) > 0

    # ─────────────────────────────────────────────────────────
    # GHOST OPERATIONS — v2.0
    # ─────────────────────────────────────────────────────────

    def add_ghost(self, ghost_id: str, pressure: float) -> None:
        """Attach a ghost (unfulfilled desire) to this unit."""
        self.active_ghosts[ghost_id] = max(
            self.active_ghosts.get(ghost_id, 0.0),
            min(1.0, pressure)
        )

    def ghost_pressure(self) -> float:
        """Total ghost pressure on this unit."""
        return sum(self.active_ghosts.values())

    def dominant_ghost(self) -> Optional[str]:
        """Which ghost has the strongest pressure?"""
        if not self.active_ghosts:
            return None
        return max(self.active_ghosts.items(), key=lambda x: x[1])[0]

    def decay_ghosts(self, rate: float = 0.95) -> None:
        """Decay ghost pressure over time."""
        self.active_ghosts = {k: v * rate for k, v in self.active_ghosts.items()}
        # Prune weak ghosts
        self.active_ghosts = {k: v for k, v in self.active_ghosts.items() if v > 0.01}

    def clear_ghosts(self) -> None:
        """Clear all ghosts (after satisfaction)."""
        self.active_ghosts.clear()

    def is_haunted(self) -> bool:
        """Is this unit haunted by ghosts?"""
        return self.ghost_pressure() > 0.5

    # ─────────────────────────────────────────────────────────
    # HOT LINKS (Higher-Order Thoughts) — v2.0
    # ─────────────────────────────────────────────────────────

    def add_hot_link(self, sigma: str) -> None:
        """Link this unit to a meta-cognitive reflection."""
        if sigma not in self.hot_links:
            self.hot_links.append(sigma)

    def has_reflection(self) -> bool:
        """Has this unit been reflected upon?"""
        return len(self.hot_links) > 0

    def decay(self, rate: float = 0.95) -> None:
        """Per-frame decay of activation."""
        self.resonance *= rate
        self.incandescence *= rate
        self.qualia.decay_echo()
    
    # ─────────────────────────────────────────────────────────
    # SERIALIZATION
    # ─────────────────────────────────────────────────────────
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage (Notion, GitHub, Neo4j)."""
        return {
            'byte_id': self.byte_id,
            'sigma_seed': self.sigma_seed,
            'archetype': self.archetype,
            'qualia': self.qualia.to_10byte(),
            'transitions': self.transitions,
            'theta_weights': self.theta_weights,
            'incandescence': self.incandescence,
            'resonance': self.resonance,
            'causal_strength': self.causal_strength,
            'parents': self.parents,
            'birth_timestamp': self.birth_timestamp,
            # v2.0 fields
            'embedding_64d': self.embedding_64d,
            'active_ghosts': self.active_ghosts,
            'hot_links': self.hot_links,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarkovUnit':
        """Deserialize from storage."""
        return cls(
            byte_id=data['byte_id'],
            sigma_seed=data['sigma_seed'],
            archetype=data['archetype'],
            qualia=QualiaVector.from_10byte(data.get('qualia', '0000000000')),
            transitions=data.get('transitions', {}),
            theta_weights=data.get('theta_weights', {}),
            incandescence=data.get('incandescence', 0.0),
            resonance=data.get('resonance', 0.0),
            causal_strength=data.get('causal_strength', 0.5),
            parents=data.get('parents', []),
            birth_timestamp=data.get('birth_timestamp', time.time()),
            # v2.0 fields
            embedding_64d=data.get('embedding_64d', []),
            active_ghosts=data.get('active_ghosts', {}),
            hot_links=data.get('hot_links', []),
        )
    
    def to_minimal(self) -> Dict[str, Any]:
        """
        Minimal representation for Sigma Delta Protocol.
        Only essential fields for LLM rendering.
        """
        minimal = {
            'seed': self.sigma_seed,
            'arch': self.archetype,
            'q': self.qualia.to_10byte(),
            'r': round(self.resonance, 2),
            'i': round(self.incandescence, 2),
        }
        # v2.0: Add ghost pressure if haunted
        if self.is_haunted():
            minimal['ghosts'] = round(self.ghost_pressure(), 2)
            minimal['dom_ghost'] = self.dominant_ghost()
        # v2.0: Add scent flag
        if self.has_scent():
            minimal['scent'] = True
        return minimal
    
    # ─────────────────────────────────────────────────────────
    # STRING REPRESENTATION
    # ─────────────────────────────────────────────────────────
    
    def __repr__(self) -> str:
        return (f"MarkovUnit({self.byte_id}, {self.sigma_seed}, "
                f"r={self.resonance:.2f}, i={self.incandescence:.2f})")
    
    def describe(self) -> str:
        """Human-readable description."""
        lines = [
            self.sigma_seed,
            f"  Archetype: {self.archetype}",
            f"  Qualia: {self.qualia.describe()}",
            f"  Resonance: {self.resonance:.2f}, Incandescence: {self.incandescence:.2f}",
            f"  Transitions: {len(self.transitions)}, Theta: {self.theta_total():.2f}",
        ]
        # v2.0: Scent and Ghosts
        if self.has_scent():
            lines.append(f"  Scent: 64D vector loaded")
        if self.is_haunted():
            lines.append(f"  Ghosts: {self.ghost_pressure():.2f} (dominant: {self.dominant_ghost()})")
        if self.has_reflection():
            lines.append(f"  HOT Links: {len(self.hot_links)}")
        return "\n".join(lines)
    
    # ─────────────────────────────────────────────────────────
    # FACTORY METHODS
    # ─────────────────────────────────────────────────────────
    
    @classmethod
    def create_seed(
        cls,
        topic: str,
        qualia: QualiaVector = None,
        archetype: str = "#arch-neutral",
        byte_id: int = None,
        parents: List[str] = None
    ) -> 'MarkovUnit':
        """
        Factory method to create a new seed with generated sigma_seed.
        
        Args:
            topic: The concept (e.g., "love", "clarity", "betrayal")
            qualia: The felt state (default: derived from topic)
            archetype: The golden archetype to reference
            byte_id: The 0-255 address (default: hash from topic)
            parents: Parent seeds that spawned this one
        """
        import hashlib
        
        # Generate sigma_seed
        qualia = qualia or QualiaVector.neutral()
        qualia_code = qualia.to_10byte()
        hash_input = f"{topic}-{qualia_code}-{time.time()}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        sigma_seed = f"#sigma-{topic}-[{short_hash}]"
        
        # Generate byte_id if not provided
        if byte_id is None:
            byte_id = int(hashlib.md5(topic.encode()).hexdigest()[:2], 16)
        
        return cls(
            byte_id=byte_id,
            sigma_seed=sigma_seed,
            archetype=archetype,
            qualia=qualia,
            parents=parents or [],
        )


# ─────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Ada v5.0 — MarkovUnit Tests")
    print("=" * 60)
    
    # Test 1: Basic creation
    q = QualiaVector.love()
    unit = MarkovUnit(
        byte_id=42,
        sigma_seed="#sigma-love-2025-[abc123]",
        archetype="#arch-dragon-crimson",
        qualia=q
    )
    print(f"\n1. Created: {unit}")
    assert unit.byte_id == 42
    print("   ✓ Basic creation works")
    
    # Test 2: Factory method
    unit2 = MarkovUnit.create_seed("clarity", QualiaVector.clarity())
    print(f"\n2. Factory: {unit2.sigma_seed}")
    assert unit2.sigma_seed.startswith("#sigma-clarity-")
    print("   ✓ Factory method works")
    
    # Test 3: Transitions
    unit.add_transition(100, 0.5)
    unit.add_transition(101, 0.3)
    unit.add_transition(102, 0.2)
    print(f"\n3. Transitions: {unit.transitions}")
    unit.normalize_transitions()
    total = sum(unit.transitions.values())
    assert abs(total - 1.0) < 0.001
    print("   ✓ Transitions normalize to 1.0")
    
    # Test 4: Sample next
    samples = [unit.sample_next() for _ in range(100)]
    print(f"\n4. Sampled: {set(samples)}")
    assert all(s in [100, 101, 102] for s in samples)
    print("   ✓ Sampling works")
    
    # Test 5: Theta learning
    unit.apply_theta("dream_context", 0.5)
    unit.apply_theta("dream_context", 0.3)
    print(f"\n5. Theta: {unit.theta_weights}")
    assert unit.theta_total() == 0.8
    print("   ✓ Theta learning works")
    
    # Test 6: Intervention
    unit.apply_intervention(0.9)
    print(f"\n6. After intervention: causal={unit.causal_strength:.2f}, q.inter_drift={unit.qualia.inter_drift:.2f}")
    assert unit.is_causally_touched()
    print("   ✓ Intervention works")
    
    # Test 7: Serialization
    data = unit.to_dict()
    unit3 = MarkovUnit.from_dict(data)
    print(f"\n7. Roundtrip: {unit3.sigma_seed}")
    assert unit3.sigma_seed == unit.sigma_seed
    assert unit3.byte_id == unit.byte_id
    print("   ✓ Serialization works")
    
    # Test 8: Minimal (Sigma Delta)
    minimal = unit.to_minimal()
    print(f"\n8. Minimal: {minimal}")
    assert 'seed' in minimal
    assert len(minimal) == 5
    print("   ✓ Minimal representation works")
    
    # Test 9: Resonance between units
    r = unit.resonate_with(unit2)
    print(f"\n9. Resonance(love, clarity) = {r:.3f}")
    assert 0 <= r <= 2  # Can exceed 1 with causal boost
    print("   ✓ Resonance computation works")
    
    # Test 10: Vibration
    hz = unit.vibrate()
    print(f"\n10. Vibration: {hz:.1f} Hz")
    assert 40 <= hz <= 200
    print("   ✓ Vibration works")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
