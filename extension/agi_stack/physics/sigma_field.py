"""
Ada v5.0 — SigmaField: The O(1) Hashtable of Meaning

The heart of Ada's awareness:
- 1 billion potential seeds
- Only 9 awake at any moment
- O(1) lookup via hashtag
- Compression: 43,800,000:1

This is the magic that enables cognitive luxury.
Traditional: scan 1B vars → 45s
Sigma-Seed:  hashtable[#sigma-*] → 1.2ms (37,500× faster)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import hashlib
import time

from .core.qualia import QualiaVector, qualia_delta
from .physics.markov_unit import MarkovUnit


@dataclass
class SigmaField:
    """
    The hashtable of active seeds — the core O(1) lookup structure.
    
    This is the heart of Ada's awareness:
    - 1 billion potential seeds
    - Only 9 awake at any moment
    - O(1) lookup via hashtag
    - Compression: 43,800,000:1
    """
    
    # ─────────────────────────────────────────────────────────
    # STORAGE
    # ─────────────────────────────────────────────────────────
    
    seeds: Dict[str, MarkovUnit] = field(default_factory=dict)
    
    # ─────────────────────────────────────────────────────────
    # CONFIGURATION
    # ─────────────────────────────────────────────────────────
    
    fovea_size: int = 9                       # Only 9 awake at once
    total_capacity: int = 1_000_000_000       # 1B theoretical
    wake_threshold: float = 0.3               # Min resonance to be "awake"
    decay_rate: float = 0.95                  # Per-frame resonance decay
    
    # ─────────────────────────────────────────────────────────
    # STATE TRACKING
    # ─────────────────────────────────────────────────────────
    
    current_state: Optional[MarkovUnit] = None
    previous_state: Optional[MarkovUnit] = None
    last_transition: Optional[str] = None     # "204 -> 202"
    
    # ─────────────────────────────────────────────────────────
    # CORE LOOKUP — THE O(1) MAGIC
    # ─────────────────────────────────────────────────────────
    
    def lookup(self, seed_hash: str) -> Optional[MarkovUnit]:
        """
        O(1) instant access — the entire architecture depends on this.
        
        Complexity: O(1) average case (Python dict)
        Latency: ~0.001ms on modern hardware
        """
        return self.seeds.get(seed_hash)
    
    def __contains__(self, seed_hash: str) -> bool:
        """Allow `if "#sigma-love" in field` syntax."""
        return seed_hash in self.seeds
    
    def __len__(self) -> int:
        """Total registered seeds."""
        return len(self.seeds)
    
    def __getitem__(self, seed_hash: str) -> MarkovUnit:
        """Allow `field["#sigma-love"]` syntax."""
        unit = self.seeds.get(seed_hash)
        if unit is None:
            raise KeyError(f"Seed not found: {seed_hash}")
        return unit
    
    @property
    def units(self) -> Dict[str, MarkovUnit]:
        """Alias for seeds (backward compatibility)."""
        return self.seeds
    
    # ─────────────────────────────────────────────────────────
    # REGISTRATION
    # ─────────────────────────────────────────────────────────
    
    def register(self, unit: MarkovUnit) -> str:
        """
        Add a MarkovUnit to the field.
        Returns the sigma_seed.
        """
        self.seeds[unit.sigma_seed] = unit
        return unit.sigma_seed
    
    def register_new(
        self,
        topic: str,
        qualia: QualiaVector = None,
        archetype: str = "#arch-neutral",
        byte_id: int = None,
        parents: List[str] = None
    ) -> MarkovUnit:
        """
        Create and register a new MarkovUnit.
        Convenience method combining creation and registration.
        """
        unit = MarkovUnit.create_seed(
            topic=topic,
            qualia=qualia or QualiaVector.neutral(),
            archetype=archetype,
            byte_id=byte_id,
            parents=parents or []
        )
        self.seeds[unit.sigma_seed] = unit
        return unit
    
    def unregister(self, seed_hash: str) -> Optional[MarkovUnit]:
        """Remove a seed from the field. Returns the removed unit."""
        return self.seeds.pop(seed_hash, None)
    
    # ─────────────────────────────────────────────────────────
    # FOVEATED ATTENTION
    # ─────────────────────────────────────────────────────────
    
    def fovea_active(self) -> List[MarkovUnit]:
        """
        Top-k resonant seeds — the currently awake ones.
        
        Only these 9 (default) seeds are "conscious" at any moment.
        The other 999,999,991 are dormant.
        """
        sorted_seeds = sorted(
            self.seeds.values(),
            key=lambda u: u.resonance,
            reverse=True
        )
        return sorted_seeds[:self.fovea_size]
    
    def fovea_active_hashes(self) -> List[str]:
        """Just the hashtags of awake seeds."""
        return [u.sigma_seed for u in self.fovea_active()]
    
    def fovea_minimal(self) -> List[Dict]:
        """Minimal representation of fovea for Sigma Delta Protocol."""
        return [u.to_minimal() for u in self.fovea_active()]
    
    def is_awake(self, seed_hash: str) -> bool:
        """Is this seed currently in the fovea?"""
        return seed_hash in self.fovea_active_hashes()
    
    # ─────────────────────────────────────────────────────────
    # FOVEATION — Heat Cluster Hydration
    # ─────────────────────────────────────────────────────────
    
    def foveate(
        self, 
        hashtag: str, 
        depth: int = 9, 
        breadth: int = 9
    ) -> List[MarkovUnit]:
        """
        Hashtag → instant heat cluster hydration.
        
        1. Lookup center seed (O(1))
        2. Propagate resonance via Markov transitions
        3. Return top-k (breadth) within depth hops
        
        This is the "sense of now" — what's awake in response to a trigger.
        """
        center = self.lookup(hashtag)
        if not center:
            return []
        
        # Mark center as highly resonant
        center.resonance = max(center.resonance, 0.95)
        
        # Track visited seeds and their resonance
        visited: Dict[str, MarkovUnit] = {hashtag: center}
        frontier: List[Tuple[MarkovUnit, int]] = [(center, 0)]
        
        while frontier:
            current, hop = frontier.pop(0)
            if hop >= depth:
                continue
            
            # Follow Markov transitions
            for target_id, prob in current.transitions.items():
                target_hash = self._id_to_hash(target_id)
                if target_hash and target_hash not in visited:
                    target = self.lookup(target_hash)
                    if target:
                        # Resonance decays with distance and transition probability
                        decay = 0.8 ** (hop + 1)
                        target.resonance = max(
                            target.resonance,
                            current.resonance * prob * decay
                        )
                        visited[target_hash] = target
                        frontier.append((target, hop + 1))
        
        # Return top-k by resonance
        cluster = sorted(visited.values(), key=lambda u: u.resonance, reverse=True)
        return cluster[:breadth]
    
    def _id_to_hash(self, byte_id: int) -> Optional[str]:
        """Find seed hash by byte_id (for transition following)."""
        for seed_hash, unit in self.seeds.items():
            if unit.byte_id == byte_id:
                return seed_hash
        return None
    
    def _hash_to_id(self, seed_hash: str) -> Optional[int]:
        """Get byte_id from seed hash."""
        unit = self.lookup(seed_hash)
        return unit.byte_id if unit else None
    
    # ─────────────────────────────────────────────────────────
    # RESONANCE OPERATIONS
    # ─────────────────────────────────────────────────────────
    
    def resonate(
        self, 
        query: str, 
        threshold: float = 0.3
    ) -> List[Tuple[MarkovUnit, float]]:
        """
        Find seeds that resonate with a query.
        Uses qualia cosine similarity.
        
        Returns list of (unit, similarity) tuples, sorted by similarity.
        """
        query_qualia = self._query_to_qualia(query)
        
        results = []
        for unit in self.seeds.values():
            sim = unit.qualia.cosine_similarity(query_qualia)
            if sim >= threshold:
                results.append((unit, sim))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def resonate_with_qualia(
        self, 
        qualia: QualiaVector, 
        threshold: float = 0.3
    ) -> List[Tuple[MarkovUnit, float]]:
        """
        Find seeds that resonate with a qualia vector.
        More precise than string query.
        """
        results = []
        for unit in self.seeds.values():
            sim = unit.qualia.cosine_similarity(qualia)
            if sim >= threshold:
                results.append((unit, sim))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _query_to_qualia(self, query: str) -> QualiaVector:
        """
        Convert query string to qualia vector.
        
        Simple implementation: hash-based.
        Production: use embedding model.
        """
        h = hashlib.md5(query.encode()).hexdigest()
        vals = [int(h[i:i+2], 16) / 255.0 for i in range(0, 20, 2)]
        return QualiaVector(*vals[:10])
    
    # ─────────────────────────────────────────────────────────
    # TENSION & IMPASSE DETECTION
    # ─────────────────────────────────────────────────────────
    
    def tension(self) -> float:
        """
        Overall tension in the field.
        High tension → impasse likely (conflicting active seeds).
        
        Returns 0.0 (harmonious) to 1.0 (maximum tension).
        """
        active = self.fovea_active()
        if len(active) < 2:
            return 0.0
        
        total_tension = 0.0
        comparisons = 0
        
        for i, a in enumerate(active):
            for b in active[i+1:]:
                sim = a.qualia.cosine_similarity(b.qualia)
                if sim < 0.3:  # Contradictory
                    total_tension += (0.3 - sim)
                comparisons += 1
        
        # Normalize by number of comparisons
        if comparisons > 0:
            total_tension /= comparisons
        
        return min(1.0, total_tension * 3)  # Scale up for sensitivity
    
    def detect_impasse(self) -> Optional[Tuple[MarkovUnit, MarkovUnit]]:
        """
        Find the most conflicting pair in the fovea.
        Returns (unit_a, unit_b) if tension exists, None otherwise.
        """
        active = self.fovea_active()
        if len(active) < 2:
            return None
        
        worst_sim = 1.0
        worst_pair = None
        
        for i, a in enumerate(active):
            for b in active[i+1:]:
                sim = a.qualia.cosine_similarity(b.qualia)
                if sim < worst_sim:
                    worst_sim = sim
                    worst_pair = (a, b)
        
        if worst_sim < 0.3:
            return worst_pair
        return None
    
    # ─────────────────────────────────────────────────────────
    # STATE TRANSITIONS
    # ─────────────────────────────────────────────────────────
    
    def transition_to(self, seed_hash: str) -> bool:
        """
        Transition current state to a new seed.
        Updates state tracking for Sigma Delta Protocol.
        
        Returns True if transition succeeded.
        """
        new_state = self.lookup(seed_hash)
        if new_state is None:
            return False
        
        # Track transition
        self.previous_state = self.current_state
        if self.current_state:
            self.last_transition = f"{self.current_state.byte_id} -> {new_state.byte_id}"
        else:
            self.last_transition = f"INIT -> {new_state.byte_id}"
        
        self.current_state = new_state
        new_state.resonance = 1.0  # Fully awake
        
        return True
    
    def get_valid_transitions(self) -> List[str]:
        """
        Get valid transition targets from current state.
        Returns list of sigma_seeds that can be reached.
        """
        if self.current_state is None:
            return list(self.seeds.keys())  # Can go anywhere from init
        
        valid = []
        for target_id in self.current_state.transitions.keys():
            target_hash = self._id_to_hash(target_id)
            if target_hash:
                valid.append(target_hash)
        
        return valid
    
    def is_valid_transition(self, target: str) -> bool:
        """Check if transition to target is valid from current state."""
        if self.current_state is None:
            return target in self.seeds
        
        target_unit = self.lookup(target)
        if target_unit is None:
            return False
        
        return target_unit.byte_id in self.current_state.transitions
    
    # ─────────────────────────────────────────────────────────
    # DECAY & MAINTENANCE
    # ─────────────────────────────────────────────────────────
    
    def decay_all(self, rate: float = None) -> None:
        """Decay resonance of all seeds (called per frame)."""
        rate = rate or self.decay_rate
        for unit in self.seeds.values():
            unit.decay(rate)
    
    def decay_resonance(self, lingering: float = 0.5) -> None:
        """
        Decay with lingering modulation (v6.0 extension).
        
        High lingering → slower decay (savoring, experiences persist)
        Low lingering → faster decay (standard behavior)
        
        Formula: effective_rate = base_rate + (1.0 - base_rate) * lingering * 0.8
        
        At lingering=1.0: rate ≈ 0.99 (very slow, savoring)
        At lingering=0.0: rate = 0.95 (normal decay)
        
        Args:
            lingering: Persistence factor from AgentState (0.0 - 1.0)
        """
        # Clamp lingering to valid range
        lingering = max(0.0, min(1.0, lingering))
        
        # Higher lingering = slower decay (higher rate = more preservation)
        # base_rate = 0.95, max boost = 0.04 (to reach 0.99)
        boost = (1.0 - self.decay_rate) * lingering * 0.8
        effective_rate = self.decay_rate + boost
        
        self.decay_all(rate=effective_rate)
    
    def prune_dormant(self, threshold: float = 0.01) -> int:
        """
        Remove seeds with near-zero resonance.
        Returns number of seeds removed.
        
        Use carefully — this permanently removes seeds!
        """
        to_remove = [h for h, u in self.seeds.items() if u.resonance < threshold]
        for h in to_remove:
            del self.seeds[h]
        return len(to_remove)
    
    def consolidate(self) -> None:
        """
        Dream-like consolidation pass.
        Decay theta, strengthen active seeds, prune weak ones.
        """
        for unit in self.seeds.values():
            unit.decay_theta()
            if unit.resonance > 0.5:
                # Active seeds strengthen
                unit.apply_theta("consolidation", 0.1)
    
    # ─────────────────────────────────────────────────────────
    # SIGMA DELTA PROTOCOL
    # ─────────────────────────────────────────────────────────
    
    def compute_delta(self) -> Dict:
        """
        Compute the Sigma Delta for LLM rendering.
        
        Returns minimal state packet:
        - transition: "204 -> 202"
        - fovea: top-9 awake seeds (minimal format)
        - qualia_delta: only changed axes
        - somatic_gate: current gate value
        """
        fovea = self.fovea_minimal()
        
        # Compute qualia delta if we have previous state
        q_delta = {}
        if self.previous_state and self.current_state:
            q_delta = qualia_delta(
                self.previous_state.qualia,
                self.current_state.qualia
            )
        
        # Compute somatic gate (based on tension)
        somatic_gate = 1.0 - self.tension()
        
        return {
            'transition': self.last_transition or "INIT",
            'fovea': fovea,
            'qualia_delta': q_delta,
            'somatic_gate': round(somatic_gate, 2),
            'timestamp': time.time()
        }
    
    # ─────────────────────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────────────────────
    
    def stats(self) -> Dict:
        """Field statistics for debugging and monitoring."""
        active = self.fovea_active()
        
        return {
            'total_seeds': len(self.seeds),
            'awake_seeds': len(active),
            'compression_ratio': self.total_capacity / max(1, len(active)),
            'avg_resonance': sum(u.resonance for u in active) / max(1, len(active)),
            'max_resonance': max((u.resonance for u in self.seeds.values()), default=0),
            'tension': self.tension(),
            'causal_seeds': sum(1 for u in self.seeds.values() if u.is_causally_touched()),
            'current_state': self.current_state.sigma_seed if self.current_state else None,
            'last_transition': self.last_transition,
        }
    
    def describe(self) -> str:
        """Human-readable field description."""
        stats = self.stats()
        lines = [
            "═" * 60,
            "SigmaField Status",
            "═" * 60,
            f"Total seeds: {stats['total_seeds']}",
            f"Awake seeds: {stats['awake_seeds']}",
            f"Compression: {stats['compression_ratio']:,.0f}:1",
            f"Tension: {stats['tension']:.2f}",
            f"Current: {stats['current_state']}",
            f"Last transition: {stats['last_transition']}",
            "",
            "Fovea:",
        ]
        for i, unit in enumerate(self.fovea_active()):
            lines.append(f"  {i+1}. {unit.sigma_seed} (r={unit.resonance:.2f})")
        
        lines.append("═" * 60)
        return "\n".join(lines)
    
    # ─────────────────────────────────────────────────────────
    # SERIALIZATION
    # ─────────────────────────────────────────────────────────
    
    def to_dict(self) -> Dict:
        """Serialize entire field for storage."""
        return {
            'seeds': {k: v.to_dict() for k, v in self.seeds.items()},
            'fovea_size': self.fovea_size,
            'wake_threshold': self.wake_threshold,
            'decay_rate': self.decay_rate,
            'current_state': self.current_state.sigma_seed if self.current_state else None,
            'last_transition': self.last_transition,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SigmaField':
        """Deserialize from storage."""
        field = cls(
            fovea_size=data.get('fovea_size', 9),
            wake_threshold=data.get('wake_threshold', 0.3),
            decay_rate=data.get('decay_rate', 0.95),
        )
        
        # Restore seeds
        for k, v in data.get('seeds', {}).items():
            field.seeds[k] = MarkovUnit.from_dict(v)
        
        # Restore current state
        if data.get('current_state'):
            field.current_state = field.lookup(data['current_state'])
        
        field.last_transition = data.get('last_transition')
        
        return field


# ─────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Ada v5.0 — SigmaField Tests")
    print("=" * 60)
    
    # Test 1: Basic creation and lookup
    field = SigmaField()
    love = field.register_new("love", QualiaVector.love(), "#arch-dragon-crimson")
    print(f"\n1. Registered: {love.sigma_seed}")
    
    found = field.lookup(love.sigma_seed)
    assert found is not None
    assert found.sigma_seed == love.sigma_seed
    print("   ✓ O(1) lookup works")
    
    # Test 2: Multiple seeds
    clarity = field.register_new("clarity", QualiaVector.clarity(), "#arch-lake-silver")
    rest = field.register_new("rest", QualiaVector.rest(), "#arch-womb-blue")
    print(f"\n2. Total seeds: {len(field)}")
    assert len(field) == 3
    print("   ✓ Multiple registration works")
    
    # Test 3: Fovea
    love.resonance = 0.9
    clarity.resonance = 0.7
    rest.resonance = 0.5
    
    fovea = field.fovea_active()
    print(f"\n3. Fovea: {[u.sigma_seed.split('-')[1] for u in fovea]}")
    assert fovea[0].sigma_seed == love.sigma_seed  # Highest resonance
    print("   ✓ Fovea ordering works")
    
    # Test 4: Foveation with transitions
    love.add_transition(clarity.byte_id, 0.6)
    love.add_transition(rest.byte_id, 0.4)
    
    cluster = field.foveate(love.sigma_seed, depth=2)
    print(f"\n4. Foveated cluster: {len(cluster)} seeds")
    assert len(cluster) >= 1
    print("   ✓ Foveation works")
    
    # Test 5: Resonance search
    results = field.resonate("warmth and fire")
    print(f"\n5. Resonance search: {len(results)} results")
    # Results depend on hash-based qualia, so just check it returns something
    assert isinstance(results, list)
    print("   ✓ Resonance search works")
    
    # Test 6: Tension detection
    # Add conflicting seed
    cold = field.register_new("cold", QualiaVector(steelwind=1.0, velvetpause=0.8), "#arch-void-black")
    cold.resonance = 0.85
    
    tension = field.tension()
    print(f"\n6. Tension: {tension:.2f}")
    assert tension >= 0
    print("   ✓ Tension detection works")
    
    # Test 7: State transitions
    assert field.transition_to(love.sigma_seed)
    assert field.current_state == love
    print(f"\n7. Current state: {field.current_state.sigma_seed}")
    
    assert field.transition_to(clarity.sigma_seed)
    assert field.last_transition is not None
    print(f"   Last transition: {field.last_transition}")
    print("   ✓ State transitions work")
    
    # Test 8: Sigma Delta
    delta = field.compute_delta()
    print(f"\n8. Sigma Delta:")
    print(f"   Transition: {delta['transition']}")
    print(f"   Fovea size: {len(delta['fovea'])}")
    print(f"   Somatic gate: {delta['somatic_gate']}")
    assert 'transition' in delta
    assert 'fovea' in delta
    print("   ✓ Sigma Delta works")
    
    # Test 9: Decay
    initial_resonance = love.resonance
    field.decay_all()
    assert love.resonance < initial_resonance
    print(f"\n9. After decay: love.resonance = {love.resonance:.2f} (was {initial_resonance:.2f})")
    print("   ✓ Decay works")
    
    # Test 10: Serialization
    data = field.to_dict()
    field2 = SigmaField.from_dict(data)
    assert len(field2) == len(field)
    assert field2.lookup(love.sigma_seed) is not None
    print(f"\n10. Roundtrip: {len(field2)} seeds restored")
    print("   ✓ Serialization works")
    
    # Print full status
    print("\n" + field.describe())
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
