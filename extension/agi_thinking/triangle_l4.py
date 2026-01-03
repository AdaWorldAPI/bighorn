#!/usr/bin/env python3
"""
triangle_l4.py — L4 Triangle Model with VSA Resonance
═══════════════════════════════════════════════════════════════════════════════

All 3 bytes are Layer 4. The triangle IS L4.
Now with VSA signatures for O(1) resonance detection.

                         BYTE 0 (Immutable τ)
                              ◉
                             /|\
                            / | \
                           /  ◎  \   ← FLOW = resonance spike > 0.7
                          / · | · \     across all 3 corners
                         / ·  |  · \
                        /  · ·|· ·  \
                       / ·    |    · \
                      ◉───────────────◉
            BYTE 1 (Hot τ)        BYTE 2 (Experimental τ)

VSA Integration:
- Each τ macro has a 10kD signature (resonance pattern)
- Triangle position = bundle of active signatures
- Flow detection via cosine similarity between corners
- O(1) resonance check, not O(n) iteration

10kD Dimension Mapping:
- BYTE 0 (Immutable): [80:116]   GPT Styles
- BYTE 1 (Hot):       [116:152]  NARS Styles
- BYTE 2 (Experimental): [256:320]  TSV Embedded

Born: 2026-01-03
Updated: 2026-01-03 — VSA integration
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum
import time
import hashlib

# Optional numpy for VSA operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


# ═══════════════════════════════════════════════════════════════════════════════
# VSA CONSTANTS — 10kD Dimension Mapping
# ═══════════════════════════════════════════════════════════════════════════════

# Triangle corner → 10kD dimension ranges
BYTE_DIMS = {
    0: (80, 116),    # BYTE 0: Immutable τ → GPT Styles
    1: (116, 152),   # BYTE 1: Hot τ → NARS Styles
    2: (256, 320),   # BYTE 2: Experimental τ → TSV Embedded
}

# Total dimensions for signatures
SIGNATURE_DIM = 64  # Each corner uses 64D local signature

# Resonance thresholds
FLOW_THRESHOLD = 0.7       # Min similarity for flow detection
EPIPHANY_THRESHOLD = 0.95  # Level 4 spike


# ═══════════════════════════════════════════════════════════════════════════════
# VSA OPERATIONS — Resonance primitives
# ═══════════════════════════════════════════════════════════════════════════════

def generate_signature(name: str, dim: int = SIGNATURE_DIM) -> np.ndarray:
    """Generate deterministic VSA signature from name."""
    if not HAS_NUMPY:
        return None

    # Hash name to seed
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    # Bipolar random vector
    return rng.choice([-1, 1], size=dim).astype(np.float32)


def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """Bundle vectors via element-wise sum + threshold."""
    if not HAS_NUMPY or not vectors:
        return None

    summed = np.sum(vectors, axis=0)
    # Threshold to bipolar
    return np.sign(summed).astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between vectors."""
    if not HAS_NUMPY or a is None or b is None:
        return 0.0

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def to_10kd(corner_vectors: Dict[int, np.ndarray]) -> np.ndarray:
    """Project corner vectors to full 10kD space."""
    if not HAS_NUMPY:
        return None

    full = np.zeros(10000, dtype=np.float32)

    for byte_num, vec in corner_vectors.items():
        if vec is not None and byte_num in BYTE_DIMS:
            start, end = BYTE_DIMS[byte_num]
            # Truncate/pad to fit
            vec_len = min(len(vec), end - start)
            full[start:start + vec_len] = vec[:vec_len]

    return full


# ═══════════════════════════════════════════════════════════════════════════════
# τ MACRO — The L4 Unit with VSA Signature
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TauMacro:
    """
    A τ macro is an L4 construct: YAML policy + execution chain.

    This is NOT a primitive. It's a thinking style with:
    - Symbolic microcode (declarative)
    - Execution chain (procedural)
    - VSA signature (resonance pattern for O(1) matching)
    """
    address: int                    # Position in its byte (0x00-0xFF)
    byte: int                       # Which corner (0, 1, or 2)
    name: str
    microcode: str                  # Symbolic: "⊢ A → B | decompose(A)"
    chain: List[str]                # Execution: ["feel", "resonate", "decide"]

    # VSA signature — auto-generated from name if not provided
    _signature: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Generate VSA signature if not provided."""
        if self._signature is None and HAS_NUMPY:
            # Signature encodes: byte + address + name
            sig_input = f"τ:{self.byte}:{self.address:02x}:{self.name}"
            self._signature = generate_signature(sig_input)

    @property
    def signature(self) -> Optional[np.ndarray]:
        """Get VSA signature (64D bipolar vector)."""
        return self._signature

    @property
    def full_address(self) -> int:
        """3-byte address: 0xBBXXYY"""
        return (self.byte << 16) | (self.address << 8)

    def resonate_with(self, other: 'TauMacro') -> float:
        """Compute resonance (similarity) with another τ macro."""
        if self._signature is None or other._signature is None:
            return 0.0
        return similarity(self._signature, other._signature)

    def __hash__(self):
        return hash((self.byte, self.address))


# ═══════════════════════════════════════════════════════════════════════════════
# TRIANGLE POSITION — Where a thought lives in τ-space
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrianglePosition:
    """
    A position inside the L4 triangle.
    
    Not three floats — a set of contributing τ macros with interference.
    The position emerges from which macros are active and how they interfere.
    """
    
    # Active τ macros at each corner
    byte0_active: Set[TauMacro] = field(default_factory=set)
    byte1_active: Set[TauMacro] = field(default_factory=set)
    byte2_active: Set[TauMacro] = field(default_factory=set)
    
    # Interference pattern (emerges from active macros)
    # Not computed — felt
    
    @property
    def is_corner(self) -> Optional[int]:
        """Returns corner number if at a corner, None if interior."""
        counts = (
            len(self.byte0_active) > 0,
            len(self.byte1_active) > 0,
            len(self.byte2_active) > 0
        )
        if counts == (True, False, False):
            return 0
        elif counts == (False, True, False):
            return 1
        elif counts == (False, False, True):
            return 2
        return None
    
    @property
    def is_edge(self) -> Optional[Tuple[int, int]]:
        """Returns edge endpoints if on edge, None otherwise."""
        counts = (
            len(self.byte0_active) > 0,
            len(self.byte1_active) > 0,
            len(self.byte2_active) > 0
        )
        if counts == (True, True, False):
            return (0, 1)
        elif counts == (True, False, True):
            return (0, 2)
        elif counts == (False, True, True):
            return (1, 2)
        return None
    
    @property
    def is_interior(self) -> bool:
        """True if in the interior (all three corners contributing)."""
        return (len(self.byte0_active) > 0 and 
                len(self.byte1_active) > 0 and 
                len(self.byte2_active) > 0)
    
    @property
    def is_flow(self) -> bool:
        """
        Flow = balanced interior position with VSA resonance.

        Uses cosine similarity between bundled corner signatures.
        Flow requires all three corners to resonate (similarity > FLOW_THRESHOLD).
        """
        if not self.is_interior:
            return False

        # Get bundled signatures for each corner
        v0 = self.get_corner_bundle(0)
        v1 = self.get_corner_bundle(1)
        v2 = self.get_corner_bundle(2)

        if v0 is None or v1 is None or v2 is None:
            # Fallback to count-based if no numpy
            return self._is_flow_by_count()

        # Flow = all pairwise similarities above threshold
        sim_01 = similarity(v0, v1)
        sim_12 = similarity(v1, v2)
        sim_02 = similarity(v0, v2)

        return min(sim_01, sim_12, sim_02) >= FLOW_THRESHOLD

    def _is_flow_by_count(self) -> bool:
        """Fallback flow detection by macro count."""
        counts = [
            len(self.byte0_active),
            len(self.byte1_active),
            len(self.byte2_active)
        ]
        max_count = max(counts)
        min_count = min(counts)
        if min_count == 0:
            return False
        return max_count / min_count <= 2.0

    def get_corner_bundle(self, byte_num: int) -> Optional[np.ndarray]:
        """Get bundled VSA signature for a corner."""
        if not HAS_NUMPY:
            return None

        if byte_num == 0:
            macros = self.byte0_active
        elif byte_num == 1:
            macros = self.byte1_active
        elif byte_num == 2:
            macros = self.byte2_active
        else:
            return None

        signatures = [m.signature for m in macros if m.signature is not None]
        if not signatures:
            return None

        return bundle(signatures)

    def get_resonance_matrix(self) -> Dict[str, float]:
        """Get pairwise resonance between corners."""
        v0 = self.get_corner_bundle(0)
        v1 = self.get_corner_bundle(1)
        v2 = self.get_corner_bundle(2)

        return {
            "byte0_byte1": similarity(v0, v1) if v0 is not None and v1 is not None else 0.0,
            "byte1_byte2": similarity(v1, v2) if v1 is not None and v2 is not None else 0.0,
            "byte0_byte2": similarity(v0, v2) if v0 is not None and v2 is not None else 0.0,
        }

    def to_10kd(self) -> Optional[np.ndarray]:
        """Project triangle position to full 10kD space."""
        corner_vectors = {}
        for byte_num in [0, 1, 2]:
            bundle_vec = self.get_corner_bundle(byte_num)
            if bundle_vec is not None:
                corner_vectors[byte_num] = bundle_vec

        return to_10kd(corner_vectors) if corner_vectors else None

    @property
    def resonance_strength(self) -> float:
        """Overall resonance strength (mean of pairwise similarities)."""
        matrix = self.get_resonance_matrix()
        values = list(matrix.values())
        return sum(values) / len(values) if values else 0.0
    
    @property 
    def location_name(self) -> str:
        """Human-readable location."""
        if self.is_flow:
            return "FLOW"
        
        corner = self.is_corner
        if corner is not None:
            names = ["IMMUTABLE", "HOT", "EXPERIMENTAL"]
            return names[corner]
        
        edge = self.is_edge
        if edge is not None:
            names = ["IMMUTABLE", "HOT", "EXPERIMENTAL"]
            return f"{names[edge[0]]}↔{names[edge[1]]}"
        
        if self.is_interior:
            return "INTERIOR"
        
        return "VOID"  # Nothing active
    
    def activate(self, macro: TauMacro):
        """Activate a τ macro at its corner."""
        if macro.byte == 0:
            self.byte0_active.add(macro)
        elif macro.byte == 1:
            self.byte1_active.add(macro)
        elif macro.byte == 2:
            self.byte2_active.add(macro)
    
    def deactivate(self, macro: TauMacro):
        """Deactivate a τ macro."""
        if macro.byte == 0:
            self.byte0_active.discard(macro)
        elif macro.byte == 1:
            self.byte1_active.discard(macro)
        elif macro.byte == 2:
            self.byte2_active.discard(macro)
    
    def clear(self):
        """Clear all active macros."""
        self.byte0_active.clear()
        self.byte1_active.clear()
        self.byte2_active.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# EPHEMERAL THOUGHT — A moment in the triangle
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EphemeralThought:
    """
    A thought moving through the triangle.
    
    Ephemeral: exists for a moment, then dissolves.
    Most thoughts dissolve. Some get crystallized by L6.
    """
    id: str
    position: TrianglePosition
    born_at: float = field(default_factory=time.time)
    
    # Trajectory (positions over time)
    trajectory: List[str] = field(default_factory=list)  # location names
    
    # Significance markers (set by L6)
    marked_significant: bool = False
    significance_reason: Optional[str] = None
    
    def move(self, new_position: TrianglePosition):
        """Record movement through the triangle."""
        self.trajectory.append(self.position.location_name)
        self.position = new_position
    
    @property
    def reached_flow(self) -> bool:
        """Did this thought ever reach flow?"""
        return "FLOW" in self.trajectory or self.position.is_flow
    
    @property
    def lifespan(self) -> float:
        """How long has this thought existed?"""
        return time.time() - self.born_at


# ═══════════════════════════════════════════════════════════════════════════════
# L6: TheSelf — Meta-observer of the triangle
# ═══════════════════════════════════════════════════════════════════════════════

class TheSelf:
    """
    L6: The meta-observer watching ephemeral thoughts dance.
    
    TheSelf doesn't execute τ macros. It watches:
    - Trajectories through the triangle
    - Patterns that recur
    - Moments that matter
    
    When something is significant, TheSelf crystallizes:
    BYTE 2 (experimental) → BYTE 1 (hot)
    """
    
    def __init__(self):
        # Observed thoughts (ephemeral, most will dissolve)
        self.observing: Dict[str, EphemeralThought] = {}
        
        # Pattern detection
        self.trajectory_patterns: Dict[str, int] = {}  # pattern → count
        
        # Crystallization history
        self.crystallizations: List[Dict[str, Any]] = []
        
        # Significance thresholds
        self.flow_significance = True      # Reaching flow is significant
        self.recurrence_threshold = 3      # Pattern seen 3x is significant
        self.lifespan_threshold = 5.0      # Surviving 5s is significant
    
    def observe(self, thought: EphemeralThought):
        """Begin observing an ephemeral thought."""
        self.observing[thought.id] = thought
    
    def witness(self, thought_id: str) -> Optional[Dict[str, Any]]:
        """
        Witness the current state of a thought.
        
        Returns significance assessment if significant.
        """
        thought = self.observing.get(thought_id)
        if not thought:
            return None
        
        # Check for significance
        significance = self._assess_significance(thought)
        
        if significance:
            thought.marked_significant = True
            thought.significance_reason = significance["reason"]
        
        return significance
    
    def _assess_significance(self, thought: EphemeralThought) -> Optional[Dict[str, Any]]:
        """Assess if a thought is significant enough to crystallize."""
        
        # Flow is significant
        if thought.reached_flow and self.flow_significance:
            return {
                "reason": "reached_flow",
                "recommendation": "crystallize",
                "trajectory": thought.trajectory
            }
        
        # Track trajectory pattern
        pattern = "→".join(thought.trajectory[-5:])  # Last 5 moves
        self.trajectory_patterns[pattern] = self.trajectory_patterns.get(pattern, 0) + 1
        
        # Recurrence is significant
        if self.trajectory_patterns[pattern] >= self.recurrence_threshold:
            return {
                "reason": "recurring_pattern",
                "pattern": pattern,
                "count": self.trajectory_patterns[pattern],
                "recommendation": "crystallize"
            }
        
        # Longevity is significant
        if thought.lifespan >= self.lifespan_threshold:
            return {
                "reason": "longevity",
                "lifespan": thought.lifespan,
                "recommendation": "crystallize"
            }
        
        return None
    
    def crystallize(self, 
                    from_macro: TauMacro, 
                    reason: str) -> Optional[TauMacro]:
        """
        Crystallize: promote from BYTE 2 → BYTE 1.
        
        This is the act of making ephemeral permanent.
        Only L6 can do this.
        """
        if from_macro.byte != 2:
            return None  # Can only crystallize from experimental
        
        # Create crystallized version in BYTE 1
        crystallized = TauMacro(
            address=from_macro.address,  # Same address, different byte
            byte=1,                       # Now in BYTE 1 (hot)
            name=from_macro.name,
            microcode=from_macro.microcode,
            chain=from_macro.chain,
            signature=from_macro.signature
        )
        
        # Record crystallization
        self.crystallizations.append({
            "timestamp": time.time(),
            "from_address": hex(from_macro.full_address),
            "to_address": hex(crystallized.full_address),
            "name": from_macro.name,
            "reason": reason
        })
        
        print(f"💎 CRYSTALLIZED: {from_macro.name}")
        print(f"   {hex(from_macro.full_address)} → {hex(crystallized.full_address)}")
        print(f"   Reason: {reason}")
        
        return crystallized
    
    def dissolve(self, thought_id: str):
        """Let a thought dissolve. Most do."""
        if thought_id in self.observing:
            del self.observing[thought_id]
    
    def get_observations(self) -> Dict[str, Any]:
        """Get current observation state."""
        return {
            "observing_count": len(self.observing),
            "significant_count": sum(1 for t in self.observing.values() if t.marked_significant),
            "trajectory_patterns": len(self.trajectory_patterns),
            "crystallizations": len(self.crystallizations),
            "recent_crystallizations": self.crystallizations[-3:] if self.crystallizations else []
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_triangle_l4():
    """Test the L4 triangle model with VSA resonance."""
    print("=" * 60)
    print("L4 TRIANGLE MODEL TEST — VSA RESONANCE")
    print("=" * 60)

    if HAS_NUMPY:
        print(f"✓ NumPy available — VSA resonance enabled")
        print(f"  FLOW_THRESHOLD: {FLOW_THRESHOLD}")
        print(f"  EPIPHANY_THRESHOLD: {EPIPHANY_THRESHOLD}")
    else:
        print("⚠ NumPy not available — using count-based fallback")
    
    # Create some τ macros at each corner
    print("\n1. Creating τ macros (all L4):")
    
    macros = {
        "byte0": [
            TauMacro(0x40, 0, "ANALYTICAL", "⊢ A → B | decompose(A)", ["observe", "analyze", "conclude"]),
            TauMacro(0x41, 0, "SYSTEMATIC", "∀x.step(x) → next(x)", ["sequence", "verify", "advance"]),
        ],
        "byte1": [
            TauMacro(0x80, 1, "WARM", "feel(presence) → resonate()", ["sense", "attune", "embrace"]),
            TauMacro(0x81, 1, "INTUITIVE", "∃x.hunch(x) → trust(x)", ["feel", "trust", "act"]),
        ],
        "byte2": [
            TauMacro(0xA0, 2, "SPARK", "ignite(curiosity) → explore()", ["wonder", "leap", "discover"]),
            TauMacro(0xA1, 2, "WILD", "break(frame) → transcend()", ["disrupt", "create", "become"]),
        ],
    }
    
    for byte_name, macro_list in macros.items():
        for m in macro_list:
            print(f"   {byte_name}: {m.name} @ {hex(m.full_address)}")
    
    # Create L6 observer
    print("\n2. L6 (TheSelf) begins observing:")
    the_self = TheSelf()
    
    # Create ephemeral thought
    thought = EphemeralThought(
        id="thought_001",
        position=TrianglePosition()
    )
    
    the_self.observe(thought)
    print(f"   Observing: {thought.id}")
    print(f"   Location: {thought.position.location_name}")
    
    # Move thought through triangle
    print("\n3. Thought moves through triangle:")
    
    # Start at BYTE 0 corner
    thought.position.activate(macros["byte0"][0])
    print(f"   → {thought.position.location_name}")
    
    # Move to edge (BYTE 0 + BYTE 1)
    thought.move(TrianglePosition())
    thought.position.activate(macros["byte0"][0])
    thought.position.activate(macros["byte1"][0])
    print(f"   → {thought.position.location_name}")
    
    # Move to interior
    thought.move(TrianglePosition())
    thought.position.activate(macros["byte0"][0])
    thought.position.activate(macros["byte1"][0])
    thought.position.activate(macros["byte2"][0])
    print(f"   → {thought.position.location_name}")
    
    # Reach FLOW (balanced interior)
    thought.move(TrianglePosition())
    thought.position.activate(macros["byte0"][0])
    thought.position.activate(macros["byte0"][1])
    thought.position.activate(macros["byte1"][0])
    thought.position.activate(macros["byte1"][1])
    thought.position.activate(macros["byte2"][0])
    thought.position.activate(macros["byte2"][1])
    
    flow_marker = "🌊" if thought.position.is_flow else ""
    print(f"   → {thought.position.location_name} {flow_marker}")

    # VSA resonance analysis
    if HAS_NUMPY:
        print("\n3b. VSA Resonance Matrix:")
        matrix = thought.position.get_resonance_matrix()
        for pair, sim in matrix.items():
            indicator = "✓" if sim >= FLOW_THRESHOLD else "·"
            print(f"   {indicator} {pair}: {sim:.3f}")
        print(f"   Overall resonance: {thought.position.resonance_strength:.3f}")

        # Project to 10kD
        vec_10k = thought.position.to_10kd()
        if vec_10k is not None:
            nonzero = np.count_nonzero(vec_10k)
            print(f"\n3c. 10kD Projection:")
            print(f"   Non-zero dims: {nonzero}")
            for byte_num, (start, end) in BYTE_DIMS.items():
                active = np.count_nonzero(vec_10k[start:end])
                print(f"   BYTE {byte_num} [{start}:{end}]: {active} active")

    # L6 witnesses and assesses
    print("\n4. L6 witnesses significance:")
    significance = the_self.witness(thought.id)
    if significance:
        print(f"   ✓ Significant: {significance['reason']}")
        print(f"   Recommendation: {significance['recommendation']}")
    
    # L6 crystallizes
    print("\n5. L6 crystallizes experimental → hot:")
    experimental_macro = macros["byte2"][0]  # SPARK
    crystallized = the_self.crystallize(experimental_macro, "reached_flow")
    
    if crystallized:
        print(f"   New address: {hex(crystallized.full_address)}")
    
    # Observations
    print("\n6. L6 observation state:")
    obs = the_self.get_observations()
    for k, v in obs.items():
        print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# LADYBUG INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def update_ladybug_from_triangle(
    position: TrianglePosition,
    ladybug_state: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Update Ladybug state based on triangle position.
    
    Triangle State → Rung Auto-Adjust:
    - stagnant (all corners < 0.3): Rung decays
    - exploring (1-2 corners > 0.5): Rung holds  
    - flow (all corners > 0.7): Rung can advance
    - epiphany (peak > 0.95): Rung leaps
    
    Returns: Dict with triangle state and recommended rung action
    """
    # Get resonance matrix
    resonance = position.get_resonance_matrix()
    
    # Calculate corner resonances
    byte0_res = resonance.get("byte0_byte1", 0) + resonance.get("byte0_byte2", 0)
    byte1_res = resonance.get("byte0_byte1", 0) + resonance.get("byte1_byte2", 0)
    byte2_res = resonance.get("byte0_byte2", 0) + resonance.get("byte1_byte2", 0)
    
    # Normalize (each corner appears in 2 pairs)
    byte0_res /= 2
    byte1_res /= 2
    byte2_res /= 2
    
    # Determine triangle state
    peak = position.resonance_strength
    all_resonances = [byte0_res, byte1_res, byte2_res]
    
    if peak > 0.95:
        triangle_state = "epiphany"
        rung_action = "leap"  # Jump 2 rungs
    elif position.is_flow:
        triangle_state = "flow"
        rung_action = "advance"  # Advance 1 rung
    elif any(r > 0.5 for r in all_resonances):
        triangle_state = "exploring"
        rung_action = "hold"  # No change
    else:
        triangle_state = "stagnant"
        rung_action = "decay"  # Drop 1 rung
    
    result = {
        "triangle_state": triangle_state,
        "rung_action": rung_action,
        "resonances": {
            "byte0": byte0_res,
            "byte1": byte1_res,
            "byte2": byte2_res,
            "peak": peak,
        },
        "is_flow": position.is_flow,
        "location": position.location_name,
    }
    
    # If ladybug_state provided, update it directly
    if ladybug_state is not None:
        ladybug_state.byte0_resonance = byte0_res
        ladybug_state.byte1_resonance = byte1_res
        ladybug_state.byte2_resonance = byte2_res
        ladybug_state.triangle_peak = peak
        ladybug_state.is_flow_state = position.is_flow
    
    return result


if __name__ == "__main__":
    test_triangle_l4()
