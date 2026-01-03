#!/usr/bin/env python3
"""
triangle_l4.py — L4 Triangle Model
═══════════════════════════════════════════════════════════════════════════════

All 3 bytes are Layer 4. The triangle IS L4.

                         BYTE 0 (Immutable τ)
                              ◉
                             /\
                            /  \
                           / ·  \
                          / · ·  \   ← Ephemeral thoughts
                         / ·  ◉·  \     dancing in the triangle
                        /  · · ·   \
                       / ·   · ·    \
                      ◉──────────────◉
            BYTE 1 (Hot τ)        BYTE 2 (Experimental τ)

Each corner = τ macro space (YAML policy + execution chain)
Interior = superposition of τ contributions
Centroid = FLOW state

L6 (TheSelf) monitors from above:
- Watches ephemeral thought-trajectories
- Identifies significant patterns
- Crystallizes: BYTE 2 → BYTE 1 when worthy

No primitives. Position in τ-space, not floats.

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import time


# ═══════════════════════════════════════════════════════════════════════════════
# τ MACRO — The L4 Unit
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TauMacro:
    """
    A τ macro is an L4 construct: YAML policy + execution chain.
    
    This is NOT a primitive. It's a thinking style with:
    - Symbolic microcode (declarative)
    - Execution chain (procedural)
    - Resonance signature (felt sense)
    """
    address: int                    # Position in its byte (0x00-0xFF)
    byte: int                       # Which corner (0, 1, or 2)
    name: str
    microcode: str                  # Symbolic: "⊢ A → B | decompose(A)"
    chain: List[str]                # Execution: ["feel", "resonate", "decide"]
    
    # Resonance signature (not a float — a pattern)
    signature: Optional[bytes] = None  # VSA-derived, for matching
    
    @property
    def full_address(self) -> int:
        """3-byte address: 0xBBXXYY"""
        return (self.byte << 16) | (self.address << 8)
    
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
        Flow = balanced interior position.
        
        Not measured by floats — measured by whether all three
        corners contribute with similar intensity (macro count as proxy).
        """
        if not self.is_interior:
            return False
        
        counts = [
            len(self.byte0_active),
            len(self.byte1_active),
            len(self.byte2_active)
        ]
        
        # Flow = no single corner dominates by more than 2x
        max_count = max(counts)
        min_count = min(counts)
        
        if min_count == 0:
            return False
        
        return max_count / min_count <= 2.0
    
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
    """Test the L4 triangle model."""
    print("=" * 60)
    print("L4 TRIANGLE MODEL TEST")
    print("=" * 60)
    
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


if __name__ == "__main__":
    test_triangle_l4()
