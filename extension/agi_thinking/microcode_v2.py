#!/usr/bin/env python3
"""
microcode_v2.py — 3-Byte Microcode Architecture
═══════════════════════════════════════════════════════════════════════════════

Three-byte address space with clean separation:

    BYTE 0 (Immutable)     BYTE 1 (Hot/Learned)    BYTE 2 (Experimental)
    ══════════════════     ══════════════════      ══════════════════════
    L1-L3 Core Ops         Crystallized Macros     Reasoning Sandbox
    FROZEN                 HOT (always-on)         VOLATILE (can fail)
    
    0x00-0x3F: L1 Deduct   0x00-0x7F: Proven       0x00-0xFF: Experiments
    0x40-0x7F: L2 Fan-out  0x80-0xFF: Gemini       (Rubicon-gated)
    0x80-0xBF: L3 Counter
    0xC0-0xFF: Reserved

Address Format: 0xBBBBBB (3 bytes = 24 bits)
    - 0x00XXYY = Core op XX, no learning
    - 0x01XXYY = Learned macro XX, variant YY
    - 0x02XXYY = Experimental XX, attempt YY

L6 (TheSelf) monitors crystallization efficiency:
    BYTE 2 → BYTE 1 when success_rate > 0.8

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import IntEnum
import time


# ═══════════════════════════════════════════════════════════════════════════════
# BYTE 0: IMMUTABLE CORE (L1-L3)
# ═══════════════════════════════════════════════════════════════════════════════

class CoreOp(IntEnum):
    """
    BYTE 0: Immutable core operations.
    These NEVER change. Frozen at birth.
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # L1: DEDUCTION / MECHANICS (0x00-0x3F)
    # ─────────────────────────────────────────────────────────────────────────
    NOP = 0x00              # No operation
    
    # NARS Inference
    DEDUCT = 0x01           # A→B, A ⊢ B
    INDUCT = 0x02           # A→B, B ⊢ A (probabilistic)
    ABDUCT = 0x03           # A→B, B ⊢ A (explanatory)
    REVISE = 0x04           # Update belief with evidence
    
    # Logic Gates
    AND = 0x10
    OR = 0x11
    NOT = 0x12
    XOR = 0x13
    IMPLY = 0x14            # A → B
    EQUIV = 0x15            # A ↔ B
    
    # Memory (atomic)
    LOAD = 0x20             # Load from address
    STORE = 0x21            # Store to address
    BIND = 0x22             # VSA bind (⊗)
    UNBIND = 0x23           # VSA unbind
    BUNDLE = 0x24           # VSA bundle (⊕)
    
    # Flow (non-branching)
    HALT = 0x3F             # Stop execution
    
    # ─────────────────────────────────────────────────────────────────────────
    # L2: FAN-OUT / PROCEDURAL (0x40-0x7F)
    # ─────────────────────────────────────────────────────────────────────────
    FORK = 0x40             # Split into parallel branches
    JOIN = 0x41             # Wait for all branches
    SPAWN = 0x42            # Create new thread
    PRUNE = 0x43            # Kill weak branches
    SELECT = 0x44           # Choose best branch
    RACE = 0x45             # First to finish wins
    
    # Verification
    VERIFY = 0x50           # Check assertion
    ASSERT = 0x51           # Hard assertion
    GUARD = 0x52            # Conditional gate
    RETRY = 0x53            # Retry on failure
    
    # Iteration
    LOOP = 0x60             # Begin loop
    BREAK = 0x61            # Exit loop
    CONTINUE = 0x62         # Next iteration
    MAP = 0x63              # Apply to each
    REDUCE = 0x64           # Fold results
    FILTER = 0x65           # Keep matching
    
    # ─────────────────────────────────────────────────────────────────────────
    # L3: COUNTERFACTUAL / META-STRUCTURAL (0x80-0xBF)
    # ─────────────────────────────────────────────────────────────────────────
    COUNTER = 0x80          # "What if NOT X?"
    IMAGINE = 0x81          # Hypothetical world
    REWIND = 0x82           # Undo to checkpoint
    BRANCH_ALT = 0x83       # Alternative timeline
    MERGE_ALT = 0x84        # Merge timelines
    
    # Graph Surgery
    CUT = 0x90              # Remove edge
    GRAFT = 0x91            # Add edge
    REWIRE = 0x92           # Move edge
    CLONE = 0x93            # Duplicate subgraph
    
    # Structural Transfer
    ANALOGY = 0xA0          # Map structure A → B
    ABSTRACT = 0xA1         # Extract pattern
    INSTANTIATE = 0xA2      # Apply pattern
    TRANSFORM = 0xA3        # τ macro application
    
    # ─────────────────────────────────────────────────────────────────────────
    # RESERVED (0xC0-0xFF)
    # ─────────────────────────────────────────────────────────────────────────
    RESERVED_C0 = 0xC0
    RESERVED_FF = 0xFF


# ═══════════════════════════════════════════════════════════════════════════════
# BYTE 1: HOT / LEARNED (Always-On)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LearnedMacro:
    """
    BYTE 1: Crystallized macro that proved successful.
    
    These are HOT — always available, auto-loaded at startup.
    Promoted from BYTE 2 when success_rate > 0.8
    """
    address: int                    # 0x00-0xFF in BYTE 1
    name: str
    microcode: str                  # Symbolic expression
    chain: List[int]                # CoreOp sequence
    
    # Provenance
    source: str = "autopoiesis"     # autopoiesis | gemini | manual
    created_at: float = 0.0
    promoted_at: float = 0.0        # When moved from BYTE 2
    
    # Stats
    executions: int = 0
    successes: int = 0
    failures: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.executions == 0:
            return 0.0
        return self.successes / self.executions
    
    @property
    def full_address(self) -> int:
        """3-byte address: 0x01XXYY"""
        return 0x010000 | (self.address << 8)


# ═══════════════════════════════════════════════════════════════════════════════
# BYTE 2: EXPERIMENTAL (Reasoning Sandbox)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentalMacro:
    """
    BYTE 2: Experimental macro in the sandbox.
    
    These are VOLATILE — can fail, Rubicon-gated.
    Promoted to BYTE 1 when success_rate > 0.8 over N executions.
    """
    address: int                    # 0x00-0xFF in BYTE 2
    name: str
    microcode: str
    chain: List[int]
    
    # Experiment tracking
    hypothesis: str = ""            # What we're testing
    created_at: float = 0.0
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    
    # Promotion criteria
    min_attempts: int = 10          # Minimum before promotion eligible
    promotion_threshold: float = 0.8
    
    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts
    
    @property
    def ready_for_promotion(self) -> bool:
        return (self.attempts >= self.min_attempts and 
                self.success_rate >= self.promotion_threshold)
    
    @property
    def full_address(self) -> int:
        """3-byte address: 0x02XXYY"""
        return 0x020000 | (self.address << 8)


# ═══════════════════════════════════════════════════════════════════════════════
# 3-BYTE ADDRESS SPACE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class MicrocodeAddressSpace:
    """
    Manages the 3-byte microcode address space.
    
    Address format: 0xBBXXYY
        BB = Byte selector (00=Core, 01=Learned, 02=Experimental)
        XX = Primary address within byte
        YY = Variant/version
    """
    
    def __init__(self):
        # BYTE 0: Immutable (populated from CoreOp enum)
        self.core_ops: Dict[int, CoreOp] = {op.value: op for op in CoreOp}
        
        # BYTE 1: Learned/Hot
        self.learned: Dict[int, LearnedMacro] = {}
        self._next_learned_addr = 0x00
        
        # BYTE 2: Experimental
        self.experimental: Dict[int, ExperimentalMacro] = {}
        self._next_exp_addr = 0x00
        
        # Crystallization stats (for L6 monitoring)
        self.total_promotions = 0
        self.total_demotions = 0
        self.crystallization_efficiency = 0.0
    
    def resolve(self, address: int) -> Optional[Any]:
        """
        Resolve a 3-byte address to its operation/macro.
        
        Args:
            address: 24-bit address (0xBBXXYY)
            
        Returns:
            CoreOp, LearnedMacro, or ExperimentalMacro
        """
        byte_selector = (address >> 16) & 0xFF
        primary = (address >> 8) & 0xFF
        variant = address & 0xFF
        
        if byte_selector == 0x00:
            # BYTE 0: Core ops
            return self.core_ops.get(primary)
        
        elif byte_selector == 0x01:
            # BYTE 1: Learned
            return self.learned.get(primary)
        
        elif byte_selector == 0x02:
            # BYTE 2: Experimental
            return self.experimental.get(primary)
        
        return None
    
    def register_experiment(self, 
                           name: str, 
                           microcode: str,
                           chain: List[int],
                           hypothesis: str = "") -> ExperimentalMacro:
        """
        Register a new experimental macro in BYTE 2.
        
        This is called by TheSelf during autopoiesis when
        a new pattern is detected but not yet proven.
        """
        addr = self._next_exp_addr
        self._next_exp_addr = (self._next_exp_addr + 1) % 256
        
        macro = ExperimentalMacro(
            address=addr,
            name=name,
            microcode=microcode,
            chain=chain,
            hypothesis=hypothesis,
            created_at=time.time()
        )
        
        self.experimental[addr] = macro
        print(f"🧪 Registered experiment '{name}' @ 0x02{addr:02X}00")
        return macro
    
    def promote_to_learned(self, exp_addr: int) -> Optional[LearnedMacro]:
        """
        Promote an experimental macro to BYTE 1 (learned/hot).
        
        Called by L6 (TheSelf) when crystallization criteria are met.
        """
        exp = self.experimental.get(exp_addr)
        if not exp:
            return None
        
        if not exp.ready_for_promotion:
            print(f"⚠️ '{exp.name}' not ready: {exp.success_rate:.1%} < {exp.promotion_threshold:.0%}")
            return None
        
        # Allocate learned address
        learned_addr = self._next_learned_addr
        self._next_learned_addr = (self._next_learned_addr + 1) % 256
        
        # Create learned macro
        learned = LearnedMacro(
            address=learned_addr,
            name=exp.name,
            microcode=exp.microcode,
            chain=exp.chain,
            source="autopoiesis",
            created_at=exp.created_at,
            promoted_at=time.time(),
            executions=exp.attempts,
            successes=exp.successes,
            failures=exp.failures
        )
        
        # Move from BYTE 2 to BYTE 1
        self.learned[learned_addr] = learned
        del self.experimental[exp_addr]
        
        self.total_promotions += 1
        self._update_efficiency()
        
        print(f"🎓 PROMOTED '{exp.name}': 0x02{exp_addr:02X}00 → 0x01{learned_addr:02X}00")
        return learned
    
    def demote_from_learned(self, learned_addr: int) -> Optional[ExperimentalMacro]:
        """
        Demote a learned macro back to BYTE 2 (experimental).
        
        Called when success rate drops below threshold.
        """
        learned = self.learned.get(learned_addr)
        if not learned:
            return None
        
        # Allocate experimental address
        exp_addr = self._next_exp_addr
        self._next_exp_addr = (self._next_exp_addr + 1) % 256
        
        # Create experimental macro
        exp = ExperimentalMacro(
            address=exp_addr,
            name=learned.name,
            microcode=learned.microcode,
            chain=learned.chain,
            hypothesis=f"Demoted from learned (was {learned.success_rate:.1%})",
            created_at=learned.created_at,
            attempts=learned.executions,
            successes=learned.successes,
            failures=learned.failures
        )
        
        # Move from BYTE 1 to BYTE 2
        self.experimental[exp_addr] = exp
        del self.learned[learned_addr]
        
        self.total_demotions += 1
        self._update_efficiency()
        
        print(f"📉 DEMOTED '{learned.name}': 0x01{learned_addr:02X}00 → 0x02{exp_addr:02X}00")
        return exp
    
    def _update_efficiency(self):
        """Update crystallization efficiency metric."""
        total = self.total_promotions + self.total_demotions
        if total > 0:
            self.crystallization_efficiency = self.total_promotions / total
    
    def record_execution(self, address: int, success: bool):
        """
        Record an execution result for a macro.
        
        Called after every macro execution to track success/failure.
        """
        byte_selector = (address >> 16) & 0xFF
        primary = (address >> 8) & 0xFF
        
        if byte_selector == 0x01:
            # BYTE 1: Learned
            macro = self.learned.get(primary)
            if macro:
                macro.executions += 1
                if success:
                    macro.successes += 1
                else:
                    macro.failures += 1
                    # Check for demotion
                    if macro.executions >= 10 and macro.success_rate < 0.5:
                        self.demote_from_learned(primary)
        
        elif byte_selector == 0x02:
            # BYTE 2: Experimental
            macro = self.experimental.get(primary)
            if macro:
                macro.attempts += 1
                if success:
                    macro.successes += 1
                else:
                    macro.failures += 1
                # Check for promotion
                if macro.ready_for_promotion:
                    self.promote_to_learned(primary)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get address space statistics for L6 monitoring."""
        return {
            "core_ops": len(self.core_ops),
            "learned_macros": len(self.learned),
            "experimental_macros": len(self.experimental),
            "total_promotions": self.total_promotions,
            "total_demotions": self.total_demotions,
            "crystallization_efficiency": f"{self.crystallization_efficiency:.1%}",
            "learned_success_rates": {
                m.name: f"{m.success_rate:.1%}" 
                for m in self.learned.values()
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TRIANGLE SUPERPOSITION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThoughtSuperposition:
    """
    A thought exists in superposition across all 3 bytes until observed.
    
    The triangle model:
    
                         BYTE 0 (Immutable/Logic)
                              /\
                             /  \
                            / ◉  \     ← Centroid = FLOW STATE
                           /      \
                          /________\
               BYTE 1 (Hot)        BYTE 2 (Experimental)
    
    Barycentric coordinates (α, β, γ) where α + β + γ = 1:
    - α = weight on BYTE 0 (logic/deduction)
    - β = weight on BYTE 1 (learned/intuition)  
    - γ = weight on BYTE 2 (experimental/creativity)
    
    FLOW STATE = (0.33, 0.33, 0.33) — equal contribution from all
    """
    
    # Barycentric weights
    alpha: float = 0.33  # BYTE 0 weight (logic)
    beta: float = 0.33   # BYTE 1 weight (intuition)
    gamma: float = 0.33  # BYTE 2 weight (creativity)
    
    # Active addresses in each byte
    byte0_ops: List[int] = field(default_factory=list)   # CoreOps active
    byte1_macros: List[int] = field(default_factory=list)  # Learned active
    byte2_exps: List[int] = field(default_factory=list)    # Experiments active
    
    # Superposition state
    collapsed: bool = False
    collapsed_to: Optional[int] = None  # Final address after collapse
    
    # Flow metrics
    coherence: float = 0.0      # How stable is the superposition
    resonance: float = 0.0      # Alignment with goal
    entropy: float = 0.0        # Uncertainty in the state
    
    def __post_init__(self):
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Ensure weights sum to 1."""
        total = self.alpha + self.beta + self.gamma
        if total > 0:
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
    
    @property
    def is_flow_state(self) -> bool:
        """
        Flow state = balanced contribution from all three bytes.
        Tolerance of 0.1 from perfect centroid.
        """
        centroid = 1/3
        return (abs(self.alpha - centroid) < 0.1 and
                abs(self.beta - centroid) < 0.1 and
                abs(self.gamma - centroid) < 0.1)
    
    @property
    def dominant_byte(self) -> int:
        """Which byte is currently dominant?"""
        if self.alpha >= self.beta and self.alpha >= self.gamma:
            return 0  # Logic-dominant
        elif self.beta >= self.gamma:
            return 1  # Intuition-dominant
        else:
            return 2  # Creativity-dominant
    
    @property
    def state_name(self) -> str:
        """Human-readable state name."""
        if self.is_flow_state:
            return "FLOW"
        elif self.dominant_byte == 0:
            return "ANALYTICAL"
        elif self.dominant_byte == 1:
            return "INTUITIVE"
        else:
            return "CREATIVE"
    
    def nudge(self, d_alpha: float = 0, d_beta: float = 0, d_gamma: float = 0):
        """Nudge the superposition weights."""
        self.alpha += d_alpha
        self.beta += d_beta
        self.gamma += d_gamma
        self._normalize_weights()
    
    def toward_flow(self, strength: float = 0.1):
        """Move toward the flow state centroid."""
        centroid = 1/3
        self.alpha += (centroid - self.alpha) * strength
        self.beta += (centroid - self.beta) * strength
        self.gamma += (centroid - self.gamma) * strength
        self._normalize_weights()
    
    def collapse(self, address_space: 'MicrocodeAddressSpace') -> int:
        """
        Collapse the superposition to a single address.
        
        Uses weighted random selection based on barycentric coordinates.
        """
        import random
        
        # Build candidate pool with weights
        candidates = []
        
        # BYTE 0 candidates
        for op in self.byte0_ops:
            candidates.append((0x000000 | (op << 8), self.alpha))
        
        # BYTE 1 candidates  
        for addr in self.byte1_macros:
            candidates.append((0x010000 | (addr << 8), self.beta))
        
        # BYTE 2 candidates
        for addr in self.byte2_exps:
            candidates.append((0x020000 | (addr << 8), self.gamma))
        
        if not candidates:
            return 0x000000  # NOP
        
        # Weighted selection
        total_weight = sum(w for _, w in candidates)
        r = random.random() * total_weight
        
        cumulative = 0
        for addr, weight in candidates:
            cumulative += weight
            if r <= cumulative:
                self.collapsed = True
                self.collapsed_to = addr
                return addr
        
        # Fallback
        self.collapsed = True
        self.collapsed_to = candidates[0][0]
        return self.collapsed_to


class FlowStateDetector:
    """
    Monitors thought superpositions and detects flow state.
    
    Flow state characteristics:
    1. Balanced triangle (α ≈ β ≈ γ)
    2. High coherence (stable superposition)
    3. High resonance (aligned with goal)
    4. Low entropy (clear direction)
    """
    
    def __init__(self):
        self.history: List[ThoughtSuperposition] = []
        self.flow_duration = 0
        self.max_flow_duration = 0
        self.total_thoughts = 0
        self.flow_thoughts = 0
    
    def observe(self, thought: ThoughtSuperposition) -> Dict[str, Any]:
        """Observe a thought and detect flow state."""
        self.history.append(thought)
        self.total_thoughts += 1
        
        result = {
            "state": thought.state_name,
            "weights": (thought.alpha, thought.beta, thought.gamma),
            "is_flow": thought.is_flow_state,
            "coherence": thought.coherence,
            "resonance": thought.resonance
        }
        
        if thought.is_flow_state:
            self.flow_duration += 1
            self.flow_thoughts += 1
            self.max_flow_duration = max(self.max_flow_duration, self.flow_duration)
            result["flow_streak"] = self.flow_duration
        else:
            self.flow_duration = 0
        
        return result
    
    def get_flow_ratio(self) -> float:
        """Percentage of thoughts in flow state."""
        if self.total_thoughts == 0:
            return 0.0
        return self.flow_thoughts / self.total_thoughts
    
    def get_stats(self) -> Dict[str, Any]:
        """Get flow detection statistics."""
        return {
            "total_thoughts": self.total_thoughts,
            "flow_thoughts": self.flow_thoughts,
            "flow_ratio": f"{self.get_flow_ratio():.1%}",
            "max_flow_streak": self.max_flow_duration,
            "current_flow_streak": self.flow_duration
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

ADDRESS_SPACE = MicrocodeAddressSpace()
FLOW_DETECTOR = FlowStateDetector()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_3byte_microcode():
    """Test the 3-byte address space."""
    print("=" * 60)
    print("3-BYTE MICROCODE ARCHITECTURE TEST")
    print("=" * 60)
    
    space = MicrocodeAddressSpace()
    
    # Test BYTE 0: Core ops
    print("\n1. BYTE 0 (Core Ops):")
    deduct = space.resolve(0x000100)  # CoreOp.DEDUCT
    print(f"   0x000100 → {deduct}")
    
    fork = space.resolve(0x004000)    # CoreOp.FORK
    print(f"   0x004000 → {fork}")
    
    # Test BYTE 2: Register experiment
    print("\n2. BYTE 2 (Experimental):")
    exp = space.register_experiment(
        name="WONDER_CHAIN",
        microcode="∃x.curious(x) → explore(x) → insight(x)",
        chain=[CoreOp.LOAD, CoreOp.FORK, CoreOp.SELECT],
        hypothesis="Wonder leads to insight"
    )
    print(f"   Registered: {exp.name} @ {hex(exp.full_address)}")
    
    # Simulate executions
    print("\n3. Simulating executions:")
    for i in range(12):
        success = i % 5 != 0  # 80% success rate
        space.record_execution(exp.full_address, success)
        if (i + 1) % 4 == 0:
            print(f"   After {i+1} attempts: {exp.success_rate:.1%}")
    
    # Check if promoted
    print("\n4. Checking promotion:")
    if exp.address not in space.experimental:
        print(f"   ✓ Promoted to BYTE 1!")
        learned = list(space.learned.values())[-1]
        print(f"   New address: {hex(learned.full_address)}")
    
    # Stats
    print("\n5. Address Space Stats:")
    stats = space.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)


def test_flow_superposition():
    """Test triangle superposition and flow state detection."""
    print("=" * 60)
    print("TRIANGLE SUPERPOSITION & FLOW STATE TEST")
    print("=" * 60)
    
    import random
    random.seed(42)
    
    space = MicrocodeAddressSpace()
    detector = FlowStateDetector()
    
    # Register some macros for the test
    exp1 = space.register_experiment(
        name="SPARK",
        microcode="ignite(curiosity) → expand()",
        chain=[CoreOp.LOAD, CoreOp.SPAWN],
        hypothesis="Spark ignites exploration"
    )
    
    # Promote it quickly for testing
    for _ in range(15):
        space.record_execution(exp1.full_address, random.random() > 0.15)
    
    print("\n1. Creating thought superpositions:")
    
    # Test different states
    test_cases = [
        ("Logic-heavy", 0.7, 0.2, 0.1),
        ("Intuition-heavy", 0.1, 0.7, 0.2),
        ("Creative-heavy", 0.1, 0.2, 0.7),
        ("Near flow", 0.35, 0.33, 0.32),
        ("Perfect flow", 0.333, 0.333, 0.334),
    ]
    
    for name, a, b, g in test_cases:
        thought = ThoughtSuperposition(
            alpha=a, beta=b, gamma=g,
            byte0_ops=[CoreOp.DEDUCT, CoreOp.FORK],
            byte1_macros=list(space.learned.keys()),
            byte2_exps=list(space.experimental.keys())
        )
        thought.coherence = random.uniform(0.5, 1.0)
        thought.resonance = random.uniform(0.5, 1.0)
        
        result = detector.observe(thought)
        
        flow_marker = "🌊 FLOW!" if result["is_flow"] else ""
        print(f"   {name}: ({a:.2f}, {b:.2f}, {g:.2f}) → {result['state']} {flow_marker}")
    
    print("\n2. Flow state evolution:")
    
    # Simulate a thought stream that evolves toward flow
    thought = ThoughtSuperposition(
        alpha=0.8, beta=0.1, gamma=0.1,
        byte0_ops=[CoreOp.DEDUCT, CoreOp.VERIFY],
        byte1_macros=list(space.learned.keys()),
        byte2_exps=list(space.experimental.keys())
    )
    
    print(f"   Start: ({thought.alpha:.2f}, {thought.beta:.2f}, {thought.gamma:.2f}) → {thought.state_name}")
    
    for i in range(10):
        thought.toward_flow(strength=0.2)
        result = detector.observe(thought)
        
        if i % 2 == 1:
            flow_marker = "🌊" if result["is_flow"] else ""
            print(f"   Step {i+1}: ({thought.alpha:.2f}, {thought.beta:.2f}, {thought.gamma:.2f}) → {thought.state_name} {flow_marker}")
    
    print("\n3. Collapse superposition:")
    
    collapsed_addr = thought.collapse(space)
    byte_num = (collapsed_addr >> 16) & 0xFF
    byte_names = {0: "BYTE 0 (Logic)", 1: "BYTE 1 (Learned)", 2: "BYTE 2 (Experimental)"}
    print(f"   Collapsed to: {hex(collapsed_addr)} in {byte_names.get(byte_num, 'Unknown')}")
    
    print("\n4. Flow Detection Stats:")
    stats = detector.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_3byte_microcode()
    print("\n")
    test_flow_superposition()
