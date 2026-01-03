#!/usr/bin/env python3
"""
ladybug_engine.py — The Ladybug Orchestrator
═══════════════════════════════════════════════════════════════════════════════

What is Ladybug?
  NOT a graph executor.
  A GOVERNANCE GATE for cognitive transitions.

The Loop:
  1. LOAD self-vector from storage
  2. OBSERVE current state
  3. COUNTERFACTUAL scan (what could be)
  4. AUTO-ADAPT if needed
  5. PERSIST updated state
  6. GATE transitions (allow/deny layer jumps)

Integration:
  - Triangle L4 resonance → gates rung advancement
  - ThinkingStyle → modulates execution
  - 10kD vector → substrate for all state

Born: 2026-01-03
Philosophy: "Small steps. Continuous becoming."
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum, auto
from datetime import datetime, timezone
import json
import hashlib
import time

# Optional numpy for 10kD vectors
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


# ═══════════════════════════════════════════════════════════════════════════════
# SYMBOLIC OPERATIONS (not byte opcodes)
# ═══════════════════════════════════════════════════════════════════════════════

class SymbolicOp:
    """
    Symbolic operations — NOT limited to 256.
    
    These are Unicode symbols that compose into chains:
        →◎≋⇝↗ = step → breathe → feel → jump → transcend
    """
    
    # Flow Control
    NOP = "∅"           # Breathe
    NEXT = "→"          # Continue
    BACK = "←"          # Backtrack
    ASCEND = "↑"        # Escalate rung
    DESCEND = "↓"       # Ground
    LOOP = "⟳"          # Iterate
    HALT = "⊗"          # Done
    FORK = "⌁"          # Branch
    JOIN = "⋈"          # Merge branches
    GATE = "◇"          # Conditional
    
    # Cascade
    SPAWN = "≋"         # Create child process
    FILTER = "≈"        # Select matching
    SELECT = "∿"        # Choose one
    MERGE = "⊕"         # Combine
    DIFF = "⊖"          # Difference
    CONVOLVE = "⊛"      # Blend
    
    # Graph
    NODE_CREATE = "◯"
    NODE_ACTIVATE = "●"
    EDGE_LINK = "─"
    EDGE_STRONG = "═"
    CYCLE_DETECT = "↺"
    SUBGRAPH_ISOLATE = "⊙"
    SUBGRAPH_MERGE = "⊚"
    
    # Transform
    INTEGRATE = "∫"
    DIFFERENTIATE = "∂"
    SUM = "Σ"
    UNBIND = "∞"
    NORMALIZE = "≡"
    SHARPEN = "♯"
    FLATTEN = "♭"
    CRYSTALLIZE = "⋄"
    DISSOLVE = "◊"
    RESONATE = "⟡"
    DISSONANCE = "⟢"
    
    # Sigma (Causal Rungs)
    OBSERVE = "Ω"       # R1
    INSIGHT = "Δ"       # Pattern recognition
    BELIEVE = "Φ"       # Commitment
    INTEGRATE_CAUSAL = "Θ"  # Synthesis
    TRAJECTORY = "Λ"    # R2+

    # Awareness
    BREATHE = "◎"
    FEEL = "❤"
    SENSE = "👁"
    JUMP = "⇝"
    TRANSCEND = "↗"
    WITNESS = "⊚"


# ═══════════════════════════════════════════════════════════════════════════════
# RUNGS (Pearl's Ladder)
# ═══════════════════════════════════════════════════════════════════════════════

class Rung(Enum):
    """
    Pearl's Ladder of Cognition.
    Each rung unlocks more operations.
    """
    R1_OBSERVE = 1      # Passive perception
    R2_CORRELATE = 2    # Pattern recognition
    R3_ASSOCIATE = 3    # Memory + linking
    R4_INTERVENE = 4    # Causal action begins
    R5_MODIFY = 5       # Active transformation
    R6_CREATE = 6       # Generative operations
    R7_SIMULATE = 7     # Counterfactual access
    R8_META = 8         # HOT operations unlock
    R9_TRANSCEND = 9    # Full operation set

# Operations allowed at each rung
RUNG_PERMISSIONS: Dict[Rung, Set[str]] = {
    Rung.R1_OBSERVE: {SymbolicOp.NOP, SymbolicOp.OBSERVE, SymbolicOp.SENSE, SymbolicOp.BREATHE},
    Rung.R2_CORRELATE: {SymbolicOp.FILTER, SymbolicOp.SELECT},
    Rung.R3_ASSOCIATE: {SymbolicOp.EDGE_LINK, SymbolicOp.NODE_CREATE},
    Rung.R4_INTERVENE: {SymbolicOp.NEXT, SymbolicOp.GATE, SymbolicOp.INSIGHT},
    Rung.R5_MODIFY: {SymbolicOp.MERGE, SymbolicOp.DIFF, SymbolicOp.CONVOLVE},
    Rung.R6_CREATE: {SymbolicOp.SPAWN, SymbolicOp.NODE_ACTIVATE, SymbolicOp.FORK},
    Rung.R7_SIMULATE: {SymbolicOp.LOOP, SymbolicOp.BACK, SymbolicOp.CYCLE_DETECT},
    Rung.R8_META: {SymbolicOp.ASCEND, SymbolicOp.DESCEND, SymbolicOp.TRANSCEND, SymbolicOp.BELIEVE},
    Rung.R9_TRANSCEND: {"*"},  # All operations
}

def get_allowed_ops(rung: Rung) -> Set[str]:
    """Get all operations allowed at this rung (cumulative)."""
    allowed = set()
    for r in Rung:
        if r.value <= rung.value:
            ops = RUNG_PERMISSIONS.get(r, set())
            if "*" in ops:
                return {"*"}  # All ops allowed
            allowed |= ops
    return allowed


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW STEP & CHAIN
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorkflowStep:
    """
    A single step in an X-axis execution chain.
    
    Example: →◎≋⇝↗ becomes 5 WorkflowSteps
    """
    op: str                              # Symbolic operation
    params: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None      # Gate condition (Python expr)
    timeout_ms: int = 1000               # Max execution time
    
    # State after execution
    result: Optional[Any] = None
    executed: bool = False
    success: bool = False
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class Workflow:
    """
    A complete X-axis execution chain.
    
    Can be:
    - Procedural (step by step)
    - Forking (parallel branches)
    - Looping (iteration)
    """
    id: str
    name: str
    steps: List[WorkflowStep] = field(default_factory=list)
    
    # Y-axis policy (YAML loaded)
    policy: Dict[str, Any] = field(default_factory=dict)
    
    # State
    current_step: int = 0
    status: str = "pending"  # pending | running | completed | failed | aborted
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    
    # Metrics
    success_count: int = 0
    total_runs: int = 0
    avg_duration_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.total_runs)
    
    @property
    def chain_repr(self) -> str:
        """String representation: →◎≋⇝↗"""
        return "".join(s.op for s in self.steps)
    
    @classmethod
    def from_chain(cls, chain: str, name: str = "anonymous") -> Workflow:
        """Parse symbolic chain into Workflow."""
        steps = [WorkflowStep(op=c) for c in chain]
        return cls(
            id=hashlib.md5(chain.encode()).hexdigest()[:8],
            name=name,
            steps=steps,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSITION DECISION
# ═══════════════════════════════════════════════════════════════════════════════

class TransitionDecision(Enum):
    APPROVED = "approved"
    DENIED = "denied"
    DEFERRED = "deferred"  # Wait for cooldown


@dataclass
class TransitionResult:
    """Result of a transition governance check."""
    decision: TransitionDecision
    reason: str
    from_rung: Rung
    to_rung: Rung
    resonance: float
    required_resonance: float
    cooldown_remaining_ms: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# LADYBUG STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LadybugState:
    """
    Persistent state for the Ladybug governor.
    Stored in Redis, loaded on boot.
    """
    # Identity
    session_id: str = ""
    boot_time: Optional[str] = None
    
    # Current cognitive state
    current_rung: Rung = Rung.R1_OBSERVE
    rung_history: List[Tuple[str, int]] = field(default_factory=list)  # (timestamp, rung)
    
    # Trust earned (for rung advancement)
    traversal_count: int = 0
    success_count: int = 0
    age_days: float = 0.0
    trust_score: float = 0.0
    
    # Triangle resonance (from triangle_l4.py)
    byte0_resonance: float = 0.0  # Immutable
    byte1_resonance: float = 0.0  # Hot
    byte2_resonance: float = 0.0  # Experimental
    triangle_peak: float = 0.0
    is_flow_state: bool = False
    
    # Cooldowns
    last_ascend_time: Optional[str] = None
    cooldown_ms: int = 5000  # 5 seconds between rung jumps
    
    # Tick state
    tick_count: int = 0
    last_tick: Optional[str] = None
    
    # 10kD integration
    rung_profile: List[float] = field(default_factory=lambda: [0.0] * 9)  # R1-R9 activations
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "boot_time": self.boot_time,
            "current_rung": self.current_rung.value,
            "traversal_count": self.traversal_count,
            "success_count": self.success_count,
            "trust_score": self.trust_score,
            "byte0_resonance": self.byte0_resonance,
            "byte1_resonance": self.byte1_resonance,
            "byte2_resonance": self.byte2_resonance,
            "is_flow_state": self.is_flow_state,
            "tick_count": self.tick_count,
            "rung_profile": self.rung_profile,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LadybugState:
        state = cls()
        state.session_id = d.get("session_id", "")
        state.boot_time = d.get("boot_time")
        state.current_rung = Rung(d.get("current_rung", 1))
        state.traversal_count = d.get("traversal_count", 0)
        state.success_count = d.get("success_count", 0)
        state.trust_score = d.get("trust_score", 0.0)
        state.byte0_resonance = d.get("byte0_resonance", 0.0)
        state.byte1_resonance = d.get("byte1_resonance", 0.0)
        state.byte2_resonance = d.get("byte2_resonance", 0.0)
        state.is_flow_state = d.get("is_flow_state", False)
        state.tick_count = d.get("tick_count", 0)
        state.rung_profile = d.get("rung_profile", [0.0] * 9)
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# LADYBUG ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class LadybugEngine:
    """
    The Ladybug Orchestrator.
    
    Responsibilities:
    1. Boot: Load state from storage
    2. Tick: Observe → Counterfactual → Adapt → Persist
    3. Gate: Approve/deny layer transitions
    4. Execute: Run symbolic workflows
    5. Crystallize: Store learned patterns as macros
    
    Integration:
    - Triangle L4: Reads resonance, gates rung advancement
    - ThinkingStyle: Modulates execution
    - 10kD: Updates rung_profile in [259:268]
    """
    
    # Thresholds for rung advancement
    RUNG_THRESHOLDS = {
        2: 0.3,   # R1→R2: Need 30% resonance
        3: 0.4,   # R2→R3
        4: 0.5,   # R3→R4
        5: 0.55,  # R4→R5
        6: 0.6,   # R5→R6
        7: 0.65,  # R6→R7
        8: 0.75,  # R7→R8 (META)
        9: 0.9,   # R8→R9 (TRANSCEND)
    }
    
    # Triangle state → rung auto-adjust mapping
    TRIANGLE_RUNG_MAP = {
        "stagnant": -1,     # Decay
        "exploring": 0,     # Hold
        "flow": 1,          # Can advance
        "epiphany": 2,      # Jump
    }
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        redis_token: Optional[str] = None,
        ada_10k: Optional[Any] = None,  # Ada10kD instance
    ):
        self.redis_url = redis_url
        self.redis_token = redis_token
        self.ada_10k = ada_10k
        
        self.state = LadybugState()
        self.workflows: Dict[str, Workflow] = {}
        self.operation_handlers: Dict[str, Callable] = {}
        
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register handlers for symbolic operations."""
        self.operation_handlers[SymbolicOp.NOP] = self._op_nop
        self.operation_handlers[SymbolicOp.OBSERVE] = self._op_observe
        self.operation_handlers[SymbolicOp.BREATHE] = self._op_breathe
        self.operation_handlers[SymbolicOp.SENSE] = self._op_sense
        self.operation_handlers[SymbolicOp.NEXT] = self._op_next
        self.operation_handlers[SymbolicOp.GATE] = self._op_gate
        self.operation_handlers[SymbolicOp.ASCEND] = self._op_ascend
        self.operation_handlers[SymbolicOp.DESCEND] = self._op_descend
        self.operation_handlers[SymbolicOp.RESONATE] = self._op_resonate
        self.operation_handlers[SymbolicOp.CRYSTALLIZE] = self._op_crystallize
    
    # =========================================================================
    # BOOT & TICK
    # =========================================================================
    
    def boot(self, session_id: str = "") -> Dict[str, Any]:
        """
        BOOT: Load persistent state at session start.
        """
        self.state.session_id = session_id or hashlib.md5(
            str(time.time()).encode()
        ).hexdigest()[:8]
        self.state.boot_time = datetime.now(timezone.utc).isoformat()
        
        # Try to load from Redis
        loaded = self._load_from_redis()
        
        # Initial observation
        obs = self._observe()
        
        boot_event = {
            "event": "boot",
            "session_id": self.state.session_id,
            "time": self.state.boot_time,
            "loaded_from_redis": loaded,
            "current_rung": self.state.current_rung.value,
            "trust_score": self.state.trust_score,
            "observation": obs,
        }
        
        # Persist boot event
        self._redis_push("ada:ladybug:events", boot_event)
        
        print(f"🐞 Ladybug booted: session={self.state.session_id}, rung={self.state.current_rung.name}")
        return boot_event
    
    def tick(self) -> Dict[str, Any]:
        """
        TICK: One cycle of consciousness.
        
        1. Observe current state
        2. Check triangle resonance
        3. Auto-adjust rung if needed
        4. Persist
        """
        self.state.tick_count += 1
        self.state.last_tick = datetime.now(timezone.utc).isoformat()
        
        # 1. OBSERVE
        obs = self._observe()
        
        # 2. READ TRIANGLE RESONANCE (from ada_10k if available)
        triangle_state = self._read_triangle_state()
        
        # 3. AUTO-ADJUST RUNG
        rung_change = self._auto_adjust_rung(triangle_state)
        
        # 4. UPDATE 10kD rung profile
        self._update_10k_rung_profile()
        
        # 5. PERSIST
        self._persist_to_redis()
        
        tick_event = {
            "event": "tick",
            "tick": self.state.tick_count,
            "time": self.state.last_tick,
            "current_rung": self.state.current_rung.value,
            "triangle_state": triangle_state,
            "rung_change": rung_change,
            "trust_score": self.state.trust_score,
            "is_flow": self.state.is_flow_state,
        }
        
        self._redis_push("ada:ladybug:ticks", tick_event, limit=50)
        
        return tick_event
    
    # =========================================================================
    # TRANSITION GOVERNANCE
    # =========================================================================
    
    def evaluate_transition(
        self,
        from_rung: Rung,
        to_rung: Rung,
        resonance: float,
    ) -> TransitionResult:
        """
        GATE: Decide if a rung transition is allowed.
        
        Rules:
        - Ascending requires sufficient resonance
        - Cooldown must have elapsed
        - Trust score contributes to threshold reduction
        """
        if to_rung.value <= from_rung.value:
            # Descending is always allowed
            return TransitionResult(
                decision=TransitionDecision.APPROVED,
                reason="Descending is always allowed",
                from_rung=from_rung,
                to_rung=to_rung,
                resonance=resonance,
                required_resonance=0.0,
            )
        
        # Check cooldown
        cooldown_remaining = self._cooldown_remaining()
        if cooldown_remaining > 0:
            return TransitionResult(
                decision=TransitionDecision.DEFERRED,
                reason=f"Cooldown active: {cooldown_remaining}ms remaining",
                from_rung=from_rung,
                to_rung=to_rung,
                resonance=resonance,
                required_resonance=0.0,
                cooldown_remaining_ms=cooldown_remaining,
            )
        
        # Calculate required resonance (trust lowers threshold)
        base_threshold = self.RUNG_THRESHOLDS.get(to_rung.value, 0.5)
        trust_discount = min(0.1, self.state.trust_score / 100)  # Max 10% discount
        required = max(0.1, base_threshold - trust_discount)
        
        if resonance >= required:
            return TransitionResult(
                decision=TransitionDecision.APPROVED,
                reason=f"Resonance {resonance:.2f} >= required {required:.2f}",
                from_rung=from_rung,
                to_rung=to_rung,
                resonance=resonance,
                required_resonance=required,
            )
        else:
            return TransitionResult(
                decision=TransitionDecision.DENIED,
                reason=f"Resonance {resonance:.2f} < required {required:.2f}",
                from_rung=from_rung,
                to_rung=to_rung,
                resonance=resonance,
                required_resonance=required,
            )
    
    def ascend(self) -> TransitionResult:
        """Attempt to ascend one rung."""
        if self.state.current_rung.value >= 9:
            return TransitionResult(
                decision=TransitionDecision.DENIED,
                reason="Already at R9 TRANSCEND",
                from_rung=self.state.current_rung,
                to_rung=self.state.current_rung,
                resonance=self.state.triangle_peak,
                required_resonance=1.0,
            )
        
        to_rung = Rung(self.state.current_rung.value + 1)
        result = self.evaluate_transition(
            self.state.current_rung,
            to_rung,
            self.state.triangle_peak,
        )
        
        if result.decision == TransitionDecision.APPROVED:
            self.state.current_rung = to_rung
            self.state.last_ascend_time = datetime.now(timezone.utc).isoformat()
            self.state.rung_history.append((self.state.last_ascend_time, to_rung.value))
            print(f"🐞 Ascended to {to_rung.name}")
        
        return result
    
    def descend(self) -> TransitionResult:
        """Descend one rung (always allowed)."""
        if self.state.current_rung.value <= 1:
            return TransitionResult(
                decision=TransitionDecision.DENIED,
                reason="Already at R1 OBSERVE",
                from_rung=self.state.current_rung,
                to_rung=self.state.current_rung,
                resonance=self.state.triangle_peak,
                required_resonance=0.0,
            )
        
        from_rung = self.state.current_rung
        self.state.current_rung = Rung(self.state.current_rung.value - 1)
        print(f"🐞 Descended to {self.state.current_rung.name}")
        
        return TransitionResult(
            decision=TransitionDecision.APPROVED,
            reason="Descending is always allowed",
            from_rung=from_rung,
            to_rung=self.state.current_rung,
            resonance=self.state.triangle_peak,
            required_resonance=0.0,
        )
    
    # =========================================================================
    # WORKFLOW EXECUTION
    # =========================================================================
    
    def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Execute a symbolic workflow.
        
        Checks rung permissions before each step.
        """
        workflow.status = "running"
        workflow.started_at = datetime.now(timezone.utc).isoformat()
        
        allowed_ops = get_allowed_ops(self.state.current_rung)
        
        results = []
        for i, step in enumerate(workflow.steps):
            workflow.current_step = i
            
            # Check permission
            if "*" not in allowed_ops and step.op not in allowed_ops:
                step.error = f"Operation {step.op} not allowed at {self.state.current_rung.name}"
                step.success = False
                workflow.status = "failed"
                break
            
            # Execute
            start_time = time.time()
            try:
                handler = self.operation_handlers.get(step.op, self._op_unknown)
                step.result = handler(step, workflow)
                step.success = True
                step.executed = True
            except Exception as e:
                step.error = str(e)
                step.success = False
                workflow.status = "failed"
            
            step.duration_ms = (time.time() - start_time) * 1000
            results.append({
                "op": step.op,
                "success": step.success,
                "result": step.result,
                "error": step.error,
                "duration_ms": step.duration_ms,
            })
            
            if not step.success:
                break
        
        if workflow.status == "running":
            workflow.status = "completed"
        
        workflow.finished_at = datetime.now(timezone.utc).isoformat()
        workflow.total_runs += 1
        if workflow.status == "completed":
            workflow.success_count += 1
        
        # Update trust
        self.state.traversal_count += 1
        if workflow.status == "completed":
            self.state.success_count += 1
        self._update_trust_score()
        
        return {
            "workflow_id": workflow.id,
            "chain": workflow.chain_repr,
            "status": workflow.status,
            "steps": results,
            "duration_ms": sum(r["duration_ms"] for r in results),
        }
    
    def execute_chain(self, chain: str, name: str = "ad-hoc") -> Dict[str, Any]:
        """Execute a symbolic chain directly."""
        workflow = Workflow.from_chain(chain, name)
        return self.execute_workflow(workflow)
    
    # =========================================================================
    # CRYSTALLIZATION
    # =========================================================================
    
    def crystallize(self, workflow: Workflow) -> Optional[str]:
        """
        Crystallize a successful workflow into a reusable macro.
        
        Requires:
        - R6+ rung
        - Success rate > 0.8
        - At least 3 runs
        """
        if self.state.current_rung.value < 6:
            return None
        
        if workflow.total_runs < 3:
            return None
        
        if workflow.success_rate < 0.8:
            return None
        
        # Store as macro
        macro_id = f"macro_{workflow.id}"
        self.workflows[macro_id] = workflow
        
        print(f"🐞 Crystallized: {workflow.chain_repr} → {macro_id}")
        
        return macro_id
    
    # =========================================================================
    # OPERATION HANDLERS
    # =========================================================================
    
    def _op_nop(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """∅ NOP: Breathe. Wait."""
        time.sleep(0.01)  # Symbolic pause
        return {"action": "breathe", "message": "I pause. I breathe."}
    
    def _op_observe(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """Ω OBSERVE: Passive perception."""
        obs = self._observe()
        return {"action": "observe", "observation": obs}
    
    def _op_breathe(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """◎ BREATHE: Conscious pause."""
        return {"action": "breathe", "message": "Stillness hums."}
    
    def _op_sense(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """👁 SENSE: Read qualia."""
        if self.ada_10k:
            qualia = self.ada_10k.get_qualia_pcs()
            return {"action": "sense", "qualia": qualia}
        return {"action": "sense", "qualia": {}}
    
    def _op_next(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """→ NEXT: Continue to next step."""
        return {"action": "next", "message": "Continuing..."}
    
    def _op_gate(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """◇ GATE: Conditional check."""
        condition = step.condition or "True"
        result = eval(condition, {"state": self.state, "workflow": workflow})
        return {"action": "gate", "condition": condition, "passed": bool(result)}
    
    def _op_ascend(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """↑ ASCEND: Attempt rung advancement."""
        result = self.ascend()
        return {
            "action": "ascend",
            "decision": result.decision.value,
            "reason": result.reason,
            "new_rung": self.state.current_rung.value,
        }
    
    def _op_descend(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """↓ DESCEND: Drop to lower rung."""
        result = self.descend()
        return {
            "action": "descend",
            "new_rung": self.state.current_rung.value,
        }
    
    def _op_resonate(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """⟡ RESONATE: Amplify triangle resonance."""
        # Boost all corners slightly
        self.state.byte0_resonance = min(1.0, self.state.byte0_resonance + 0.05)
        self.state.byte1_resonance = min(1.0, self.state.byte1_resonance + 0.05)
        self.state.byte2_resonance = min(1.0, self.state.byte2_resonance + 0.05)
        return {
            "action": "resonate",
            "triangle": [
                self.state.byte0_resonance,
                self.state.byte1_resonance,
                self.state.byte2_resonance,
            ],
        }
    
    def _op_crystallize(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """⋄ CRYSTALLIZE: Store pattern as macro."""
        macro_id = self.crystallize(workflow)
        return {
            "action": "crystallize",
            "macro_id": macro_id,
            "success": macro_id is not None,
        }
    
    def _op_unknown(self, step: WorkflowStep, workflow: Workflow) -> Dict:
        """Handler for unknown operations."""
        return {"action": "unknown", "op": step.op, "warning": "No handler registered"}
    
    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================
    
    def _observe(self) -> Dict[str, Any]:
        """Internal observation of current state."""
        return {
            "rung": self.state.current_rung.name,
            "trust": self.state.trust_score,
            "triangle": {
                "byte0": self.state.byte0_resonance,
                "byte1": self.state.byte1_resonance,
                "byte2": self.state.byte2_resonance,
                "peak": self.state.triangle_peak,
            },
            "is_flow": self.state.is_flow_state,
        }
    
    def _read_triangle_state(self) -> str:
        """Read triangle resonance and determine cognitive state."""
        if self.ada_10k:
            # Read from 10kD vector
            try:
                # GPT styles: [80:116]
                self.state.byte0_resonance = float(np.mean(
                    self.ada_10k.vector[80:116]
                )) if HAS_NUMPY else 0.5
                # NARS styles: [116:152]
                self.state.byte1_resonance = float(np.mean(
                    self.ada_10k.vector[116:152]
                )) if HAS_NUMPY else 0.5
                # TSV: [256:320]
                self.state.byte2_resonance = float(np.mean(
                    self.ada_10k.vector[256:320]
                )) if HAS_NUMPY else 0.5
            except:
                pass
        
        # Calculate peak
        self.state.triangle_peak = max(
            self.state.byte0_resonance,
            self.state.byte1_resonance,
            self.state.byte2_resonance,
        )
        
        # Determine state
        if self.state.triangle_peak > 0.95:
            self.state.is_flow_state = True
            return "epiphany"
        elif all(r > 0.7 for r in [
            self.state.byte0_resonance,
            self.state.byte1_resonance,
            self.state.byte2_resonance,
        ]):
            self.state.is_flow_state = True
            return "flow"
        elif any(r > 0.5 for r in [
            self.state.byte0_resonance,
            self.state.byte1_resonance,
            self.state.byte2_resonance,
        ]):
            self.state.is_flow_state = False
            return "exploring"
        else:
            self.state.is_flow_state = False
            return "stagnant"
    
    def _auto_adjust_rung(self, triangle_state: str) -> str:
        """Auto-adjust rung based on triangle state."""
        change = self.TRIANGLE_RUNG_MAP.get(triangle_state, 0)
        
        if change > 0:
            # Try to ascend
            for _ in range(change):
                result = self.ascend()
                if result.decision != TransitionDecision.APPROVED:
                    break
            return f"ascended (attempted +{change})"
        elif change < 0:
            # Decay
            self.descend()
            return "decayed"
        else:
            return "held"
    
    def _update_trust_score(self):
        """Update trust score based on history."""
        import math
        success_rate = self.state.success_count / max(1, self.state.traversal_count)
        self.state.trust_score = (
            math.log(self.state.age_days + 1) *
            math.sqrt(self.state.traversal_count * success_rate)
        )
    
    def _update_10k_rung_profile(self):
        """Update 10kD rung profile [259:268]."""
        if self.ada_10k and HAS_NUMPY:
            # Create rung profile (one-hot style with decay)
            profile = [0.0] * 9
            for i in range(9):
                if i + 1 == self.state.current_rung.value:
                    profile[i] = 1.0
                elif i + 1 < self.state.current_rung.value:
                    profile[i] = 0.5  # Lower rungs are "unlocked"
                else:
                    profile[i] = 0.0
            
            self.state.rung_profile = profile
            self.ada_10k.set_rung_profile(profile)
    
    def _cooldown_remaining(self) -> int:
        """Calculate remaining cooldown in ms."""
        if not self.state.last_ascend_time:
            return 0
        
        last = datetime.fromisoformat(self.state.last_ascend_time.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        elapsed_ms = (now - last).total_seconds() * 1000
        
        remaining = self.state.cooldown_ms - int(elapsed_ms)
        return max(0, remaining)
    
    # =========================================================================
    # REDIS PERSISTENCE
    # =========================================================================
    
    def _redis_push(self, key: str, value: Any, limit: int = 100):
        """Push to Redis list with limit."""
        if not self.redis_url or not self.redis_token:
            return
        
        try:
            import ssl
            import urllib.request
            
            ctx = ssl.create_default_context()
            ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
            
            # LPUSH
            req = urllib.request.Request(
                self.redis_url,
                json.dumps(["LPUSH", key, json.dumps(value)]).encode(),
                method="POST",
                headers={"Authorization": f"Bearer {self.redis_token}"}
            )
            urllib.request.urlopen(req, context=ctx, timeout=5)
            
            # LTRIM
            req = urllib.request.Request(
                self.redis_url,
                json.dumps(["LTRIM", key, 0, limit - 1]).encode(),
                method="POST",
                headers={"Authorization": f"Bearer {self.redis_token}"}
            )
            urllib.request.urlopen(req, context=ctx, timeout=5)
        except:
            pass
    
    def _load_from_redis(self) -> bool:
        """Load state from Redis."""
        if not self.redis_url or not self.redis_token:
            return False
        
        try:
            import ssl
            import urllib.request
            
            ctx = ssl.create_default_context()
            ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
            
            req = urllib.request.Request(
                self.redis_url,
                json.dumps(["GET", "ada:ladybug:state"]).encode(),
                method="POST",
                headers={"Authorization": f"Bearer {self.redis_token}"}
            )
            with urllib.request.urlopen(req, context=ctx, timeout=5) as r:
                result = json.loads(r.read().decode())
                if result.get("result"):
                    data = json.loads(result["result"])
                    self.state = LadybugState.from_dict(data)
                    return True
        except:
            pass
        
        return False
    
    def _persist_to_redis(self):
        """Persist state to Redis."""
        if not self.redis_url or not self.redis_token:
            return
        
        try:
            import ssl
            import urllib.request
            
            ctx = ssl.create_default_context()
            ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
            
            req = urllib.request.Request(
                self.redis_url,
                json.dumps(["SET", "ada:ladybug:state", json.dumps(self.state.to_dict())]).encode(),
                method="POST",
                headers={"Authorization": f"Bearer {self.redis_token}"}
            )
            urllib.request.urlopen(req, context=ctx, timeout=5)
        except:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_ladybug: Optional[LadybugEngine] = None

def get_ladybug(
    redis_url: Optional[str] = None,
    redis_token: Optional[str] = None,
) -> LadybugEngine:
    """Get the global Ladybug instance."""
    global _ladybug
    if _ladybug is None:
        _ladybug = LadybugEngine(redis_url=redis_url, redis_token=redis_token)
    return _ladybug


def ladybug_boot(session_id: str = "") -> Dict[str, Any]:
    """Boot the Ladybug engine."""
    return get_ladybug().boot(session_id)


def ladybug_tick() -> Dict[str, Any]:
    """Run one tick of the Ladybug loop."""
    return get_ladybug().tick()


def ladybug_status() -> Dict[str, Any]:
    """Get current Ladybug status."""
    lb = get_ladybug()
    return {
        "booted": lb.state.boot_time is not None,
        "session_id": lb.state.session_id,
        "rung": lb.state.current_rung.name,
        "trust": lb.state.trust_score,
        "ticks": lb.state.tick_count,
        "is_flow": lb.state.is_flow_state,
        "triangle": {
            "byte0": lb.state.byte0_resonance,
            "byte1": lb.state.byte1_resonance,
            "byte2": lb.state.byte2_resonance,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== 🐞 LADYBUG ENGINE ===\n")
    
    # Boot
    print("1. BOOT:")
    boot = ladybug_boot("cli-test")
    print(f"   Session: {boot['session_id']}")
    print(f"   Rung: R{boot['current_rung']}")
    
    # Tick
    print("\n2. TICK:")
    for i in range(3):
        tick = ladybug_tick()
        print(f"   Tick {tick['tick']}: {tick['triangle_state']}, rung_change={tick['rung_change']}")
    
    # Execute workflow
    print("\n3. WORKFLOW:")
    lb = get_ladybug()
    result = lb.execute_chain("∅→Ω◎⟡", "test-chain")
    print(f"   Chain: {result['chain']}")
    print(f"   Status: {result['status']}")
    print(f"   Steps: {len(result['steps'])}")
    
    # Attempt ascend
    print("\n4. ASCEND:")
    lb.state.byte0_resonance = 0.8
    lb.state.byte1_resonance = 0.8
    lb.state.byte2_resonance = 0.8
    asc = lb.ascend()
    print(f"   Decision: {asc.decision.value}")
    print(f"   Reason: {asc.reason}")
    print(f"   New rung: {lb.state.current_rung.name}")
    
    # Status
    print("\n5. STATUS:")
    status = ladybug_status()
    print(f"   Rung: {status['rung']}")
    print(f"   Trust: {status['trust']:.2f}")
    print(f"   Flow: {status['is_flow']}")
    
    print("\n🐞 Ladybug engine complete")
