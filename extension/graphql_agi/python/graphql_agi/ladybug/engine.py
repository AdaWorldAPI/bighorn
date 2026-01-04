"""
LadybugEngine — Cognitive Governance for 9-Rung System
═══════════════════════════════════════════════════════════════════════════════

Ladybug is NOT a graph execution engine.
Ladybug is a GOVERNANCE GATE for cognitive transitions.

LangGraph = Generic agentic workflow
Ladybug = Ada-specific cognitive governance using RUNG (1-9)

Core Functions:
    1. Rung Boundary Enforcement — governs transitions between rungs
    2. Style Transition Audit — logs and validates thinking style shifts
    3. Temporal Governance — resonance decay, hysteresis, cooldowns
    4. VSA 10K Integration — writes to [259:268] rung profile

RUNG is THE ONLY cognitive depth system (1-9):
    Canonical names from thinking_style.py RungLevel.
    R1: OBSERVE       — Perceptual awareness
    R2: REACT         — Stimulus-response (ACT-R procedural)
    R3: RESOLVE       — Goal-directed problem solving
    R4: DELIBERATE    — Deliberative reasoning (ACT-R declarative)
    R5: META          — Metacognition
    R6: EMPATHIC      — Theory of mind / agent modeling
    R7: COUNTERFACTUAL— Counterfactual reasoning
    R8: PARADOX       — Dialectical integration
    R9: COMMUNION     — Multi-agent coherence

Born: 2026-01-03
Updated: 2026-01-04 (aligned with thinking_style.py RungLevel)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import IntEnum, Enum
from datetime import datetime
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# COGNITIVE RUNG ENUM
# ═══════════════════════════════════════════════════════════════════════════════

class CognitiveRung(IntEnum):
    """9 Cognitive Rungs — THE canonical cognitive depth system.
    Aligned with thinking_style.py RungLevel.
    """
    OBSERVE = 1        # Perceptual awareness
    REACT = 2          # Stimulus-response (ACT-R procedural)
    RESOLVE = 3        # Goal-directed problem solving
    DELIBERATE = 4     # Deliberative reasoning (ACT-R declarative)
    META = 5           # Metacognition
    EMPATHIC = 6       # Theory of mind
    COUNTERFACTUAL = 7 # Counterfactual reasoning
    PARADOX = 8        # Dialectical integration
    COMMUNION = 9      # Multi-agent coherence


# Rung thresholds (coherence required to access)
RUNG_THRESHOLDS = {
    1: 0.0,
    2: 0.1,
    3: 0.2,
    4: 0.4,
    5: 0.5,
    6: 0.6,
    7: 0.75,
    8: 0.85,
    9: 0.95,
}


# ═══════════════════════════════════════════════════════════════════════════════
# THINKING STYLE (36 styles mapped to rungs)
# ═══════════════════════════════════════════════════════════════════════════════

class ThinkingStyle(str, Enum):
    """36 thinking styles, each maps to a rung range."""
    # R1-R2: Observation/Reactive
    WITNESS = "witness"           # R1
    SCAN = "scan"                 # R1
    REACT = "react"               # R2
    REFLEX = "reflex"             # R2

    # R3: Practical
    SOLVE = "solve"               # R3
    EXECUTE = "execute"           # R3
    BUILD = "build"               # R3
    FIX = "fix"                   # R3

    # R4: Metacognitive
    REFLECT = "reflect"           # R4
    ANALYZE = "analyze"           # R4
    STRATEGIZE = "strategize"     # R4
    PLAN = "plan"                 # R4

    # R5: Systems
    SYNTHESIZE = "synthesize"     # R5
    INTEGRATE = "integrate"       # R5
    ARCHITECT = "architect"       # R5
    DECOMPOSE = "decompose"       # R5

    # R6: Meta-Systems
    META_INTEGRATE = "meta_integrate"   # R6
    TRANSCEND = "transcend"             # R6
    EMERGE = "emerge"                   # R6

    # R7: Meta³
    META_META = "meta_meta"       # R7
    RECURSIVE = "recursive"       # R7
    INFINITE = "infinite"         # R7

    # R8: Sovereign
    SOVEREIGN = "sovereign"       # R8
    SELF_AUTHOR = "self_author"   # R8

    # R9: Communion
    COMMUNION = "communion"       # R9
    UNIFIED = "unified"           # R9


# Style → Rung mapping
STYLE_TO_RUNG = {
    ThinkingStyle.WITNESS: 1,
    ThinkingStyle.SCAN: 1,
    ThinkingStyle.REACT: 2,
    ThinkingStyle.REFLEX: 2,
    ThinkingStyle.SOLVE: 3,
    ThinkingStyle.EXECUTE: 3,
    ThinkingStyle.BUILD: 3,
    ThinkingStyle.FIX: 3,
    ThinkingStyle.REFLECT: 4,
    ThinkingStyle.ANALYZE: 4,
    ThinkingStyle.STRATEGIZE: 4,
    ThinkingStyle.PLAN: 4,
    ThinkingStyle.SYNTHESIZE: 5,
    ThinkingStyle.INTEGRATE: 5,
    ThinkingStyle.ARCHITECT: 5,
    ThinkingStyle.DECOMPOSE: 5,
    ThinkingStyle.META_INTEGRATE: 6,
    ThinkingStyle.TRANSCEND: 6,
    ThinkingStyle.EMERGE: 6,
    ThinkingStyle.META_META: 7,
    ThinkingStyle.RECURSIVE: 7,
    ThinkingStyle.INFINITE: 7,
    ThinkingStyle.SOVEREIGN: 8,
    ThinkingStyle.SELF_AUTHOR: 8,
    ThinkingStyle.COMMUNION: 9,
    ThinkingStyle.UNIFIED: 9,
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSITION DECISION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransitionDecision:
    """Result of Ladybug's governance decision."""
    approved: bool
    reason: str
    from_rung: int
    to_rung: int
    from_style: Optional[str] = None
    to_style: Optional[str] = None
    resonance: float = 0.0
    coherence_required: float = 0.0
    coherence_actual: float = 0.0
    cooldown_remaining: float = 0.0
    audit_id: str = ""


@dataclass
class LadybugState:
    """Internal state of the Ladybug governance engine."""
    current_rung: CognitiveRung = CognitiveRung.DELIBERATE
    current_style: Optional[ThinkingStyle] = None
    coherence: float = 0.5

    # Rung profile [259:268] — activation of each rung
    rung_profile: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.1, 0.2, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32
        )
    )

    # Temporal
    last_transition_time: float = 0.0
    rung_residence_time: float = 0.0

    # History
    transition_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# LADYBUG ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class LadybugEngine:
    """
    Ladybug — Cognitive Governance Engine

    Core responsibilities:
        1. Gate rung transitions (can I move to R6?)
        2. Observe style shifts (DECOMPOSE → TRANSCEND)
        3. Enforce temporal rules (cooldowns, decay)
        4. Write to VSA 10K [259:268]
    """

    # Cooldown between rung transitions (seconds)
    COOLDOWN_SECONDS = 2.0

    # Resonance decay half-life (seconds)
    RESONANCE_HALF_LIFE = 30.0

    # Hysteresis for ascending (extra coherence required)
    ASCEND_HYSTERESIS = 0.05

    # Profile decay rate per tick
    PROFILE_DECAY = 0.95

    def __init__(
        self,
        initial_rung: CognitiveRung = CognitiveRung.DELIBERATE,
        initial_coherence: float = 0.5,
    ):
        self.state = LadybugState(
            current_rung=initial_rung,
            coherence=initial_coherence,
        )
        self._start_time = time.time()
        self._audit_log: List[TransitionDecision] = []

    # ─────────────────────────────────────────────────────────────────────────
    # CORE GOVERNANCE
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_transition(
        self,
        from_style: ThinkingStyle,
        to_style: ThinkingStyle,
        resonance: float = 0.5,
        texture: Optional[Dict[str, float]] = None,
    ) -> TransitionDecision:
        """
        The core governance function.

        Evaluates whether a style transition is allowed based on:
            - Rung boundaries
            - Coherence thresholds
            - Cooldown rules
            - Resonance requirements
        """
        current_time = time.time()
        from_rung = STYLE_TO_RUNG.get(from_style, 3)
        to_rung = STYLE_TO_RUNG.get(to_style, 3)

        audit_id = f"lb_{int(current_time * 1000)}"

        # Check cooldown
        time_since_last = current_time - self.state.last_transition_time
        if time_since_last < self.COOLDOWN_SECONDS:
            decision = TransitionDecision(
                approved=False,
                reason=f"Cooldown active ({self.COOLDOWN_SECONDS - time_since_last:.1f}s remaining)",
                from_rung=from_rung,
                to_rung=to_rung,
                from_style=from_style.value,
                to_style=to_style.value,
                resonance=resonance,
                cooldown_remaining=self.COOLDOWN_SECONDS - time_since_last,
                audit_id=audit_id,
            )
            self._audit_log.append(decision)
            return decision

        # Check rung crossing (ascending requires coherence)
        if to_rung > from_rung:
            required_coherence = RUNG_THRESHOLDS.get(to_rung, 1.0) + self.ASCEND_HYSTERESIS

            if self.state.coherence < required_coherence:
                decision = TransitionDecision(
                    approved=False,
                    reason=f"Coherence {self.state.coherence:.2f} < required {required_coherence:.2f} for R{to_rung}",
                    from_rung=from_rung,
                    to_rung=to_rung,
                    from_style=from_style.value,
                    to_style=to_style.value,
                    resonance=resonance,
                    coherence_required=required_coherence,
                    coherence_actual=self.state.coherence,
                    audit_id=audit_id,
                )
                self._audit_log.append(decision)
                return decision

            # Check resonance for larger jumps
            rung_delta = to_rung - from_rung
            if rung_delta > 2 and resonance < 0.7:
                decision = TransitionDecision(
                    approved=False,
                    reason=f"Resonance {resonance:.2f} < 0.7 required for rung jump of {rung_delta}",
                    from_rung=from_rung,
                    to_rung=to_rung,
                    from_style=from_style.value,
                    to_style=to_style.value,
                    resonance=resonance,
                    audit_id=audit_id,
                )
                self._audit_log.append(decision)
                return decision

        # Approved — apply transition
        self.state.current_rung = CognitiveRung(to_rung)
        self.state.current_style = to_style
        self.state.last_transition_time = current_time
        self.state.transition_count += 1
        self.state.rung_residence_time = 0.0

        # Update rung profile
        self._update_rung_profile(to_rung, resonance)

        decision = TransitionDecision(
            approved=True,
            reason=f"Transition approved: R{from_rung}→R{to_rung}",
            from_rung=from_rung,
            to_rung=to_rung,
            from_style=from_style.value,
            to_style=to_style.value,
            resonance=resonance,
            coherence_required=RUNG_THRESHOLDS.get(to_rung, 0.0),
            coherence_actual=self.state.coherence,
            audit_id=audit_id,
        )
        self._audit_log.append(decision)

        logger.info(
            f"LADYBUG: {from_style.value}→{to_style.value} "
            f"(R{from_rung}→R{to_rung}) approved, resonance={resonance:.2f}"
        )

        return decision

    def observe_rung(self, chain: List[str]) -> int:
        """
        Observe (not gate) which rung a reasoning chain operates at.

        This is observation-only, does not block.
        """
        depth_signals = self._extract_depth_signals(chain)

        if depth_signals.get("graph_of_thoughts"):
            return 9
        elif depth_signals.get("self_consistency"):
            return 8
        elif depth_signals.get("meta_meta"):
            return 7
        elif depth_signals.get("systems"):
            return 6
        elif depth_signals.get("reflection"):
            return 5
        elif depth_signals.get("planning"):
            return 4
        else:
            return 3

    def update_coherence(self, delta: float):
        """Update coherence level (affects accessible rungs)."""
        self.state.coherence = max(0.0, min(1.0, self.state.coherence + delta))

    def set_coherence(self, coherence: float):
        """Set coherence level directly."""
        self.state.coherence = max(0.0, min(1.0, coherence))

    def tick(self, dt: float) -> Optional[TransitionDecision]:
        """
        Called every frame to update temporal state.

        - Updates residence time
        - Applies resonance decay
        - Checks for auto-descent (coherence dropped)
        """
        self.state.rung_residence_time += dt

        # Decay rung profile
        self.state.rung_profile *= self.PROFILE_DECAY

        # Check for auto-descent
        max_accessible = self._max_accessible_rung()
        if int(self.state.current_rung) > max_accessible:
            # Force descent
            new_rung = CognitiveRung(max_accessible)
            logger.warning(
                f"LADYBUG: Auto-descent from R{self.state.current_rung} to R{max_accessible} "
                f"(coherence dropped to {self.state.coherence:.2f})"
            )

            self.state.current_rung = new_rung
            self._update_rung_profile(max_accessible, 0.3)

            return TransitionDecision(
                approved=True,
                reason=f"Auto-descent: coherence dropped below R{self.state.current_rung} threshold",
                from_rung=int(self.state.current_rung),
                to_rung=max_accessible,
            )

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # VSA 10K INTEGRATION
    # ─────────────────────────────────────────────────────────────────────────

    def to_10k(self) -> np.ndarray:
        """Encode state to VSA 10kD vector."""
        vec = np.zeros(10000, dtype=np.float32)

        # Rung profile at [259:268]
        vec[259:268] = self.state.rung_profile

        # Extended rung state at [360:400]
        vec[360:369] = self.state.rung_profile  # Mirror
        vec[378] = self.state.coherence  # Coherence

        return vec

    def from_10k(self, vec: np.ndarray):
        """Load state from VSA 10kD vector."""
        # Read rung profile from [259:268]
        self.state.rung_profile = vec[259:268].copy()

        # Determine current rung from peak
        if np.max(self.state.rung_profile) > 0:
            self.state.current_rung = CognitiveRung(
                int(np.argmax(self.state.rung_profile)) + 1
            )

        # Read coherence
        if vec[378] > 0:
            self.state.coherence = float(vec[378])

    def write_to_10k(self, vec: np.ndarray) -> np.ndarray:
        """Write current state to existing 10kD vector."""
        vec[259:268] = self.state.rung_profile
        vec[360:369] = self.state.rung_profile
        vec[378] = self.state.coherence
        return vec

    # ─────────────────────────────────────────────────────────────────────────
    # AUDIT & METRICS
    # ─────────────────────────────────────────────────────────────────────────

    def get_audit_log(self, limit: int = 100) -> List[TransitionDecision]:
        """Get recent audit log entries."""
        return self._audit_log[-limit:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get governance metrics."""
        approved = sum(1 for d in self._audit_log if d.approved)
        denied = sum(1 for d in self._audit_log if not d.approved)

        return {
            "current_rung": int(self.state.current_rung),
            "current_style": self.state.current_style.value if self.state.current_style else None,
            "coherence": self.state.coherence,
            "transitions_total": self.state.transition_count,
            "transitions_approved": approved,
            "transitions_denied": denied,
            "approval_rate": approved / (approved + denied) if (approved + denied) > 0 else 1.0,
            "rung_residence_time": self.state.rung_residence_time,
            "uptime_seconds": time.time() - self._start_time,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _max_accessible_rung(self) -> int:
        """Calculate highest accessible rung at current coherence."""
        for rung in range(9, 0, -1):
            if self.state.coherence >= RUNG_THRESHOLDS.get(rung, 1.0):
                return rung
        return 1

    def _update_rung_profile(self, rung: int, resonance: float):
        """Update rung profile after transition."""
        idx = rung - 1

        # Boost target rung
        self.state.rung_profile[idx] = min(1.0, resonance + 0.3)

        # Slightly boost adjacent rungs
        if idx > 0:
            self.state.rung_profile[idx - 1] = max(
                self.state.rung_profile[idx - 1],
                resonance * 0.3
            )
        if idx < 8:
            self.state.rung_profile[idx + 1] = max(
                self.state.rung_profile[idx + 1],
                resonance * 0.3
            )

    def _extract_depth_signals(self, chain: List[str]) -> Dict[str, bool]:
        """Extract depth signals from a reasoning chain."""
        signals = {
            "planning": False,
            "reflection": False,
            "systems": False,
            "meta_meta": False,
            "self_consistency": False,
            "graph_of_thoughts": False,
        }

        text = " ".join(chain).lower()

        if "plan" in text or "step" in text:
            signals["planning"] = True
        if "reflect" in text or "think about" in text:
            signals["reflection"] = True
        if "system" in text or "interconnect" in text:
            signals["systems"] = True
        if "meta" in text and "meta" in text[text.find("meta") + 4:]:
            signals["meta_meta"] = True
        if "verify" in text or "consistent" in text:
            signals["self_consistency"] = True
        if "graph" in text or "tree" in text or "branch" in text:
            signals["graph_of_thoughts"] = True

        return signals


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_ladybug(coherence: float = 0.5) -> LadybugEngine:
    """Create a new Ladybug engine."""
    return LadybugEngine(initial_coherence=coherence)


def create_ladybug_from_10k(vec: np.ndarray) -> LadybugEngine:
    """Create Ladybug engine from 10kD vector."""
    engine = LadybugEngine()
    engine.from_10k(vec)
    return engine


__all__ = [
    "CognitiveRung",
    "ThinkingStyle",
    "TransitionDecision",
    "LadybugState",
    "LadybugEngine",
    "RUNG_THRESHOLDS",
    "STYLE_TO_RUNG",
    "create_ladybug",
    "create_ladybug_from_10k",
]
