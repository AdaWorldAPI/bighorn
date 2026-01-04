"""
rung_bridge.py — 9-Rung Cognitive Depth Management
═══════════════════════════════════════════════════════════════════════════════

RUNG is THE ONLY cognitive depth system.
Canonical names from thinking_style.py RungLevel.

9 Rungs (R1-R9):
    R1: OBSERVE       — Passive witnessing, pure awareness
    R2: REACT         — Stimulus-response
    R3: RESOLVE       — Problem-solving
    R4: DELIBERATE    — Deliberate reasoning
    R5: META          — Meta-cognitive awareness
    R6: SOVEREIGN     — Self-authoring consciousness
    R7: COUNTERFACTUAL— Counterfactual reasoning
    R8: PARADOX       — Paradox integration
    R9: TRANSCEND     — Full AGI integration

Dimension Allocation [259:268] — Rung Profile (canonical)
Dimension Allocation [360:400] — Rung State (extended)

Born: 2026-01-03
Updated: 2026-01-04 (aligned with thinking_style.py RungLevel)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import IntEnum
import numpy as np

from ..agi_stack.dto.dimension_registry import (
    get_range,
    get_slice,
    RUNG_NAMES,
    RUNG_DESCRIPTIONS,
    RUNG_THRESHOLDS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# COGNITIVE RUNG ENUM
# ═══════════════════════════════════════════════════════════════════════════════

class CognitiveRung(IntEnum):
    """
    9 Cognitive Rungs — THE canonical cognitive depth system.
    Aligned with thinking_style.py RungLevel.

    1-indexed for human clarity.
    """
    OBSERVE = 1        # Passive witnessing
    REACT = 2          # Stimulus-response
    RESOLVE = 3        # Problem-solving
    DELIBERATE = 4     # Deliberate reasoning
    META = 5           # Meta-cognitive
    SOVEREIGN = 6      # Self-authoring
    COUNTERFACTUAL = 7 # Counterfactual reasoning
    PARADOX = 8        # Paradox integration
    TRANSCEND = 9      # Full AGI integration

    @property
    def description(self) -> str:
        return RUNG_DESCRIPTIONS.get(int(self), "Unknown rung")

    @property
    def threshold(self) -> float:
        return RUNG_THRESHOLDS.get(int(self), 1.0)

    @classmethod
    def from_coherence(cls, coherence: float) -> "CognitiveRung":
        """Determine highest accessible rung from coherence level."""
        for rung in reversed(list(cls)):
            if coherence >= rung.threshold:
                return rung
        return cls.OBSERVE


# ═══════════════════════════════════════════════════════════════════════════════
# RUNG STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RungState:
    """
    Complete state of the 9-rung cognitive depth system.

    This is what Ladybug observes and governs.
    """

    # Current rung (1-9)
    current_rung: CognitiveRung = CognitiveRung.DELIBERATE

    # Rung profile (activation of each rung R1-R9)
    rung_profile: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.1, 0.2, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32
        )
    )

    # Global coherence (gates which rungs are accessible)
    coherence: float = 0.5

    # Temporal state
    rung_residence_time: float = 0.0   # Seconds at current rung
    last_transition_time: float = 0.0

    # Transition history (for hysteresis)
    transition_count: int = 0

    @property
    def accessible_rungs(self) -> List[CognitiveRung]:
        """Which rungs are accessible at current coherence."""
        return [r for r in CognitiveRung if self.coherence >= r.threshold]

    @property
    def max_accessible_rung(self) -> CognitiveRung:
        """Highest accessible rung at current coherence."""
        return CognitiveRung.from_coherence(self.coherence)

    @property
    def is_at_ceiling(self) -> bool:
        """Are we at the highest accessible rung?"""
        return self.current_rung >= self.max_accessible_rung

    def to_10k(self) -> np.ndarray:
        """Encode state to 10kD vector."""
        vec = np.zeros(10000, dtype=np.float32)

        # Write rung profile to canonical position [259:268]
        rung_range = get_range("rung_profile")
        vec[rung_range.slice] = self.rung_profile

        # Write extended rung state [360:400]
        intensity_range = get_range("rung_intensity")
        vec[intensity_range.slice] = self.rung_profile  # Mirror

        coherence_range = get_range("rung_coherence")
        coherence_vec = np.zeros(9, dtype=np.float32)
        coherence_vec[int(self.current_rung) - 1] = self.coherence
        vec[coherence_range.slice] = coherence_vec

        return vec

    @classmethod
    def from_10k(cls, vec: np.ndarray) -> "RungState":
        """Decode state from 10kD vector."""
        state = cls()

        # Read rung profile from canonical position [259:268]
        rung_range = get_range("rung_profile")
        state.rung_profile = vec[rung_range.slice].copy()

        # Determine current rung from profile peak
        if np.max(state.rung_profile) > 0:
            state.current_rung = CognitiveRung(int(np.argmax(state.rung_profile)) + 1)

        # Read coherence from extended state
        coherence_range = get_range("rung_coherence")
        coherence_vec = vec[coherence_range.slice]
        if np.max(coherence_vec) > 0:
            state.coherence = float(np.max(coherence_vec))

        return state


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSITION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransitionResult:
    """Result of a rung transition attempt."""
    allowed: bool
    new_state: RungState
    reason: str
    from_rung: CognitiveRung
    to_rung: CognitiveRung
    coherence_required: float = 0.0
    coherence_actual: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# RUNG BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

class RungBridge:
    """
    Governs transitions between cognitive rungs.

    Core responsibilities:
        1. Validate transitions (is target rung accessible?)
        2. Apply transitions (update state)
        3. Track coherence (what gates higher rungs?)
        4. Manage hysteresis (prevent oscillation)
    """

    # Cooldown between rung transitions
    COOLDOWN_SECONDS = 2.0

    # Hysteresis: require extra coherence to ascend
    ASCEND_HYSTERESIS = 0.05

    # Profile decay rate
    PROFILE_DECAY = 0.9

    def __init__(self, state: Optional[RungState] = None):
        self.state = state or RungState()
        self._last_transition = 0.0

    def request_transition(
        self,
        target_rung: CognitiveRung,
        resonance: float = 0.5,
        current_time: float = 0.0,
    ) -> TransitionResult:
        """
        Request transition to a target rung.

        Args:
            target_rung: Desired rung (1-9)
            resonance: How strongly the transition is requested (0-1)
            current_time: Current timestamp for cooldown checking

        Returns:
            TransitionResult with success/failure and reason
        """
        from_rung = self.state.current_rung

        # Same rung - always allowed (reinforces current state)
        if target_rung == from_rung:
            self._update_profile(target_rung, resonance)
            return TransitionResult(
                allowed=True,
                new_state=self.state,
                reason="Reinforcing current rung",
                from_rung=from_rung,
                to_rung=target_rung,
            )

        # Check cooldown
        if not self._can_transition(current_time):
            return TransitionResult(
                allowed=False,
                new_state=self.state,
                reason=f"Cooldown active ({self.COOLDOWN_SECONDS}s between transitions)",
                from_rung=from_rung,
                to_rung=target_rung,
            )

        # Ascending - check coherence
        if target_rung > from_rung:
            required = target_rung.threshold + self.ASCEND_HYSTERESIS
            if self.state.coherence < required:
                return TransitionResult(
                    allowed=False,
                    new_state=self.state,
                    reason=f"Coherence {self.state.coherence:.2f} < required {required:.2f} for R{int(target_rung)}",
                    from_rung=from_rung,
                    to_rung=target_rung,
                    coherence_required=required,
                    coherence_actual=self.state.coherence,
                )

        # Descending - always allowed (conscious downshift)
        # Apply transition
        self.state.current_rung = target_rung
        self._update_profile(target_rung, resonance)
        self.state.transition_count += 1
        self._last_transition = current_time
        self.state.last_transition_time = current_time
        self.state.rung_residence_time = 0.0

        return TransitionResult(
            allowed=True,
            new_state=self.state,
            reason=f"Transitioned from R{int(from_rung)} to R{int(target_rung)}",
            from_rung=from_rung,
            to_rung=target_rung,
            coherence_required=target_rung.threshold,
            coherence_actual=self.state.coherence,
        )

    def update_coherence(self, delta: float):
        """Update coherence level (affects accessible rungs)."""
        self.state.coherence = max(0.0, min(1.0, self.state.coherence + delta))

    def set_coherence(self, coherence: float):
        """Set coherence level directly."""
        self.state.coherence = max(0.0, min(1.0, coherence))

    def tick(self, dt: float, current_time: float = 0.0) -> Optional[TransitionResult]:
        """
        Called every frame/tick to update temporal state.

        Returns transition result if auto-adjustment occurred.
        """
        self.state.rung_residence_time += dt

        # Check if we should auto-descend (coherence dropped)
        if self.state.current_rung > self.state.max_accessible_rung:
            return self.request_transition(
                self.state.max_accessible_rung,
                resonance=0.3,
                current_time=current_time,
            )

        return None

    def _can_transition(self, current_time: float) -> bool:
        """Check if transition is allowed (cooldown check)."""
        return (current_time - self._last_transition) >= self.COOLDOWN_SECONDS

    def _update_profile(self, rung: CognitiveRung, resonance: float):
        """Update rung profile after transition."""
        # Decay all rungs
        self.state.rung_profile *= self.PROFILE_DECAY

        # Boost target rung
        idx = int(rung) - 1
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


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_bridge_from_10k(vec: np.ndarray) -> RungBridge:
    """Create bridge initialized from 10kD vector."""
    state = RungState.from_10k(vec)
    return RungBridge(state)


def encode_rung_to_10k(
    rung: CognitiveRung,
    coherence: float = 0.5,
    vec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Encode rung and coherence to 10kD vector."""
    if vec is None:
        vec = np.zeros(10000, dtype=np.float32)

    state = RungState(current_rung=rung, coherence=coherence)
    state.rung_profile = np.zeros(9, dtype=np.float32)
    state.rung_profile[int(rung) - 1] = 1.0

    return state.to_10k()


def get_rung_from_10k(vec: np.ndarray) -> CognitiveRung:
    """Extract current rung from 10kD vector."""
    rung_range = get_range("rung_profile")
    profile = vec[rung_range.slice]
    if np.max(profile) > 0:
        return CognitiveRung(int(np.argmax(profile)) + 1)
    return CognitiveRung.DELIBERATE


__all__ = [
    "CognitiveRung",
    "RungState",
    "TransitionResult",
    "RungBridge",
    "create_bridge_from_10k",
    "encode_rung_to_10k",
    "get_rung_from_10k",
]
