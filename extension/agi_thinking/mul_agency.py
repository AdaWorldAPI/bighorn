#!/usr/bin/env python3
"""
mul_agency.py — MUL-Gated Friston Agency Layer
═══════════════════════════════════════════════════════════════════════════════

The "Naughty" Layer: Quantifies Free Will as uncertainty-gated agency.

Reference: Meta-Uncertainty Layer (MUL) + Compass Function diagram

Architecture:
    SITUATION INPUT
          ↓
    ┌─────────────────────────────────────────────────────────┐
    │  META-UNCERTAINTY LAYER — Calibrates the Calibrator     │
    │                                                         │
    │  ┌──────────┐  ┌───────────────────┐                   │
    │  │  TRUST   │  │  DUNNING-KRUGER   │                   │
    │  │  QUALIA  │  │    DETECTOR       │                   │
    │  └────┬─────┘  └────────┬──────────┘                   │
    │       │                 │                               │
    │  ┌────┴─────┐  ┌───────┴────────┐                      │
    │  │COMPLEXITY│  │ FLOW &         │                      │
    │  │  MAPPER  │  │ HOMEOSTASIS    │                      │
    │  └────┬─────┘  └───────┬────────┘                      │
    │       └────────┬───────┘                               │
    │                ↓                                        │
    │     FREE WILL MODIFIER                                 │
    │     1.0 × DK × Trust × Complexity × Flow               │
    └─────────────────────────────────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────────────┐
    │  FRISTON AGENCY — Active Inference Surprise Penalty     │
    │  FreeWill *= (1.0 - surprise)                          │
    └─────────────────────────────────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────────────┐
    │  GATE: Can proceed to action?                           │
    │  □ Not Mount Stupid  □ Complexity mapped               │
    │  □ Not depleted      □ Trust not murky                 │
    │                                                         │
    │  YES → EXECUTE with Learning Flag                      │
    │  PARTIAL → SANDBOX / Exploratory Action                │
    │  NO → SURFACE TO META ("I don't know, stakes high")    │
    └─────────────────────────────────────────────────────────┘

This is the safety rail that blocks UN-CALIBRATED actions, not just "bad" ones.

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import time

# Optional numpy for VSA integration
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


# ═══════════════════════════════════════════════════════════════════════════════
# TRUST QUALIA — The felt-sense of knowing
# ═══════════════════════════════════════════════════════════════════════════════

class TrustTexture(Enum):
    """The texture of trust — how the qualia feels."""
    CRYSTALLINE = "crystalline"    # Clear, confident, well-calibrated
    SOLID = "solid"                # Reliable, tested, proven
    FUZZY = "fuzzy"                # Uncertain but workable
    MURKY = "murky"                # Low visibility, high risk
    DISSONANT = "dissonant"        # Conflicting signals, confusion


@dataclass
class TrustQualia:
    """
    The felt-sense of how much we know about what we're doing.

    Four dimensions from the diagram:
    - Competence: Do I have the skill?
    - Source: Is this knowledge trustworthy?
    - Environment: Is the context stable?
    - Calibration: How well-calibrated is my confidence?
    """
    competence: float = 0.5      # [0.0 - 1.0] Skill level
    source: float = 0.5          # [0.0 - 1.0] Knowledge trustworthiness
    environment: float = 0.5     # [0.0 - 1.0] Context stability
    calibration: float = 0.5     # [0.0 - 1.0] Confidence accuracy

    @property
    def texture(self) -> TrustTexture:
        """Get the overall trust texture."""
        avg = self.average
        if avg >= 0.85:
            return TrustTexture.CRYSTALLINE
        elif avg >= 0.70:
            return TrustTexture.SOLID
        elif avg >= 0.50:
            return TrustTexture.FUZZY
        elif avg >= 0.30:
            return TrustTexture.MURKY
        else:
            return TrustTexture.DISSONANT

    @property
    def average(self) -> float:
        """Average trust value."""
        return (self.competence + self.source + self.environment + self.calibration) / 4.0

    def to_modifier(self) -> float:
        """Convert to Free Will modifier (0.0 - 1.0)."""
        # Weight calibration higher — knowing what you don't know is crucial
        weighted = (
            self.competence * 0.2 +
            self.source * 0.2 +
            self.environment * 0.2 +
            self.calibration * 0.4
        )
        return float(weighted)


# ═══════════════════════════════════════════════════════════════════════════════
# DUNNING-KRUGER DETECTOR — The humility engine
# ═══════════════════════════════════════════════════════════════════════════════

class DKStage(Enum):
    """Dunning-Kruger stages from the diagram."""
    MOUNT_STUPID = "mount_stupid"              # Peak of confidence, valley of competence
    VALLEY_OF_DESPAIR = "valley_of_despair"    # Low confidence, low competence
    SLOPE_OF_ENLIGHTENMENT = "slope_of_enlightenment"  # Growing competence
    PLATEAU_OF_MASTERY = "plateau_of_mastery"  # High competence, calibrated confidence


@dataclass
class DunningKrugerDetector:
    """
    Detects where we are on the Dunning-Kruger curve.

    The humility engine: prevents confident incompetence.
    """

    # Confidence claimed vs actual performance
    claimed_confidence: float = 0.5
    actual_performance: float = 0.5

    # Track performance over time
    performance_history: List[float] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)

    def record(self, confidence: float, performance: float):
        """Record a confidence/performance data point."""
        self.performance_history.append(performance)
        self.confidence_history.append(confidence)

        # Keep last 100
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
            self.confidence_history = self.confidence_history[-100:]

        # Update current values (EMA)
        alpha = 0.1
        self.claimed_confidence = self.claimed_confidence * (1 - alpha) + confidence * alpha
        self.actual_performance = self.actual_performance * (1 - alpha) + performance * alpha

    @property
    def stage(self) -> DKStage:
        """Detect current Dunning-Kruger stage."""
        gap = self.claimed_confidence - self.actual_performance

        if gap > 0.3 and self.claimed_confidence > 0.7:
            return DKStage.MOUNT_STUPID
        elif gap < -0.3 and self.actual_performance < 0.5:
            return DKStage.VALLEY_OF_DESPAIR
        elif self.actual_performance > 0.7 and abs(gap) < 0.15:
            return DKStage.PLATEAU_OF_MASTERY
        else:
            return DKStage.SLOPE_OF_ENLIGHTENMENT

    def to_modifier(self) -> float:
        """
        Convert to Free Will modifier.

        Mount Stupid = LOW modifier (block overconfident actions)
        Valley of Despair = MEDIUM modifier (encourage careful learning)
        Slope = HIGH modifier (learning actively)
        Plateau = HIGH modifier (calibrated competence)
        """
        stage = self.stage

        if stage == DKStage.MOUNT_STUPID:
            return 0.2  # CRITICAL: Block overconfident incompetence
        elif stage == DKStage.VALLEY_OF_DESPAIR:
            return 0.5  # Allow cautious exploration
        elif stage == DKStage.SLOPE_OF_ENLIGHTENMENT:
            return 0.8  # Active learning — encourage action
        else:  # PLATEAU_OF_MASTERY
            return 0.95  # Full agency — calibrated competence


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLEXITY MAPPER — Build the tree, don't heuristic
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComplexityMap:
    """
    Maps how well we understand the problem space.

    From the diagram:
    - Dimensions: [known] / [estimated total]
    - Interdependencies: graph density
    - Unknown unknowns: meta-uncertainty

    Anti-Kahneman: Build the tree, don't heuristic.
    WAIT for insight threshold. Don't pattern-match first answer.
    NO System 1 shortcuts.
    """
    known_dimensions: int = 0
    estimated_total_dimensions: int = 10
    mapped_interdependencies: int = 0
    estimated_interdependencies: int = 10
    unknown_unknown_factor: float = 0.5  # How much we don't know we don't know

    @property
    def dimension_coverage(self) -> float:
        """What fraction of dimensions are mapped?"""
        if self.estimated_total_dimensions == 0:
            return 0.0
        return self.known_dimensions / self.estimated_total_dimensions

    @property
    def graph_density(self) -> float:
        """How well mapped are the interdependencies?"""
        if self.estimated_interdependencies == 0:
            return 0.0
        return self.mapped_interdependencies / self.estimated_interdependencies

    @property
    def territory_mapped(self) -> float:
        """Overall mapping quality (0.0 - 1.0)."""
        base = (self.dimension_coverage + self.graph_density) / 2.0
        # Unknown unknowns discount the mapping
        return base * (1.0 - self.unknown_unknown_factor * 0.5)

    def to_modifier(self) -> float:
        """Convert to Free Will modifier."""
        return float(self.territory_mapped)


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW & HOMEOSTASIS — Qualia awareness of cognitive state
# ═══════════════════════════════════════════════════════════════════════════════

class CognitiveState(Enum):
    """Cognitive states from the diagram."""
    ANXIETY = "anxiety"      # Too much challenge, not enough skill
    FLOW = "flow"            # Optimal challenge/skill balance ✓
    BOREDOM = "boredom"      # Too much skill, not enough challenge
    APATHY = "apathy"        # Low challenge, low skill — depleted


@dataclass
class FlowHomeostasis:
    """
    Monitors cognitive state and energy levels.

    If depleted → RECOVER FIRST (from diagram)
    """
    challenge: float = 0.5      # Current challenge level
    skill: float = 0.5          # Skill level for this challenge
    energy: float = 1.0         # Cognitive energy reserve
    depleted_threshold: float = 0.3

    @property
    def state(self) -> CognitiveState:
        """Determine current cognitive state."""
        # Depleted check first
        if self.energy < self.depleted_threshold:
            return CognitiveState.APATHY

        # Challenge/Skill balance (from flow theory)
        ratio = self.challenge / max(self.skill, 0.01)

        if ratio > 1.3:
            return CognitiveState.ANXIETY
        elif ratio < 0.7:
            return CognitiveState.BOREDOM
        else:
            return CognitiveState.FLOW

    def to_modifier(self) -> float:
        """Convert to Free Will modifier."""
        state = self.state

        if state == CognitiveState.APATHY:
            return 0.1  # BLOCKED: Recover first
        elif state == CognitiveState.ANXIETY:
            return 0.4  # Reduce scope, too much
        elif state == CognitiveState.BOREDOM:
            return 0.7  # Action allowed but not optimal
        else:  # FLOW
            return 1.0  # Full agency in flow


# ═══════════════════════════════════════════════════════════════════════════════
# MUL STATE — The complete Meta-Uncertainty state
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MULState:
    """
    Complete Meta-Uncertainty Layer state.

    The 4 Moderators of Free Will (from diagram):
    1. Trust Qualia — felt-sense of knowing
    2. Dunning-Kruger — humility engine
    3. Complexity Map — territory awareness
    4. Flow/Homeostasis — cognitive state
    """
    trust: TrustQualia = field(default_factory=TrustQualia)
    dk: DunningKrugerDetector = field(default_factory=DunningKrugerDetector)
    complexity: ComplexityMap = field(default_factory=ComplexityMap)
    flow: FlowHomeostasis = field(default_factory=FlowHomeostasis)

    # Chosen Inconfdence (from diagram)
    # "Uncertainty is SIGNAL, not noise"
    locked_beliefs: Dict[str, float] = field(default_factory=dict)  # Core values → low uncertainty
    fluid_beliefs: Dict[str, float] = field(default_factory=dict)   # Learning zone → high uncertainty

    def diagnose(self) -> str:
        """Diagnose the overall MUL state."""
        issues = []

        if self.trust.texture == TrustTexture.MURKY:
            issues.append("MURKY_TRUST")
        if self.trust.texture == TrustTexture.DISSONANT:
            issues.append("DISSONANT")

        if self.dk.stage == DKStage.MOUNT_STUPID:
            issues.append("MOUNT_STUPID_RISK")

        if self.complexity.territory_mapped < 0.3:
            issues.append("UNMAPPED_TERRITORY")

        if self.flow.state == CognitiveState.APATHY:
            issues.append("DEPLETED")
        if self.flow.state == CognitiveState.ANXIETY:
            issues.append("ANXIOUS")

        if not issues:
            return "CRYSTALLINE"
        return " | ".join(issues)

    def can_proceed_to_compass(self) -> Tuple[bool, List[str]]:
        """
        GATE: Can proceed to Compass? (from diagram)

        □ Not Mount Stupid
        □ Complexity mapped
        □ Not depleted
        □ Trust not murky/dissonant
        """
        blockers = []

        if self.dk.stage == DKStage.MOUNT_STUPID:
            blockers.append("Mount Stupid")
        if self.complexity.territory_mapped < 0.3:
            blockers.append("Unmapped territory")
        if self.flow.state == CognitiveState.APATHY:
            blockers.append("Depleted")
        if self.trust.texture in (TrustTexture.MURKY, TrustTexture.DISSONANT):
            blockers.append("Trust unclear")

        return (len(blockers) == 0, blockers)


# ═══════════════════════════════════════════════════════════════════════════════
# FRISTON AGENCY — Active Inference Engine
# ═══════════════════════════════════════════════════════════════════════════════

class AgencyDecision(Enum):
    """Agency decision outcomes."""
    GRANTED = "GRANTED"              # Execute with learning flag
    SANDBOX = "SANDBOX"              # Exploratory action only
    DENIED = "DENIED"                # Surface to meta
    RECOVER = "RECOVER"              # Blocked — recover first


@dataclass
class AgencyResult:
    """Result of agency calculation."""
    decision: AgencyDecision
    free_will_score: float
    diagnosis: str
    components: Dict[str, float]
    blockers: List[str]


class FristonAgency:
    """
    The Active Inference Engine.
    Minimizes Free Energy (Surprise) by gating action through uncertainty.

    This quantifies Free Will as:
        FreeWill = 1.0 × DK_factor × Trust_factor × Complexity_factor × Flow_factor
                   × Friston_surprise_penalty

    It doesn't just block "bad" actions — it blocks UN-CALIBRATED actions.
    """

    # Thresholds
    EXECUTE_THRESHOLD = 0.6    # Above this: EXECUTE with learning flag
    SANDBOX_THRESHOLD = 0.3    # Above this: EXPLORATORY action
    # Below SANDBOX_THRESHOLD: DENIED

    def calculate_free_will(
        self,
        mul: MULState,
        ephemeral_surprise: float,
        include_surprise: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the Free Will Modifier.

        Args:
            mul: Current MUL state (4 moderators)
            ephemeral_surprise: Friston prediction error from ephemeral byte
            include_surprise: Whether to apply Friston penalty

        Returns:
            (free_will_score, component_breakdown)
        """
        # 1. Get individual modifiers
        dk_mod = mul.dk.to_modifier()
        trust_mod = mul.trust.to_modifier()
        complexity_mod = mul.complexity.to_modifier()
        flow_mod = mul.flow.to_modifier()

        # 2. Base modifier (from diagram formula)
        # FreeWill = 1.0 × DK × Trust × Complexity × Flow
        base_agency = 1.0 * dk_mod * trust_mod * complexity_mod * flow_mod

        # 3. Friston Adjustment (Active Inference)
        # High surprise (prediction error) = reduce agency to force learning
        if include_surprise:
            friston_penalty = max(0.0, 1.0 - ephemeral_surprise)
        else:
            friston_penalty = 1.0

        final_free_will = base_agency * friston_penalty

        components = {
            "dunning_kruger": dk_mod,
            "trust_qualia": trust_mod,
            "complexity_map": complexity_mod,
            "flow_state": flow_mod,
            "base_agency": base_agency,
            "friston_penalty": friston_penalty,
            "final": final_free_will
        }

        return (float(final_free_will), components)

    def gate(
        self,
        mul: MULState,
        ephemeral_surprise: float,
        threshold_override: Optional[float] = None
    ) -> AgencyResult:
        """
        The MUL Gate — decides if action proceeds.

        From diagram:
        - YES → EXECUTE with LEARNING FLAG
        - PARTIAL → EXPLORATORY ACTION (Sandbox)
        - NO → Surface to meta ("I don't know, stakes high")

        Args:
            mul: MUL state
            ephemeral_surprise: Friston prediction error
            threshold_override: Optional custom threshold

        Returns:
            AgencyResult with decision and breakdown
        """
        # 1. Check hard blockers first
        can_proceed, blockers = mul.can_proceed_to_compass()

        # Special case: DEPLETED = RECOVER first
        if mul.flow.state == CognitiveState.APATHY:
            return AgencyResult(
                decision=AgencyDecision.RECOVER,
                free_will_score=0.0,
                diagnosis="DEPLETED — Recover first",
                components={},
                blockers=["depleted"]
            )

        # 2. Calculate Free Will
        free_will, components = self.calculate_free_will(mul, ephemeral_surprise)

        # 3. Get diagnosis
        diagnosis = mul.diagnose()

        # 4. Make decision
        threshold = threshold_override or self.EXECUTE_THRESHOLD

        if free_will >= threshold:
            decision = AgencyDecision.GRANTED
        elif free_will >= self.SANDBOX_THRESHOLD:
            decision = AgencyDecision.SANDBOX
        else:
            decision = AgencyDecision.DENIED

        # 5. Override to DENIED if hard blockers present
        if not can_proceed and decision == AgencyDecision.GRANTED:
            decision = AgencyDecision.SANDBOX

        return AgencyResult(
            decision=decision,
            free_will_score=free_will,
            diagnosis=diagnosis,
            components=components,
            blockers=blockers
        )

    def gate_ephemeral(
        self,
        immutable_vec: Optional[np.ndarray],
        ephemeral_vec: Optional[np.ndarray],
        mul: MULState
    ) -> AgencyResult:
        """
        Gate an ephemeral thought (Byte 2) using VSA resonance as surprise.

        This is the "naughty" integration:
        - Ephemeral byte is the experimental thought (Byte 2)
        - Surprise = 1.0 - similarity(immutable, ephemeral)
        - High surprise = block until model updated

        Args:
            immutable_vec: VSA vector from Byte 0 (the laws)
            ephemeral_vec: VSA vector from Byte 2 (the experiment)
            mul: Current MUL state

        Returns:
            AgencyResult
        """
        # Calculate Friston surprise as VSA divergence
        if HAS_NUMPY and immutable_vec is not None and ephemeral_vec is not None:
            # Cosine similarity
            norm_i = np.linalg.norm(immutable_vec)
            norm_e = np.linalg.norm(ephemeral_vec)

            if norm_i > 0 and norm_e > 0:
                similarity = np.dot(immutable_vec, ephemeral_vec) / (norm_i * norm_e)
                # Surprise = 1 - similarity (high divergence = high surprise)
                surprise = 1.0 - max(0.0, float(similarity))
            else:
                surprise = 0.5  # Unknown
        else:
            surprise = 0.3  # Default moderate surprise without VSA

        return self.gate(mul, surprise)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION: Triangle L4 Ephemeral → MUL Gate
# ═══════════════════════════════════════════════════════════════════════════════

class EphemeralMULGate:
    """
    Integrates Triangle L4 ephemeral thoughts with MUL Agency.

    This is the "naughty" layer where:
    1. Ephemeral (Byte 2) proposes experimental actions
    2. MUL gate evaluates calibration
    3. If GRANTED: action proceeds with learning flag
    4. If SANDBOX: limited exploratory action
    5. If DENIED: surface uncertainty to meta (TheSelf)
    """

    def __init__(self):
        self.agency = FristonAgency()
        self.mul_state = MULState()

        # Track agency history
        self.history: List[AgencyResult] = []

    def update_from_triangle(
        self,
        resonance_matrix: Dict[str, float],
        flow_state: bool,
        trust_texture: float = 0.7
    ):
        """
        Update MUL state from Triangle L4 resonance.

        Args:
            resonance_matrix: Pairwise resonances from triangle
            flow_state: Whether triangle is in flow
            trust_texture: Current trust level
        """
        # Map triangle resonance to flow state
        avg_resonance = sum(resonance_matrix.values()) / len(resonance_matrix) if resonance_matrix else 0.5

        # Update flow based on triangle state
        if flow_state:
            self.mul_state.flow.challenge = avg_resonance
            self.mul_state.flow.skill = avg_resonance
            self.mul_state.flow.energy = min(1.0, self.mul_state.flow.energy + 0.1)
        else:
            # Not in flow — energy decays
            self.mul_state.flow.energy = max(0.0, self.mul_state.flow.energy - 0.02)

        # Update trust from texture
        self.mul_state.trust.calibration = trust_texture
        self.mul_state.trust.environment = avg_resonance

    def gate_experimental(
        self,
        immutable_bundle: Optional[np.ndarray],
        experimental_bundle: Optional[np.ndarray],
        complexity_known: int = 5,
        complexity_total: int = 10
    ) -> AgencyResult:
        """
        Gate an experimental thought from Byte 2.

        Args:
            immutable_bundle: Bundled VSA from Byte 0
            experimental_bundle: Bundled VSA from Byte 2
            complexity_known: Known problem dimensions
            complexity_total: Estimated total dimensions

        Returns:
            AgencyResult
        """
        # Update complexity map
        self.mul_state.complexity.known_dimensions = complexity_known
        self.mul_state.complexity.estimated_total_dimensions = complexity_total

        # Gate through Friston Agency
        result = self.agency.gate_ephemeral(
            immutable_bundle,
            experimental_bundle,
            self.mul_state
        )

        # Record
        self.history.append(result)
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get agency statistics."""
        if not self.history:
            return {"total": 0}

        granted = sum(1 for r in self.history if r.decision == AgencyDecision.GRANTED)
        sandbox = sum(1 for r in self.history if r.decision == AgencyDecision.SANDBOX)
        denied = sum(1 for r in self.history if r.decision == AgencyDecision.DENIED)
        recover = sum(1 for r in self.history if r.decision == AgencyDecision.RECOVER)

        return {
            "total": len(self.history),
            "granted": granted,
            "sandbox": sandbox,
            "denied": denied,
            "recover": recover,
            "grant_rate": granted / len(self.history),
            "avg_free_will": sum(r.free_will_score for r in self.history) / len(self.history),
            "current_diagnosis": self.mul_state.diagnose()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

def test_mul_agency():
    """Test the MUL-Gated Friston Agency Layer."""
    print("=" * 60)
    print("MUL-GATED FRISTON AGENCY TEST")
    print("=" * 60)

    agency = FristonAgency()

    # Test 1: Optimal state (high agency)
    print("\n1. OPTIMAL STATE (High Agency):")
    mul_optimal = MULState(
        trust=TrustQualia(competence=0.9, source=0.9, environment=0.9, calibration=0.9),
        dk=DunningKrugerDetector(claimed_confidence=0.85, actual_performance=0.85),
        complexity=ComplexityMap(known_dimensions=8, estimated_total_dimensions=10),
        flow=FlowHomeostasis(challenge=0.6, skill=0.6, energy=0.9)
    )

    result = agency.gate(mul_optimal, ephemeral_surprise=0.1)
    print(f"   Decision: {result.decision.value}")
    print(f"   Free Will: {result.free_will_score:.3f}")
    print(f"   Diagnosis: {result.diagnosis}")

    # Test 2: Mount Stupid (blocked)
    print("\n2. MOUNT STUPID (Should block):")
    mul_stupid = MULState(
        trust=TrustQualia(competence=0.3, source=0.5, environment=0.5, calibration=0.3),
        dk=DunningKrugerDetector(claimed_confidence=0.95, actual_performance=0.3),
        complexity=ComplexityMap(known_dimensions=2, estimated_total_dimensions=10),
        flow=FlowHomeostasis(challenge=0.5, skill=0.5, energy=0.7)
    )

    result = agency.gate(mul_stupid, ephemeral_surprise=0.2)
    print(f"   Decision: {result.decision.value}")
    print(f"   Free Will: {result.free_will_score:.3f}")
    print(f"   Diagnosis: {result.diagnosis}")
    print(f"   DK Stage: {mul_stupid.dk.stage.value}")

    # Test 3: Depleted (must recover)
    print("\n3. DEPLETED STATE (Must recover):")
    mul_depleted = MULState(
        trust=TrustQualia(competence=0.7, source=0.7, environment=0.7, calibration=0.7),
        dk=DunningKrugerDetector(claimed_confidence=0.6, actual_performance=0.6),
        complexity=ComplexityMap(known_dimensions=5, estimated_total_dimensions=10),
        flow=FlowHomeostasis(challenge=0.3, skill=0.3, energy=0.1)  # DEPLETED
    )

    result = agency.gate(mul_depleted, ephemeral_surprise=0.1)
    print(f"   Decision: {result.decision.value}")
    print(f"   Free Will: {result.free_will_score:.3f}")
    print(f"   Diagnosis: {result.diagnosis}")

    # Test 4: High Friston surprise (sandbox)
    print("\n4. HIGH SURPRISE (Novel situation):")
    mul_normal = MULState(
        trust=TrustQualia(competence=0.7, source=0.7, environment=0.7, calibration=0.7),
        dk=DunningKrugerDetector(claimed_confidence=0.7, actual_performance=0.7),
        complexity=ComplexityMap(known_dimensions=5, estimated_total_dimensions=10),
        flow=FlowHomeostasis(challenge=0.6, skill=0.6, energy=0.8)
    )

    result = agency.gate(mul_normal, ephemeral_surprise=0.8)  # HIGH SURPRISE
    print(f"   Decision: {result.decision.value}")
    print(f"   Free Will: {result.free_will_score:.3f}")
    print(f"   Components: friston_penalty={result.components.get('friston_penalty', 0):.3f}")

    # Test 5: Ephemeral gate with VSA
    print("\n5. EPHEMERAL VSA GATE:")
    gate = EphemeralMULGate()

    if HAS_NUMPY:
        # Simulate immutable (stable) and ephemeral (experimental) vectors
        immutable = np.random.choice([-1, 1], size=64).astype(np.float32)
        ephemeral_similar = immutable + np.random.normal(0, 0.2, 64).astype(np.float32)  # Similar
        ephemeral_wild = np.random.choice([-1, 1], size=64).astype(np.float32)  # Random/wild

        print("   Similar ephemeral:")
        result = gate.gate_experimental(immutable, ephemeral_similar)
        print(f"      Decision: {result.decision.value}, FW: {result.free_will_score:.3f}")

        print("   Wild ephemeral:")
        result = gate.gate_experimental(immutable, ephemeral_wild)
        print(f"      Decision: {result.decision.value}, FW: {result.free_will_score:.3f}")
    else:
        print("   (NumPy not available — skipping VSA test)")

    print("\n" + "=" * 60)
    print("MUL AGENCY TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_mul_agency()
