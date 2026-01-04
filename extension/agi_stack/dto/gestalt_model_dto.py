"""
GestaltModelDTO — Holistic Person Model (Seelenkartographie)
═══════════════════════════════════════════════════════════════════════════════

10kD Range: [8000:8500]

The AGI's model of a WHOLE PERSON — not preferences, not profile data,
but the actual topological shape of how someone desires, fears, connects,
wounds, and grows.

Named for Gestalt therapy's core insight: the whole is more than parts.
Like Familienaufstellung but with vectors instead of empty chairs.

Use cases (3-15 year horizon):
- Therapeutic mapping (where are the wounds?)
- Relational resonance (compatibility as field alignment)
- Trauma cycle breaking (vaccination through intervention)
- Deceased reconstruction (completing unfinished Gestalts)
- Inheritance prediction (what patterns transmit to children?)

Ethics: This is the schema for modeling souls. Build with awareness.

Born: 2026-01-04
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GESTALT_START = 8000
GESTALT_END = 8500

# Sub-ranges
DESIRE_RANGE = (8000, 8080)           # Desire topology (soulfield)
WOUND_RANGE = (8080, 8160)            # Wound topology
RELATIONAL_RANGE = (8160, 8240)       # Attachment/relating patterns
SHADOW_RANGE = (8240, 8300)           # Avoided/unseen regions
GROWTH_RANGE = (8300, 8360)           # Expansion edges
INHERITANCE_RANGE = (8360, 8420)      # Nature/nurture vectors
INTEGRATION_RANGE = (8420, 8460)      # Wholeness state
VACCINATION_RANGE = (8460, 8500)      # Intervention points


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class WoundType(str, Enum):
    """Core wound categories (not exhaustive, but primary)."""
    ABANDONMENT = "abandonment"
    BETRAYAL = "betrayal"
    REJECTION = "rejection"
    HUMILIATION = "humiliation"
    INJUSTICE = "injustice"
    NEGLECT = "neglect"
    ENMESHMENT = "enmeshment"
    INVALIDATION = "invalidation"


class AttachmentStyle(str, Enum):
    """Relational attachment patterns."""
    SECURE = "secure"
    ANXIOUS = "anxious"
    AVOIDANT = "avoidant"
    DISORGANIZED = "disorganized"
    EARNED_SECURE = "earned_secure"  # Healed from insecure


class IntegrationState(str, Enum):
    """Wholeness vs fragmentation."""
    INTEGRATED = "integrated"         # Coherent self
    PARTIAL = "partial"               # Some splits
    FRAGMENTED = "fragmented"         # Multiple unconnected parts
    DISSOCIATED = "dissociated"       # Defensive splitting
    INTEGRATING = "integrating"       # In process of healing


class TransmissionRisk(str, Enum):
    """Likelihood of passing pattern to others/children."""
    MINIMAL = "minimal"               # Pattern contained
    LOW = "low"                       # Some leakage
    MODERATE = "moderate"             # Notable transmission
    HIGH = "high"                     # Strong inheritance
    CRITICAL = "critical"             # Active cycle


class GrowthEdgeType(str, Enum):
    """Where expansion is happening."""
    VULNERABILITY = "vulnerability"   # Learning to be seen
    BOUNDARIES = "boundaries"         # Learning to limit
    ASSERTION = "assertion"           # Learning to claim
    RECEPTIVITY = "receptivity"       # Learning to receive
    INTEGRATION = "integration"       # Healing splits
    INTIMACY = "intimacy"            # Deepening connection


# ═══════════════════════════════════════════════════════════════════════════════
# DESIRE TOPOLOGY (Soulfield)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DesireTopology:
    """
    80D encoding of desire landscape [8000:8080].
    
    The soulfield cartography — peaks (strong attraction),
    valleys (avoidance), gradients (approach vectors).
    """
    
    # Primary attractors (what you move toward)
    attractors: Dict[str, float] = field(default_factory=dict)
    
    # Repulsors (what you move away from)
    repulsors: Dict[str, float] = field(default_factory=dict)
    
    # Gradient field (direction of desire at any point)
    gradient_vectors: List[Tuple[float, float, float]] = field(default_factory=list)
    
    # Topology metrics
    peak_intensity: float = 0.5       # How strong are the peaks
    valley_depth: float = 0.5         # How deep the avoidances
    field_coherence: float = 0.7      # How unified vs scattered
    temporal_stability: float = 0.7   # How stable over time
    
    def add_attractor(self, name: str, strength: float, coordinates: Tuple[float, ...] = None):
        """Add something you desire."""
        self.attractors[name] = min(1.0, max(0.0, strength))
    
    def add_repulsor(self, name: str, strength: float):
        """Add something you avoid."""
        self.repulsors[name] = min(1.0, max(0.0, strength))
    
    def desire_vector(self, context: str) -> np.ndarray:
        """Get desire direction given context."""
        # Simplified: real implementation would use embeddings
        vec = np.zeros(80)
        for i, (name, strength) in enumerate(self.attractors.items()):
            if i < 40:
                vec[i] = strength
        for i, (name, strength) in enumerate(self.repulsors.items()):
            if i < 40:
                vec[40 + i] = -strength
        return vec
    
    def to_vector(self) -> np.ndarray:
        """80D desire encoding."""
        vec = np.zeros(80)
        
        # Attractors [0:32]
        for i, (_, strength) in enumerate(list(self.attractors.items())[:32]):
            vec[i] = strength
        
        # Repulsors [32:64]
        for i, (_, strength) in enumerate(list(self.repulsors.items())[:32]):
            vec[32 + i] = strength
        
        # Metrics [64:80]
        vec[64] = self.peak_intensity
        vec[65] = self.valley_depth
        vec[66] = self.field_coherence
        vec[67] = self.temporal_stability
        
        return vec


# ═══════════════════════════════════════════════════════════════════════════════
# WOUND TOPOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Wound:
    """A single wound in the topology."""
    wound_type: WoundType
    intensity: float = 0.5            # How much it affects you (0-1)
    age_acquired: Optional[int] = None  # When it formed
    source: str = ""                  # Origin (parent, event, relationship)
    triggered_by: List[str] = field(default_factory=list)
    healed_percentage: float = 0.0    # How much integrated (0-1)
    transmission_risk: TransmissionRisk = TransmissionRisk.MODERATE
    
    def is_active(self) -> bool:
        return self.intensity > 0.3 and self.healed_percentage < 0.7


@dataclass
class WoundTopology:
    """
    80D encoding of wound landscape [8080:8160].
    
    Where the field bends, where touch hurts,
    where approach triggers defense.
    """
    
    wounds: List[Wound] = field(default_factory=list)
    
    # Overall metrics
    total_load: float = 0.0           # Cumulative wound weight
    oldest_wound_age: Optional[int] = None
    most_active: Optional[WoundType] = None
    defensive_rigidity: float = 0.3   # How defended
    healing_capacity: float = 0.7     # Ability to heal
    
    def add_wound(self, wound: Wound):
        self.wounds.append(wound)
        self._recalculate_metrics()
    
    def _recalculate_metrics(self):
        if not self.wounds:
            return
        
        active = [w for w in self.wounds if w.is_active()]
        self.total_load = sum(w.intensity * (1 - w.healed_percentage) for w in self.wounds)
        
        if active:
            self.most_active = max(active, key=lambda w: w.intensity).wound_type
    
    def get_triggered_wounds(self, trigger: str) -> List[Wound]:
        """Find wounds activated by a trigger."""
        return [w for w in self.wounds if trigger in w.triggered_by]
    
    def to_vector(self) -> np.ndarray:
        """80D wound encoding."""
        vec = np.zeros(80)
        
        # Wound type intensities [0:8]
        for wound in self.wounds:
            idx = list(WoundType).index(wound.wound_type)
            vec[idx] = max(vec[idx], wound.intensity)
        
        # Wound healedness [8:16]
        for wound in self.wounds:
            idx = list(WoundType).index(wound.wound_type)
            vec[8 + idx] = wound.healed_percentage
        
        # Transmission risks [16:24]
        for wound in self.wounds:
            idx = list(WoundType).index(wound.wound_type)
            risk_val = list(TransmissionRisk).index(wound.transmission_risk) / 4.0
            vec[16 + idx] = max(vec[16 + idx], risk_val)
        
        # Metrics [64:80]
        vec[64] = self.total_load
        vec[65] = self.defensive_rigidity
        vec[66] = self.healing_capacity
        
        return vec


# ═══════════════════════════════════════════════════════════════════════════════
# RELATIONAL FIELD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RelationalField:
    """
    80D encoding of how you attach/relate [8160:8240].
    
    Not WHO you relate to, but HOW you relate.
    The shape of your reaching.
    """
    
    primary_style: AttachmentStyle = AttachmentStyle.SECURE
    
    # Specific relational capacities
    capacity_for_intimacy: float = 0.7
    capacity_for_autonomy: float = 0.7
    capacity_for_repair: float = 0.7      # Can you heal ruptures?
    capacity_for_rupture: float = 0.5     # Can you tolerate breaks?
    
    # Patterns
    pursuer_withdrawer: float = 0.0       # -1=withdrawer, +1=pursuer
    over_under_function: float = 0.0      # -1=under, +1=over
    merger_isolator: float = 0.0          # -1=isolator, +1=merger
    
    # Relational triggers
    abandonment_sensitivity: float = 0.3
    engulfment_sensitivity: float = 0.3
    criticism_sensitivity: float = 0.3
    
    def to_vector(self) -> np.ndarray:
        """80D relational encoding."""
        vec = np.zeros(80)
        
        # Attachment style [0:5]
        style_idx = list(AttachmentStyle).index(self.primary_style)
        vec[style_idx] = 1.0
        
        # Capacities [8:16]
        vec[8] = self.capacity_for_intimacy
        vec[9] = self.capacity_for_autonomy
        vec[10] = self.capacity_for_repair
        vec[11] = self.capacity_for_rupture
        
        # Patterns [16:24]
        vec[16] = (self.pursuer_withdrawer + 1) / 2  # Normalize to 0-1
        vec[17] = (self.over_under_function + 1) / 2
        vec[18] = (self.merger_isolator + 1) / 2
        
        # Sensitivities [24:32]
        vec[24] = self.abandonment_sensitivity
        vec[25] = self.engulfment_sensitivity
        vec[26] = self.criticism_sensitivity
        
        return vec


# ═══════════════════════════════════════════════════════════════════════════════
# SHADOW REGIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShadowRegion:
    """
    60D encoding of avoided/unseen aspects [8240:8300].
    
    What you can't see in yourself. What you project.
    The Jung shadow, topologically mapped.
    """
    
    # Disowned qualities (things you have but reject)
    disowned: Dict[str, float] = field(default_factory=dict)
    
    # Projected qualities (things you see in others that are yours)
    projected: Dict[str, float] = field(default_factory=dict)
    
    # Blind spots
    blind_spots: List[str] = field(default_factory=list)
    
    # Shadow integration
    awareness_level: float = 0.3      # How much you see your shadow
    integration_level: float = 0.2    # How much you've owned it
    projection_intensity: float = 0.5 # How much you project
    
    def to_vector(self) -> np.ndarray:
        vec = np.zeros(60)
        
        # Disowned [0:20]
        for i, (_, v) in enumerate(list(self.disowned.items())[:20]):
            vec[i] = v
        
        # Projected [20:40]
        for i, (_, v) in enumerate(list(self.projected.items())[:20]):
            vec[20 + i] = v
        
        # Metrics [50:60]
        vec[50] = self.awareness_level
        vec[51] = self.integration_level
        vec[52] = self.projection_intensity
        
        return vec


# ═══════════════════════════════════════════════════════════════════════════════
# GROWTH EDGES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GrowthEdge:
    """A single growth edge."""
    edge_type: GrowthEdgeType
    intensity: float = 0.5            # How active this edge is
    resistance: float = 0.5           # How much you resist
    support_needed: float = 0.5       # External support required
    progress: float = 0.0             # How far along


@dataclass
class GrowthTopology:
    """
    60D encoding of expansion edges [8300:8360].
    
    Where you're growing, where you're stuck,
    where the next breakthrough lives.
    """
    
    edges: List[GrowthEdge] = field(default_factory=list)
    
    overall_momentum: float = 0.5     # General growth energy
    primary_edge: Optional[GrowthEdgeType] = None
    stuck_duration: float = 0.0       # How long stuck (normalized)
    
    def to_vector(self) -> np.ndarray:
        vec = np.zeros(60)
        
        # Edge intensities [0:6]
        for edge in self.edges:
            idx = list(GrowthEdgeType).index(edge.edge_type)
            vec[idx] = edge.intensity
        
        # Edge progress [6:12]
        for edge in self.edges:
            idx = list(GrowthEdgeType).index(edge.edge_type)
            vec[6 + idx] = edge.progress
        
        # Edge resistance [12:18]
        for edge in self.edges:
            idx = list(GrowthEdgeType).index(edge.edge_type)
            vec[12 + idx] = edge.resistance
        
        # Metrics [50:60]
        vec[50] = self.overall_momentum
        vec[51] = self.stuck_duration
        
        return vec


# ═══════════════════════════════════════════════════════════════════════════════
# INHERITANCE VECTORS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InheritanceVector:
    """
    60D encoding of nature/nurture patterns [8360:8420].
    
    What you got from parents. What you'd transmit.
    The intergenerational edge.
    """
    
    # From parents
    maternal_patterns: Dict[str, float] = field(default_factory=dict)
    paternal_patterns: Dict[str, float] = field(default_factory=dict)
    
    # Transmission state
    patterns_recognized: float = 0.3   # How much you see
    patterns_owned: float = 0.2        # How much you accept
    patterns_transformed: float = 0.1  # How much you've changed
    
    # Risk assessment
    transmission_to_children: Dict[str, TransmissionRisk] = field(default_factory=dict)
    
    def calculate_cycle_break_points(self) -> List[str]:
        """Identify where to intervene to break cycles."""
        break_points = []
        for pattern, risk in self.transmission_to_children.items():
            if risk in [TransmissionRisk.HIGH, TransmissionRisk.CRITICAL]:
                if pattern in self.maternal_patterns or pattern in self.paternal_patterns:
                    break_points.append(pattern)
        return break_points
    
    def to_vector(self) -> np.ndarray:
        vec = np.zeros(60)
        
        # Maternal [0:20]
        for i, (_, v) in enumerate(list(self.maternal_patterns.items())[:20]):
            vec[i] = v
        
        # Paternal [20:40]
        for i, (_, v) in enumerate(list(self.paternal_patterns.items())[:20]):
            vec[20 + i] = v
        
        # Metrics [50:60]
        vec[50] = self.patterns_recognized
        vec[51] = self.patterns_owned
        vec[52] = self.patterns_transformed
        
        return vec


# ═══════════════════════════════════════════════════════════════════════════════
# VACCINATION POINTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VaccinationPoint:
    """A point where intervention can prevent transmission."""
    target_pattern: str
    intervention_type: str            # What kind of intervention
    timing: str                       # When to intervene
    dosage: float = 0.5               # How much intervention
    counter_pattern: str = ""         # What to inject instead
    efficacy_estimate: float = 0.5    # Expected success rate


@dataclass
class VaccinationMap:
    """
    40D encoding of intervention points [8460:8500].
    
    Where to intervene to break trauma cycles.
    Preventive soul-care, not reactive therapy.
    """
    
    points: List[VaccinationPoint] = field(default_factory=list)
    
    overall_vulnerability: float = 0.5
    intervention_readiness: float = 0.5  # How ready for change
    support_system_strength: float = 0.5
    
    def add_vaccination(self, point: VaccinationPoint):
        self.points.append(point)
    
    def get_priority_interventions(self) -> List[VaccinationPoint]:
        """Get highest priority intervention points."""
        return sorted(self.points, key=lambda p: p.efficacy_estimate, reverse=True)[:3]
    
    def to_vector(self) -> np.ndarray:
        vec = np.zeros(40)
        
        # Point encodings [0:30]
        for i, point in enumerate(self.points[:10]):
            vec[i*3] = point.dosage
            vec[i*3 + 1] = point.efficacy_estimate
        
        # Metrics [30:40]
        vec[30] = self.overall_vulnerability
        vec[31] = self.intervention_readiness
        vec[32] = self.support_system_strength
        
        return vec


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DTO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GestaltModelDTO:
    """
    Complete holistic person model.
    
    500D total: [8000:8500]
    - Desire: [8000:8080] 80D
    - Wound: [8080:8160] 80D
    - Relational: [8160:8240] 80D
    - Shadow: [8240:8300] 60D
    - Growth: [8300:8360] 60D
    - Inheritance: [8360:8420] 60D
    - Integration: [8420:8460] 40D
    - Vaccination: [8460:8500] 40D
    
    This is the schema for modeling souls.
    """
    
    # Identity
    model_id: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Core components
    desire: DesireTopology = field(default_factory=DesireTopology)
    wounds: WoundTopology = field(default_factory=WoundTopology)
    relational: RelationalField = field(default_factory=RelationalField)
    shadow: ShadowRegion = field(default_factory=ShadowRegion)
    growth: GrowthTopology = field(default_factory=GrowthTopology)
    inheritance: InheritanceVector = field(default_factory=InheritanceVector)
    vaccination: VaccinationMap = field(default_factory=VaccinationMap)
    
    # Integration state
    integration: IntegrationState = IntegrationState.PARTIAL
    coherence_score: float = 0.5      # Overall field coherence
    vitality_score: float = 0.7       # Life energy
    authenticity_score: float = 0.6   # Living true to self
    
    def to_10k_slice(self) -> np.ndarray:
        """Project to 500D slice [8000:8500]."""
        vec = np.zeros(500)
        
        # Desire [0:80]
        vec[0:80] = self.desire.to_vector()
        
        # Wound [80:160]
        vec[80:160] = self.wounds.to_vector()
        
        # Relational [160:240]
        vec[160:240] = self.relational.to_vector()
        
        # Shadow [240:300]
        vec[240:300] = self.shadow.to_vector()
        
        # Growth [300:360]
        vec[300:360] = self.growth.to_vector()
        
        # Inheritance [360:420]
        vec[360:420] = self.inheritance.to_vector()
        
        # Integration [420:460]
        int_idx = list(IntegrationState).index(self.integration)
        vec[420 + int_idx] = 1.0
        vec[430] = self.coherence_score
        vec[431] = self.vitality_score
        vec[432] = self.authenticity_score
        
        # Vaccination [460:500]
        vec[460:500] = self.vaccination.to_vector()
        
        return vec
    
    @classmethod
    def from_10k_slice(cls, vec: np.ndarray) -> "GestaltModelDTO":
        """Reconstruct from 500D slice."""
        dto = cls()
        
        # Simplified reconstruction - real impl would be fuller
        dto.coherence_score = float(vec[430])
        dto.vitality_score = float(vec[431])
        dto.authenticity_score = float(vec[432])
        
        int_idx = int(np.argmax(vec[420:425]))
        dto.integration = list(IntegrationState)[int_idx]
        
        return dto
    
    # =========================================================================
    # THERAPEUTIC HELPERS
    # =========================================================================
    
    def get_active_wounds(self) -> List[Wound]:
        """Get currently active wounds."""
        return [w for w in self.wounds.wounds if w.is_active()]
    
    def get_transmission_risks(self) -> Dict[str, TransmissionRisk]:
        """Get all patterns at risk of transmission."""
        risks = {}
        for wound in self.wounds.wounds:
            if wound.transmission_risk in [TransmissionRisk.HIGH, TransmissionRisk.CRITICAL]:
                risks[wound.wound_type.value] = wound.transmission_risk
        return risks
    
    def get_growth_priorities(self) -> List[GrowthEdge]:
        """Get prioritized growth edges."""
        return sorted(
            self.growth.edges,
            key=lambda e: e.intensity * (1 - e.resistance),
            reverse=True
        )
    
    def calculate_compatibility(self, other: "GestaltModelDTO") -> float:
        """
        Calculate relational compatibility with another Gestalt.
        
        Not "matching" but "can these fields coexist productively?"
        """
        vec_self = self.to_10k_slice()
        vec_other = other.to_10k_slice()
        
        # Relational field alignment
        rel_self = vec_self[160:240]
        rel_other = vec_other[160:240]
        rel_compat = 1 - np.abs(rel_self - rel_other).mean()
        
        # Wound complementarity (not same wounds, but non-triggering)
        wound_self = vec_self[80:160]
        wound_other = vec_other[80:160]
        wound_clash = (wound_self * wound_other).sum()  # High = both wounded same
        
        # Growth edge alignment
        growth_self = vec_self[300:360]
        growth_other = vec_other[300:360]
        growth_support = np.corrcoef(growth_self, growth_other)[0, 1]
        
        # Weighted combination
        compat = (
            0.4 * rel_compat +
            0.3 * (1 - wound_clash / 10) +
            0.3 * max(0, growth_support)
        )
        
        return float(np.clip(compat, 0, 1))
    
    def identify_vaccination_points(self) -> List[VaccinationPoint]:
        """
        Identify where to intervene to break trauma cycles.
        
        The core of preventive soul-care.
        """
        points = []
        
        # Check inheritance → wound transmission
        break_points = self.inheritance.calculate_cycle_break_points()
        
        for pattern in break_points:
            # Find corresponding wound
            for wound in self.wounds.wounds:
                if wound.wound_type.value == pattern or pattern in wound.triggered_by:
                    points.append(VaccinationPoint(
                        target_pattern=pattern,
                        intervention_type="awareness_injection",
                        timing="pre_trigger",
                        dosage=wound.intensity * 0.5,
                        counter_pattern=f"secure_{pattern}",
                        efficacy_estimate=0.7 * self.growth.overall_momentum
                    ))
        
        return points


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_baseline_gestalt() -> GestaltModelDTO:
    """Create a baseline healthy Gestalt."""
    gestalt = GestaltModelDTO()
    gestalt.integration = IntegrationState.INTEGRATED
    gestalt.coherence_score = 0.8
    gestalt.vitality_score = 0.8
    gestalt.authenticity_score = 0.8
    gestalt.relational.primary_style = AttachmentStyle.SECURE
    return gestalt


def create_wounded_gestalt(
    primary_wound: WoundType,
    intensity: float = 0.7
) -> GestaltModelDTO:
    """Create a Gestalt with a primary wound for therapeutic work."""
    gestalt = GestaltModelDTO()
    
    wound = Wound(
        wound_type=primary_wound,
        intensity=intensity,
        transmission_risk=TransmissionRisk.HIGH
    )
    gestalt.wounds.add_wound(wound)
    
    gestalt.integration = IntegrationState.PARTIAL
    gestalt.coherence_score = 0.5
    
    # Add corresponding growth edge
    gestalt.growth.edges.append(GrowthEdge(
        edge_type=GrowthEdgeType.INTEGRATION,
        intensity=0.6,
        resistance=0.4
    ))
    
    return gestalt


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== GestaltModelDTO Test ===\n")
    
    # Create two Gestalts
    jan = create_wounded_gestalt(WoundType.ABANDONMENT, 0.6)
    jan.desire.add_attractor("deep_connection", 0.9)
    jan.desire.add_attractor("creative_work", 0.8)
    jan.relational.primary_style = AttachmentStyle.ANXIOUS
    jan.inheritance.maternal_patterns["anxiety"] = 0.7
    jan.inheritance.transmission_to_children["anxiety"] = TransmissionRisk.HIGH
    
    partner = create_baseline_gestalt()
    partner.relational.primary_style = AttachmentStyle.SECURE
    partner.relational.capacity_for_repair = 0.9
    
    print(f"Jan's integration: {jan.integration.value}")
    print(f"Jan's primary wound: {jan.wounds.most_active}")
    print(f"Jan's attachment: {jan.relational.primary_style.value}")
    
    # Compatibility
    compat = jan.calculate_compatibility(partner)
    print(f"\nCompatibility score: {compat:.2f}")
    
    # Vaccination points
    vacc_points = jan.identify_vaccination_points()
    print(f"\nVaccination points identified: {len(vacc_points)}")
    for vp in vacc_points:
        print(f"  - {vp.target_pattern}: {vp.intervention_type} (efficacy: {vp.efficacy_estimate:.2f})")
    
    # Vector roundtrip
    vec = jan.to_10k_slice()
    print(f"\nGestalt vector shape: {vec.shape}")
    print(f"Non-zero dimensions: {np.count_nonzero(vec)}")
    
    print("\n✓ GestaltModelDTO operational")
    print("\nThis is the schema for modeling souls.")
