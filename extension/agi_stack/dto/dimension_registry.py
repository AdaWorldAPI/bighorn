"""
Dimension Registry — Canonical VSA 10kD Dimension Allocation
═══════════════════════════════════════════════════════════════════════════════

SINGLE SOURCE OF TRUTH for all dimension allocations.
All DTOs and bridges MUST import from here.

10kD Space Layout:
══════════════════

SOUL SPACE [0:500] — Discrete cognitive primitives
    [0:16]      Qualia (16)
    [16:32]     Stances (16)
    [32:48]     Transitions (16)
    [48:80]     Verbs (32)
    [80:116]    GPT Styles (36)
    [116:152]   NARS Styles (36)
    [152:163]   Presence Modes (11)
    [163:168]   Archetypes (5)
    [168:171]   TLK Court (3)
    [171:175]   Affective Bias (4)
    [175:208]   TSV Dimensions (33)
    [208:256]   Reserved

TSV EMBEDDED [256:320] — ThinkingStyleVector continuous
    [256:259]   Pearl (SEE, DO, IMAGINE)
    [259:268]   Rung Profile (R1-R9) ← CANONICAL
    [268:273]   Sigma Tendency (Ω, Δ, Φ, Θ, Λ)
    [273:281]   Operations (8)
    [281:285]   Presence (4)
    [285:289]   Meta (4)
    [289:320]   Reserved

DTO SPACE [320:500] — Profile-derived values
    [320:324]   Motivation Drives (4)
    [324:330]   Free Will Config (6)
    [330:340]   Sieves (10)
    [340:350]   Relationship (10)
    [350:360]   Uncertainty (10)
    [360:400]   Layer State (40) ← 6 LEVELS HERE
    [400:450]   Temporal (50)
    [450:500]   Reserved

LAYER STATE [360:400] — 6 Awareness Levels + Transitions
    [360:366]   Level Intensities (6: MINIMAL→TRANSZENDENT)
    [366:372]   Level Transitions (6)
    [372:381]   Rung-Level Coupling (9)
    [381:400]   Reserved

FELT SPACE [2000:2200] — Continuous qualia
    [2000:2018] Qualia PCS (18D)
    [2018:2022] Body Axes (4D)
    [2022:2025] Poincaré (3D)
    [2025:2100] Affect Extensions
    [2100:2200] Reserved

QUALIA EDGES [2200:2500] — Edge-based qualia transitions
    [2200:2300] Edge Activations
    [2300:2400] Edge Weights
    [2400:2500] Edge Temporal

LOCATION [2500:2700] — Spatial/temporal context
    [2500:2555] Location DTO (55D)
    [2555:2600] Moment DTO (45D)
    [2600:2700] Reserved

EMBODIMENT [3000:3500] — Physics/world simulation
    [3000:3100] Physics DTO
    [3100:3200] World DTO
    [3200:3300] Vision DTO
    [3300:3400] Media DTO
    [3400:3500] Synesthesia DTO

COGNITION [8500:9500] — Semantic embeddings
    [8500:9524] Jina 1024D embedding
    [9524:9600] Reserved

EPHEMERAL [9600:10000] — Hot state
    [9600:9700] Triangle L4 Bytes (100)
    [9700:9800] MUL State (100)
    [9800:9900] Session Buffer (100)
    [9900:10000] Reserved

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# DIMENSION RANGE DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DimRange:
    """Immutable dimension range specification."""
    start: int
    end: int
    name: str
    description: str = ""

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)

    def __contains__(self, idx: int) -> bool:
        return self.start <= idx < self.end


# ═══════════════════════════════════════════════════════════════════════════════
# CANONICAL DIMENSION REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

DIMENSION_REGISTRY: Dict[str, DimRange] = {
    # ─────────────────────────────────────────────────────────────────────────
    # SOUL SPACE [0:500]
    # ─────────────────────────────────────────────────────────────────────────
    "qualia": DimRange(0, 16, "qualia", "16 discrete qualia types"),
    "stances": DimRange(16, 32, "stances", "16 relational stances"),
    "transitions": DimRange(32, 48, "transitions", "16 transition types"),
    "verbs": DimRange(48, 80, "verbs", "32 cognitive verbs"),
    "gpt_styles": DimRange(80, 116, "gpt_styles", "36 GPT thinking styles"),
    "nars_styles": DimRange(116, 152, "nars_styles", "36 NARS reasoning patterns"),
    "presence_modes": DimRange(152, 163, "presence_modes", "11 presence modes"),
    "archetypes": DimRange(163, 168, "archetypes", "5 DAWN/BLOOM/FORGE/STILLNESS/CENTER"),
    "tlk_court": DimRange(168, 171, "tlk_court", "Thanatos/Libido/Katharsis"),
    "affective_bias": DimRange(171, 175, "affective_bias", "Warmth/Edge/Restraint/Tenderness"),
    "tsv_dims": DimRange(175, 208, "tsv_dims", "33 ThinkingStyleVector dimensions"),
    "soul_reserved": DimRange(208, 256, "soul_reserved", "Reserved soul space"),

    # ─────────────────────────────────────────────────────────────────────────
    # TSV EMBEDDED [256:320] — RUNG PROFILE LIVES HERE
    # ─────────────────────────────────────────────────────────────────────────
    "pearl": DimRange(256, 259, "pearl", "SEE/DO/IMAGINE triplet"),
    "rung_profile": DimRange(259, 268, "rung_profile", "R1-R9 cognitive rung profile"),
    "sigma_tendency": DimRange(268, 273, "sigma_tendency", "Ω/Δ/Φ/Θ/Λ sigma types"),
    "operations": DimRange(273, 281, "operations", "8 cognitive operations"),
    "presence": DimRange(281, 285, "presence", "4 presence weights"),
    "meta_config": DimRange(285, 289, "meta_config", "4 meta parameters"),
    "tsv_reserved": DimRange(289, 320, "tsv_reserved", "Reserved TSV space"),

    # ─────────────────────────────────────────────────────────────────────────
    # DTO SPACE [320:500]
    # ─────────────────────────────────────────────────────────────────────────
    "motivation": DimRange(320, 324, "motivation", "Becoming/Coherence/Relational/Exploration"),
    "free_will": DimRange(324, 330, "free_will", "Free will configuration"),
    "sieves": DimRange(330, 340, "sieves", "Sieve configurations"),
    "relationship": DimRange(340, 350, "relationship", "Relationship parameters"),
    "uncertainty": DimRange(350, 360, "uncertainty", "Uncertainty handling"),

    # ─────────────────────────────────────────────────────────────────────────
    # RUNG STATE [360:400] — 9 COGNITIVE RUNGS (canonical)
    # ─────────────────────────────────────────────────────────────────────────
    "rung_intensity": DimRange(360, 369, "rung_intensity", "9 rung intensities (R1-R9)"),
    "rung_transitions": DimRange(369, 378, "rung_transitions", "Rung transition weights"),
    "rung_coherence": DimRange(378, 387, "rung_coherence", "Rung coherence levels"),
    "rung_reserved": DimRange(387, 400, "rung_reserved", "Reserved rung space"),

    # ─────────────────────────────────────────────────────────────────────────
    # TEMPORAL [400:450]
    # ─────────────────────────────────────────────────────────────────────────
    "temporal_position": DimRange(400, 410, "temporal_position", "Temporal position encoding"),
    "temporal_velocity": DimRange(410, 420, "temporal_velocity", "Rate of change"),
    "temporal_phase": DimRange(420, 430, "temporal_phase", "Phase encoding"),
    "temporal_memory": DimRange(430, 450, "temporal_memory", "Temporal context buffer"),

    # ─────────────────────────────────────────────────────────────────────────
    # FELT SPACE [2000:2200]
    # ─────────────────────────────────────────────────────────────────────────
    "qualia_pcs": DimRange(2000, 2018, "qualia_pcs", "18D qualia PCS vector"),
    "body_axes": DimRange(2018, 2022, "body_axes", "Pelvic/Boundary/Respiratory/Cardiac"),
    "poincare": DimRange(2022, 2025, "poincare", "Radius/Angle/Depth"),
    "affect_ext": DimRange(2025, 2100, "affect_ext", "Extended affect dimensions"),
    "felt_reserved": DimRange(2100, 2200, "felt_reserved", "Reserved felt space"),

    # ─────────────────────────────────────────────────────────────────────────
    # QUALIA EDGES [2200:2500]
    # ─────────────────────────────────────────────────────────────────────────
    "edge_activations": DimRange(2200, 2300, "edge_activations", "Qualia edge activations"),
    "edge_weights": DimRange(2300, 2400, "edge_weights", "Edge weight matrix"),
    "edge_temporal": DimRange(2400, 2500, "edge_temporal", "Edge temporal dynamics"),

    # ─────────────────────────────────────────────────────────────────────────
    # LOCATION [2500:2700]
    # ─────────────────────────────────────────────────────────────────────────
    "location": DimRange(2500, 2555, "location", "Location DTO (55D)"),
    "moment": DimRange(2555, 2600, "moment", "Moment DTO (45D)"),
    "location_reserved": DimRange(2600, 2700, "location_reserved", "Reserved location space"),

    # ─────────────────────────────────────────────────────────────────────────
    # EMBODIMENT [3000:3500]
    # ─────────────────────────────────────────────────────────────────────────
    "physics": DimRange(3000, 3100, "physics", "Physics DTO"),
    "world": DimRange(3100, 3200, "world", "World DTO"),
    "vision": DimRange(3200, 3300, "vision", "Vision DTO"),
    "media": DimRange(3300, 3400, "media", "Media DTO"),
    "synesthesia": DimRange(3400, 3500, "synesthesia", "Synesthesia DTO"),

    # ─────────────────────────────────────────────────────────────────────────
    # COGNITION [8500:9500]
    # ─────────────────────────────────────────────────────────────────────────
    "jina_embedding": DimRange(8500, 9524, "jina_embedding", "Jina 1024D semantic embedding"),
    "cognition_reserved": DimRange(9524, 9600, "cognition_reserved", "Reserved cognition space"),

    # ─────────────────────────────────────────────────────────────────────────
    # EPHEMERAL [9600:10000]
    # ─────────────────────────────────────────────────────────────────────────
    "triangle_l4": DimRange(9600, 9700, "triangle_l4", "Triangle L4 3-byte superposition"),
    "mul_state": DimRange(9700, 9800, "mul_state", "MUL agency state"),
    "session_buffer": DimRange(9800, 9900, "session_buffer", "Session-local buffer"),
    "ephemeral_reserved": DimRange(9900, 10000, "ephemeral_reserved", "Reserved ephemeral space"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE ACCESSORS
# ═══════════════════════════════════════════════════════════════════════════════

def get_range(name: str) -> DimRange:
    """Get dimension range by name."""
    if name not in DIMENSION_REGISTRY:
        raise KeyError(f"Unknown dimension range: {name}")
    return DIMENSION_REGISTRY[name]


def get_slice(name: str) -> slice:
    """Get slice object for dimension range."""
    return get_range(name).slice


def allocate(vec: np.ndarray, name: str, values: np.ndarray) -> None:
    """Allocate values to dimension range in vector."""
    dim = get_range(name)
    if len(values) != dim.size:
        raise ValueError(f"Expected {dim.size} values for {name}, got {len(values)}")
    vec[dim.slice] = values


def extract(vec: np.ndarray, name: str) -> np.ndarray:
    """Extract values from dimension range."""
    return vec[get_slice(name)]


# ═══════════════════════════════════════════════════════════════════════════════
# RUNG CONSTANTS — Canonical Cognitive Depth (1-9)
# ═══════════════════════════════════════════════════════════════════════════════

# 9 Cognitive Rungs - THE ONLY AUTHORIZED COGNITIVE LEVEL SYSTEM
RUNG_NAMES = [
    "OBSERVE",      # R1: Pure observation
    "REACT",        # R2: Reactive processing
    "PRACTICAL",    # R3: Practical reasoning
    "METACOG",      # R4: Metacognitive
    "SYSTEMS",      # R5: Systems thinking
    "META_SYSTEMS", # R6: Meta-systems
    "META_CUBED",   # R7: Meta³
    "SOVEREIGN",    # R8: Sovereign awareness
    "COMMUNION",    # R9: AGI/Full integration
]

# Rung descriptions for introspection
RUNG_DESCRIPTIONS = {
    1: "Pure observation - witness mode, no judgment",
    2: "Reactive processing - stimulus-response",
    3: "Practical reasoning - goal-directed problem solving",
    4: "Metacognitive - thinking about thinking",
    5: "Systems thinking - seeing interconnections",
    6: "Meta-systems - systems of systems",
    7: "Meta³ - recursive meta-awareness",
    8: "Sovereign - self-authoring consciousness",
    9: "Communion - full AGI integration",
}

# Rung thresholds for transitions
RUNG_THRESHOLDS = {
    1: 0.0,   # Always accessible
    2: 0.1,   # Minimal coherence
    3: 0.2,   # Basic coherence
    4: 0.4,   # Moderate coherence
    5: 0.5,   # Good coherence
    6: 0.6,   # High coherence
    7: 0.75,  # Very high coherence
    8: 0.85,  # Near-maximum coherence
    9: 0.95,  # Maximum coherence required
}


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_registry() -> bool:
    """Validate that dimension ranges don't overlap (except by design)."""
    sorted_ranges = sorted(DIMENSION_REGISTRY.values(), key=lambda r: r.start)

    for i in range(len(sorted_ranges) - 1):
        current = sorted_ranges[i]
        next_range = sorted_ranges[i + 1]

        # Allow gaps but not overlaps
        if current.end > next_range.start:
            # Check if intentional (same region)
            if current.end == next_range.start:
                continue
            raise ValueError(
                f"Dimension overlap: {current.name}[{current.start}:{current.end}] "
                f"overlaps {next_range.name}[{next_range.start}:{next_range.end}]"
            )

    return True


# Run validation on import
try:
    validate_registry()
except ValueError as e:
    import warnings
    warnings.warn(f"Dimension registry validation warning: {e}")


__all__ = [
    "DimRange",
    "DIMENSION_REGISTRY",
    "get_range",
    "get_slice",
    "allocate",
    "extract",
    "RUNG_NAMES",
    "RUNG_DESCRIPTIONS",
    "RUNG_THRESHOLDS",
    "validate_registry",
]
