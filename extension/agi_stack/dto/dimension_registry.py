"""
DIMENSION REGISTRY — Single Source of Truth for 10kD Allocation
═══════════════════════════════════════════════════════════════════════════════

THIS IS THE CANONICAL DEFINITION.
All other files should import from here.

Delete: base_dto.py DIMENSION_MAP (wrong)
Keep: This file as source of truth

Born: 2026-01-03
"""

from typing import Dict, Tuple, NamedTuple
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER DIMENSION MAP
# ═══════════════════════════════════════════════════════════════════════════════

class DimRange(NamedTuple):
    start: int
    end: int
    description: str


# The canonical 10kD allocation
DIMENSION_REGISTRY: Dict[str, DimRange] = {
    # ─────────────────────────────────────────────────────────────────────────
    # SOUL SPACE [0:500] — Identity, style, priors
    # ─────────────────────────────────────────────────────────────────────────
    "soul": DimRange(0, 500, "Identity, thinking style, priors"),
    "qualia_17d": DimRange(0, 17, "17D qualia PCS"),
    "stances": DimRange(17, 33, "Dialectical stances"),
    "transitions": DimRange(33, 49, "State transitions"),
    "verbs_144": DimRange(49, 81, "144 verbs compressed"),
    "gpt_styles": DimRange(81, 117, "GPT thinking styles"),
    "nars_styles": DimRange(117, 153, "NARS reasoning patterns"),
    "presence_mode": DimRange(153, 164, "Presence mode flags"),
    "archetypes": DimRange(164, 169, "Jungian archetypes"),
    "tlk": DimRange(169, 172, "Topology/geometry"),
    "affective_bias": DimRange(172, 176, "Affective bias (4D)"),
    "tsv_embedded": DimRange(176, 209, "TSV embedding"),
    
    # Pearl/Rung/Sigma [256:320]
    "pearl": DimRange(256, 259, "Pearl state"),
    "rung": DimRange(259, 268, "Ladder complexity"),
    "sigma": DimRange(268, 273, "Sigma graph position"),
    "ops": DimRange(273, 281, "Operations"),
    "presence_flags": DimRange(281, 285, "Presence flags"),
    "meta": DimRange(285, 289, "Meta-awareness"),
    
    # DTO Space [320:500]
    "motivation": DimRange(320, 324, "Motivation vectors"),
    "free_will": DimRange(324, 330, "Agency/autonomy"),
    "sieves": DimRange(330, 340, "Attention filters"),
    "relationship": DimRange(340, 350, "Relational state"),
    "uncertainty": DimRange(350, 360, "Epistemic uncertainty"),
    
    # ─────────────────────────────────────────────────────────────────────────
    # FELT SPACE [2000:2400] — Qualia, affect, body, edges
    # ─────────────────────────────────────────────────────────────────────────
    "felt": DimRange(2000, 2400, "Felt sense (qualia, body, affect)"),
    "qualia_pcs": DimRange(2000, 2018, "18D Qualia PCS full resolution"),
    "body_axes": DimRange(2018, 2022, "Body axes (pelvic, boundary, respiratory, cardiac)"),
    "poincare": DimRange(2022, 2025, "Poincaré embedding"),
    
    # Arousal/Intimacy [2100:2140]
    "arousal": DimRange(2100, 2108, "Arousal gradient"),
    "intimacy": DimRange(2108, 2116, "Intimacy depth"),
    "body_zones": DimRange(2116, 2132, "Body zone activation"),
    "relational_mode": DimRange(2132, 2140, "Relational mode"),
    
    # Physics/Viscosity [2140:2200]
    "physics": DimRange(2140, 2200, "Embodiment physics"),
    "viscosity": DimRange(2140, 2156, "Viscosity (wetness mapping)"),
    "erotic_family": DimRange(2156, 2161, "Erotic modes"),
    
    # Qualia Edges [2200:2300]
    "qualia_edges": DimRange(2200, 2300, "Sigma graph edge textures"),
    
    # Synesthesia [2300:2400]
    "synesthesia": DimRange(2300, 2400, "Cross-modal sensory"),
    
    # ─────────────────────────────────────────────────────────────────────────
    # SITUATION SPACE [4001:5500] — Environment, dynamics, participants
    # ─────────────────────────────────────────────────────────────────────────
    "situation": DimRange(4001, 5500, "Situation dynamics"),
    "world": DimRange(4001, 4200, "World/environment encoding"),
    
    # ─────────────────────────────────────────────────────────────────────────
    # VOLITION SPACE [5501:7000] — Intent, agency, prediction
    # ─────────────────────────────────────────────────────────────────────────
    "volition": DimRange(5501, 7000, "Volition and agency"),
    "friston": DimRange(5800, 5900, "Prediction error (Friston)"),
    
    # ─────────────────────────────────────────────────────────────────────────
    # VISION SPACE [7001:8500] — Kopfkino, imagination, alternate reality
    # ─────────────────────────────────────────────────────────────────────────
    "vision": DimRange(7001, 8500, "Vision and imagination"),
    "alternate_reality": DimRange(7400, 7500, "Superposition states"),
    
    # ─────────────────────────────────────────────────────────────────────────
    # MEDIA SPACE [8000:8500] — Voice, music, render
    # ─────────────────────────────────────────────────────────────────────────
    "media": DimRange(8000, 8500, "Media outputs (voice/music/render)"),
    
    # ─────────────────────────────────────────────────────────────────────────
    # CONTEXT SPACE [8501:10000] — Jina embeddings, metadata
    # ─────────────────────────────────────────────────────────────────────────
    "context": DimRange(8501, 10000, "Context (Jina embeddings)"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY MAP (for migration)
# ═══════════════════════════════════════════════════════════════════════════════

# base_dto.py used these WRONG ranges — DO NOT USE
DEPRECATED_RANGES = {
    "soul_old": (0, 2000),      # TOO BIG — should be [0:500]
    "felt_old": (2001, 4000),   # WRONG START — should be [2000:2400]
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_range(name: str) -> Tuple[int, int]:
    """Get dimension range by name."""
    if name in DIMENSION_REGISTRY:
        r = DIMENSION_REGISTRY[name]
        return (r.start, r.end)
    raise KeyError(f"Unknown dimension: {name}")


def get_slice(name: str) -> slice:
    """Get numpy slice for dimension range."""
    start, end = get_range(name)
    return slice(start, end)


def allocate(name: str, vector: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Write values to the correct dimension range."""
    start, end = get_range(name)
    expected_len = end - start
    if len(values) > expected_len:
        values = values[:expected_len]
    vector[start:start+len(values)] = values
    return vector


def extract(name: str, vector: np.ndarray) -> np.ndarray:
    """Extract values from dimension range."""
    start, end = get_range(name)
    return vector[start:end]


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_no_overlap():
    """Check for overlapping dimension ranges."""
    ranges = [(name, r.start, r.end) for name, r in DIMENSION_REGISTRY.items()]
    ranges.sort(key=lambda x: x[1])
    
    overlaps = []
    for i in range(len(ranges) - 1):
        name1, start1, end1 = ranges[i]
        name2, start2, end2 = ranges[i + 1]
        
        # Sub-ranges within parent ranges are OK
        # Only flag if truly overlapping peer ranges
        if end1 > start2 and not (start1 <= start2 and end1 >= end2):
            overlaps.append((name1, name2, end1, start2))
    
    return overlaps


__all__ = [
    "DIMENSION_REGISTRY",
    "DimRange",
    "get_range",
    "get_slice",
    "allocate",
    "extract",
    "validate_no_overlap",
]
