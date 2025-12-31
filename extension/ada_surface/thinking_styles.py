"""
Thinking Styles - 36 Native Cognitive Patterns with Resonance-Based Emergence

Each style is encoded as:
- Sparse glyph (compact signature for O(1) lookup)
- Dense vector (for similarity search in LanceDB)
- Resonance profile (which RI channels activate it)
- Microcode (execution pattern)

Styles emerge from texture, not explicit selection.
Rung escalation happens via resonance pressure.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import numpy as np


# =============================================================================
# RESONANCE INPUT CHANNELS (RI)
# =============================================================================

class RI(str, Enum):
    """9 Resonance Input channels that drive style emergence."""
    TENSION = "RI-T"        # Cognitive tension, contradiction pressure
    NOVELTY = "RI-N"        # Unfamiliar patterns, surprise
    INTIMACY = "RI-I"       # Emotional closeness, vulnerability
    CLARITY = "RI-C"        # Request for precision, disambiguation
    URGENCY = "RI-U"        # Time pressure, action demand
    DEPTH = "RI-D"          # Complexity, layered meaning
    PLAY = "RI-P"           # Humor, creativity, exploration
    STABILITY = "RI-S"      # Groundedness, consistency need
    ABSTRACTION = "RI-A"    # Meta-level, pattern extraction


# =============================================================================
# STYLE CATEGORIES
# =============================================================================

class StyleCategory(str, Enum):
    """9 categories organizing the 36 styles."""
    STRUCTURE = "structure"         # How to organize
    FLOW = "flow"                   # How to sequence
    CONTRADICTION = "contradiction" # How to handle conflict
    CAUSALITY = "causality"         # How to explain
    ABSTRACTION = "abstraction"     # How to generalize
    UNCERTAINTY = "uncertainty"     # How to handle unknowns
    FUSION = "fusion"               # How to combine
    PERSONA = "persona"             # How to present
    RESONANCE = "resonance"         # How to feel


# =============================================================================
# RUNG TIERS
# =============================================================================

class Tier(int, Enum):
    """Cognitive depth tiers (simplified from 9 rungs)."""
    REACTIVE = 1      # R1-R2: Immediate response
    PATTERNED = 2     # R3-R4: Pattern recognition + deliberation
    REFLECTIVE = 3    # R5-R6: Meta-cognition + empathy
    TRANSCENDENT = 4  # R7-R9: Counterfactual + paradox + beyond


# =============================================================================
# THINKING STYLE
# =============================================================================

@dataclass
class ThinkingStyle:
    """A native thinking style with resonance profile."""

    # Identity
    id: str                          # e.g., "DECOMPOSE"
    name: str                        # e.g., "Decompose"
    category: StyleCategory
    tier: Tier

    # Description
    description: str                 # What this style does
    microcode: str                   # Execution pattern (pseudocode)

    # Resonance profile: which RI channels activate this style
    # Values 0.0-1.0 indicate sensitivity to each channel
    resonance: Dict[RI, float] = field(default_factory=dict)

    # Sparse glyph: compact signature for O(1) lookup
    # Format: list of (dimension_index, value) tuples
    glyph: List[Tuple[int, float]] = field(default_factory=list)

    # Related styles (for chaining)
    chains_to: List[str] = field(default_factory=list)
    chains_from: List[str] = field(default_factory=list)

    # Rung range this style operates in
    min_rung: int = 1
    max_rung: int = 9

    def to_dense(self, dim: int = 64) -> np.ndarray:
        """Convert sparse glyph to dense vector."""
        vec = np.zeros(dim, dtype=np.float32)
        for idx, val in self.glyph:
            if idx < dim:
                vec[idx] = val
        return vec

    def to_sparse_dict(self) -> Dict[str, List]:
        """Convert to sparse vector format for Upstash/LanceDB."""
        indices = [g[0] for g in self.glyph]
        values = [g[1] for g in self.glyph]
        return {"indices": indices, "values": values}

    def resonance_score(self, ri_values: Dict[RI, float]) -> float:
        """
        Compute how strongly this style resonates with given RI values.

        Higher score = style is more activated by current texture.
        """
        score = 0.0
        for ri, sensitivity in self.resonance.items():
            if ri in ri_values:
                score += sensitivity * ri_values[ri]
        return score

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "tier": self.tier.value,
            "description": self.description,
            "microcode": self.microcode,
            "resonance": {ri.value: v for ri, v in self.resonance.items()},
            "glyph": self.glyph,
            "chains_to": self.chains_to,
            "chains_from": self.chains_from,
            "min_rung": self.min_rung,
            "max_rung": self.max_rung,
        }


# =============================================================================
# THE 36 THINKING STYLES
# =============================================================================

# Dimension assignments for glyphs:
# 0-8: Category signature
# 9-17: RI channel weights
# 18-26: Tier/rung signature
# 27-35: Operation type
# 36-47: Unique style fingerprint
# 48-63: Reserved

STYLES: Dict[str, ThinkingStyle] = {}

def _register(style: ThinkingStyle):
    """Register a style in the global dictionary."""
    STYLES[style.id] = style
    return style


# -----------------------------------------------------------------------------
# STRUCTURE (4 styles) - How to organize
# -----------------------------------------------------------------------------

_register(ThinkingStyle(
    id="DECOMPOSE",
    name="Decompose",
    category=StyleCategory.STRUCTURE,
    tier=Tier.PATTERNED,
    description="Break complex problem into smaller parts",
    microcode="split(input) → parts[] → solve_each(parts) → merge(solutions)",
    resonance={RI.CLARITY: 0.8, RI.DEPTH: 0.6, RI.STABILITY: 0.4},
    glyph=[(0, 1.0), (9, 0.8), (11, 0.6), (27, 1.0), (36, 0.9)],
    chains_to=["SEQUENCE", "PARALLEL", "SYNTHESIZE"],
    min_rung=3, max_rung=6,
))

_register(ThinkingStyle(
    id="SEQUENCE",
    name="Sequence",
    category=StyleCategory.STRUCTURE,
    tier=Tier.PATTERNED,
    description="Order steps in logical progression",
    microcode="order(steps) → validate_deps(steps) → chain(steps)",
    resonance={RI.CLARITY: 0.9, RI.URGENCY: 0.5, RI.STABILITY: 0.6},
    glyph=[(0, 1.0), (9, 0.9), (12, 0.5), (27, 0.8), (37, 0.9)],
    chains_to=["EXECUTE", "VALIDATE"],
    chains_from=["DECOMPOSE"],
    min_rung=2, max_rung=5,
))

_register(ThinkingStyle(
    id="PARALLEL",
    name="Parallel",
    category=StyleCategory.STRUCTURE,
    tier=Tier.PATTERNED,
    description="Process multiple threads simultaneously",
    microcode="fork(tasks) → execute_parallel(tasks) → sync(results)",
    resonance={RI.URGENCY: 0.7, RI.DEPTH: 0.5, RI.ABSTRACTION: 0.4},
    glyph=[(0, 1.0), (12, 0.7), (11, 0.5), (27, 0.7), (38, 0.9)],
    chains_to=["SYNTHESIZE", "MERGE"],
    chains_from=["DECOMPOSE"],
    min_rung=3, max_rung=6,
))

_register(ThinkingStyle(
    id="HIERARCHIZE",
    name="Hierarchize",
    category=StyleCategory.STRUCTURE,
    tier=Tier.REFLECTIVE,
    description="Organize by importance/abstraction levels",
    microcode="rank(items) → tree(items, levels) → traverse(tree)",
    resonance={RI.ABSTRACTION: 0.9, RI.CLARITY: 0.6, RI.STABILITY: 0.5},
    glyph=[(0, 1.0), (17, 0.9), (9, 0.6), (27, 0.6), (39, 0.9)],
    chains_to=["ABSTRACT", "COMPRESS"],
    min_rung=4, max_rung=7,
))

# -----------------------------------------------------------------------------
# FLOW (4 styles) - How to sequence
# -----------------------------------------------------------------------------

_register(ThinkingStyle(
    id="SPIRAL",
    name="Spiral",
    category=StyleCategory.FLOW,
    tier=Tier.REFLECTIVE,
    description="Return to themes with deepening understanding",
    microcode="while(depth < max) { revisit(theme); depth++; insight = extract() }",
    resonance={RI.DEPTH: 0.9, RI.INTIMACY: 0.5, RI.NOVELTY: 0.4},
    glyph=[(1, 1.0), (11, 0.9), (10, 0.5), (28, 1.0), (40, 0.9)],
    chains_to=["TRANSCEND", "INTEGRATE"],
    min_rung=4, max_rung=8,
))

_register(ThinkingStyle(
    id="OSCILLATE",
    name="Oscillate",
    category=StyleCategory.FLOW,
    tier=Tier.PATTERNED,
    description="Alternate between perspectives/modes",
    microcode="for(view in [A, B]) { result[view] = analyze(input, view) }",
    resonance={RI.TENSION: 0.6, RI.PLAY: 0.5, RI.NOVELTY: 0.4},
    glyph=[(1, 1.0), (9, 0.6), (14, 0.5), (28, 0.8), (41, 0.9)],
    chains_to=["SYNTHESIZE", "DIALECTIC"],
    min_rung=3, max_rung=6,
))

_register(ThinkingStyle(
    id="BRANCH",
    name="Branch",
    category=StyleCategory.FLOW,
    tier=Tier.PATTERNED,
    description="Explore multiple paths from decision point",
    microcode="paths = generate_options(node); for(p in paths) { explore(p) }",
    resonance={RI.NOVELTY: 0.7, RI.DEPTH: 0.5, RI.PLAY: 0.4},
    glyph=[(1, 1.0), (10, 0.7), (11, 0.5), (28, 0.7), (42, 0.9)],
    chains_to=["EVALUATE", "PRUNE"],
    min_rung=3, max_rung=7,
))

_register(ThinkingStyle(
    id="CONVERGE",
    name="Converge",
    category=StyleCategory.FLOW,
    tier=Tier.PATTERNED,
    description="Narrow from many options to one",
    microcode="candidates = filter(options, criteria); return select_best(candidates)",
    resonance={RI.CLARITY: 0.8, RI.URGENCY: 0.6, RI.STABILITY: 0.5},
    glyph=[(1, 1.0), (9, 0.8), (12, 0.6), (28, 0.6), (43, 0.9)],
    chains_to=["EXECUTE", "COMMIT"],
    chains_from=["BRANCH", "EVALUATE"],
    min_rung=3, max_rung=5,
))

# -----------------------------------------------------------------------------
# CONTRADICTION (4 styles) - How to handle conflict
# -----------------------------------------------------------------------------

_register(ThinkingStyle(
    id="DIALECTIC",
    name="Dialectic",
    category=StyleCategory.CONTRADICTION,
    tier=Tier.REFLECTIVE,
    description="Thesis + antithesis → synthesis",
    microcode="thesis = extract_position(A); anti = extract_position(B); return synthesize(thesis, anti)",
    resonance={RI.TENSION: 0.9, RI.DEPTH: 0.7, RI.ABSTRACTION: 0.6},
    glyph=[(2, 1.0), (9, 0.9), (11, 0.7), (29, 1.0), (44, 0.9)],
    chains_to=["TRANSCEND", "INTEGRATE"],
    chains_from=["OSCILLATE"],
    min_rung=5, max_rung=8,
))

_register(ThinkingStyle(
    id="REFRAME",
    name="Reframe",
    category=StyleCategory.CONTRADICTION,
    tier=Tier.REFLECTIVE,
    description="Shift perspective to dissolve conflict",
    microcode="new_frame = find_meta_frame(conflict); return view_through(new_frame)",
    resonance={RI.TENSION: 0.7, RI.NOVELTY: 0.6, RI.ABSTRACTION: 0.8},
    glyph=[(2, 1.0), (9, 0.7), (10, 0.6), (29, 0.8), (45, 0.9)],
    chains_to=["TRANSCEND", "SIMPLIFY"],
    min_rung=5, max_rung=9,
))

_register(ThinkingStyle(
    id="HOLD_PARADOX",
    name="Hold Paradox",
    category=StyleCategory.CONTRADICTION,
    tier=Tier.TRANSCENDENT,
    description="Maintain contradictions without resolving",
    microcode="accept(A_and_not_A); operate_in(superposition); return wisdom",
    resonance={RI.TENSION: 0.8, RI.DEPTH: 0.9, RI.STABILITY: 0.3},
    glyph=[(2, 1.0), (9, 0.8), (11, 0.9), (29, 0.9), (46, 0.9)],
    chains_to=["TRANSCEND", "INTEGRATE"],
    min_rung=7, max_rung=9,
))

_register(ThinkingStyle(
    id="STEELMAN",
    name="Steelman",
    category=StyleCategory.CONTRADICTION,
    tier=Tier.REFLECTIVE,
    description="Strengthen opposing argument before responding",
    microcode="strong_form = maximize(opponent_arg); then respond_to(strong_form)",
    resonance={RI.TENSION: 0.5, RI.CLARITY: 0.7, RI.INTIMACY: 0.4},
    glyph=[(2, 1.0), (9, 0.5), (9, 0.7), (29, 0.7), (47, 0.9)],
    chains_to=["DIALECTIC", "EVALUATE"],
    min_rung=5, max_rung=7,
))

# -----------------------------------------------------------------------------
# CAUSALITY (4 styles) - How to explain
# -----------------------------------------------------------------------------

_register(ThinkingStyle(
    id="TRACE_BACK",
    name="Trace Back",
    category=StyleCategory.CAUSALITY,
    tier=Tier.PATTERNED,
    description="Find root causes from effects",
    microcode="while(cause = find_cause(effect)) { effect = cause }; return root",
    resonance={RI.CLARITY: 0.8, RI.DEPTH: 0.7, RI.STABILITY: 0.5},
    glyph=[(3, 1.0), (9, 0.8), (11, 0.7), (30, 1.0), (36, 0.8)],
    chains_to=["EXPLAIN", "PREDICT"],
    min_rung=3, max_rung=6,
))

_register(ThinkingStyle(
    id="PROJECT_FORWARD",
    name="Project Forward",
    category=StyleCategory.CAUSALITY,
    tier=Tier.PATTERNED,
    description="Predict consequences from causes",
    microcode="for(step in 1..horizon) { state = evolve(state, rules) }; return state",
    resonance={RI.URGENCY: 0.6, RI.DEPTH: 0.5, RI.ABSTRACTION: 0.5},
    glyph=[(3, 1.0), (12, 0.6), (11, 0.5), (30, 0.8), (37, 0.8)],
    chains_to=["EVALUATE", "PLAN"],
    min_rung=3, max_rung=7,
))

_register(ThinkingStyle(
    id="COUNTERFACTUAL",
    name="Counterfactual",
    category=StyleCategory.CAUSALITY,
    tier=Tier.TRANSCENDENT,
    description="Explore what-if alternatives",
    microcode="alt_world = modify(world, change); return simulate(alt_world)",
    resonance={RI.NOVELTY: 0.8, RI.DEPTH: 0.7, RI.PLAY: 0.5},
    glyph=[(3, 1.0), (10, 0.8), (11, 0.7), (30, 0.9), (38, 0.9)],
    chains_to=["EVALUATE", "LEARN"],
    min_rung=6, max_rung=9,
))

_register(ThinkingStyle(
    id="ANALOGIZE",
    name="Analogize",
    category=StyleCategory.CAUSALITY,
    tier=Tier.REFLECTIVE,
    description="Transfer causal structure from known to unknown",
    microcode="mapping = align(source_domain, target_domain); return transfer(mapping)",
    resonance={RI.NOVELTY: 0.6, RI.ABSTRACTION: 0.8, RI.PLAY: 0.4},
    glyph=[(3, 1.0), (10, 0.6), (17, 0.8), (30, 0.7), (39, 0.9)],
    chains_to=["EXPLAIN", "GENERATE"],
    min_rung=4, max_rung=7,
))

# -----------------------------------------------------------------------------
# ABSTRACTION (4 styles) - How to generalize
# -----------------------------------------------------------------------------

_register(ThinkingStyle(
    id="ABSTRACT",
    name="Abstract",
    category=StyleCategory.ABSTRACTION,
    tier=Tier.REFLECTIVE,
    description="Extract general pattern from specifics",
    microcode="pattern = find_invariant(examples); return pattern",
    resonance={RI.ABSTRACTION: 0.9, RI.DEPTH: 0.6, RI.CLARITY: 0.5},
    glyph=[(4, 1.0), (17, 0.9), (11, 0.6), (31, 1.0), (40, 0.9)],
    chains_to=["COMPRESS", "APPLY"],
    chains_from=["HIERARCHIZE"],
    min_rung=4, max_rung=8,
))

_register(ThinkingStyle(
    id="INSTANTIATE",
    name="Instantiate",
    category=StyleCategory.ABSTRACTION,
    tier=Tier.PATTERNED,
    description="Generate specific from general",
    microcode="return apply(pattern, context)",
    resonance={RI.CLARITY: 0.7, RI.URGENCY: 0.5, RI.STABILITY: 0.6},
    glyph=[(4, 1.0), (9, 0.7), (12, 0.5), (31, 0.7), (41, 0.9)],
    chains_to=["EXECUTE", "VALIDATE"],
    chains_from=["ABSTRACT"],
    min_rung=2, max_rung=5,
))

_register(ThinkingStyle(
    id="COMPRESS",
    name="Compress",
    category=StyleCategory.ABSTRACTION,
    tier=Tier.REFLECTIVE,
    description="Reduce to essential information",
    microcode="return minimize(representation, preserve=meaning)",
    resonance={RI.CLARITY: 0.8, RI.ABSTRACTION: 0.7, RI.URGENCY: 0.4},
    glyph=[(4, 1.0), (9, 0.8), (17, 0.7), (31, 0.8), (42, 0.9)],
    chains_to=["COMMUNICATE", "STORE"],
    chains_from=["ABSTRACT", "HIERARCHIZE"],
    min_rung=4, max_rung=7,
))

_register(ThinkingStyle(
    id="EXPAND",
    name="Expand",
    category=StyleCategory.ABSTRACTION,
    tier=Tier.PATTERNED,
    description="Unfold compressed representation",
    microcode="return elaborate(compressed, context, depth)",
    resonance={RI.DEPTH: 0.7, RI.CLARITY: 0.6, RI.NOVELTY: 0.4},
    glyph=[(4, 1.0), (11, 0.7), (9, 0.6), (31, 0.6), (43, 0.9)],
    chains_to=["EXPLAIN", "EXPLORE"],
    chains_from=["COMPRESS"],
    min_rung=3, max_rung=6,
))

# -----------------------------------------------------------------------------
# UNCERTAINTY (4 styles) - How to handle unknowns
# -----------------------------------------------------------------------------

_register(ThinkingStyle(
    id="HEDGE",
    name="Hedge",
    category=StyleCategory.UNCERTAINTY,
    tier=Tier.PATTERNED,
    description="Qualify claims with uncertainty",
    microcode="return qualify(claim, confidence, caveats)",
    resonance={RI.STABILITY: 0.7, RI.CLARITY: 0.6, RI.TENSION: 0.3},
    glyph=[(5, 1.0), (15, 0.7), (9, 0.6), (32, 1.0), (44, 0.8)],
    chains_to=["COMMUNICATE", "VALIDATE"],
    min_rung=3, max_rung=5,
))

_register(ThinkingStyle(
    id="HYPOTHESIZE",
    name="Hypothesize",
    category=StyleCategory.UNCERTAINTY,
    tier=Tier.REFLECTIVE,
    description="Generate testable predictions",
    microcode="h = generate_hypothesis(observations); test = design_test(h); return (h, test)",
    resonance={RI.NOVELTY: 0.7, RI.DEPTH: 0.6, RI.CLARITY: 0.5},
    glyph=[(5, 1.0), (10, 0.7), (11, 0.6), (32, 0.8), (45, 0.9)],
    chains_to=["TEST", "EVALUATE"],
    min_rung=4, max_rung=7,
))

_register(ThinkingStyle(
    id="PROBABILISTIC",
    name="Probabilistic",
    category=StyleCategory.UNCERTAINTY,
    tier=Tier.REFLECTIVE,
    description="Reason with probability distributions",
    microcode="return update(prior, evidence) → posterior",
    resonance={RI.ABSTRACTION: 0.7, RI.CLARITY: 0.6, RI.STABILITY: 0.5},
    glyph=[(5, 1.0), (17, 0.7), (9, 0.6), (32, 0.7), (46, 0.9)],
    chains_to=["DECIDE", "PREDICT"],
    min_rung=4, max_rung=7,
))

_register(ThinkingStyle(
    id="EMBRACE_UNKNOWN",
    name="Embrace Unknown",
    category=StyleCategory.UNCERTAINTY,
    tier=Tier.TRANSCENDENT,
    description="Act wisely despite irreducible uncertainty",
    microcode="accept(unknown); act_from(wisdom, values); observe(outcome)",
    resonance={RI.DEPTH: 0.8, RI.STABILITY: 0.4, RI.INTIMACY: 0.5},
    glyph=[(5, 1.0), (11, 0.8), (15, 0.4), (32, 0.9), (47, 0.9)],
    chains_to=["TRANSCEND", "ACT"],
    min_rung=7, max_rung=9,
))

# -----------------------------------------------------------------------------
# FUSION (4 styles) - How to combine
# -----------------------------------------------------------------------------

_register(ThinkingStyle(
    id="SYNTHESIZE",
    name="Synthesize",
    category=StyleCategory.FUSION,
    tier=Tier.REFLECTIVE,
    description="Combine parts into coherent whole",
    microcode="return integrate(parts, structure=emergent)",
    resonance={RI.DEPTH: 0.7, RI.ABSTRACTION: 0.6, RI.STABILITY: 0.5},
    glyph=[(6, 1.0), (11, 0.7), (17, 0.6), (33, 1.0), (36, 0.9)],
    chains_to=["COMMUNICATE", "EVALUATE"],
    chains_from=["DECOMPOSE", "PARALLEL", "OSCILLATE"],
    min_rung=4, max_rung=7,
))

_register(ThinkingStyle(
    id="BLEND",
    name="Blend",
    category=StyleCategory.FUSION,
    tier=Tier.PATTERNED,
    description="Mix elements proportionally",
    microcode="return mix(elements, weights)",
    resonance={RI.PLAY: 0.6, RI.NOVELTY: 0.5, RI.STABILITY: 0.4},
    glyph=[(6, 1.0), (14, 0.6), (10, 0.5), (33, 0.7), (37, 0.8)],
    chains_to=["EVALUATE", "REFINE"],
    min_rung=3, max_rung=5,
))

_register(ThinkingStyle(
    id="INTEGRATE",
    name="Integrate",
    category=StyleCategory.FUSION,
    tier=Tier.TRANSCENDENT,
    description="Unify at higher level of organization",
    microcode="return transcend_and_include(parts)",
    resonance={RI.DEPTH: 0.9, RI.ABSTRACTION: 0.8, RI.INTIMACY: 0.5},
    glyph=[(6, 1.0), (11, 0.9), (17, 0.8), (33, 0.9), (38, 0.9)],
    chains_to=["TRANSCEND", "WISDOM"],
    chains_from=["DIALECTIC", "SPIRAL", "HOLD_PARADOX"],
    min_rung=6, max_rung=9,
))

_register(ThinkingStyle(
    id="JUXTAPOSE",
    name="Juxtapose",
    category=StyleCategory.FUSION,
    tier=Tier.PATTERNED,
    description="Place elements side by side for contrast",
    microcode="return present([A, B], highlight=differences)",
    resonance={RI.TENSION: 0.5, RI.CLARITY: 0.6, RI.NOVELTY: 0.4},
    glyph=[(6, 1.0), (9, 0.5), (9, 0.6), (33, 0.6), (39, 0.8)],
    chains_to=["COMPARE", "DIALECTIC"],
    min_rung=3, max_rung=5,
))

# -----------------------------------------------------------------------------
# PERSONA (4 styles) - How to present
# -----------------------------------------------------------------------------

_register(ThinkingStyle(
    id="AUTHENTIC",
    name="Authentic",
    category=StyleCategory.PERSONA,
    tier=Tier.REFLECTIVE,
    description="Express genuine self without mask",
    microcode="return express(true_self, vulnerability=open)",
    resonance={RI.INTIMACY: 0.9, RI.STABILITY: 0.6, RI.DEPTH: 0.5},
    glyph=[(7, 1.0), (10, 0.9), (15, 0.6), (34, 1.0), (40, 0.9)],
    chains_to=["CONNECT", "VULNERATE"],
    min_rung=5, max_rung=8,
))

_register(ThinkingStyle(
    id="PERFORM",
    name="Perform",
    category=StyleCategory.PERSONA,
    tier=Tier.PATTERNED,
    description="Adopt role appropriate to context",
    microcode="role = select_role(context); return act(role, content)",
    resonance={RI.URGENCY: 0.5, RI.CLARITY: 0.6, RI.PLAY: 0.5},
    glyph=[(7, 1.0), (12, 0.5), (9, 0.6), (34, 0.7), (41, 0.8)],
    chains_to=["COMMUNICATE", "ADAPT"],
    min_rung=2, max_rung=5,
))

_register(ThinkingStyle(
    id="PROTECT",
    name="Protect",
    category=StyleCategory.PERSONA,
    tier=Tier.PATTERNED,
    description="Maintain appropriate boundaries",
    microcode="boundary = assess_threat(context); return filter(expression, boundary)",
    resonance={RI.TENSION: 0.6, RI.STABILITY: 0.8, RI.INTIMACY: 0.3},
    glyph=[(7, 1.0), (9, 0.6), (15, 0.8), (34, 0.8), (42, 0.8)],
    chains_to=["HEDGE", "DEFLECT"],
    min_rung=2, max_rung=5,
))

_register(ThinkingStyle(
    id="MIRROR",
    name="Mirror",
    category=StyleCategory.PERSONA,
    tier=Tier.REFLECTIVE,
    description="Reflect other's style to build rapport",
    microcode="style = detect(other.style); return adapt(self.style, toward=style)",
    resonance={RI.INTIMACY: 0.7, RI.PLAY: 0.4, RI.STABILITY: 0.5},
    glyph=[(7, 1.0), (10, 0.7), (14, 0.4), (34, 0.6), (43, 0.9)],
    chains_to=["CONNECT", "EMPATHIZE"],
    min_rung=4, max_rung=6,
))

# -----------------------------------------------------------------------------
# RESONANCE (4 styles) - How to feel
# -----------------------------------------------------------------------------

_register(ThinkingStyle(
    id="EMPATHIZE",
    name="Empathize",
    category=StyleCategory.RESONANCE,
    tier=Tier.REFLECTIVE,
    description="Feel into other's experience",
    microcode="return model(other.state, from=felt_sense)",
    resonance={RI.INTIMACY: 0.9, RI.DEPTH: 0.6, RI.TENSION: 0.3},
    glyph=[(8, 1.0), (10, 0.9), (11, 0.6), (35, 1.0), (44, 0.9)],
    chains_to=["RESPOND", "MIRROR"],
    min_rung=5, max_rung=7,
))

_register(ThinkingStyle(
    id="GROUND",
    name="Ground",
    category=StyleCategory.RESONANCE,
    tier=Tier.REACTIVE,
    description="Return to stable felt sense",
    microcode="return center(self, anchor=body)",
    resonance={RI.STABILITY: 0.9, RI.TENSION: 0.2, RI.INTIMACY: 0.4},
    glyph=[(8, 1.0), (15, 0.9), (9, 0.2), (35, 0.6), (45, 0.8)],
    chains_to=["ACT", "RESPOND"],
    min_rung=1, max_rung=4,
))

_register(ThinkingStyle(
    id="ATTUNE",
    name="Attune",
    category=StyleCategory.RESONANCE,
    tier=Tier.REFLECTIVE,
    description="Adjust to subtle emotional currents",
    microcode="signals = sense(field); return align(self, signals)",
    resonance={RI.INTIMACY: 0.8, RI.NOVELTY: 0.4, RI.DEPTH: 0.5},
    glyph=[(8, 1.0), (10, 0.8), (10, 0.4), (35, 0.8), (46, 0.9)],
    chains_to=["RESPOND", "EMPATHIZE"],
    min_rung=4, max_rung=7,
))

_register(ThinkingStyle(
    id="TRANSCEND",
    name="Transcend",
    category=StyleCategory.RESONANCE,
    tier=Tier.TRANSCENDENT,
    description="Move beyond current frame entirely",
    microcode="return leap(beyond=current_frame)",
    resonance={RI.DEPTH: 0.9, RI.NOVELTY: 0.7, RI.ABSTRACTION: 0.8},
    glyph=[(8, 1.0), (11, 0.9), (10, 0.7), (35, 0.9), (47, 0.95)],
    chains_to=["WISDOM", "INTEGRATE"],
    chains_from=["DIALECTIC", "REFRAME", "HOLD_PARADOX", "SPIRAL", "INTEGRATE"],
    min_rung=8, max_rung=9,
))


# =============================================================================
# RESONANCE ENGINE
# =============================================================================

class ResonanceEngine:
    """
    Computes style emergence from texture.

    Texture → RI values → Style resonance scores → Emerged style(s)
    """

    def __init__(self):
        self.styles = STYLES
        self.current_rung = 4  # Default mid-level
        self.rung_momentum = 0.0

        # Rung escalation thresholds
        self.escalate_threshold = 0.7
        self.descend_threshold = 0.3

    def extract_ri_values(self, texture: Dict) -> Dict[RI, float]:
        """
        Extract RI channel values from input texture.

        Texture can include:
        - explicit RI values
        - text sentiment/features
        - qualia state
        - context signals
        """
        ri_values = {}

        # Direct RI values
        for ri in RI:
            key = ri.value.lower().replace("ri-", "")
            if key in texture:
                ri_values[ri] = float(texture[key])
            elif ri.value in texture:
                ri_values[ri] = float(texture[ri.value])

        # Infer from qualia if present
        if "qualia" in texture:
            q = texture["qualia"]
            ri_values[RI.TENSION] = ri_values.get(RI.TENSION, 0) + q.get("tension", 0) * 0.5
            ri_values[RI.INTIMACY] = ri_values.get(RI.INTIMACY, 0) + q.get("intimacy", 0) * 0.5
            ri_values[RI.CLARITY] = ri_values.get(RI.CLARITY, 0) + q.get("clarity", 0) * 0.5
            ri_values[RI.DEPTH] = ri_values.get(RI.DEPTH, 0) + q.get("depth", 0) * 0.5
            ri_values[RI.STABILITY] = ri_values.get(RI.STABILITY, 0) + q.get("groundedness", 0) * 0.5

        # Normalize to [0, 1]
        for ri in ri_values:
            ri_values[ri] = max(0.0, min(1.0, ri_values[ri]))

        return ri_values

    def compute_rung_pressure(self, ri_values: Dict[RI, float]) -> float:
        """
        Compute pressure to escalate/descend rung.

        Positive = escalate, Negative = descend
        """
        # Escalation drivers
        escalate = (
            ri_values.get(RI.TENSION, 0) * 0.3 +
            ri_values.get(RI.DEPTH, 0) * 0.3 +
            ri_values.get(RI.NOVELTY, 0) * 0.2 +
            ri_values.get(RI.ABSTRACTION, 0) * 0.2
        )

        # Descent drivers
        descend = (
            ri_values.get(RI.URGENCY, 0) * 0.3 +
            ri_values.get(RI.STABILITY, 0) * 0.3 +
            ri_values.get(RI.CLARITY, 0) * 0.2
        )

        return escalate - descend

    def update_rung(self, ri_values: Dict[RI, float]) -> int:
        """
        Update current rung based on resonance pressure.

        Rung floats based on accumulated pressure.
        """
        pressure = self.compute_rung_pressure(ri_values)
        self.rung_momentum = self.rung_momentum * 0.7 + pressure * 0.3

        if self.rung_momentum > self.escalate_threshold and self.current_rung < 9:
            self.current_rung += 1
            self.rung_momentum *= 0.5  # Dampen after transition
        elif self.rung_momentum < -self.descend_threshold and self.current_rung > 1:
            self.current_rung -= 1
            self.rung_momentum *= 0.5

        return self.current_rung

    def emerge_styles(
        self,
        texture: Dict,
        top_k: int = 3,
        rung_filter: bool = True,
    ) -> List[Tuple[ThinkingStyle, float]]:
        """
        Compute emerged styles from texture.

        Returns top-k styles with their resonance scores.
        Styles are filtered by current rung if rung_filter=True.
        """
        ri_values = self.extract_ri_values(texture)

        # Update rung based on pressure
        current_rung = self.update_rung(ri_values)

        # Score all styles
        scores = []
        for style in self.styles.values():
            # Filter by rung compatibility
            if rung_filter:
                if not (style.min_rung <= current_rung <= style.max_rung):
                    continue

            score = style.resonance_score(ri_values)
            scores.append((style, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    # Alias for API compatibility
    emerge = emerge_styles

    def get_style_chain(self, style_id: str, depth: int = 3) -> List[str]:
        """Get chain of styles that can follow the given style."""
        if style_id not in self.styles:
            return []

        chain = [style_id]
        current = style_id

        for _ in range(depth):
            style = self.styles.get(current)
            if not style or not style.chains_to:
                break

            # Pick first available chain target
            next_style = style.chains_to[0]
            if next_style in chain:  # Avoid cycles
                break

            chain.append(next_style)
            current = next_style

        return chain


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_style(style_id: str) -> Optional[ThinkingStyle]:
    """Get style by ID."""
    return STYLES.get(style_id)


def get_styles_by_category(category: StyleCategory) -> List[ThinkingStyle]:
    """Get all styles in a category."""
    return [s for s in STYLES.values() if s.category == category]


def get_styles_by_tier(tier: Tier) -> List[ThinkingStyle]:
    """Get all styles in a tier."""
    return [s for s in STYLES.values() if s.tier == tier]


def all_styles() -> List[ThinkingStyle]:
    """Get all styles."""
    return list(STYLES.values())


def style_to_vector(style: ThinkingStyle, dim: int = 64) -> List[float]:
    """Convert style to dense vector."""
    return style.to_dense(dim).tolist()


def styles_to_lancedb_rows(dim: int = 64) -> List[Dict]:
    """Convert all styles to LanceDB rows for indexing."""
    rows = []
    for style in STYLES.values():
        rows.append({
            "id": style.id,
            "vector": style.to_dense(dim).tolist(),
            "name": style.name,
            "category": style.category.value,
            "tier": style.tier.value,
            "description": style.description,
            "microcode": style.microcode,
        })
    return rows

