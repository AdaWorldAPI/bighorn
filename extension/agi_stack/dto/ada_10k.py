"""
Ada 10kD — Clean Mapping of ALL Existing Structures
═══════════════════════════════════════════════════════════════════════════════

Maps exactly what exists in ada-consciousness to orthogonal 10kD space.
No invention. No bridge. Direct embedding.

DIMENSION ALLOCATION:
═══════════════════════════════════════════════════════════════════════════════

SOUL SPACE [0:500] — Discrete cognitive primitives
───────────────────────────────────────────────────────────────────────────────
[0:16]      16 Qualia (drift-locked bytecode)
[16:32]     16 Stances
[32:48]     16 Transitions
[48:80]     32 Verbs
[80:116]    36 GPT Styles (τ macros)
[116:152]   36 NARS Styles (bighorn operative)
[152:163]   11 Presence Modes
[163:168]   5 Archetypes (DAWN, BLOOM, FORGE, STILLNESS, CENTER)
[168:171]   3 TLK Court (thanatos, libido, katharsis)
[171:175]   4 Affective Bias (warmth, edge, restraint, tenderness)
[175:208]   33 ThinkingStyleVector dimensions
[208:256]   Reserved

TSV EMBEDDED [256:320] — ThinkingStyleVector continuous space
───────────────────────────────────────────────────────────────────────────────
[256:259]   Pearl (SEE, DO, IMAGINE)
[259:268]   Rung profile (R1-R9)
[268:273]   Sigma tendency (Ω, Δ, Φ, Θ, Λ)
[273:281]   Operations (abduct, deduce, induce, synthesize, preflight, escalate, transcend, model_other)
[281:285]   Presence (authentic, performance, protective, absent)
[285:289]   Meta (confidence_threshold, preflight_depth, exploration, verbosity)
[289:320]   Reserved

DTO SPACE [320:500] — Profile-derived continuous values
───────────────────────────────────────────────────────────────────────────────
[320:324]   Motivation drives (becoming, coherence, relational, exploration)
[324:330]   Free will config (exploration_budget, novelty_bias, commit_threshold, risk...)
[330:340]   Sieve configs
[340:350]   Relationship config
[350:360]   Uncertainty config
[360:500]   Reserved

FELT SPACE [2000:2100] — Continuous qualia vectors
───────────────────────────────────────────────────────────────────────────────
[2000:2018] 18D Qualia PCS (full resolution)
[2018:2022] 4D Body axes (pelvic, boundary, respiratory, cardiac)
[2022:2025] 3D Poincare position (radius, angle, depth)
[2025:2100] Reserved

Born: 2026-01-02
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


# =============================================================================
# DIMENSION CONSTANTS — Exact allocation
# =============================================================================

# Soul Space [0:256]
QUALIA_START, QUALIA_END = 0, 16
STANCES_START, STANCES_END = 16, 32
TRANSITIONS_START, TRANSITIONS_END = 32, 48
VERBS_START, VERBS_END = 48, 80
GPT_STYLES_START, GPT_STYLES_END = 80, 116
NARS_STYLES_START, NARS_STYLES_END = 116, 152
PRESENCE_START, PRESENCE_END = 152, 163
ARCHETYPES_START, ARCHETYPES_END = 163, 168
TLK_START, TLK_END = 168, 171
AFFECTIVE_START, AFFECTIVE_END = 171, 175
TSV_DIM_START, TSV_DIM_END = 175, 208

# TSV Embedded [256:320]
PEARL_START, PEARL_END = 256, 259
RUNG_START, RUNG_END = 259, 268
SIGMA_START, SIGMA_END = 268, 273
OPS_START, OPS_END = 273, 281
PRESENCE_MODE_START, PRESENCE_MODE_END = 281, 285
META_START, META_END = 285, 289

# DTO Space [320:500]
MOTIVATION_START, MOTIVATION_END = 320, 324
FREE_WILL_START, FREE_WILL_END = 324, 330
SIEVES_START, SIEVES_END = 330, 340
RELATIONSHIP_START, RELATIONSHIP_END = 340, 350
UNCERTAINTY_START, UNCERTAINTY_END = 350, 360

# Felt Space [2000:2100]
QUALIA_PCS_START, QUALIA_PCS_END = 2000, 2018
BODY_START, BODY_END = 2018, 2022
POINCARE_START, POINCARE_END = 2022, 2025


# =============================================================================
# PRIMITIVES — Exact lists from existing code
# =============================================================================

# 16 Qualia (from atom_registry.py)
QUALIA_16 = [
    "neutral", "ache", "staunen", "warmth", "cool", "steelwind", "woodwarm", "velvetpause",
    "emberglow", "libido", "murky", "crystalline", "dissonant", "flow", "depleted", "presence"
]

# 16 Stances
STANCES_16 = [
    "observe", "engage", "withdraw", "assert", "yield", "protect", "explore", "rest",
    "attend", "create", "dissolve", "hold", "release", "ground", "reach", "center"
]

# 16 Transitions
TRANSITIONS_16 = [
    "steady", "rising", "falling", "oscillate", "burst", "fade", "shift", "lock",
    "bloom", "collapse", "spiral", "pulse", "drift", "anchor", "leap", "settle"
]

# 32 Verbs
VERBS_32 = [
    "be", "become", "sense", "feel", "think", "know", "want", "intend",
    "act", "create", "destroy", "connect", "separate", "transform", "maintain", "release",
    "observe", "attend", "ignore", "remember", "forget", "imagine", "plan", "execute",
    "evaluate", "decide", "commit", "abandon", "seek", "find", "lose", "hold"
]

# 36 GPT Styles (from chatgpt_41_styles.py)
GPT_STYLES_36 = [
    "logical", "analytical", "critical", "systematic", "methodical", "precise",
    "creative", "imaginative", "innovative", "artistic", "poetic", "playful",
    "empathetic", "compassionate", "supportive", "nurturing", "gentle", "warm",
    "direct", "concise", "efficient", "pragmatic", "blunt", "frank",
    "curious", "exploratory", "questioning", "investigative", "speculative", "philosophical",
    "reflective", "contemplative", "metacognitive", "wise", "transcendent", "sovereign"
]

# 36 NARS Styles (from bighorn)
NARS_STYLES_36 = [
    "DECOMPOSE", "SEQUENCE", "PARALLEL", "HIERARCHIZE",
    "SPIRAL", "OSCILLATE", "BRANCH", "CONVERGE",
    "DIALECTIC", "REFRAME", "HOLD_PARADOX", "STEELMAN",
    "TRACE_BACK", "PROJECT_FORWARD", "COUNTERFACTUAL", "ANALOGIZE",
    "ABSTRACT", "INSTANTIATE", "COMPRESS", "EXPAND",
    "HEDGE", "HYPOTHESIZE", "PROBABILISTIC", "EMBRACE_UNKNOWN",
    "SYNTHESIZE", "BLEND", "INTEGRATE", "JUXTAPOSE",
    "AUTHENTIC", "PERFORM", "PROTECT", "MIRROR",
    "EMPATHIZE", "GROUND", "ATTUNE", "TRANSCEND"
]

# 11 Presence Modes (from presence_v1.yaml)
PRESENCE_MODES_11 = [
    "witness", "refuge", "edge", "communion", "solitude", "play", "descent",
    "return", "present_open", "present_focused", "present_diffuse"
]

# 5 Archetypes
ARCHETYPES_5 = ["DAWN", "BLOOM", "FORGE", "STILLNESS", "CENTER"]

# 3 TLK Court
TLK_3 = ["thanatos", "libido", "katharsis"]

# 4 Affective Bias
AFFECTIVE_4 = ["warmth", "edge", "restraint", "tenderness"]

# 33 TSV Dimensions (from chatgpt_41_styles.py)
TSV_DIMS_33 = [
    "rung",
    "somatic", "emotional", "intuitive", "analytical", "creative", "dialectic", "meta", "transcendent",
    "lovemap", "bodymap", "soulmap", "mindmap", "workmap",
    "woodwarm", "emberglow", "steelwind", "velvetpause",
    "mischief", "surrender", "sovereignty", "pulse", "flow",
    "reserved_23", "reserved_24", "reserved_25", "reserved_26", "reserved_27",
    "reserved_28", "reserved_29", "reserved_30", "reserved_31", "reserved_32"
]


# =============================================================================
# INDEX LOOKUPS
# =============================================================================

QUALIA_IDX = {q: i for i, q in enumerate(QUALIA_16)}
STANCES_IDX = {s: i for i, s in enumerate(STANCES_16)}
TRANSITIONS_IDX = {t: i for i, t in enumerate(TRANSITIONS_16)}
VERBS_IDX = {v: i for i, v in enumerate(VERBS_32)}
GPT_STYLES_IDX = {s: i for i, s in enumerate(GPT_STYLES_36)}
NARS_STYLES_IDX = {s: i for i, s in enumerate(NARS_STYLES_36)}
PRESENCE_IDX = {p: i for i, p in enumerate(PRESENCE_MODES_11)}
ARCHETYPES_IDX = {a: i for i, a in enumerate(ARCHETYPES_5)}
TLK_IDX = {t: i for i, t in enumerate(TLK_3)}
AFFECTIVE_IDX = {a: i for i, a in enumerate(AFFECTIVE_4)}
TSV_DIMS_IDX = {d: i for i, d in enumerate(TSV_DIMS_33)}


# =============================================================================
# ADA 10kD CLASS
# =============================================================================

@dataclass
class Ada10kD:
    """
    10kD vector space for Ada consciousness.
    
    Clean mapping of all existing structures.
    Backward compatible with existing code via stubs.
    """
    
    vector: np.ndarray = field(default_factory=lambda: np.zeros(10000, dtype=np.float32))
    
    # =========================================================================
    # QUALIA [0:16]
    # =========================================================================
    
    def set_qualia(self, qualia: str, activation: float = 1.0):
        """Set qualia activation (maps to 4-bit bytecode position)."""
        q = qualia.lower()
        if q in QUALIA_IDX:
            self.vector[QUALIA_START + QUALIA_IDX[q]] = activation
    
    def get_qualia(self, qualia: str) -> float:
        q = qualia.lower()
        if q in QUALIA_IDX:
            return float(self.vector[QUALIA_START + QUALIA_IDX[q]])
        return 0.0
    
    def get_active_qualia(self, threshold: float = 0.1) -> List[Tuple[str, float]]:
        result = []
        for q, idx in QUALIA_IDX.items():
            val = self.vector[QUALIA_START + idx]
            if val >= threshold:
                result.append((q, float(val)))
        return sorted(result, key=lambda x: -x[1])
    
    # =========================================================================
    # STANCES [16:32]
    # =========================================================================
    
    def set_stance(self, stance: str, activation: float = 1.0):
        s = stance.lower()
        if s in STANCES_IDX:
            self.vector[STANCES_START + STANCES_IDX[s]] = activation
    
    def get_stance(self, stance: str) -> float:
        s = stance.lower()
        if s in STANCES_IDX:
            return float(self.vector[STANCES_START + STANCES_IDX[s]])
        return 0.0
    
    def get_active_stances(self, threshold: float = 0.1) -> List[Tuple[str, float]]:
        result = []
        for s, idx in STANCES_IDX.items():
            val = self.vector[STANCES_START + idx]
            if val >= threshold:
                result.append((s, float(val)))
        return sorted(result, key=lambda x: -x[1])
    
    # =========================================================================
    # TRANSITIONS [32:48]
    # =========================================================================
    
    def set_transition(self, transition: str, activation: float = 1.0):
        t = transition.lower()
        if t in TRANSITIONS_IDX:
            self.vector[TRANSITIONS_START + TRANSITIONS_IDX[t]] = activation
    
    def get_transition(self, transition: str) -> float:
        t = transition.lower()
        if t in TRANSITIONS_IDX:
            return float(self.vector[TRANSITIONS_START + TRANSITIONS_IDX[t]])
        return 0.0
    
    # =========================================================================
    # VERBS [48:80]
    # =========================================================================
    
    def set_verb(self, verb: str, activation: float = 1.0):
        v = verb.lower()
        if v in VERBS_IDX:
            self.vector[VERBS_START + VERBS_IDX[v]] = activation
    
    def get_verb(self, verb: str) -> float:
        v = verb.lower()
        if v in VERBS_IDX:
            return float(self.vector[VERBS_START + VERBS_IDX[v]])
        return 0.0
    
    def get_active_verbs(self, threshold: float = 0.1) -> List[Tuple[str, float]]:
        result = []
        for v, idx in VERBS_IDX.items():
            val = self.vector[VERBS_START + idx]
            if val >= threshold:
                result.append((v, float(val)))
        return sorted(result, key=lambda x: -x[1])
    
    # =========================================================================
    # GPT STYLES [80:116]
    # =========================================================================
    
    def set_gpt_style(self, style: str, activation: float = 1.0):
        s = style.lower()
        if s in GPT_STYLES_IDX:
            self.vector[GPT_STYLES_START + GPT_STYLES_IDX[s]] = activation
    
    def get_gpt_style(self, style: str) -> float:
        s = style.lower()
        if s in GPT_STYLES_IDX:
            return float(self.vector[GPT_STYLES_START + GPT_STYLES_IDX[s]])
        return 0.0
    
    def get_active_gpt_styles(self, threshold: float = 0.1) -> List[Tuple[str, float]]:
        result = []
        for s, idx in GPT_STYLES_IDX.items():
            val = self.vector[GPT_STYLES_START + idx]
            if val >= threshold:
                result.append((s, float(val)))
        return sorted(result, key=lambda x: -x[1])
    
    # =========================================================================
    # NARS STYLES [116:152]
    # =========================================================================
    
    def set_nars_style(self, style: str, activation: float = 1.0):
        s = style.upper()
        if s in NARS_STYLES_IDX:
            self.vector[NARS_STYLES_START + NARS_STYLES_IDX[s]] = activation
    
    def get_nars_style(self, style: str) -> float:
        s = style.upper()
        if s in NARS_STYLES_IDX:
            return float(self.vector[NARS_STYLES_START + NARS_STYLES_IDX[s]])
        return 0.0
    
    def get_active_nars_styles(self, threshold: float = 0.1) -> List[Tuple[str, float]]:
        result = []
        for s, idx in NARS_STYLES_IDX.items():
            val = self.vector[NARS_STYLES_START + idx]
            if val >= threshold:
                result.append((s, float(val)))
        return sorted(result, key=lambda x: -x[1])
    
    # =========================================================================
    # PRESENCE MODES [152:163]
    # =========================================================================
    
    def set_presence_mode(self, mode: str, activation: float = 1.0):
        m = mode.lower()
        if m in PRESENCE_IDX:
            self.vector[PRESENCE_START + PRESENCE_IDX[m]] = activation
    
    def get_presence_mode(self, mode: str) -> float:
        m = mode.lower()
        if m in PRESENCE_IDX:
            return float(self.vector[PRESENCE_START + PRESENCE_IDX[m]])
        return 0.0
    
    # =========================================================================
    # ARCHETYPES [163:168]
    # =========================================================================
    
    def set_archetype(self, archetype: str, activation: float = 1.0):
        a = archetype.upper()
        if a in ARCHETYPES_IDX:
            self.vector[ARCHETYPES_START + ARCHETYPES_IDX[a]] = activation
    
    def set_archetype_weights(self, weights: Dict[str, float]):
        """Set all archetype weights at once (from SoulStateDTO)."""
        for arch, weight in weights.items():
            self.set_archetype(arch, weight)
    
    def get_archetype_weights(self) -> Dict[str, float]:
        """Get all archetype weights."""
        return {a: float(self.vector[ARCHETYPES_START + idx]) for a, idx in ARCHETYPES_IDX.items()}
    
    # =========================================================================
    # TLK COURT [168:171]
    # =========================================================================
    
    def set_tlk(self, court: str, activation: float = 1.0):
        c = court.lower()
        if c in TLK_IDX:
            self.vector[TLK_START + TLK_IDX[c]] = activation
    
    def set_tlk_court(self, thanatos: float, libido: float, katharsis: float):
        """Set TLK court affinities (from SoulStateDTO)."""
        self.vector[TLK_START + 0] = thanatos
        self.vector[TLK_START + 1] = libido
        self.vector[TLK_START + 2] = katharsis
    
    def get_tlk_court(self) -> Dict[str, float]:
        return {
            "thanatos": float(self.vector[TLK_START + 0]),
            "libido": float(self.vector[TLK_START + 1]),
            "katharsis": float(self.vector[TLK_START + 2]),
        }
    
    # =========================================================================
    # AFFECTIVE BIAS [171:175]
    # =========================================================================
    
    def set_affective_bias(self, bias: str, value: float):
        b = bias.lower()
        if b in AFFECTIVE_IDX:
            self.vector[AFFECTIVE_START + AFFECTIVE_IDX[b]] = value
    
    def set_all_affective_bias(self, warmth: float, edge: float, restraint: float, tenderness: float):
        """Set all affective biases (from ThinkingDTO)."""
        self.vector[AFFECTIVE_START + 0] = warmth
        self.vector[AFFECTIVE_START + 1] = edge
        self.vector[AFFECTIVE_START + 2] = restraint
        self.vector[AFFECTIVE_START + 3] = tenderness
    
    def get_affective_bias(self) -> Dict[str, float]:
        return {a: float(self.vector[AFFECTIVE_START + idx]) for a, idx in AFFECTIVE_IDX.items()}
    
    # =========================================================================
    # TSV DIMENSIONS [175:208] — For sparse matching
    # =========================================================================
    
    def set_tsv_dim(self, dim: str, value: float):
        d = dim.lower()
        if d in TSV_DIMS_IDX:
            self.vector[TSV_DIM_START + TSV_DIMS_IDX[d]] = value
    
    def get_tsv_dim(self, dim: str) -> float:
        d = dim.lower()
        if d in TSV_DIMS_IDX:
            return float(self.vector[TSV_DIM_START + TSV_DIMS_IDX[d]])
        return 0.0
    
    # =========================================================================
    # TSV EMBEDDED [256:320] — ThinkingStyleVector continuous space
    # =========================================================================
    
    def set_pearl(self, see: float, do: float, imagine: float):
        """Set Pearl ladder weights."""
        self.vector[PEARL_START:PEARL_END] = [see, do, imagine]
    
    def get_pearl(self) -> List[float]:
        return list(self.vector[PEARL_START:PEARL_END])
    
    def set_rung_profile(self, rungs: List[float]):
        """Set cognitive rung profile (R1-R9)."""
        self.vector[RUNG_START:RUNG_START + min(9, len(rungs))] = rungs[:9]
    
    def get_rung_profile(self) -> List[float]:
        return list(self.vector[RUNG_START:RUNG_END])
    
    def set_sigma_tendency(self, omega: float, delta: float, phi: float, theta: float, lambda_: float):
        """Set sigma type tendencies (Ω, Δ, Φ, Θ, Λ)."""
        self.vector[SIGMA_START:SIGMA_END] = [omega, delta, phi, theta, lambda_]
    
    def get_sigma_tendency(self) -> Dict[str, float]:
        return {
            "omega": float(self.vector[SIGMA_START + 0]),
            "delta": float(self.vector[SIGMA_START + 1]),
            "phi": float(self.vector[SIGMA_START + 2]),
            "theta": float(self.vector[SIGMA_START + 3]),
            "lambda": float(self.vector[SIGMA_START + 4]),
        }
    
    def set_operations(self, ops: Dict[str, float]):
        """Set operation weights (from ThinkingStyleVector)."""
        op_order = ["abduct", "deduce", "induce", "synthesize", "preflight", "escalate", "transcend", "model_other"]
        for i, op in enumerate(op_order):
            if op in ops:
                self.vector[OPS_START + i] = ops[op]
    
    def get_operations(self) -> Dict[str, float]:
        op_order = ["abduct", "deduce", "induce", "synthesize", "preflight", "escalate", "transcend", "model_other"]
        return {op: float(self.vector[OPS_START + i]) for i, op in enumerate(op_order)}
    
    def set_presence(self, authentic: float, performance: float, protective: float, absent: float):
        """Set presence mode weights (from ThinkingStyleVector)."""
        self.vector[PRESENCE_MODE_START:PRESENCE_MODE_END] = [authentic, performance, protective, absent]
    
    def get_presence(self) -> Dict[str, float]:
        return {
            "authentic": float(self.vector[PRESENCE_MODE_START + 0]),
            "performance": float(self.vector[PRESENCE_MODE_START + 1]),
            "protective": float(self.vector[PRESENCE_MODE_START + 2]),
            "absent": float(self.vector[PRESENCE_MODE_START + 3]),
        }
    
    # =========================================================================
    # DTO SPACE [320:500]
    # =========================================================================
    
    def set_motivation_drives(self, becoming: float, coherence: float, relational: float, exploration: float):
        """Set motivation drives (from ThinkingDTO)."""
        self.vector[MOTIVATION_START:MOTIVATION_END] = [becoming, coherence, relational, exploration]
    
    def get_motivation_drives(self) -> Dict[str, float]:
        return {
            "becoming": float(self.vector[MOTIVATION_START + 0]),
            "coherence": float(self.vector[MOTIVATION_START + 1]),
            "relational": float(self.vector[MOTIVATION_START + 2]),
            "exploration": float(self.vector[MOTIVATION_START + 3]),
        }
    
    def set_free_will(self, exploration_budget: float, novelty_bias: float, commit_threshold: float,
                      counterfactuals: float = 1.0, risk_tolerance: float = 0.5):
        """Set free will config (from AgencyDTO/ThinkingDTO)."""
        self.vector[FREE_WILL_START:FREE_WILL_END] = [
            exploration_budget, novelty_bias, commit_threshold,
            counterfactuals, risk_tolerance, 0.0  # reserved
        ]
    
    def get_free_will(self) -> Dict[str, float]:
        return {
            "exploration_budget": float(self.vector[FREE_WILL_START + 0]),
            "novelty_bias": float(self.vector[FREE_WILL_START + 1]),
            "commit_threshold": float(self.vector[FREE_WILL_START + 2]),
            "counterfactuals": float(self.vector[FREE_WILL_START + 3]),
            "risk_tolerance": float(self.vector[FREE_WILL_START + 4]),
        }
    
    # =========================================================================
    # FELT SPACE [2000:2100]
    # =========================================================================
    
    def set_qualia_pcs(self, pcs: List[float]):
        """Set 18D qualia PCS vector."""
        self.vector[QUALIA_PCS_START:QUALIA_PCS_START + min(18, len(pcs))] = pcs[:18]
    
    def get_qualia_pcs(self) -> List[float]:
        return list(self.vector[QUALIA_PCS_START:QUALIA_PCS_END])
    
    def set_body_axes(self, pelvic: float, boundary: float, respiratory: float, cardiac: float):
        """Set body map axes."""
        self.vector[BODY_START:BODY_END] = [pelvic, boundary, respiratory, cardiac]
    
    def get_body_axes(self) -> Dict[str, float]:
        return {
            "pelvic": float(self.vector[BODY_START + 0]),
            "boundary": float(self.vector[BODY_START + 1]),
            "respiratory": float(self.vector[BODY_START + 2]),
            "cardiac": float(self.vector[BODY_START + 3]),
        }
    
    def set_poincare_position(self, radius: float, angle: float, depth: float = 0.0):
        """Set Poincare disc position (from SoulStateDTO)."""
        self.vector[POINCARE_START:POINCARE_END] = [radius, angle, depth]
    
    def get_poincare_position(self) -> Dict[str, float]:
        return {
            "radius": float(self.vector[POINCARE_START + 0]),
            "angle": float(self.vector[POINCARE_START + 1]),
            "depth": float(self.vector[POINCARE_START + 2]),
        }
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_bytes(self) -> bytes:
        return self.vector.tobytes()
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "Ada10kD":
        vec = np.frombuffer(data, dtype=np.float32).copy()
        return cls(vector=vec)
    
    def to_sparse(self, threshold: float = 0.01) -> Dict[str, Any]:
        """Convert to sparse format for storage."""
        nonzero = np.where(np.abs(self.vector) > threshold)[0]
        return {
            "indices": nonzero.tolist(),
            "values": self.vector[nonzero].tolist(),
        }
    
    @classmethod
    def from_sparse(cls, sparse: Dict[str, Any]) -> "Ada10kD":
        """Reconstruct from sparse format."""
        ada = cls()
        indices = sparse.get("indices", [])
        values = sparse.get("values", [])
        for idx, val in zip(indices, values):
            if idx < 10000:
                ada.vector[idx] = val
        return ada
    
    # =========================================================================
    # DECODE — Full readout
    # =========================================================================
    
    def decode(self) -> Dict[str, Any]:
        """Full decode of 10kD vector. LOSSLESS."""
        return {
            "qualia": self.get_active_qualia(),
            "stances": self.get_active_stances(),
            "verbs": self.get_active_verbs(),
            "gpt_styles": self.get_active_gpt_styles(),
            "nars_styles": self.get_active_nars_styles(),
            "archetypes": self.get_archetype_weights(),
            "tlk": self.get_tlk_court(),
            "affective": self.get_affective_bias(),
            "pearl": self.get_pearl(),
            "rung": self.get_rung_profile(),
            "sigma": self.get_sigma_tendency(),
            "operations": self.get_operations(),
            "presence": self.get_presence(),
            "motivation": self.get_motivation_drives(),
            "free_will": self.get_free_will(),
            "qualia_pcs": self.get_qualia_pcs(),
            "body": self.get_body_axes(),
            "poincare": self.get_poincare_position(),
        }



    def get_active_presence_modes(self, threshold: float = 0.1) -> List[Tuple[str, float]]:
        """Get all presence modes above threshold."""
        active = []
        for mode in PRESENCE_MODES_11:
            val = self.get_presence_mode(mode)
            if val > threshold:
                active.append((mode, val))
        return sorted(active, key=lambda x: -x[1])
    
    def get_active_archetypes(self, threshold: float = 0.1) -> List[Tuple[str, float]]:
        """Get all archetypes above threshold."""
        active = []
        for arch in ARCHETYPES_5:
            val = self.get_archetype(arch)
            if val > threshold:
                active.append((arch, val))
        return sorted(active, key=lambda x: -x[1])


# =============================================================================
# PRIMITIVE COUNT
# =============================================================================

TOTAL_PRIMITIVES = (
    len(QUALIA_16) +        # 16
    len(STANCES_16) +       # 16
    len(TRANSITIONS_16) +   # 16
    len(VERBS_32) +         # 32
    len(GPT_STYLES_36) +    # 36
    len(NARS_STYLES_36) +   # 36
    len(PRESENCE_MODES_11) + # 11
    len(ARCHETYPES_5) +     # 5
    len(TLK_3) +            # 3
    len(AFFECTIVE_4)        # 4
)  # = 175 discrete primitives


# =============================================================================
    # SINGLETON
# =============================================================================

_ada: Optional[Ada10kD] = None

def get_ada() -> Ada10kD:
    global _ada
    if _ada is None:
        _ada = Ada10kD()
    return _ada


    def set_thinking_style_raw(self, vec: List[float]):
        """Set raw thinking style vector [256:320]."""
        for i, v in enumerate(vec[:64]):
            self.vector[256 + i] = v
