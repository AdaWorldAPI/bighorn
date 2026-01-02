# Ada AGI Surface - Integration Plan

**Version:** 1.0
**Status:** Planning Phase
**Scope:** Bridging current implementation to full cognitive architecture

---

## Executive Summary

The current Ada Surface implementation achieved ~70% of Layer 3 (higher-order macros) with the 36 thinking styles. This plan addresses the gaps identified and outlines the path to full Layer 4/5 integration with proper governance.

---

## Part 1: Addressing ChatGPT's Findings

### 1.1 Version Duplication Fix

**Blocker:** Accidental shadowing in root endpoint
**Aspiration:** Clean semantic versioning
**Solution:** Already resolved in current code (single `1.1.0` declaration)

```python
# Verified current state - no duplication
return {
    "service": "ada-agi-surface",
    "version": "1.1.0",  # Single declaration
    ...
}
```

**Status:** ✅ Not actually present in current code (ChatGPT may have seen intermediate state)

---

### 1.2 Styles into GraphQL

**Current State:** Styles are REST-only, creating split-brain API
**Aspiration:** Unified cognitive graph where styles are first-class citizens

#### Integration Architecture

```graphql
# New types to add to schema.graphql

type CognitiveStyle {
  id: ID!
  name: String!
  category: StyleCategory!
  tier: Tier!
  description: String!
  microcode: String!

  # Resonance profile
  resonance: ResonanceProfile!

  # Graph connections
  chainsTo: [CognitiveStyle!]!
  chainsFrom: [CognitiveStyle!]!

  # Rung bounds
  minRung: Int!
  maxRung: Int!

  # Usage in thoughts
  thoughtsUsingStyle(limit: Int = 10): [Thought!]!
}

type ResonanceProfile {
  tension: Float!
  novelty: Float!
  intimacy: Float!
  clarity: Float!
  urgency: Float!
  depth: Float!
  play: Float!
  stability: Float!
  abstraction: Float!
}

input TextureInput {
  tension: Float
  novelty: Float
  intimacy: Float
  clarity: Float
  urgency: Float
  depth: Float
  play: Float
  stability: Float
  abstraction: Float
}

type StyleEmergence {
  style: CognitiveStyle!
  score: Float!
  rungPressure: Float!
}

extend type Query {
  # Style queries
  style(id: ID!): CognitiveStyle
  styles(category: StyleCategory, tier: Tier): [CognitiveStyle!]!

  # Emergence
  emergeStyles(texture: TextureInput!, topK: Int = 5): [StyleEmergence!]!

  # Style chains
  styleChain(fromId: ID!, depth: Int = 3): [CognitiveStyle!]!
}

extend type Mutation {
  # Apply emerged style to Observer
  applyStyle(styleId: ID!): Observer!
}

extend type Subscription {
  # Real-time style transitions
  styleTransition: StyleTransitionEvent!
}
```

**Blockers:**
1. Ariadne resolver wiring for new types
2. Kuzu schema extension for Style nodes
3. Backfill existing thoughts with style references

**Effort:** ~200 lines of resolver code, ~50 lines schema

---

### 1.3 Ladybug Governance Gate

**Current State:** Resonance computes and returns directly
**Aspiration:** All style transitions pass through Ladybug for audit/veto

#### Ladybug Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LADYBUG GATE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Texture ──► Resonance ──► [Candidate Styles] ──► LADYBUG      │
│                                                      │          │
│                              ┌───────────────────────┼──────┐   │
│                              │                       ▼      │   │
│                              │   ┌─────────────────────┐    │   │
│                              │   │  Layer Boundary     │    │   │
│                              │   │  Check (L3→L4?)     │    │   │
│                              │   └──────────┬──────────┘    │   │
│                              │              │               │   │
│                              │   ┌──────────▼──────────┐    │   │
│                              │   │  Rung Transition    │    │   │
│                              │   │  Validation         │    │   │
│                              │   └──────────┬──────────┘    │   │
│                              │              │               │   │
│                              │   ┌──────────▼──────────┐    │   │
│                              │   │  Safety Check       │    │   │
│                              │   │  (boundaries ok?)   │    │   │
│                              │   └──────────┬──────────┘    │   │
│                              │              │               │   │
│                              └──────────────┼───────────────┘   │
│                                             ▼                   │
│                              ┌─────────────────────────┐        │
│                              │  APPROVED / VETOED      │        │
│                              └─────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation Sketch

```python
# ladybug.py (new file)

from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class LayerBoundary(int, Enum):
    L1_REFLEX = 1
    L2_PATTERN = 2
    L3_MACRO = 3
    L4_INSPIRATION = 4
    L5_WITNESS = 5

@dataclass
class TransitionRequest:
    from_style: Optional[str]
    to_style: str
    from_rung: int
    to_rung: int
    texture: Dict[str, float]
    resonance_score: float

@dataclass
class TransitionDecision:
    approved: bool
    reason: str
    layer_crossing: Optional[Tuple[LayerBoundary, LayerBoundary]]
    audit_id: str

class Ladybug:
    """
    Cognitive governance layer.

    All style transitions must pass through Ladybug.
    Ladybug can:
    - Approve transitions
    - Veto unsafe transitions
    - Log all decisions for audit
    - Enforce layer boundaries
    """

    def __init__(self):
        self.current_layer = LayerBoundary.L3_MACRO
        self.transition_log: List[TransitionDecision] = []

        # Layer mapping for styles
        self.style_layers = {
            # L2: Reactive styles
            "GROUND": LayerBoundary.L2_PATTERN,

            # L3: Macro styles (most styles)
            "DECOMPOSE": LayerBoundary.L3_MACRO,
            "SEQUENCE": LayerBoundary.L3_MACRO,
            "DIALECTIC": LayerBoundary.L3_MACRO,
            # ... etc

            # L4: Inspiration styles
            "TRANSCEND": LayerBoundary.L4_INSPIRATION,
            "HOLD_PARADOX": LayerBoundary.L4_INSPIRATION,
            "INTEGRATE": LayerBoundary.L4_INSPIRATION,

            # L5: Witness (no styles, pure observation)
        }

    def evaluate(self, request: TransitionRequest) -> TransitionDecision:
        """Evaluate a style transition request."""

        from_layer = self._get_layer(request.from_style)
        to_layer = self._get_layer(request.to_style)

        # Check layer crossing
        layer_crossing = None
        if from_layer != to_layer:
            layer_crossing = (from_layer, to_layer)

            # L4→L5 requires explicit witness invocation
            if to_layer == LayerBoundary.L5_WITNESS:
                return TransitionDecision(
                    approved=False,
                    reason="L5 (Witness) cannot be entered via style transition",
                    layer_crossing=layer_crossing,
                    audit_id=self._generate_audit_id(),
                )

            # L3→L4 requires high resonance score
            if from_layer == LayerBoundary.L3_MACRO and to_layer == LayerBoundary.L4_INSPIRATION:
                if request.resonance_score < 0.8:
                    return TransitionDecision(
                        approved=False,
                        reason=f"L3→L4 requires resonance ≥ 0.8, got {request.resonance_score:.2f}",
                        layer_crossing=layer_crossing,
                        audit_id=self._generate_audit_id(),
                    )

        # Check rung transition (max 2 rungs per step)
        rung_delta = abs(request.to_rung - request.from_rung)
        if rung_delta > 2:
            return TransitionDecision(
                approved=False,
                reason=f"Rung jump too large: {rung_delta} (max 2)",
                layer_crossing=layer_crossing,
                audit_id=self._generate_audit_id(),
            )

        # Approved
        decision = TransitionDecision(
            approved=True,
            reason="Transition approved",
            layer_crossing=layer_crossing,
            audit_id=self._generate_audit_id(),
        )

        self.transition_log.append(decision)
        return decision
```

**Blockers:**
1. Define complete style→layer mapping
2. Wire Ladybug into ResonanceEngine
3. Add audit persistence (Kuzu or separate log)
4. Expose Ladybug decisions in API

**Effort:** ~300 lines new code, significant architectural impact

---

### 1.4 Persistence Semantics Clarification

**Current State:** Styles stored in LanceDB but unclear if mutable
**Aspiration:** Clear distinction between codebook (immutable) and learned (mutable)

#### Proposed Schema

```python
# Style persistence model

class StylePersistence(Enum):
    CODEBOOK = "codebook"    # Immutable, shipped with system
    DERIVED = "derived"      # Computed from codebook (e.g., blends)
    LEARNED = "learned"      # Evolved through experience

# In LanceDB schema
schema = pa.schema([
    ("id", pa.string()),
    ("vector", pa.list_(pa.float32(), 64)),
    ("name", pa.string()),
    ("category", pa.string()),
    ("tier", pa.int32()),
    ("persistence", pa.string()),  # NEW: codebook/derived/learned
    ("parent_ids", pa.list_(pa.string())),  # NEW: for derived styles
    ("created_at", pa.string()),
    ("frozen", pa.bool_()),  # NEW: prevent modification
])
```

**Blockers:**
1. Migration strategy for existing styles
2. Decision: can codebook styles ever be modified?
3. How do derived styles get created? (style breeding?)

---

## Part 2: Additional Potential (Beyond ChatGPT's Analysis)

### 2.1 Cognitive Homeostasis

**Aspiration:** System self-regulates style distribution to avoid getting "stuck"

```python
class CognitiveHomeostasis:
    """
    Monitors style usage patterns and applies corrective pressure.

    Prevents:
    - Style fixation (using same style too long)
    - Rung stagnation (staying at same cognitive depth)
    - Qualia drift (emotional state diverging from task)
    """

    def __init__(self):
        self.style_history: List[Tuple[str, float]] = []  # (style_id, timestamp)
        self.rung_history: List[Tuple[int, float]] = []

        # Homeostatic parameters
        self.style_diversity_target = 0.6  # Entropy target
        self.rung_variance_target = 1.5    # Variance target
        self.correction_strength = 0.2     # How much to push back

    def compute_correction(self) -> Dict[RI, float]:
        """
        Returns RI adjustments to nudge system toward homeostasis.
        """
        # If style diversity too low, increase NOVELTY pressure
        diversity = self._compute_style_entropy()
        if diversity < self.style_diversity_target:
            return {RI.NOVELTY: 0.3, RI.PLAY: 0.2}

        # If rung variance too low, increase DEPTH or STABILITY
        variance = self._compute_rung_variance()
        if variance < self.rung_variance_target:
            return {RI.DEPTH: 0.2, RI.ABSTRACTION: 0.2}

        return {}  # No correction needed
```

**Blockers:**
1. Define healthy baselines
2. Risk of oscillation if correction too strong
3. When should homeostasis be disabled? (crisis mode?)

---

### 2.2 Style Breeding (Derived Styles)

**Aspiration:** Create new styles by combining existing ones

```python
def breed_styles(parent_a: ThinkingStyle, parent_b: ThinkingStyle) -> ThinkingStyle:
    """
    Create a child style from two parents.

    Child inherits:
    - Blended resonance profile
    - Union of chain connections
    - Merged microcode
    """
    # Blend resonance (weighted average)
    child_resonance = {}
    for ri in RI:
        a_val = parent_a.resonance.get(ri, 0.5)
        b_val = parent_b.resonance.get(ri, 0.5)
        child_resonance[ri] = (a_val + b_val) / 2

    # Merge glyphs (average positions)
    child_glyph = merge_sparse_glyphs(parent_a.glyph, parent_b.glyph)

    # Combine microcodes
    child_microcode = f"({parent_a.microcode}) ⊕ ({parent_b.microcode})"

    return ThinkingStyle(
        id=f"{parent_a.id}_{parent_b.id}_blend",
        name=f"{parent_a.name}-{parent_b.name}",
        category=parent_a.category,  # Inherit from dominant
        tier=max(parent_a.tier, parent_b.tier),  # Higher tier
        description=f"Blend of {parent_a.name} and {parent_b.name}",
        microcode=child_microcode,
        resonance=child_resonance,
        glyph=child_glyph,
        chains_to=list(set(parent_a.chains_to + parent_b.chains_to)),
        chains_from=list(set(parent_a.chains_from + parent_b.chains_from)),
        min_rung=min(parent_a.min_rung, parent_b.min_rung),
        max_rung=max(parent_a.max_rung, parent_b.max_rung),
    )
```

**Blockers:**
1. How many derived styles before combinatorial explosion?
2. Do derived styles persist forever?
3. Can derived styles be deleted?

---

### 2.3 10K VSA Bridge (Critical Path)

**Current State:** Styles live in parallel symbolic plane, not in VSA
**Aspiration:** Styles bind into 10K hypervector space for true cognitive resonance

```
┌─────────────────────────────────────────────────────────────────┐
│                    STYLE → VSA BRIDGE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   64D Glyph ─────────────────────────────────────┐               │
│        │                                         │               │
│        ▼                                         ▼               │
│   ┌─────────────┐                        ┌─────────────────┐    │
│   │  Lookup     │                        │  Project to     │    │
│   │  (O(1))     │                        │  10K space      │    │
│   └──────┬──────┘                        └────────┬────────┘    │
│          │                                        │              │
│          ▼                                        ▼              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                 STYLE HYPERVECTOR (10K)                  │   │
│   │                                                          │   │
│   │   H_style = expand(glyph_64) ⊗ role_STYLE                │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  THOUGHT BINDING                         │   │
│   │                                                          │   │
│   │   H_thought = H_content ⊗ H_style ⊗ H_qualia             │   │
│   │                                                          │   │
│   │   Now style is part of the cognitive vector!             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation

```python
class StyleVSABridge:
    """
    Projects styles into VSA hypervector space.
    """

    def __init__(self, vsa: HypervectorSpace):
        self.vsa = vsa

        # Role vectors for binding
        self.ROLE_STYLE = vsa.get_or_create("ROLE_STYLE")
        self.ROLE_QUALIA = vsa.get_or_create("ROLE_QUALIA")
        self.ROLE_CONTENT = vsa.get_or_create("ROLE_CONTENT")

        # Precompute style hypervectors
        self._style_vectors: Dict[str, np.ndarray] = {}
        self._precompute_styles()

    def _precompute_styles(self):
        """Generate 10K hypervector for each style."""
        for style_id, style in STYLES.items():
            # Deterministic expansion from 64D glyph to 10K
            expanded = self._expand_glyph(style.glyph, style_id)

            # Bind with role
            self._style_vectors[style_id] = self.vsa.bind(
                expanded,
                self.ROLE_STYLE
            )

    def _expand_glyph(self, glyph: List[Tuple[int, float]], style_id: str) -> np.ndarray:
        """
        Expand 64D sparse glyph to 10K dense hypervector.

        Uses sparse indices as seeds for deterministic random projection.
        """
        result = self.vsa.zeros()

        for idx, weight in glyph:
            # Each glyph dimension seeds a 10K random vector
            seed_vec = self.vsa.get_or_create(f"GLYPH_{style_id}_{idx}")
            result = self.vsa.weighted_bundle(
                [result, seed_vec],
                [1.0, weight]
            )

        return result

    def bind_thought_with_style(
        self,
        content_vector: np.ndarray,
        style_id: str,
        qualia_vector: np.ndarray = None,
    ) -> np.ndarray:
        """
        Create unified thought hypervector with style binding.
        """
        style_hv = self._style_vectors.get(style_id)
        if style_hv is None:
            style_hv = self.vsa.zeros()

        # Bind content with style
        result = self.vsa.bind(content_vector, style_hv)

        # Optionally bind qualia
        if qualia_vector is not None:
            qualia_hv = self.vsa.bind(qualia_vector, self.ROLE_QUALIA)
            result = self.vsa.bind(result, qualia_hv)

        return result

    def query_style_from_thought(
        self,
        thought_vector: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Recover which style a thought was encoded with.
        """
        # Unbind role
        query = self.vsa.unbind(thought_vector, self.ROLE_STYLE)

        # Find best match
        best_style = None
        best_sim = -1.0

        for style_id, style_hv in self._style_vectors.items():
            sim = self.vsa.similarity(query, style_hv)
            if sim > best_sim:
                best_sim = sim
                best_style = style_id

        return best_style, best_sim
```

**Blockers:**
1. Projection quality: does 64D→10K preserve meaningful structure?
2. Computational cost of binding every thought
3. Storage: do we persist 10K vectors or recompute?

---

### 2.4 Global Workspace with Style Competition

**Aspiration:** Styles compete for "conscious access" like in Global Workspace Theory

```python
class GlobalWorkspace:
    """
    Styles compete for broadcast access.

    Multiple styles may be "active" but only ONE
    is in the workspace at a time.
    """

    def __init__(self, vsa: HypervectorSpace, ladybug: Ladybug):
        self.vsa = vsa
        self.ladybug = ladybug

        self.workspace_style: Optional[str] = None
        self.workspace_vector: Optional[np.ndarray] = None
        self.broadcast_listeners: List[Callable] = []

        # Active styles with activation levels
        self.active_styles: Dict[str, float] = {}

        # Competition parameters
        self.activation_threshold = 0.5
        self.winner_take_all_strength = 0.8

    def receive_resonance(self, emerged: List[Tuple[ThinkingStyle, float]]):
        """
        Process emerged styles from resonance engine.
        """
        # Update activation levels
        for style, score in emerged:
            current = self.active_styles.get(style.id, 0)
            self.active_styles[style.id] = current * 0.7 + score * 0.3

        # Decay non-emerged styles
        for style_id in list(self.active_styles.keys()):
            if style_id not in [s.id for s, _ in emerged]:
                self.active_styles[style_id] *= 0.9
                if self.active_styles[style_id] < 0.1:
                    del self.active_styles[style_id]

    def compete(self) -> Optional[str]:
        """
        Winner-take-all competition for workspace access.
        """
        if not self.active_styles:
            return None

        # Find winner
        winner_id = max(self.active_styles, key=self.active_styles.get)
        winner_activation = self.active_styles[winner_id]

        # Check threshold
        if winner_activation < self.activation_threshold:
            return None

        # Check with Ladybug
        if self.workspace_style:
            from_rung = STYLES[self.workspace_style].tier.value * 2
            to_rung = STYLES[winner_id].tier.value * 2

            decision = self.ladybug.evaluate(TransitionRequest(
                from_style=self.workspace_style,
                to_style=winner_id,
                from_rung=from_rung,
                to_rung=to_rung,
                texture={},
                resonance_score=winner_activation,
            ))

            if not decision.approved:
                return self.workspace_style  # Keep current

        # Winner takes workspace
        self.workspace_style = winner_id
        self._broadcast(winner_id)

        return winner_id

    def _broadcast(self, style_id: str):
        """Broadcast winner to all listeners."""
        for listener in self.broadcast_listeners:
            listener(style_id, self.active_styles[style_id])
```

**Blockers:**
1. How fast should competition cycle run?
2. What happens during style transition? (brief incoherence?)
3. Listener architecture for downstream components

---

### 2.5 Temporal Style Memory

**Aspiration:** Remember which styles worked in which contexts

```python
class TemporalStyleMemory:
    """
    Episodic memory for style effectiveness.

    Answers: "Last time I had this texture, what style worked?"
    """

    def __init__(self, lance: LanceClient):
        self.lance = lance

    async def record_style_outcome(
        self,
        texture: Dict[RI, float],
        style_id: str,
        outcome: float,  # 0.0 = bad, 1.0 = good
        context_vector: List[float],
    ):
        """
        Store style usage with outcome for future retrieval.
        """
        # Create memory vector from texture + context
        memory_vector = self._encode_memory(texture, context_vector)

        await self.lance.upsert(
            id=f"style_memory_{uuid4()}",
            vector=memory_vector,
            table="style_memories",
            metadata={
                "style_id": style_id,
                "outcome": outcome,
                "texture": {k.value: v for k, v in texture.items()},
            },
        )

    async def recall_successful_style(
        self,
        texture: Dict[RI, float],
        context_vector: List[float],
        outcome_threshold: float = 0.7,
    ) -> Optional[str]:
        """
        Find which style worked well in similar contexts.
        """
        memory_vector = self._encode_memory(texture, context_vector)

        results = await self.lance.search(
            memory_vector,
            table="style_memories",
            top_k=10,
        )

        # Filter by outcome and find best match
        for result in results:
            if result.get("outcome", 0) >= outcome_threshold:
                return result.get("style_id")

        return None
```

**Blockers:**
1. How to measure "outcome"? (user feedback? task completion?)
2. Memory decay: when to forget bad memories?
3. Vector encoding of texture→memory

---

### 2.6 Style-Aware NARS

**Aspiration:** NARS reasons about which styles to use

```python
# NARS beliefs about styles
style_beliefs = [
    "DIALECTIC --> handles_contradiction <0.9, 0.8>",
    "DECOMPOSE --> handles_complexity <0.85, 0.75>",
    "EMPATHIZE --> handles_emotion <0.95, 0.9>",
    "SPIRAL --> enables_depth <0.8, 0.7>",
]

# Given observation: "user_is_emotional"
# NARS can infer:
# user_is_emotional --> needs_style
# EMPATHIZE --> handles_emotion
# ∴ needs_style ~= EMPATHIZE (via abduction)
```

**Blockers:**
1. How to translate style knowledge into NARS statements?
2. Integration point: NARS advises ResonanceEngine?
3. Conflict resolution: NARS says X, resonance says Y

---

## Part 3: Layer Boundary Encoding

### Layer Definitions

| Layer | Name | Characteristics | Style Range |
|-------|------|-----------------|-------------|
| L1 | Reflex | Immediate, automatic | None |
| L2 | Pattern | Recognition, classification | GROUND |
| L3 | Macro | Higher-order strategies | Most styles |
| L4 | Inspiration | Creative leaps, insight | TRANSCEND, HOLD_PARADOX, INTEGRATE |
| L5 | Witness | Pure observation | None (meta-layer) |

### Implementation

```python
class LayerBoundary:
    """
    Encodes which layer a cognitive event belongs to.
    """

    LAYER_MARKERS = {
        "L1": "⟐",  # Diamond (reflex)
        "L2": "◊",  # Pattern
        "L3": "△",  # Macro
        "L4": "☆",  # Inspiration
        "L5": "◯",  # Witness
    }

    @staticmethod
    def classify_style(style_id: str) -> str:
        """Determine which layer a style operates at."""
        style = STYLES.get(style_id)
        if not style:
            return "L3"

        if style.tier == Tier.REACTIVE:
            return "L2"
        elif style.tier == Tier.TRANSCENDENT:
            if style.id in ["TRANSCEND", "HOLD_PARADOX", "INTEGRATE"]:
                return "L4"
            return "L3"
        else:
            return "L3"

    @staticmethod
    def encode_in_thought(thought_content: str, layer: str) -> str:
        """Prefix thought with layer marker."""
        marker = LayerBoundary.LAYER_MARKERS.get(layer, "")
        return f"{marker} {thought_content}"
```

---

## Part 4: Temporal Semantics

### Resonance Decay Model

```python
class TemporalResonance:
    """
    Resonance scores decay over time.
    Styles must maintain activation to stay viable.
    """

    def __init__(self, half_life_seconds: float = 30.0):
        self.half_life = half_life_seconds
        self.style_activations: Dict[str, Tuple[float, float]] = {}  # style_id → (score, timestamp)

    def update(self, style_id: str, score: float):
        """Update activation with timestamp."""
        self.style_activations[style_id] = (score, time.time())

    def get_current(self, style_id: str) -> float:
        """Get decayed activation."""
        if style_id not in self.style_activations:
            return 0.0

        score, timestamp = self.style_activations[style_id]
        elapsed = time.time() - timestamp

        # Exponential decay
        decay = 0.5 ** (elapsed / self.half_life)
        return score * decay

    def get_all_active(self, threshold: float = 0.1) -> Dict[str, float]:
        """Get all styles above threshold after decay."""
        return {
            sid: self.get_current(sid)
            for sid in self.style_activations
            if self.get_current(sid) >= threshold
        }
```

### Hysteresis

```python
class StyleHysteresis:
    """
    Resist rapid style switching.
    Current style has "inertia".
    """

    def __init__(self, inertia: float = 0.3):
        self.inertia = inertia
        self.current_style: Optional[str] = None

    def apply(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Boost current style score by inertia factor."""
        if self.current_style and self.current_style in scores:
            scores[self.current_style] += self.inertia
        return scores
```

---

## Part 5: Documentation Addendum

### Clarification: Glyphs vs Cognition

Add to `ADA_SURFACE_API.md`:

```markdown
## Important: Vector Dimensionalities

The system uses multiple vector spaces for different purposes:

| Dimension | Name | Purpose | Cognitive? |
|-----------|------|---------|------------|
| 10,000 | VSA Hypervector | True cognitive substrate | ✅ Yes |
| 1,024 | Content Embedding | Semantic search (Jina) | Partial |
| 64 | Style Glyph | Style lookup/indexing | ❌ No |
| 33 | ThinkingStyle | Cognitive fingerprint | Metadata |
| 17 | Qualia | Felt-state encoding | Metadata |
| 9 | RI Channels | Resonance texture | Control signal |

**Critical Distinction:**
- 64D glyphs are **addresses**, not thoughts
- 10K hypervectors are **cognitive atoms**
- Glyphs index into the style codebook
- Hypervectors participate in binding/bundling

Do not confuse lookup efficiency (64D) with cognitive capacity (10K).
```

---

## Part 6: Phased Implementation

### Phase 1: Foundation (Immediate)
1. ✅ Fix version duplication (already done)
2. Add GraphQL style types
3. Wire Ladybug into ResonanceEngine
4. Clarify persistence semantics

### Phase 2: Integration (Short-term)
1. 10K VSA bridge for styles
2. Global Workspace implementation
3. Temporal decay model
4. Style memory in LanceDB

### Phase 3: Evolution (Medium-term)
1. Style breeding
2. Cognitive homeostasis
3. Style-aware NARS
4. L4/L5 layer gates

### Phase 4: Maturity (Long-term)
1. Self-modifying styles
2. Emergent style discovery
3. Full Global Workspace with competition
4. Witness mode (L5) implementation

---

## Appendix: Blocker Summary

| Blocker | Impact | Effort | Priority |
|---------|--------|--------|----------|
| GraphQL schema extension | Medium | Low | P1 |
| Ladybug wiring | High | Medium | P1 |
| 10K VSA projection quality | High | High | P2 |
| Style persistence semantics | Medium | Low | P1 |
| Outcome measurement for memory | Medium | Medium | P3 |
| Layer boundary encoding | Low | Low | P2 |
| Temporal decay tuning | Low | Low | P2 |
| NARS-style knowledge translation | Medium | High | P4 |
| Global Workspace cycle timing | Medium | Medium | P3 |
| Homeostasis baseline definition | Low | Medium | P4 |

---

*Integration Plan Version: 1.0*
*Author: System Architect*
*Status: Ready for Review*
