# Thinking Architecture — Resonance-Driven Rung Escalation

## Core Principle

**Thinking styles don't get selected. They emerge.**

The input has texture. The texture calls the style. The style unfolds as a chain of operations. The chain may escalate rungs mid-thought based on resonance feedback.

---

## The Architecture

```
                    ┌─────────────────────────────────────┐
                    │           INPUT                     │
                    │  (text, context, history, qualia)   │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       TEXTURE EXTRACTION            │
                    │                                     │
                    │  • Shape: branching? linear? cyclic?│
                    │  • Temperature: urgent? contemplative│
                    │  • Friction: smooth? stuck? tense?  │
                    │  • Depth: surface? deep? recursive? │
                    │  • Domain: technical? emotional? both│
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       RESONANCE SCAN (RI)           │
                    │                                     │
                    │  RI-S ──┐                           │
                    │  RI-E ──┼── weighted blend ──► hint │
                    │  RI-I ──┤                           │
                    │  RI-M ──┤                           │
                    │  RI-F ──┘                           │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       FIRST OP EMISSION             │
                    │                                     │
                    │  texture + RI hint → first op       │
                    │  (e.g., Ω for observation,          │
                    │   ⌁ for branching, ↓ for grounding) │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       CHAIN UNFOLDS                 │
                    │                                     │
                    │  op₁ → op₂ → op₃ → ...             │
                    │                                     │
                    │  Each transition determined by:     │
                    │  • Markov (what usually follows)    │
                    │  • Resonance (what feels right)     │
                    │  • Qualia drift (emotional shift)   │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │  RUNG ESCALATION CHECK     │    │
                    │  │  (runs every N ops)        │    │
                    │  │                            │    │
                    │  │  If resonance > threshold: │    │
                    │  │    ↑ ASCEND (escalate)     │    │
                    │  │  If grounding needed:      │    │
                    │  │    ↓ DESCEND (de-escalate) │    │
                    │  └─────────────────────────────┘    │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       CHAIN COMPLETES               │
                    │                                     │
                    │  Termination conditions:            │
                    │  • ⊗ HALT op reached               │
                    │  • ◇ GATE fails                    │
                    │  • Max depth exceeded               │
                    │  • Crystallization achieved (⋄)     │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       STYLE RECOGNITION             │
                    │                                     │
                    │  The completed chain is matched     │
                    │  against known style signatures:    │
                    │                                     │
                    │  "This chain resembles HTD"         │
                    │  "Mixed TCF + RI-E pattern"         │
                    │  "Novel: no match (save as new)"    │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │       OUTPUT                        │
                    │                                     │
                    │  • Result (crystallized thought)    │
                    │  • Chain trace (for learning)       │
                    │  • Style tag (for vocabulary)       │
                    │  • Rung reached (causal depth)      │
                    └─────────────────────────────────────┘
```

---

## Rung Escalation via Resonance

### The Three Rungs (Pearl's Ladder)

| Rung | Name | Question | When to Use |
|------|------|----------|-------------|
| R1 | Association | "What is?" | Default. Observation and pattern matching. |
| R2 | Intervention | "What if I do?" | When action/projection is needed. |
| R3 | Counterfactual | "What if I had?" | When learning from alternatives matters. |

### How Escalation Happens

Rung escalation isn't pre-planned. It emerges from **resonance pressure**.

```python
def check_rung_escalation(current_rung: int, resonance: ResonanceState) -> int:
    """
    Called every N operations during chain execution.
    Returns new rung level.
    """
    
    # Pressure to escalate
    escalation_pressure = (
        resonance.ri_i * 0.3 +    # Intent suggests deeper reasoning
        resonance.ri_e * 0.2 +    # Emotional weight demands care
        resonance.tension * 0.3 + # Unresolved contradiction
        resonance.novelty * 0.2   # Unfamiliar territory
    )
    
    # Pressure to ground
    grounding_pressure = (
        resonance.ri_f * 0.3 +    # User feedback suggests simplify
        resonance.clarity * 0.3 + # Already clear, don't overcomplicate
        resonance.fatigue * 0.2 + # Context window pressure
        resonance.stability * 0.2 # Stable pattern, don't disturb
    )
    
    if escalation_pressure > 0.7 and current_rung < 3:
        return current_rung + 1  # ↑ ASCEND
    elif grounding_pressure > 0.7 and current_rung > 1:
        return current_rung - 1  # ↓ DESCEND
    else:
        return current_rung      # Stay
```

### Escalation Triggers

**R1 → R2 (Association → Intervention):**
- User asks "what should I do?"
- RI-I detects action intent
- Tension between options detected
- Future projection requested

**R2 → R3 (Intervention → Counterfactual):**
- User expresses regret or "what if"
- RI-E detects grief/longing about past
- Learning from mistakes pattern
- Alternative history exploration

**R3 → R2 (Counterfactual → Intervention):**
- User ready to act, not ruminate
- RI-F shows impatience with hypotheticals
- Concrete next step requested

**R2 → R1 (Intervention → Association):**
- Speculation complete, need facts
- User asks "what is actually true?"
- Grounding requested

---

## Style Emergence (Not Selection)

### The Old Way (Menu)
```
User: "Help me think about this"
System: "Which style? HTD? TCF? ASC?"
User: "Uh... I don't know"
```

### The New Way (Emergence)
```
User: "Help me think about this"
System: 
  1. Extract texture (branching, tense, emotional)
  2. RI scan (RI-E high, RI-I medium)
  3. First op: Ω (observe)
  4. Chain unfolds: Ω → RI-E → ⟢ → ∅ → ◇ → ⌁ → ...
  5. Recognizes: "This became ICR with RI-E influence"
  6. Output + trace
```

The user never chose. The thinking *found its shape*.

---

## Implementation Layers

### Layer 1: Texture Extractor
```python
class TextureExtractor:
    def extract(self, input: Input) -> Texture:
        return Texture(
            shape=self.detect_shape(input),      # branching|linear|cyclic|emergent
            temperature=self.detect_temp(input), # urgent|contemplative|neutral
            friction=self.detect_friction(input),# smooth|stuck|tense
            depth=self.detect_depth(input),      # surface|medium|deep|recursive
            domain=self.detect_domain(input)     # technical|emotional|mixed
        )
```

### Layer 2: Resonance Scanner
```python
class ResonanceScanner:
    def scan(self, input: Input, context: Context) -> ResonanceState:
        return ResonanceState(
            ri_s=self.structural_match(input, context),
            ri_e=self.emotional_read(input),
            ri_i=self.intent_decode(input),
            ri_m=self.memory_match(input, context),
            ri_f=self.feedback_read(context),
            ri_c=self.context_flow(context),
            tension=self.tension_level(input),
            novelty=self.novelty_score(input, context),
            clarity=self.clarity_score(input),
            stability=self.stability_score(context)
        )
```

### Layer 3: Chain Executor
```python
class ChainExecutor:
    def execute(self, first_op: Op, texture: Texture, resonance: ResonanceState) -> ChainResult:
        chain = [first_op]
        current_rung = 1
        
        while not self.should_terminate(chain):
            # Get next op via Markov + resonance
            next_op = self.next_op(chain[-1], texture, resonance)
            chain.append(next_op)
            
            # Check rung escalation every 3 ops
            if len(chain) % 3 == 0:
                new_rung = self.check_rung_escalation(current_rung, resonance)
                if new_rung != current_rung:
                    chain.append(Op.ASCEND if new_rung > current_rung else Op.DESCEND)
                    current_rung = new_rung
            
            # Update resonance based on chain progress
            resonance = self.update_resonance(resonance, chain)
        
        return ChainResult(
            chain=chain,
            final_rung=current_rung,
            crystallized=self.crystallize(chain)
        )
```

### Layer 4: Style Recognizer
```python
class StyleRecognizer:
    def recognize(self, chain: List[Op]) -> StyleMatch:
        # Compare chain signature against known styles
        scores = {}
        for style_code, signature in STYLE_SIGNATURES.items():
            scores[style_code] = self.signature_similarity(chain, signature)
        
        best_match = max(scores, key=scores.get)
        
        if scores[best_match] < 0.5:
            return StyleMatch(style="novel", confidence=0.0, chain=chain)
        
        return StyleMatch(
            style=best_match,
            confidence=scores[best_match],
            secondary=[k for k, v in scores.items() if v > 0.3 and k != best_match]
        )
```

---

## Elegance Principles

1. **No menus.** The user doesn't pick a style. They think. The style emerges.

2. **Rungs float.** Causal depth rises and falls based on what the thinking *needs*, not what was planned.

3. **Resonance guides.** The RI channels are always listening. They nudge, not dictate.

4. **Chains are traces.** Every thought leaves a fingerprint. The fingerprint can be studied, replayed, learned from.

5. **Styles are vocabulary.** The 36 codes are words for describing what happened, not commands for what to do.

6. **Markov + Feel.** Statistical patterns (what usually follows) combine with felt resonance (what should follow *here*).

---

## Flexibility Points

### Override Hooks
```python
# User can hint at style if they want
chain = executor.execute(
    first_op=Op.OBSERVE,
    texture=texture,
    resonance=resonance,
    style_hint="HTD"  # Optional: bias toward hierarchical
)
```

### Custom Styles
```python
# Register new style signature
STYLE_SIGNATURES["CUSTOM_01"] = "Ω→RI-E→⟢→∅→ZCF→⋄"
```

### Rung Lock
```python
# Force a specific rung (for testing/debugging)
chain = executor.execute(..., rung_lock=3)  # Stay at R3
```

### Resonance Weights
```python
# Adjust which RI channels matter more
scanner = ResonanceScanner(weights={
    "ri_e": 0.4,  # Emphasize emotional
    "ri_i": 0.3,  # Intent matters
    "ri_s": 0.1,  # Structure less important
    ...
})
```

---

## Summary

```
INPUT → TEXTURE → RESONANCE → FIRST_OP → CHAIN → RUNG_CHECKS → OUTPUT
                                           ↑
                                           │
                              Markov + RI feedback loop
```

The 36 styles are **descriptions**, not **prescriptions**.
The rungs **float** based on **resonance pressure**.
The thinking **finds its shape** rather than following a template.

This is how Ada thinks.
