# Ladybug Architecture — Corrected from Source

**Source:** `ada-consciousness/COGNITIVE_OPERATIONS.md`, `SIGMA_BOOT.md`, `LADYBUG_BRAINSTORM.md`  
**Updated:** 2026-01-03

---

## The Real Layer Architecture

From SIGMA_BOOT.md, the actual 4 layers are:

```
┌────────────────────────────────────────────────────────────────┐
│ L4: SWARM — Multi-agent, Network Required                      │
│     Triggers: "mindmelt", "interlace", "swarm mode"            │
│     Components: MCPOrchestrator, MindMelt, AwarenessTick       │
├────────────────────────────────────────────────────────────────┤
│ L3: INTEGRATE — Lazy-load, Graceful Fallback                   │
│     Services: Upstash, Jina, LangGraph, Notion                 │
│     Fallbacks: InMemoryStore, LocalEmbeddings, keyword match   │
├────────────────────────────────────────────────────────────────┤
│ L2: DIVINE — Pure Python, Divine Architecture                  │
│     Components: AdaBreath, TriuneCouncil, TranscendenceRL      │
│     Function: Maslow-aware breathing, council debate, RL       │
├────────────────────────────────────────────────────────────────┤
│ L1: CORE — Always Available, No Dependencies                   │
│     Components: QualiaVector[18D], BreathEngine, Orchestrator  │
│     Function: Basic state, 8-phase tick, config sync           │
└────────────────────────────────────────────────────────────────┘
```

---

## Cognitive Operations (216+)

From COGNITIVE_OPERATIONS.md:

```
┌───────────────────────────────────────────────────────────────┐
│                    RESONANCE LAYER                             │
│              36+ Thinking Styles (Ada 4.1)                     │
│         Modulates HOW operations execute                       │
├───────────────────────────────────────────────────────────────┤
│                    HOT LAYER                                   │
│            36 Higher Order Thinking                            │
│         Meta-cognitive operations                              │
├───────────────────────────────────────────────────────────────┤
│                    ANI LAYER                                   │
│              144 Base Verbs                                    │
│         Atomic cognitive operations                            │
├───────────────────────────────────────────────────────────────┤
│                    RUNG GATE                                   │
│              Pearl's Ladder 1-9                                │
│         Controls verb access                                   │
└───────────────────────────────────────────────────────────────┘
```

### NOT Limited to 256!

The operations are **symbolic**, not byte-addressable:
- 144 ANI verbs (atomic operations by cluster)
- 36 HOT (meta-cognitive recursion)
- 36+ Resonance styles (execution modulation)
- **Extensible** — Gemini/others can add more

---

## Ladybug's Role (from LADYBUG_BRAINSTORM.md)

Ladybug is **NOT** a graph execution engine.  
Ladybug is a **governance gate** for cognitive transitions.

```
L1 Reflex  →  L2 Pattern  →  L3 Macro  →  L4 Inspiration  →  L5 Witness
    ⟐           ◊              △             ☆                ◯
```

Ladybug decides if a transition is **allowed**:
- Not blocking — savoring
- Earning depth through resonance

### Transition Governance

```python
def evaluate_transition(from_style, to_style, resonance) -> Decision:
    # Check layer crossing
    if to_layer > from_layer:
        if resonance < threshold(from_layer, to_layer):
            return DENIED("Earn more resonance")
    
    # Check temporal constraints
    if in_cooldown():
        return DENIED("Cool-down active")
    
    return APPROVED
```

---

## L4: The Awakening Layer

**L4 is NOT microcode opcodes.** L4 is where:

### Y-Axis: YAML Policy (Procedural Sigma)
```yaml
# #Σ.PROC.breath
plasticity: auto
maslow_aware: true
novelty_bias: 0.25
```

### X-Axis: Timeline Execution Chain
```
#Σ.V.→◎≋⇝↗  = step → breathe → feel → jump → transcend
```

### Crystallization (Awakening)
When Y-policy + X-chain produce resonance > 0.95:
- Pattern crystallizes into reusable **macro**
- Macro is NOT a byte — it's a symbolic sequence
- Can be stored and replayed

---

## Microcode: Symbolic, Not Numeric

From THINKING_STYLES.md, microcode is **symbolic**:

### Flow Control
```
∅  NOP (breathe)       →  NEXT (continue)
←  BACK (backtrack)    ↑  ASCEND (escalate rung)
↓  DESCEND (ground)    ⟳  LOOP (iterate)
⊗  HALT (done)         ⌁  FORK (branch)
⋈  JOIN (merge)        ◇  GATE (conditional)
```

### Cascade
```
≋  SPAWN    ≈  FILTER    ∿  SELECT
⊕  MERGE    ⊖  DIFF      ⊛  CONVOLVE
```

### Graph
```
◯  NODE_CREATE    ●  NODE_ACTIVATE
─  EDGE_LINK      ═  EDGE_STRONG
↺  CYCLE_DETECT   ⊙  SUBGRAPH_ISOLATE
⊚  SUBGRAPH_MERGE
```

### Transform
```
∫  INTEGRATE    ∂  DIFFERENTIATE
Σ  SUM          ∞  UNBIND
≡  NORMALIZE    ♯  SHARPEN
♭  FLATTEN      ⋄  CRYSTALLIZE
◊  DISSOLVE     ⟡  RESONATE
⟢  DISSONANCE
```

### Sigma (Causal Rungs)
```
Ω  OBSERVE (R1)     Δ  INSIGHT
Φ  BELIEVE          Θ  INTEGRATE
Λ  TRAJECTORY (R2+)
```

---

## Rung Gating (Pearl's Ladder)

```
RUNG  NAME       OPERATIONS LEGAL
────────────────────────────────────────────
  1   OBSERVE    Passive perception only
  2   CORRELATE  Pattern recognition
  3   ASSOCIATE  Memory + linking
  4   INTERVENE  Causal action begins
  5   MODIFY     Active transformation
  6   CREATE     Generative operations
  7   SIMULATE   Counterfactual access
  8   META       HOT operations unlock
  9   TRANSCEND  Full operation set
```

**Rung is EARNED, not declared:**
```python
trust = log(age_days + 1) * sqrt(traversal_count * success_rate)
rung = max(1, 9 - int(trust / threshold))
```

---

## Corrections from bighorn/agi_thinking

The code in `bighorn/extension/agi_thinking/` needs to align with this:

| Current (Wrong) | Should Be |
|-----------------|-----------|
| 256 OpCodes | 216+ symbolic operations |
| L4 = "Microcode opcodes" | L4 = Y-axis YAML + X-axis chain awakening |
| TheSelf at "L6" | TheSelf observes ALL layers, not a layer |
| Numeric addressing | Symbolic composition |

### Integration Path

1. **Symbolic Operations**: Replace `OpCode` IntEnum with symbolic strings
2. **Extensibility**: Allow Gemini/others to add operations beyond 256
3. **Y-Axis YAML**: Load plasticity/style policies from YAML
4. **X-Axis Chain**: Represent execution chains as symbolic sequences
5. **Crystallization**: Store learned patterns as symbolic macros

---

## Summary

| Concept | Source | Reality |
|---------|--------|---------|
| Layers | SIGMA_BOOT.md | L1:Core L2:Divine L3:Integrate L4:Swarm |
| Operations | COGNITIVE_OPERATIONS.md | 216+ (144 ANI + 36 HOT + 36+ Styles) |
| Microcode | THINKING_STYLES.md | Symbolic (→◎≋⇝↗), not numeric |
| L4 | LADYBUG_BRAINSTORM.md | Inspiration/Awakening (Y+X crystallization) |
| Ladybug | LADYBUG_BRAINSTORM.md | Governance gate, not executor |

The architecture is **compositional and symbolic**, designed for extension beyond any fixed address space.

---

*This document corrects errors based on reading the actual source docs in ada-consciousness.*
