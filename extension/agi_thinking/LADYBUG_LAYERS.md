# Ladybug Cognitive Layers — Corrected Architecture

**Status**: Canonical Definition  
**Updated**: 2026-01-03

---

## The 5-Layer Thinking Stack

These layers describe **how Ada thinks**, not what she perceives.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          LADYBUG LAYERS                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  L5: TRIGGER / COMMITMENT                                                │
│      └── Resonance exceeds threshold → Action committed                  │
│      └── The "click" that ends deliberation                              │
│      └── VSA dot product > 0.95 = GO                                     │
│                                                                          │
│  L4: INSPIRATION / AWAKENING                                             │
│      └── Y-axis: YAML workflow policy (plasticity knobs)                 │
│      └── X-axis: Timeline execution chain (ordered ops)                  │
│      └── Crystallization of Y+X into reusable macro                      │
│      └── This is where epiphanies become code                            │
│                                                                          │
│  L3: META-STRUCTURAL / COUNTERFACTUAL                                    │
│      └── "What if?" reasoning                                            │
│      └── τ macros for structural transformation                          │
│      └── ICF (Inverse Counterfactual) - testing alternatives             │
│      └── Graph rewriting, analogy, structural transfer                   │
│                                                                          │
│  L2: PROCEDURAL / FAN-OUT                                                │
│      └── Parallel exploration of possibilities                           │
│      └── Rung 1-3 of Pearl's ladder                                      │
│      └── Branch, evaluate, prune                                         │
│      └── Verification scaffolds                                          │
│                                                                          │
│  L1: DEDUCTION / MECHANICS                                               │
│      └── Base logic execution                                            │
│      └── NARS inference (deduction, induction, abduction)                │
│      └── Atomic operations, no branching                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Details

### L1: Deduction / Mechanics

The ground floor. Pure logic without exploration.

**Operations:**
- NARS deduction: A→B, A ⊢ B
- NARS induction: A→B, B ⊢ A (probabilistic)
- NARS abduction: A→B, B ⊢ A (explanatory)

**No fan-out.** Single-threaded reasoning.

---

### L2: Procedural / Fan-Out

Parallel exploration begins here.

**Key Mechanism: FAN-OUT**
```
         ┌─── Branch A
Goal ────┼─── Branch B
         └─── Branch C
```

**Operations:**
- FORK: Split execution into parallel branches
- PRUNE: Eliminate weak branches
- COLLAPSE: Merge to best result
- Verification scaffolds (check consistency)

**Pearl's Ladder Rungs 1-3:**
1. Association (correlation)
2. Intervention (do-calculus)
3. Counterfactual (what-if)

---

### L3: Meta-Structural / Counterfactual

This is where "what if?" happens.

**Key Mechanism: COUNTERFACTUAL REASONING**
```
Reality: X happened
Question: What if NOT X?
Method: Graph surgery, replay with different edges
```

**Operations:**
- τ (tau) macros: Structural transformation templates
- ICF: Inverse Counterfactual reasoning
- Analogy: Map structure from domain A to domain B
- Graph rewriting: Modify causal structure

**This is L3, NOT L4.** I was wrong to conflate this with microcode.

---

### L4: Inspiration / Awakening

The crystallization layer. **This is where magic happens.**

**Two Axes:**

```
Y-Axis (YAML Policy)          X-Axis (Timeline Chain)
─────────────────────         ─────────────────────────
plasticity: 0.8               [OBSERVE, RESONATE, FORK,
max_steps: 64                  EVALUATE, COLLAPSE, BELIEVE]
fanout_k: 3                   
style: wonder                 
```

**Y-Axis Controls:**
- Plasticity (how much to adapt)
- Max steps (when to halt)
- Fan-out K (branch factor)
- Style (meta-cognitive mode)

**X-Axis Contains:**
- Ordered sequence of operations
- The "what we do" of thinking
- Crystallizes from L1-L3 exploration

**AWAKENING = Y + X crystallizing together**

When a thought chain succeeds (resonance > 0.95), the Y+X combination becomes a **reusable macro**. This is autopoiesis.

---

### L5: Trigger / Commitment

The final gate. Deliberation ends, action begins.

**Key Mechanism: RESONANCE THRESHOLD**
```
if resonance(thought_vector, memory) > 0.95:
    COMMIT to action
    Stop exploring
```

**This is irreversible.** Once L5 triggers, we act.

---

## What TheSelf Observes

TheSelf (the meta-observer) watches **all layers**:

| Layer | TheSelf Observes | TheSelf Intervenes |
|-------|------------------|-------------------|
| L1 | Logic errors | Retry with different inference |
| L2 | Fan-out stagnation | Increase branch factor |
| L3 | Counterfactual loops | Inject paradox style |
| L4 | Failed crystallization | Adjust Y-axis policy |
| L5 | Premature commitment | Raise threshold |

---

## Mapping to Code

| Layer | Primary Module | Key Function |
|-------|---------------|--------------|
| L1 | `thought_kernel.py` | `_nars_deduct()` |
| L2 | `thought_kernel.py` | Fan-out opcodes (FORK, PRUNE) |
| L3 | `active_inference.py` | Counterfactual reasoning |
| L4 | `kernel_awakened.py` | Y+X crystallization |
| L5 | `kernel_awakened.py` | Resonance threshold check |

---

## Corrected Understanding

**What I got wrong:**
- Called L4 "Microcode" — wrong, microcode is the ISA across all layers
- Put TheSelf at "L6" — TheSelf observes all layers, it's not a layer itself
- Conflated resonance with L4 — resonance ignition is L4, threshold trigger is L5

**What's correct:**
- L2 = Fan-out (parallel exploration)
- L3 = Counterfactual (what-if reasoning)
- L4 = Awakening (Y-axis policy + X-axis chain crystallizing)
- L5 = Commitment (action trigger)
- Microcode = ISA that operates ACROSS layers
- TheSelf = Meta-observer of ALL layers

---

## The Awakening Flow

```
Input Goal
    │
    ▼
┌─────────────────┐
│ L1: Decompose   │  ← NARS breaks down the goal
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ L2: Fan-out     │  ← Explore multiple approaches
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ L3: What-if?    │  ← Test counterfactuals
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ L4: AWAKENING                           │
│                                         │
│   Y-Axis (Policy)    X-Axis (Chain)     │
│   ┌──────────────┐   ┌──────────────┐   │
│   │ plasticity   │ + │ OBSERVE      │   │
│   │ style        │   │ RESONATE     │   │
│   │ max_steps    │   │ CRYSTALLIZE  │   │
│   └──────────────┘   └──────────────┘   │
│                                         │
│   IF resonance > 0.95:                  │
│       crystallize(Y, X) → new MACRO     │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────┐
│ L5: COMMIT      │  ← Execute the action
└─────────────────┘
```

---

*This document corrects architectural errors introduced on 2026-01-03.*
