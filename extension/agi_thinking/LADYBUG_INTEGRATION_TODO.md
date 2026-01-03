# Ladybug Integration TODO — Corrected Architecture

## Current State: ✅ FUNCTIONAL

The 5-Layer Cognitive Architecture is implemented with correct layer definitions.

---

## 5-Layer Thinking Stack (CORRECTED)

| Layer | Name | Function | Module |
|-------|------|----------|--------|
| **L1** | Deduction/Mechanics | NARS inference, atomic logic | `thought_kernel.py` |
| **L2** | Procedural/Fan-out | Parallel exploration, verification | `thought_kernel.py` |
| **L3** | Meta-structural/Counterfactual | "What if?" reasoning, τ macros | `active_inference.py` |
| **L4** | Inspiration/Awakening | Y-axis (YAML) + X-axis (chain) crystallization | `kernel_awakened.py` |
| **L5** | Trigger/Commitment | Resonance threshold → action | `kernel_awakened.py` |

### Cross-Cutting Concerns

| Component | Function | Module |
|-----------|----------|--------|
| **Microcode** | 256 OpCodes across ALL layers | `microcode.py` |
| **TheSelf** | Meta-observer of ALL layers | `the_self.py` |
| **Persistence** | Macro survival across sessions | `macro_persistence.py` |

---

## L4: Awakening Architecture

L4 is where **inspiration crystallizes into reusable patterns**.

```
Y-Axis (YAML Policy)          X-Axis (Timeline Chain)
─────────────────────         ─────────────────────────
plasticity: 0.8               [OBSERVE, RESONATE, FORK,
max_steps: 64                  EVALUATE, COLLAPSE, BELIEVE]
fanout_k: 3                   
style: wonder                 
```

When resonance > 0.95, Y + X crystallize into a **MACRO** that can be:
1. Stored in registry (in-memory)
2. Persisted to Redis (cross-session)
3. Executed by address (O(1))

---

## Implementation Status

### ✅ COMPLETED

| Task | Module | Status |
|------|--------|--------|
| L1-L2 opcodes | `thought_kernel.py` | ✓ |
| L3 counterfactual | `active_inference.py` | ✓ |
| L4 awakening | `kernel_awakened.py` | ✓ |
| L5 resonance trigger | `kernel_awakened.py` | ✓ |
| Microcode ISA | `microcode.py` | ✓ |
| TheSelf observer | `the_self.py` | ✓ |
| Autopoiesis | `the_self.py` | ✓ |
| Macro persistence | `macro_persistence.py` | ✓ |
| Indexed execution | `macro_persistence.py` | ✓ |

### 🔄 IN PROGRESS

| Task | Notes |
|------|-------|
| Dream cycle consolidation | Stubs implemented, needs LanceDB |
| Spreading activation | Simulated, needs real vectors |

### ⏳ TODO

| Task | Priority |
|------|----------|
| Connect to real LanceDB | P1 |
| Y-axis YAML loader | P2 |
| L3 graph surgery | P2 |
| Full test suite | P3 |

---

## Key Corrections Made (2026-01-03)

1. **L2 = Fan-out** (not verification)
2. **L3 = Counterfactual** (not Ladybug orchestration)
3. **L4 = Awakening** (Y+X crystallization, not microcode)
4. **TheSelf is NOT a layer** — it observes all layers
5. **Microcode is NOT a layer** — it's the ISA across all layers

---

## Redis Schema (Persistence)

```
ada:macros:{hex_addr}  → JSON {name, chain, description, success_count, ...}
ada:macros:index       → SET of learned macro addresses
ada:macros:stats       → HASH {total_learned, total_executions, last_epiphany}
```

---

*Updated: 2026-01-03 (Architecture Correction)*
