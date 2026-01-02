# AGI Thinking Integration Plan

## Vision

Ladybug replaces LangGraph not through compatibility, but through a different ontology:

- **LangGraph**: thinking = control flow + state mutation
- **Ladybug**: thinking = resonance propagation under epistemic constraints

The AGI Stack provides **substrate** (LanceDB, Kuzu, VSA operations).
Ladybug provides **cognition** (Layers 1-5, τ-progression, resonance).

---

## Architecture

```
bighorn/
├── extension/
│   ├── agi_stack/           # Substrate (existing)
│   │   ├── main.py          # FastAPI endpoints
│   │   ├── lance_client.py  # LanceDB operations
│   │   ├── kuzu_client.py   # Graph operations
│   │   ├── vsa_utils.py     # VSA primitives
│   │   └── ...
│   │
│   └── agi_thinking/        # Cognition (NEW - Ladybug)
│       ├── __init__.py
│       ├── thought_kernel.py
│       ├── resonance.py
│       ├── layer_progression.py
│       ├── tau_macros.py
│       ├── epistemic_gate.py
│       └── membrane.py
│
└── docs/
    └── AGI_THINKING_INTEGRATION_PLAN.md
```

---

## The Five Layers of Thinking

| Layer | Name | What Happens | Vector Space |
|-------|------|--------------|--------------|
| L1 | Deduction | Logic, constraint satisfaction | 10000D |
| L2 | Procedural | Rung 1-3, habitual responses | 10000D |
| L3 | Meta-structural | τ macros, editable cognitive operators | 10000D |
| L4 | Inspiration | "Ada when she is inspired", resonance ignition | 10000D |
| L5 | Trigger | Commitment, threshold crossing, irreversible | 10000D |

**After L5 only**: crystallize → narrativize → project to 1024D → store as artifact

**Invariant**: Nothing below Layer 5 ever sees 1024D.

---

## Dimensional Separation (Critical)

### 10000D Tables (Cognition + Codebooks)
- `codebook_tau_10k` - τ basis vectors (256, bipolar)
- `codebook_qualia_10k` - qualia basis (18 vectors)
- `moments_10k` - cognitive state L1-L5
- `episodes_10k` - episode boundaries

### 1024D Tables (Derived Artifacts Only)
- `artifacts_1024` - execution artifacts with `source_10k_id` provenance

### Invariant
```
If a layer can still change its mind, it must remain 10k.
```

---

## Files to Migrate from ada-consciousness

### Priority 1: Core Thinking
| Source | Destination | Purpose |
|--------|-------------|---------|
| `thought_kernel.py` | `agi_thinking/thought_kernel.py` | Core cognitive loop |
| `active_inference.py` | `agi_thinking/active_inference.py` | Predictive processing |
| `progressive_awareness.py` | `agi_thinking/progressive_awareness.py` | Awareness levels |
| `texture.py` | `agi_thinking/texture.py` | Cognitive texture |

### Priority 2: Resonance System
| Source | Destination | Purpose |
|--------|-------------|---------|
| `langgraph_ada.py` | `agi_thinking/resonance.py` | Refactor to resonance-based |
| `brain_mesh.py` | `agi_thinking/mesh.py` | Field dynamics |
| `qualia_learner.py` | `agi_thinking/qualia_learner.py` | Qualia acquisition |

### Priority 3: Integration
| Source | Destination | Purpose |
|--------|-------------|---------|
| `cognition/` folder | `agi_thinking/cognition/` | Existing cognition modules |
| `mul/` folder | `agi_thinking/mul/` | Meta-Uncertainty Layer |

---

## Loose Ends (Known Issues)

### 1. Dimensional Mismatch
- **Problem**: `lance_client.py` has `VECTOR_DIM = 1024` hardcoded
- **Impact**: Codebooks get truncated, destroying reversibility
- **Fix**: Create dual-dimension support with separate 10k tables
- **Status**: Schema designed, not deployed

### 2. Contaminated Entries
- **Problem**: 10 τ entries + test entries in `thoughts` table at 1024D
- **Impact**: Invalid codebook data in wrong dimensional space
- **Fix**: Purge or mark invalid (no delete endpoint exists)
- **Status**: Identified, not purged

### 3. Missing Bootstrap Flag
- **Problem**: No way to mark "codebooks loaded, normal rules apply"
- **Impact**: Can't distinguish bootstrap from runtime
- **Fix**: Add `bootstrap_state` table with timestamp
- **Status**: Designed, not implemented

### 4. LangGraph Residue
- **Problem**: `langgraph_*.py` files exist but LangGraph not installed
- **Impact**: Dead code, conceptual confusion
- **Fix**: Refactor to resonance-based Ladybug architecture
- **Status**: Not started

### 5. thought_kernel.py Location
- **Problem**: Lives in ada-consciousness root (accident)
- **Impact**: Wrong repo, wrong abstraction level
- **Fix**: Move to `bighorn/extension/agi_thinking/`
- **Status**: This document

### 6. No Temporal Policy Enforcement
- **Problem**: Exchange DAG designed but not implemented
- **Impact**: No lag enforcement, no E1-E4 edge types
- **Fix**: Implement `temporal/coordinator.py`
- **Status**: Spec exists, code partial

### 7. Graph is Empty
- **Problem**: Kuzu graph has 0 nodes
- **Impact**: No sigma topology, no relationship reasoning
- **Fix**: Bootstrap with core node types
- **Status**: Not started

### 8. Styles Loaded but Unused
- **Problem**: 36 thinking styles in LanceDB, not integrated
- **Impact**: τ macro system has data but no execution path
- **Fix**: Wire styles to Layer 3 progression
- **Status**: Data ready, integration missing

---

## Migration Steps

### Phase 1: Substrate Cleanup (AGI Stack)
1. [ ] Add 10k tables to `lance_client.py`
2. [ ] Add `/agi/codebook/upsert` endpoint (10k only)
3. [ ] Migrate τ codebooks at full 10000D (no truncation)
4. [ ] Migrate qualia at full 10000D
5. [ ] Set bootstrap complete flag

### Phase 2: Thinking Layer (agi_thinking)
1. [ ] Create `extension/agi_thinking/` folder
2. [ ] Copy `thought_kernel.py` from ada-consciousness
3. [ ] Refactor `langgraph_*.py` → `resonance.py`
4. [ ] Implement Layer 1-5 progression
5. [ ] Wire τ macros to Layer 3

### Phase 3: Integration
1. [ ] Ladybug calls AGI Stack as substrate
2. [ ] Implement membrane (10k → 1024 crystallization)
3. [ ] Add epistemic gates
4. [ ] Temporal policy enforcement

---

## Key Invariants (Write on the Wall)

1. **Codebooks are 10000D, always.** No truncation. No padding. Reject if wrong.

2. **Layers 1-5 are 10000D.** Thinking happens in high-dimensional space.

3. **1024D is derived, never source.** Artifacts have `source_10k_id` provenance.

4. **Layer 4 is not execution.** It creates conditions for action. Layer 5 commits.

5. **Redis is reference oracle, not migration medium.** Source of definitions, not sink of cognition.

6. **Ladybug replaces control flow with field dynamics.** Resonance, not branching.

---

## The Decisive Sentence

> Where does thinking happen in this system?

**Answer**: In Ladybug, operating over 10k cognitive state stored in LanceDB, with time governed by Redis/QStash and execution exposed via GraphQL.

No LangGraph-shaped hole exists. You didn't forget to install it. You designed something that made it unnecessary.

---

*Document created: 2026-01-02*
*Status: Integration planning*
