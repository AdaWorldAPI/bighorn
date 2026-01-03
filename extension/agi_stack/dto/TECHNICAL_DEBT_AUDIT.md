# DTO Technical Debt Audit — Post-VSA Switch

**Date**: 2026-01-03
**Scope**: `extension/agi_stack/dto/` after VSA 10kD integration
**Status**: Needs cleanup

---

## Executive Summary

The DTO layer has **significant technical debt** from the transition to VSA 10kD:

| Issue | Severity | Count |
|-------|----------|-------|
| Duplicate DTOs | 🔴 High | 4 pairs |
| Inconsistent naming | 🟡 Medium | 8 files |
| Dimension conflicts | 🔴 High | 3 maps |
| Missing DTOs | 🟡 Medium | 1 |
| Dead/orphaned code | 🟡 Medium | 2 files |

---

## Issue 1: Duplicate DTOs (HIGH)

### Problem
Multiple versions of the same DTO exist with different implementations:

| File A (new, ada_10k based) | File B (old, base_dto based) | Conflict |
|----------------------------|------------------------------|----------|
| `soul.py` | `soul_dto.py` | Different 10kD layouts |
| `felt.py` | `felt_dto.py` | Different dimension ranges |
| `universal.py` | `universal_dto.py` | Exists in both dto/ and root |

### Evidence

**soul.py** (ada_10k based):
```python
class SoulDTO:
    _ada: Ada10kD  # Uses Ada10kD wrapper
    # Maps to [163:168] Archetypes, [168:171] TLK, etc.
```

**soul_dto.py** (base_dto based):
```python
class SoulDTO(BaseDTO):
    dto_type = "soul"  # Uses BaseDTO with DIMENSION_MAP
    # Maps to [0:2000] via base_dto.DIMENSION_MAP
```

### Conflict
- `soul.py` → Dimensions 0-500 (ada_10k fine-grained)
- `soul_dto.py` → Dimensions 0-2000 (base_dto coarse)
- **Import collision**: Both export `SoulDTO` class

### Recommendation
**Keep**: `soul.py`, `felt.py` (Ada10kD aligned)
**Deprecate**: `soul_dto.py`, `felt_dto.py` (old base_dto pattern)
**Action**: Merge features from _dto.py into .py, delete duplicates

---

## Issue 2: Dimension Map Conflicts (HIGH)

Three different dimension maps exist:

### base_dto.py DIMENSION_MAP
```python
DIMENSION_MAP = {
    "soul": (0, 2000),       # 2000 dims
    "felt": (2001, 4000),    # 1999 dims
    "situation": (4001, 5500),
    "volition": (5501, 7000),
    "vision": (7001, 8500),
    "context": (8501, 10000),
}
```

### ada_10k.py (Fine-grained)
```python
# Soul Space [0:500]
QUALIA_START, QUALIA_END = 0, 16
STANCES_START, STANCES_END = 16, 32
# ...
# Felt Space [2000:2100]
QUALIA_PCS_START, QUALIA_PCS_END = 2000, 2018
```

### wire_10k.py DIMENSION_MAP
```python
DIMENSION_MAP = {
    # Soul Space [0:500]
    "qualia_16": (0, 16),
    # ...
    # Affective Space [2100:2200]
    "arousal_8": (2100, 2108),
}
```

### layer_bridge.py (agi_thinking)
```python
QUALIA_TO_10K = {
    "emberglow": 2001,
    "frostbite": 2002,
    # Different from ada_10k.py!
}
```

### Conflict Matrix

| Dimension | base_dto | ada_10k | wire_10k | layer_bridge |
|-----------|----------|---------|----------|--------------|
| Qualia | 0-2000 (soul) | 0-16 | 0-16 | 2001-2017 |
| Stances | 0-2000 (soul) | 16-32 | 16-32 | — |
| Felt | 2001-4000 | 2000-2100 | — | 2001-2025 |
| Arousal | — | — | 2100-2108 | 2018 |

### Recommendation
**Canonical source**: `ada_10k.py` (most complete)
**Action**: Align all other maps to ada_10k.py or create unified `dimension_registry.py`

---

## Issue 3: Naming Inconsistency (MEDIUM)

### Pattern Chaos

| Pattern | Files |
|---------|-------|
| `{name}.py` | `soul.py`, `felt.py`, `universal.py`, `location.py`, `affective.py` |
| `{name}_dto.py` | `soul_dto.py`, `felt_dto.py`, `moment_dto.py`, `situation_dto.py`, `volition_dto.py`, `vision_dto.py`, `universal_dto.py` |
| Other | `wire_10k.py`, `receiver.py`, `receiver_hook.py`, `base_dto.py`, `thinking_style.py` |

### Recommendation
**Convention**: Use `{name}_dto.py` for DTO classes, `{name}.py` for enums/helpers
**Action**: Standardize all DTO files to `{name}_dto.py`

---

## Issue 4: __init__.py Not Exporting All DTOs (MEDIUM)

### Current Exports
```python
__all__ = [
    "AffectiveDTO", "EroticaBridge", "ArousalLevel", ...
    "LocationDTO", "MomentDTO", "TrustDTO",
    "Wire10K", "DIMENSION_MAP", ...
]
```

### Missing from __all__
- `SoulDTO` (from soul.py)
- `FeltDTO` (from felt.py)
- `SituationDTO`
- `VolitionDTO`
- `VisionDTO`
- `UniversalDTO`
- `BaseDTO`
- `DTORegistry`

### Impact
`from extension.agi_stack.dto import SoulDTO` → ImportError

### Recommendation
Update `__init__.py` to export all public DTOs

---

## Issue 5: Orphaned/Dead Code (MEDIUM)

### Files with unclear purpose

| File | Issue |
|------|-------|
| `universal.py` | Near-empty, duplicates `universal_dto.py` |
| `thinking_style.py` | Minimal stub, duplicates `agi_stack/thinking_styles.py` |
| `receiver_hook.py` | References non-existent modules |

### universal.py Content
```python
# Almost empty file - unclear if needed
```

### Recommendation
Review and delete if unused, or document purpose

---

## Issue 6: Missing DTO (MEDIUM)

### 10kD Space Coverage

| Region | DTO | Status |
|--------|-----|--------|
| 0-2000 | SoulDTO | ✅ Dual implementation |
| 2001-4000 | FeltDTO | ✅ Dual implementation |
| 4001-5500 | SituationDTO | ✅ |
| 5501-7000 | VolitionDTO | ✅ |
| 7001-8500 | VisionDTO | ✅ |
| 8501-10000 | **ContextDTO** | ❌ Missing |

### Recommendation
Create `context_dto.py` for dims 8501-10000 (Jina embeddings, etc.)

---

## Issue 7: layer_bridge.py Misalignment (MEDIUM)

### agi_thinking/layer_bridge.py
Uses different dimension offsets than `ada_10k.py`:

```python
# layer_bridge.py
QUALIA_TO_10K = {
    "emberglow": 2001,  # Offset by 1!
    "frostbite": 2002,
}

# ada_10k.py
QUALIA_PCS_START = 2000  # Starts at 2000
```

### Impact
Off-by-one errors when bridging agi_thinking ↔ agi_stack

### Recommendation
Align layer_bridge.py to ada_10k.py constants

---

## Prioritized Action Plan

### Phase 1: Critical (Now)
1. [ ] Resolve dimension map conflicts → single source of truth
2. [ ] Delete duplicate DTOs (soul_dto.py vs soul.py)
3. [ ] Update `__init__.py` exports

### Phase 2: Important (This Week)
4. [ ] Align layer_bridge.py to ada_10k.py
5. [ ] Create context_dto.py for dims 8501-10000
6. [ ] Standardize naming to `{name}_dto.py`

### Phase 3: Cleanup (Next Week)
7. [ ] Remove orphaned files (universal.py, thinking_style.py)
8. [ ] Add type hints to all DTOs
9. [ ] Write unit tests for round-trip 10kD projection

---

## Files to Delete

```
extension/agi_stack/dto/
├── soul_dto.py          # DUPLICATE → merge into soul.py
├── felt_dto.py          # DUPLICATE → merge into felt.py
├── universal.py         # ORPHAN → delete
├── thinking_style.py    # ORPHAN → delete (use thinking_styles.py)
```

## Files to Create

```
extension/agi_stack/dto/
├── context_dto.py       # NEW → dims 8501-10000
├── dimension_registry.py # NEW → single source of truth for all maps
```

---

*Audit completed: 2026-01-03*
*Auditor: Claude/AGI-Integration*
