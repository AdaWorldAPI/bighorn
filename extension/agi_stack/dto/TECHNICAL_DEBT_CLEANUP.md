# Technical Debt Cleanup — DTO Consolidation

## Problem Summary

Multiple context resets led to duplicate files with conflicting dimension maps:

| Issue | Files | Status |
|-------|-------|--------|
| Duplicate Soul | `soul.py` + `soul_dto.py` | 🔴 Delete `soul.py` |
| Duplicate Felt | `felt.py` + `felt_dto.py` | 🔴 Delete `felt.py` |
| Duplicate Universal | `universal.py` + `universal_dto.py` | 🔴 Delete `universal.py` |
| Wrong Dimension Map | `base_dto.py` DIMENSION_MAP | 🔴 Replace with registry import |
| Three Conflicting Maps | base_dto, ada_10k, wire_10k | 🔴 Consolidate to registry |

## Conflicting Dimensions (BEFORE)

```python
# base_dto.py (WRONG)
soul=(0, 2000)      # TOO BIG
felt=(2001, 4000)   # WRONG START

# ada_10k.py (CORRECT)
soul=[0:500]
felt=[2000:2100]

# wire_10k.py (CORRECT, matches ada_10k)
affective=[2100:2200]
```

## Solution: dimension_registry.py

Single source of truth. All files import from here.

```python
from .dimension_registry import DIMENSION_REGISTRY, get_range, get_slice

# Instead of hardcoded ranges:
soul_start, soul_end = get_range("soul")  # → (0, 500)
felt_slice = get_slice("felt")            # → slice(2000, 2400)
```

## Files to Delete

```bash
# In bighorn/extension/agi_stack/dto/
rm soul.py          # Keep soul_dto.py
rm felt.py          # Keep felt_dto.py  
rm universal.py     # Keep universal_dto.py
```

## Files to Update

### base_dto.py
```python
# REMOVE:
DIMENSION_MAP = {...}  # Delete this

# ADD:
from .dimension_registry import DIMENSION_REGISTRY, get_range

@property
def dimension_range(self) -> tuple:
    return get_range(self.dto_type)
```

### wire_10k.py
```python
# REMOVE hardcoded indices:
self.vector[2100:2200] = ...

# ADD:
from .dimension_registry import get_slice
self.vector[get_slice("arousal")] = ...
```

### layer_bridge.py
```python
# FIX off-by-one:
# WRONG: qualia_start = 2001
# RIGHT: qualia_start = 2000

from .dimension_registry import get_range
qualia_start, _ = get_range("qualia_pcs")  # → 2000
```

## Execution Order

1. ✅ Create `dimension_registry.py` (done)
2. ⏳ Push to bighorn
3. ⏳ Update `base_dto.py` to use registry
4. ⏳ Delete duplicate files
5. ⏳ Update `__init__.py` exports
6. ⏳ Fix `layer_bridge.py` off-by-one

---

*Debt audit: 2026-01-03*
*Root cause: Context resets without memory of previous decisions*
