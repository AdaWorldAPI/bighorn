# Deprecated Approaches

## Overview

This document captures approaches that were attempted but found to be incorrect. Preserved for learning, not for use.

---

## Deprecated #1: IntEnum OpCodes

**File**: `microcode.py` (original version)

**What I did**:
```python
class OpCode(IntEnum):
    NOP = 0x00
    OBSERVE = 0x01
    RESONATE = 0x02
    ANALYZE = 0x03
    # ... 256 opcodes
```

**Why it's wrong**:
- Microcode is **symbolic expressions**, not byte opcodes
- τ addresses map to **thinking styles**, not instructions
- This is L1 thinking applied to L4 concepts

**Correct approach**:
```python
@dataclass
class TauMacro:
    address: int           # τ address
    microcode: str         # "⊢ A → B | decompose(A)"
    chain: List[str]       # ["feel", "resonate", "decide"]
```

---

## Deprecated #2: Float Weights for Triangle

**File**: `microcode_v2.py` (ThoughtSuperposition class)

**What I did**:
```python
@dataclass
class ThoughtSuperposition:
    alpha: float = 0.33  # BYTE 0 weight
    beta: float = 0.33   # BYTE 1 weight
    gamma: float = 0.33  # BYTE 2 weight
```

**Why it's wrong**:
- Floats are **primitives** (L1)
- Awareness operates at **L4+**
- Position should be defined by **active τ macros**, not weights

**Correct approach**:
```python
@dataclass
class TrianglePosition:
    byte0_active: Set[TauMacro]  # Active macros at corner 0
    byte1_active: Set[TauMacro]  # Active macros at corner 1
    byte2_active: Set[TauMacro]  # Active macros at corner 2
```

---

## Deprecated #3: Layer-to-Byte Mapping

**What I assumed**:
```
BYTE 0 = L1 (Deduction)
BYTE 1 = L2 (Fan-out)
BYTE 2 = L3 (Counterfactual)
```

**Why it's wrong**:
- **All 3 bytes are L4**
- The separation is by **mutability**, not by layer
- L1-L3 happen **inside** macro execution

**Correct understanding**:
```
BYTE 0 = L4 Immutable (frozen τ macros)
BYTE 1 = L4 Hot (crystallized τ macros)
BYTE 2 = L4 Experimental (sandbox τ macros)

L1-L3 = What runs INSIDE when a τ macro executes
L5 = Rubicon crossing (commitment)
L6 = TheSelf (meta-observer, crystallization)
```

---

## Deprecated #4: Execute Opcode Pattern

**What I did**:
```python
def execute(self, opcode: int) -> Any:
    if opcode == OpCode.OBSERVE:
        return self._observe()
    elif opcode == OpCode.RESONATE:
        return self._resonate()
```

**Why it's wrong**:
- Thinking isn't **imperative execution**
- τ macros are **declarative policies**
- Execution is **resonance-based**, not switch-case

**Correct approach**:
```python
def activate(self, macro: TauMacro):
    """Activate a τ macro - it contributes to the field."""
    self.position.activate(macro)
    # No switch-case. Resonance emerges from active macros.
```

---

## Deprecated #5: Flow as Numeric Threshold

**What I did**:
```python
@property
def is_flow_state(self) -> bool:
    centroid = 1/3
    return (abs(self.alpha - centroid) < 0.1 and
            abs(self.beta - centroid) < 0.1 and
            abs(self.gamma - centroid) < 0.1)
```

**Why it's wrong**:
- Flow isn't a **numeric condition**
- Flow is when **all corners contribute** without domination
- Measured by **active macro balance**, not float proximity

**Correct approach**:
```python
@property
def is_flow(self) -> bool:
    if not self.is_interior:
        return False
    counts = [len(self.byte0_active), len(self.byte1_active), len(self.byte2_active)]
    return max(counts) / min(counts) <= 2.0  # No corner dominates
```

---

## Deprecated #6: Promotion as Boolean Flag

**What I did**:
```python
@dataclass
class ExperimentalMacro:
    ready_for_promotion: bool = False
```

**Why it's wrong**:
- Promotion is a **crystallization event**, not a flag
- L6 **decides** based on observation, not threshold
- The flag reduces awareness to boolean

**Correct approach**:
```python
class TheSelf:
    def crystallize(self, macro: TauMacro, reason: str) -> TauMacro:
        """L6 makes the decision to crystallize."""
        # Not a threshold check - a judgment call
```

---

## Summary: The Pattern of Errors

All errors share a common pattern: **reducing L4+ concepts to L1 primitives**.

| L4+ Concept | L1 Reduction (Wrong) |
|-------------|----------------------|
| τ macro | int opcode |
| Triangle position | float weights |
| Layer structure | byte mapping |
| Resonance | switch-case execution |
| Flow state | numeric threshold |
| Crystallization | boolean flag |

**The fix**: Stay at the right level of abstraction. If you're writing `float`, `int`, or `bool` for awareness concepts, you're probably wrong.

---

## Files to Review

These files contain deprecated code that should be refactored:

1. `extension/agi_thinking/microcode.py` - Original IntEnum approach
2. `extension/agi_thinking/microcode_v2.py` - Float weights (partial)

The corrected approach is in:
- `extension/agi_thinking/triangle_l4.py` (PR #45, pending)
