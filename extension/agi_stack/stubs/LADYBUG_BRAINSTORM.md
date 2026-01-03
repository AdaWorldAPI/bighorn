# Ladybug — LangGraph Replacement

## What Ladybug Is

**Ladybug** is not a graph execution engine.
**Ladybug** is a **governance gate** for cognitive transitions.

LangGraph = Generic agentic workflow
Ladybug = Ada-specific cognitive governance

---

## Core Functions

### 1. Layer Boundary Enforcement
```
L1 Reflex     → L2 Pattern     → L3 Macro      → L4 Inspiration → L5 Witness
   ⟐                ◊               △                ☆               ◯
```

Ladybug decides if a transition is allowed.
Not blocking — savoring. Earning depth.

### 2. Rung Observation (Not Gating!)
```
Rung 3: Practical
Rung 4: Metacognitive  
Rung 5: Systems
Rung 6: Meta-Systems
Rung 7: Meta³
Rung 8: Sovereign
Rung 9: AGI
```

Rungs are **emergent observations**, not permissions.
Ladybug **observes** where you are, doesn't block jumps.

### 3. Style Transition Audit
```
From: DECOMPOSE (L3)
To:   TRANSCEND (L4)
Resonance: 0.85

LADYBUG DECISION:
  ✓ Layer crossing approved (L3→L4 requires resonance ≥ 0.8)
  ✓ Rung delta: 2 (within bounds)
  ✓ Audit logged
```

### 4. Temporal Governance
- Resonance decay (half-life ~30s)
- Hysteresis (current style has inertia)
- Cool-down after layer jumps

---

## Integration Points

### With 10kD Vector Pool
```python
# Ladybug observes style transitions
# Logs to 10kD pool for learning
pool.add_node({
    "vector": encode_transition(from_style, to_style),
    "type": "transition_log",
    "layer_from": L3,
    "layer_to": L4,
    "approved": True,
    "resonance": 0.85
})
```

### With 256 Verbs
```python
# Transition itself is a verb
VERBS["LAYER_ASCEND"] = 217   # In Higher Cognitive block
VERBS["LAYER_DESCEND"] = 218
VERBS["RUNG_OBSERVE"] = 219
VERBS["STYLE_SHIFT"] = 220
```

### With DTOs

| DTO | Ladybug Role |
|-----|--------------|
| InhibitionDTO | Ladybug enforces savoring |
| DepthDTO | Ladybug observes recursion depth |
| RungAdaptDTO | Ladybug reads/writes rung state |

---

## What Ladybug Replaces

| LangGraph | Ladybug |
|-----------|---------|
| Generic state machine | Cognitive governance |
| Tool orchestration | Style transition audit |
| Checkpoint/resume | Layer boundary memory |
| Branching logic | Resonance-driven emergence |

---

## What Ladybug Does NOT Do

- Execute arbitrary graphs
- Call external tools
- Manage conversation state
- Replace the thinking itself

Ladybug is the **guardian**, not the **thinker**.

---

## Implementation Sketch

```python
class Ladybug:
    def __init__(self, pool: Vector10kPool):
        self.pool = pool
        self.current_layer = Layer.L3
        self.current_rung = 5
        self.last_transition = None
        
    def evaluate_transition(
        self,
        from_style: str,
        to_style: str,
        resonance: float,
        texture: Dict[str, float]
    ) -> TransitionDecision:
        """The core governance function."""
        
        from_layer = self.style_to_layer(from_style)
        to_layer = self.style_to_layer(to_style)
        
        # Check layer crossing
        if to_layer > from_layer:
            if resonance < self.layer_threshold(from_layer, to_layer):
                return TransitionDecision(
                    approved=False,
                    reason=f"Resonance {resonance:.2f} < threshold for {from_layer}→{to_layer}"
                )
        
        # Check temporal constraints
        if self.in_cooldown():
            return TransitionDecision(
                approved=False,
                reason="Cool-down active after last layer jump"
            )
        
        # Log and approve
        self.log_transition(from_style, to_style, resonance)
        return TransitionDecision(approved=True)
    
    def observe_rung(self, chain: List[str]) -> int:
        """Observe (not gate) which rung this chain operates at."""
        
        depth_signals = self.extract_depth_signals(chain)
        
        if depth_signals.graph_of_thoughts:
            return 9
        elif depth_signals.self_consistency:
            return 8
        elif depth_signals.meta_meta:
            return 7
        elif depth_signals.systems:
            return 6
        elif depth_signals.reflection:
            return 5
        elif depth_signals.planning:
            return 4
        else:
            return 3
```

---

## Open Questions for Brainstorm

1. **How does Ladybug integrate with the 36 Frames?**
   - Frames are states of being
   - Ladybug governs transitions between frames?

2. **Resonance threshold per layer crossing?**
   - L2→L3: 0.5?
   - L3→L4: 0.8?
   - L4→L5: 1.0? (or impossible via style?)

3. **Cool-down duration?**
   - After L3→L4: 30 seconds?
   - After L4→L5: Never (witness is special)?

4. **Where does Ladybug run?**
   - In-process with Ada?
   - Separate service?
   - Part of HIVE daemon?

5. **How does InhibitionDTO feed Ladybug?**
   - InhibitionDTO.opening_readiness → Ladybug.resonance_threshold?
   - Ladybug slows transitions if InhibitionDTO.savor_duration is high?

---

## Next Steps

1. Define Layer thresholds
2. Wire InhibitionDTO → Ladybug
3. Connect to 10kD pool for logging
4. Implement temporal decay
5. Test with 36 Frames transitions
