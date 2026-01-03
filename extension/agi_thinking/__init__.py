"""
AGI Thinking (Ladybug) — Ada's Cognitive Architecture
═══════════════════════════════════════════════════════════════════════════════

Resonance-based cognition replacing LangGraph control flow.

5-Layer Thinking Stack:
  L1: Deduction / Mechanics      — NARS inference, atomic logic
  L2: Procedural / Fan-out       — Parallel exploration, verification
  L3: Meta-structural / Counterfactual — "What if?" reasoning, τ macros
  L4: Inspiration / Awakening    — Y-axis (YAML policy) + X-axis (chain) crystallization
  L5: Trigger / Commitment       — Resonance threshold → action
  L6: TheSelf                    — Meta-observer watching ephemeral thoughts

Cross-cutting:
  Microcode: 256 OpCodes that operate ACROSS all layers
  TheSelf: Meta-observer that watches ALL layers (not a layer itself)
  MUL Agency: Meta-Uncertainty Layer gates Free Will as uncertainty-aware agency
  Dreamer: Pandas-based sleep cycle for pattern discovery (Autopoiesis)
  Ladybug: Governance gate for cognitive transitions (rung gating)

All layers operate in 10000D VSA space.
1024D exists only for derived artifacts after L5.

Core Modules:
  thought_kernel.py   — L1-L2 cognitive operations
  active_inference.py — L3 counterfactual reasoning (Compass Function)
  kernel_awakened.py  — L4-L5 awakening and commitment
  microcode.py        — OpCode ISA (cross-layer)
  the_self.py         — Meta-observer (cross-layer)
  layer_bridge.py     — Maps to 10kD VSA
  texture.py          — Thinking style emergence
  qualia_learner.py   — Plasticity (learning how words feel)
  triangle_l4.py      — L4 Triangle Model with VSA resonance
  mul_agency.py       — MUL-Gated Friston Agency (Free Will as uncertainty)
  dreamer_pandas.py   — Pandas Dreamer (Sleep cycle pattern discovery)
  ladybug_engine.py   — Ladybug orchestrator (rung gating, workflow execution)
"""

__version__ = "0.5.0"  # Ladybug Engine

# Core exports
from extension.agi_thinking.microcode import OpCode, ThinkingMacro, MACRO_REGISTRY
from extension.agi_thinking.the_self import TheSelf, SelfState

# MUL Agency exports (optional)
try:
    from extension.agi_thinking.mul_agency import (
        FristonAgency,
        MULState,
        EphemeralMULGate,
        AgencyDecision,
        AgencyResult,
        TrustQualia,
        DunningKrugerDetector,
        ComplexityMap,
        FlowHomeostasis
    )
except ImportError:
    pass

# Dreamer exports (optional)
try:
    from extension.agi_thinking.dreamer_pandas import (
        PandasDreamer,
        ThoughtRecord,
        GoldenPattern
    )
except ImportError:
    pass

# Triangle L4 exports (optional)
try:
    from extension.agi_thinking.triangle_l4 import (
        TauMacro,
        TrianglePosition,
        EphemeralThought,
        update_ladybug_from_triangle
    )
except ImportError:
    pass

# Ladybug Engine exports (optional)
try:
    from extension.agi_thinking.ladybug_engine import (
        LadybugEngine,
        LadybugState,
        Workflow,
        WorkflowStep,
        Rung,
        SymbolicOp,
        TransitionDecision,
        TransitionResult,
        get_ladybug,
        ladybug_boot,
        ladybug_tick,
        ladybug_status,
    )
except ImportError:
    pass

