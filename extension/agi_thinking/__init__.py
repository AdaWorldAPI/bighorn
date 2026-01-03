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

Cross-cutting:
  Microcode: 256 OpCodes that operate ACROSS all layers
  TheSelf: Meta-observer that watches ALL layers (not a layer itself)

All layers operate in 10000D VSA space.
1024D exists only for derived artifacts after L5.

Core Modules:
  thought_kernel.py   — L1-L2 cognitive operations
  active_inference.py — L3 counterfactual reasoning
  kernel_awakened.py  — L4-L5 awakening and commitment
  microcode.py        — OpCode ISA (cross-layer)
  the_self.py         — Meta-observer (cross-layer)
  layer_bridge.py     — Maps to 10kD VSA
  texture.py          — Thinking style emergence
  qualia_learner.py   — Plasticity (learning how words feel)
"""

__version__ = "0.3.0"

# Core exports
from extension.agi_thinking.microcode import OpCode, ThinkingMacro, MACRO_REGISTRY
from extension.agi_thinking.the_self import TheSelf, SelfState

