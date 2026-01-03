"""
Ladybug — Cognitive Governance for 9-Rung System

Ladybug is NOT a graph execution engine.
Ladybug is a GOVERNANCE GATE for cognitive transitions.

Components:
    - LadybugEngine: Core governance (rung transitions, coherence)
    - LadybugDebugger: Query analysis and debugging
"""

from .engine import (
    CognitiveRung,
    ThinkingStyle,
    TransitionDecision,
    LadybugState,
    LadybugEngine,
    RUNG_THRESHOLDS,
    STYLE_TO_RUNG,
    create_ladybug,
    create_ladybug_from_10k,
)

from .debugger import (
    LadybugDebugger,
    LadybugConfig,
    QueryAnalysis,
    QueryMetrics,
)

__all__ = [
    # Engine (cognitive governance)
    "CognitiveRung",
    "ThinkingStyle",
    "TransitionDecision",
    "LadybugState",
    "LadybugEngine",
    "RUNG_THRESHOLDS",
    "STYLE_TO_RUNG",
    "create_ladybug",
    "create_ladybug_from_10k",
    # Debugger
    "LadybugDebugger",
    "LadybugConfig",
    "QueryAnalysis",
    "QueryMetrics",
]
