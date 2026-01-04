"""
bridge/orchestrator.py — REDIRECT HOOK
======================================

This module has been superseded by core.orchestrator (v9).

The v9 orchestrator adds BREATH as phase 0, implementing:
"breath modulates qualia BEFORE cognition begins"

This hook maintains backward compatibility while logging usage
to identify any remaining consumers.

Migration:
    OLD: from bridge.orchestrator import SigmaOrchestrator, create_orchestrator
    NEW: from core.orchestrator import SigmaOrchestrator, create_orchestrator

Archive: bridge/_deprecated/orchestrator_v6.py
"""

import warnings
import logging

# Set up deprecation logging
_logger = logging.getLogger("ada.deprecation")

def _log_deprecation(caller_info: str = ""):
    """Log deprecation for tracking migration progress."""
    msg = f"bridge.orchestrator imported (deprecated) {caller_info}"
    _logger.warning(msg)
    warnings.warn(
        "bridge.orchestrator is deprecated. Use core.orchestrator instead. "
        "See bridge/_deprecated/orchestrator_v6.py for archived code.",
        DeprecationWarning,
        stacklevel=3
    )

# Log on import
_log_deprecation()

# Re-export everything from core.orchestrator for compatibility
from ..core.orchestrator import (
    # Enums
    OrchestratorPhase,
    BreathVerb,
    # Classes
    PhaseResult,
    TickResult,
    SigmaOrchestrator,
    # Factory
    create_orchestrator,
    # Handlers
    make_breath_aware_perceive_handler,
    make_breath_aware_affect_handler,
    make_flow_handler,
    make_relate_handler,
)

__all__ = [
    "OrchestratorPhase",
    "BreathVerb",
    "PhaseResult", 
    "TickResult",
    "SigmaOrchestrator",
    "create_orchestrator",
    "make_breath_aware_perceive_handler",
    "make_breath_aware_affect_handler", 
    "make_flow_handler",
    "make_relate_handler",
]
