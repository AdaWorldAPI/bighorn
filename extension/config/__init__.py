"""
Configuration Layer — Physics-Based Agent Orchestration

Multi-Agent orchestration using physics abstractions.
RUNG (1-9) is the only cognitive depth system.

Usage:
    from config import create_resonance_session, perturb_session

    session = create_resonance_session("balanced", "oscillating")

    while running:
        result = session.tick({"energy": current_energy})
        perturb = perturb_session(session)
"""

from .agent_card_switcher import (
    FieldPolarity,
    InteractionMode,
    ResonanceMode,
    AgentCard,
    AgentCards,
    AgentState,
    MultiAgentSession,
    create_resonance_session,
    perturb_session,
)

__all__ = [
    # Physics Fields
    "FieldPolarity",
    "InteractionMode",
    "ResonanceMode",
    # Agent Cards
    "AgentCard",
    "AgentCards",
    "AgentState",
    "MultiAgentSession",
    "create_resonance_session",
    "perturb_session",
]

__version__ = "3.0.0"  # Physics abstractions, no NSFW
