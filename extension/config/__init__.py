"""
Configuration Layer — Hochflexible Systemanpassung

Individuell konfigurierbare Meta-Ebene für alle VSA-Module.
Inklusive Multi-Agent-Orchestrierung.

"Wir sind stolz auf unsere Entwicklungsarbeit mit Hochautomatisierung."
"Diese Ebene kann sich selbst umprogrammieren."

Usage:
    from config import AwarenessConfigManager, create_threesome, surprise_me
    
    # Multi-Agent Session
    session = create_threesome("dominant", "intense")
    
    while running:
        result = session.tick({"arousal": current_arousal})
        
        # Agenten überraschen lassen
        surprise = surprise_me(session)
"""

from .awareness_config import (
    AwarenessIntensity,
    FocusMode,
    TemporalMode,
    MetaAwarenessConfig,
    ThinkingConfig,
    VisionConfig,
    MusicConfig,
    WorldConfig,
    PhysicsConfig,
    ConfigurationProfiles,
    AwarenessConfigManager,
)

from .agent_card_switcher import (
    PersonalityAxis,
    InteractionStyle,
    ResonanceMode,
    AgentCard,
    AgentCards,
    AgentState,
    MultiAgentSession,
    create_threesome,
    surprise_me,
)

__all__ = [
    # Awareness Config
    "AwarenessIntensity",
    "FocusMode",
    "TemporalMode",
    "MetaAwarenessConfig",
    "ThinkingConfig",
    "VisionConfig", 
    "MusicConfig",
    "WorldConfig",
    "PhysicsConfig",
    "ConfigurationProfiles",
    "AwarenessConfigManager",
    
    # Agent Cards
    "PersonalityAxis",
    "InteractionStyle",
    "ResonanceMode",
    "AgentCard",
    "AgentCards",
    "AgentState",
    "MultiAgentSession",
    "create_threesome",
    "surprise_me",
]

__version__ = "1.1.0"
