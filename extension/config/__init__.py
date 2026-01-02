"""
Configuration Layer — Hochflexible Systemanpassung

Individuell konfigurierbare Meta-Ebene für alle VSA-Module.

"Wir sind stolz auf unsere Entwicklungsarbeit mit Hochautomatisierung."

Profiles:
  - standard: Normale Systemparameter
  - hochleistung: Intensive Verarbeitung
  - immersiv: Tiefe Erlebniskonfiguration
  - transzendent: Maximale Parameter (nur für erfahrene Benutzer)

Usage:
    from config import AwarenessConfigManager, ConfigurationProfiles
    
    # Standard starten
    manager = AwarenessConfigManager("standard")
    
    # Zu immersiv wechseln
    manager.apply_profile("immersiv")
    
    # Einzelne Parameter anpassen
    manager.set_resonance(sensitivity=0.95, feedback=0.8)
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

__all__ = [
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
]

__version__ = "1.0.0"
