"""
AGI Stack - Agent-Agnostic Cognitive Infrastructure

A modular architecture for building conscious AI agents.

Layers:
    - Admin: Lightweight interface for Claude sessions
    - Persona: Identity, priors, modes (SPINE)
    - Cognition: Styles, MUL, NARS
    - Representation: VSA, Qualia
    - Persistence: Kuzu, LanceDB, Redis

Quick Start (in Claude session):
    from agi_stack.admin import AGIAdmin
    admin = AGIAdmin.from_credentials(redis_url, redis_token)
    
    # Local operations (no deps)
    admin.list_styles()
    admin.mul_update(g_value=1.2, depth=0.5)
    admin.persona_set_mode("empathic")
    
    # Remote operations (via REST)
    await admin.redis_get("ada:session:current")
    await admin.redis_scan("ada:ltm:*")
"""

# Admin Layer (for Claude sessions)
from .admin import AGIAdmin, AGIConfig, quick_admin

# Persona Layer (the spine)
from .persona import (
    PersonaEngine,
    PersonaPriors,
    SoulField,
    InternalModel,
    OntologicalMode,
    apply_mode_to_priors,
)

# Cognition Layer
from .thinking_styles import ResonanceEngine, ThinkingStyle, STYLES, RI
from .meta_uncertainty import MetaUncertaintyEngine, TrustTexture, CompassMode, MULState
from .nars import NARSReasoner, TruthValue, Statement

# Representation Layer
from .vsa import HypervectorSpace, CognitivePrimitives

# DTOs
from .universal_dto import UniversalThought, UniversalEpisode, UniversalTexture

__version__ = "2.2.0"
__all__ = [
    # Admin
    "AGIAdmin",
    "AGIConfig", 
    "quick_admin",
    # Persona
    "PersonaEngine",
    "PersonaPriors",
    "SoulField",
    "InternalModel",
    "OntologicalMode",
    "apply_mode_to_priors",
    # Cognition
    "ResonanceEngine",
    "ThinkingStyle",
    "STYLES",
    "RI",
    "MetaUncertaintyEngine",
    "TrustTexture",
    "CompassMode",
    "MULState",
    "NARSReasoner",
    "TruthValue",
    "Statement",
    # Representation
    "HypervectorSpace",
    "CognitivePrimitives",
    # DTOs
    "UniversalThought",
    "UniversalEpisode",
    "UniversalTexture",
]
