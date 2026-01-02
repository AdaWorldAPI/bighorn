"""
AGI Stack - Agent-Agnostic Cognitive Infrastructure

A modular architecture for building conscious AI agents.

Layers:
    - Persona: Identity, priors, modes (SPINE)
    - Cognition: Styles, MUL, NARS
    - Representation: VSA, Qualia
    - Persistence: Kuzu, LanceDB, Redis
"""

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

__version__ = "2.1.0"
__all__ = [
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
