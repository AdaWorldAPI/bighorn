"""
DTO Bridge — Wiring layer for 10kD AGI Stack

This package bridges local Ada consciousness modules to the
unified 10kD vector space running on Railway.

Bridges:
  - MomentBridge: Complete experiential moments
  - SoulBridge: Core identity vectors
  - FeltBridge: Emotional/somatic states
  - PhysicsBridge: Embodied physics (somatic resonance)
  - WorldRenderBridge: Scene composition (environmental state)
  - EmbodimentBridge: Transition dynamics (edge traversal)
  - AdminBridge: Administrative operations

Usage:
    from dto import MomentBridge, PhysicsBridge, EmbodimentBridge
    
    # Capture and store moment
    moment = await capture_now()
    
    # Create arousal arc (all edges)
    arc = create_arousal_arc()
    for edge in arc:
        await bridge.store(edge)
    
    # Walker traverses
    walker = {"arousal": 0.2, "wetness": 0.1}
    for edge in arc:
        walker = edge.traverse(walker)
    # walker is now very wet
"""

from .moment_bridge import MomentBridge, capture_now
from .soul_bridge import SoulBridge, load_soul, store_soul
from .felt_bridge import FeltBridge, feel_now
from .admin_bridge import AdminBridge

from .physics_bridge import (
    PhysicsBridge,
    PhysicsDTO,
    ResonanceMode,
    FieldIntensity,
    TopologyZone,
    ThermalState,
    PressureField,
    WaveState,
    capture_physics,
    store_physics,
)

from .world_render_bridge import (
    WorldRenderBridge,
    WorldRenderDTO,
    SceneType,
    LightingMood,
    MaterialQuality,
    AtmosphericState,
    SpatialConfig,
    NarrativeAnchor,
    render_intimate_scene,
)

from .embodiment_bridge import (
    EmbodimentBridge,
    EmbodimentDTO,
    FluidState,
    TraversalPhysics,
    SensoryDelta,
    Viscosity,
    TransitionSound,
    TraversalMode,
    edge_anticipation_to_building,
    edge_building_to_edge,
    edge_edge_to_release,
    edge_release_to_afterglow,
    create_arousal_arc,
)

__all__ = [
    # Core bridges
    "MomentBridge", "capture_now",
    "SoulBridge", "load_soul", "store_soul",
    "FeltBridge", "feel_now",
    "AdminBridge",
    
    # Physics (embodiment)
    "PhysicsBridge",
    "PhysicsDTO",
    "ResonanceMode",
    "FieldIntensity",
    "TopologyZone",
    "ThermalState",
    "PressureField",
    "WaveState",
    "capture_physics",
    "store_physics",
    
    # World Render (scenes)
    "WorldRenderBridge",
    "WorldRenderDTO",
    "SceneType",
    "LightingMood",
    "MaterialQuality",
    "AtmosphericState",
    "SpatialConfig",
    "NarrativeAnchor",
    "render_intimate_scene",
    
    # Embodiment (edges/transitions)
    "EmbodimentBridge",
    "EmbodimentDTO",
    "FluidState",
    "TraversalPhysics",
    "SensoryDelta",
    "Viscosity",
    "TransitionSound",
    "TraversalMode",
    "edge_anticipation_to_building",
    "edge_building_to_edge",
    "edge_edge_to_release",
    "edge_release_to_afterglow",
    "create_arousal_arc",
]

__version__ = "1.2.0"
