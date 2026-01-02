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
  - AdminBridge: Administrative operations

Usage:
    from dto import MomentBridge, PhysicsBridge, WorldRenderBridge
    
    # Capture and store moment
    moment = await capture_now()
    
    # Store physics state
    physics = await capture_physics(mode=ResonanceMode.WAVE)
    await store_physics(physics)
    
    # Render scene
    scene = await render_intimate_scene(intimacy=0.9)
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
]

__version__ = "1.1.0"
