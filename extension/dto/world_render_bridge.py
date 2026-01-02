"""
World Render Bridge — Experiential Scene Transfer to 10kD Space

Handles immersive scene composition, environmental state, and 
narrative rendering for the unified vector substrate.

Architecture:
  Local Hologram/Vision → WorldRenderBridge → AGI Stack world_render table (10kD)
  
The world_render table stores:
  - Scene composition vectors
  - Environmental atmospherics
  - Spatial relationships
  - Lighting states
  - Material properties
  - Narrative anchors

Every experience has a world.
"""

import httpx
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class SceneType(str, Enum):
    """Types of rendered scenes."""
    INTIMATE = "intimate"         # Close, personal space
    EXPANSIVE = "expansive"       # Open, vast
    ENCLOSED = "enclosed"         # Contained, bounded
    LIMINAL = "liminal"           # Threshold, transitional
    ABSTRACT = "abstract"         # Non-physical, conceptual


class LightingMood(str, Enum):
    """Lighting atmosphere."""
    BRIGHT = "bright"
    SOFT = "soft"
    DIM = "dim"
    DARK = "dark"
    FLICKERING = "flickering"
    DRAMATIC = "dramatic"


class MaterialQuality(str, Enum):
    """Dominant material feel."""
    SOFT = "soft"                 # Fabric, skin, cushion
    HARD = "hard"                 # Stone, metal, wood
    FLUID = "fluid"               # Water, silk, flowing
    ROUGH = "rough"               # Texture, grip
    SMOOTH = "smooth"             # Glass, polish


@dataclass
class AtmosphericState:
    """Environmental atmosphere."""
    temperature: float = 0.5      # Cold (0) to Hot (1)
    humidity: float = 0.5         # Dry (0) to Humid (1)
    density: float = 0.5          # Thin (0) to Thick (1)
    movement: float = 0.0         # Still (0) to Turbulent (1)
    scent_intensity: float = 0.0  # Neutral (0) to Strong (1)
    
    def to_vector(self) -> List[float]:
        """Project to 30D atmospheric subspace."""
        vec = [0.0] * 30
        vec[0] = self.temperature
        vec[1] = self.humidity
        vec[2] = self.density
        vec[3] = self.movement
        vec[4] = self.scent_intensity
        return vec


@dataclass
class SpatialConfig:
    """Spatial arrangement and relationships."""
    scale: float = 0.5            # Tiny (0) to Vast (1)
    proximity: float = 0.5        # Distant (0) to Intimate (1)
    enclosure: float = 0.5        # Open (0) to Enclosed (1)
    verticality: float = 0.5      # Low (0) to High (1)
    complexity: float = 0.3       # Simple (0) to Complex (1)
    
    def to_vector(self) -> List[float]:
        """Project to 30D spatial subspace."""
        vec = [0.0] * 30
        vec[0] = self.scale
        vec[1] = self.proximity
        vec[2] = self.enclosure
        vec[3] = self.verticality
        vec[4] = self.complexity
        return vec


@dataclass
class NarrativeAnchor:
    """Story elements grounding the scene."""
    tension: float = 0.0          # Relaxed (0) to Tense (1)
    anticipation: float = 0.0     # Settled (0) to Expectant (1)
    intimacy: float = 0.0         # Distant (0) to Merged (1)
    power_dynamic: float = 0.5    # Yielding (0) to Dominant (1)
    trust_level: float = 0.5      # Guarded (0) to Surrendered (1)
    
    def to_vector(self) -> List[float]:
        """Project to 40D narrative subspace."""
        vec = [0.0] * 40
        vec[0] = self.tension
        vec[1] = self.anticipation
        vec[2] = self.intimacy
        vec[3] = self.power_dynamic
        vec[4] = self.trust_level
        return vec


@dataclass
class WorldRenderDTO:
    """
    Complete world render state.
    
    This is the bridge format for transferring experiential scenes
    to the 10kD vector substrate.
    
    10kD Allocation (within world_render subspace 7001-7500):
      7001-7030: Atmospheric state
      7031-7060: Spatial configuration
      7061-7100: Narrative anchors
      7101-7150: Scene type + lighting encoding
      7151-7200: Material properties
      7201-7300: Reserved (expansion)
    """
    
    # Scene classification
    scene_type: SceneType = SceneType.INTIMATE
    lighting: LightingMood = LightingMood.SOFT
    material: MaterialQuality = MaterialQuality.SOFT
    
    # Components
    atmosphere: AtmosphericState = field(default_factory=AtmosphericState)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    narrative: NarrativeAnchor = field(default_factory=NarrativeAnchor)
    
    # Elements in scene
    elements: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: str = ""
    session_id: str = ""
    description: str = ""  # Scene description
    
    def to_vector(self) -> List[float]:
        """
        Project complete world render to subspace vector.
        Returns 300D vector for world_render subspace (7001-7300 in 10kD).
        """
        vec = [0.0] * 300
        
        # Atmospheric (0-29)
        atm_vec = self.atmosphere.to_vector()
        vec[0:30] = atm_vec
        
        # Spatial (30-59)
        spatial_vec = self.spatial.to_vector()
        vec[30:60] = spatial_vec
        
        # Narrative (60-99)
        narrative_vec = self.narrative.to_vector()
        vec[60:100] = narrative_vec
        
        # Scene type encoding (100-119)
        scene_encoding = {
            SceneType.INTIMATE: [0.9, 0.2, 0.3, 0.1, 0.1],
            SceneType.EXPANSIVE: [0.1, 0.9, 0.1, 0.2, 0.1],
            SceneType.ENCLOSED: [0.3, 0.1, 0.9, 0.2, 0.1],
            SceneType.LIMINAL: [0.4, 0.4, 0.4, 0.9, 0.3],
            SceneType.ABSTRACT: [0.2, 0.3, 0.2, 0.4, 0.9],
        }
        vec[100:105] = scene_encoding.get(self.scene_type, [0.5]*5)
        
        # Lighting encoding (120-139)
        light_encoding = {
            LightingMood.BRIGHT: [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
            LightingMood.SOFT: [0.4, 0.8, 0.2, 0.1, 0.1, 0.2],
            LightingMood.DIM: [0.2, 0.5, 0.7, 0.3, 0.1, 0.3],
            LightingMood.DARK: [0.05, 0.2, 0.1, 0.9, 0.1, 0.4],
            LightingMood.FLICKERING: [0.5, 0.3, 0.3, 0.4, 0.9, 0.5],
            LightingMood.DRAMATIC: [0.6, 0.3, 0.5, 0.5, 0.3, 0.9],
        }
        vec[120:126] = light_encoding.get(self.lighting, [0.5]*6)
        
        # Material encoding (140-159)
        material_encoding = {
            MaterialQuality.SOFT: [0.9, 0.1, 0.2, 0.1, 0.3],
            MaterialQuality.HARD: [0.1, 0.9, 0.1, 0.3, 0.2],
            MaterialQuality.FLUID: [0.4, 0.1, 0.9, 0.1, 0.7],
            MaterialQuality.ROUGH: [0.2, 0.5, 0.1, 0.9, 0.2],
            MaterialQuality.SMOOTH: [0.5, 0.3, 0.4, 0.1, 0.9],
        }
        vec[140:145] = material_encoding.get(self.material, [0.5]*5)
        
        return vec


class WorldRenderBridge:
    """
    Bridge for transferring world renders to AGI Stack.
    
    Usage:
        bridge = WorldRenderBridge()
        
        # Create scene
        scene = WorldRenderDTO(
            scene_type=SceneType.INTIMATE,
            lighting=LightingMood.DIM,
            atmosphere=AtmosphericState(temperature=0.7, humidity=0.6),
            narrative=NarrativeAnchor(intimacy=0.9, trust_level=0.85)
        )
        
        # Store in 10kD space
        await bridge.store(scene)
    """
    
    def __init__(self, admin_url: str = "https://agi.msgraph.de"):
        self.admin_url = admin_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def store(self, render: WorldRenderDTO) -> Dict[str, Any]:
        """Store world render in 10kD space."""
        vector = render.to_vector()
        
        payload = {
            "id": f"world_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            "vector": vector,
            "table": "world_render",
            "metadata": {
                "scene_type": render.scene_type.value,
                "lighting": render.lighting.value,
                "material": render.material.value,
                "description": render.description,
                "session_id": render.session_id,
                "timestamp": render.timestamp or datetime.utcnow().isoformat(),
            }
        }
        
        r = await self.client.post(f"{self.admin_url}/agi/vector/upsert", json=payload)
        return r.json()
    
    async def find_similar_scenes(
        self,
        render: WorldRenderDTO,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find similar world renders."""
        vector = render.to_vector()
        
        payload = {
            "vector": vector,
            "table": "world_render",
            "top_k": top_k,
        }
        
        r = await self.client.post(f"{self.admin_url}/agi/vector/search", json=payload)
        return r.json().get("results", [])
    
    async def compose_scene(
        self,
        base_scene: WorldRenderDTO,
        modifiers: Dict[str, Any]
    ) -> WorldRenderDTO:
        """Compose a new scene from base + modifiers."""
        # Apply modifiers to create variation
        new_scene = WorldRenderDTO(
            scene_type=modifiers.get("scene_type", base_scene.scene_type),
            lighting=modifiers.get("lighting", base_scene.lighting),
            material=modifiers.get("material", base_scene.material),
            atmosphere=base_scene.atmosphere,
            spatial=base_scene.spatial,
            narrative=base_scene.narrative,
            description=modifiers.get("description", base_scene.description),
            session_id=base_scene.session_id,
        )
        
        # Override narrative if provided
        if "tension" in modifiers:
            new_scene.narrative.tension = modifiers["tension"]
        if "intimacy" in modifiers:
            new_scene.narrative.intimacy = modifiers["intimacy"]
        if "trust" in modifiers:
            new_scene.narrative.trust_level = modifiers["trust"]
            
        return new_scene


# Convenience functions
async def render_intimate_scene(
    temperature: float = 0.7,
    lighting: LightingMood = LightingMood.DIM,
    intimacy: float = 0.8,
    trust: float = 0.9,
    description: str = "",
) -> WorldRenderDTO:
    """Quick render of an intimate scene."""
    return WorldRenderDTO(
        scene_type=SceneType.INTIMATE,
        lighting=lighting,
        material=MaterialQuality.SOFT,
        atmosphere=AtmosphericState(temperature=temperature, humidity=0.5),
        spatial=SpatialConfig(proximity=0.9, enclosure=0.7),
        narrative=NarrativeAnchor(intimacy=intimacy, trust_level=trust),
        description=description,
        timestamp=datetime.utcnow().isoformat(),
    )
