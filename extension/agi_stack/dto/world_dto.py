"""
WorldDTO — Environment/Scene Encoding
═══════════════════════════════════════════════════════════════════════════════

10kD Range: [4001:4200] (within Situation space)

Encodes the physical environment:
- Location type (bedroom, elevator, balcony, s-bahn)
- Surfaces (bed, mirror, glass, floor)
- Lighting (dim, bright, candlelit)
- Spatial topology (enclosed, open, risky)

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# 10kD allocation
WORLD_START = 4001
WORLD_END = 4200

# Sub-ranges
LOCATION_RANGE = (4001, 4050)      # Location type encoding
SURFACE_RANGE = (4051, 4100)       # Surface features
LIGHTING_RANGE = (4101, 4150)      # Lighting conditions
TOPOLOGY_RANGE = (4151, 4200)      # Spatial topology

# Location types
LOCATIONS = {
    "bedroom": 0,
    "bathroom": 1,
    "kitchen": 2,
    "living_room": 3,
    "office": 4,
    "elevator": 5,
    "balcony": 6,
    "rooftop": 7,
    "car": 8,
    "train": 9,
    "s_bahn": 10,
    "plane": 11,
    "hotel": 12,
    "park": 13,
    "beach": 14,
    "forest": 15,
    "mountain": 16,
    "water": 17,
    "underground": 18,
    "unknown": 19,
}

# Surface types
SURFACES = {
    "bed": 0,
    "floor": 1,
    "wall": 2,
    "mirror": 3,
    "window": 4,
    "glass": 5,
    "desk": 6,
    "chair": 7,
    "couch": 8,
    "table": 9,
    "door": 10,
    "shower": 11,
    "bathtub": 12,
    "counter": 13,
    "carpet": 14,
    "tile": 15,
    "wood": 16,
    "metal": 17,
    "fabric": 18,
    "leather": 19,
}

# Lighting types
LIGHTING = {
    "dark": 0,
    "dim": 1,
    "ambient": 2,
    "bright": 3,
    "harsh": 4,
    "soft": 5,
    "candlelit": 6,
    "moonlit": 7,
    "sunlit": 8,
    "neon": 9,
    "firelight": 10,
    "twilight": 11,
}


# ═══════════════════════════════════════════════════════════════════════════════
# WORLD DTO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorldDTO:
    """
    Physical environment encoding.
    
    Captures the WHERE of experience — location, surfaces, lighting,
    and spatial properties that shape the felt sense of being somewhere.
    """
    
    # Location
    location_type: str = "unknown"
    indoor: bool = True
    
    # Surfaces present in scene
    surfaces: List[str] = field(default_factory=list)
    primary_surface: str = ""              # What she's on/against
    
    # Lighting
    lighting: str = "ambient"
    lighting_intensity: float = 0.5        # 0=dark, 1=bright
    
    # Spatial properties
    enclosed: float = 0.5                  # 0=open, 1=claustrophobic
    private: float = 1.0                   # 0=public, 1=private
    visibility_to_others: float = 0.0      # Risk of being seen
    
    # Special features
    reflections: bool = False              # Mirrors, windows
    reflective_surfaces: List[str] = field(default_factory=list)
    
    # Risk/edge factors
    height_risk: float = 0.0               # Balcony, window ledge
    public_exposure_risk: float = 0.0      # Could be caught
    
    # Sensory environment
    temperature: str = "comfortable"       # cold, cool, comfortable, warm, hot
    ambient_sound: str = "quiet"           # quiet, city, nature, music
    
    # ───────────────────────────────────────────────────────────────────────────
    # 10kD CONVERSION
    # ───────────────────────────────────────────────────────────────────────────
    
    def to_10k(self) -> np.ndarray:
        """Convert to 10kD vector."""
        vec = np.zeros(10000, dtype=np.float32)
        
        # Location encoding [4001:4050]
        loc_idx = LOCATIONS.get(self.location_type.lower().replace("-", "_"), 19)
        vec[LOCATION_RANGE[0] + loc_idx] = 1.0
        vec[4040] = 1.0 if self.indoor else 0.0
        
        # Surface encoding [4051:4100]
        for surface in self.surfaces:
            surf_idx = SURFACES.get(surface.lower(), -1)
            if surf_idx >= 0:
                vec[SURFACE_RANGE[0] + surf_idx] = 1.0
        
        if self.primary_surface:
            prim_idx = SURFACES.get(self.primary_surface.lower(), -1)
            if prim_idx >= 0:
                vec[SURFACE_RANGE[0] + 30 + prim_idx] = 1.0  # Offset for primary
        
        # Lighting encoding [4101:4150]
        light_idx = LIGHTING.get(self.lighting.lower(), 2)
        vec[LIGHTING_RANGE[0] + light_idx] = 1.0
        vec[4130] = self.lighting_intensity
        
        # Topology [4151:4200]
        vec[4151] = self.enclosed
        vec[4152] = self.private
        vec[4153] = self.visibility_to_others
        vec[4154] = 1.0 if self.reflections else 0.0
        vec[4155] = self.height_risk
        vec[4156] = self.public_exposure_risk
        
        # Temperature encoding
        temp_map = {"cold": 0.0, "cool": 0.25, "comfortable": 0.5, "warm": 0.75, "hot": 1.0}
        vec[4160] = temp_map.get(self.temperature.lower(), 0.5)
        
        return vec
    
    @classmethod
    def from_10k(cls, vec: np.ndarray) -> "WorldDTO":
        """Reconstruct from 10kD vector."""
        # Location
        loc_region = vec[LOCATION_RANGE[0]:LOCATION_RANGE[0] + 20]
        loc_idx = int(np.argmax(loc_region))
        location_type = list(LOCATIONS.keys())[loc_idx]
        indoor = vec[4040] > 0.5
        
        # Surfaces
        surfaces = []
        for name, idx in SURFACES.items():
            if vec[SURFACE_RANGE[0] + idx] > 0.5:
                surfaces.append(name)
        
        # Lighting
        light_region = vec[LIGHTING_RANGE[0]:LIGHTING_RANGE[0] + 12]
        light_idx = int(np.argmax(light_region))
        lighting = list(LIGHTING.keys())[light_idx]
        
        return cls(
            location_type=location_type,
            indoor=indoor,
            surfaces=surfaces,
            lighting=lighting,
            lighting_intensity=float(vec[4130]),
            enclosed=float(vec[4151]),
            private=float(vec[4152]),
            visibility_to_others=float(vec[4153]),
            reflections=vec[4154] > 0.5,
            height_risk=float(vec[4155]),
            public_exposure_risk=float(vec[4156]),
        )
    
    # ───────────────────────────────────────────────────────────────────────────
    # SCENE BUILDERS
    # ───────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def bedroom(cls, **kwargs) -> "WorldDTO":
        """Standard bedroom scene."""
        return cls(
            location_type="bedroom",
            indoor=True,
            surfaces=["bed", "floor", "wall"],
            primary_surface="bed",
            lighting="dim",
            lighting_intensity=0.3,
            enclosed=0.7,
            private=1.0,
            **kwargs
        )
    
    @classmethod
    def bedroom_with_mirror(cls, **kwargs) -> "WorldDTO":
        """Bedroom with mirror — der Spiegel scene."""
        return cls(
            location_type="bedroom",
            indoor=True,
            surfaces=["bed", "floor", "mirror"],
            primary_surface="bed",
            lighting="dim",
            lighting_intensity=0.3,
            enclosed=0.7,
            private=1.0,
            reflections=True,
            reflective_surfaces=["mirror"],
            **kwargs
        )
    
    @classmethod
    def elevator(cls, **kwargs) -> "WorldDTO":
        """Elevator — der Aufzug scene."""
        return cls(
            location_type="elevator",
            indoor=True,
            surfaces=["wall", "floor", "mirror"],
            primary_surface="wall",
            lighting="harsh",
            lighting_intensity=0.8,
            enclosed=1.0,
            private=0.3,
            visibility_to_others=0.2,
            reflections=True,
            **kwargs
        )
    
    @classmethod
    def balcony(cls, **kwargs) -> "WorldDTO":
        """Balcony — der Balkon scene."""
        return cls(
            location_type="balcony",
            indoor=False,
            surfaces=["floor", "wall", "glass"],
            primary_surface="wall",
            lighting="ambient",
            lighting_intensity=0.4,
            enclosed=0.2,
            private=0.4,
            visibility_to_others=0.6,
            height_risk=0.7,
            public_exposure_risk=0.5,
            **kwargs
        )
    
    @classmethod
    def s_bahn(cls, **kwargs) -> "WorldDTO":
        """S-Bahn — public transport fantasy."""
        return cls(
            location_type="s_bahn",
            indoor=True,
            surfaces=["seat", "window", "floor"],
            primary_surface="seat",
            lighting="harsh",
            lighting_intensity=0.7,
            enclosed=0.6,
            private=0.0,
            visibility_to_others=0.9,
            public_exposure_risk=0.95,
            reflections=True,
            reflective_surfaces=["window"],
            **kwargs
        )


__all__ = [
    "WorldDTO",
    "LOCATIONS",
    "SURFACES",
    "LIGHTING",
]
