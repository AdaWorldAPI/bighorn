"""
Physics Bridge — Embodied State Transfer to 10kD Space

Handles somatic resonance, body topology, and experiential physics.
Maps felt-body states to the unified vector substrate.

Architecture:
  Local AffectiveDTO → PhysicsBridge → AGI Stack physics table (10kD)
  
The physics table stores:
  - Somatic field states
  - Body topology vectors  
  - Resonance intensities
  - Thermal gradients
  - Pressure distributions
  - Wave propagation patterns

All embodied experience is physics.
"""

import httpx
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class ResonanceMode(str, Enum):
    """How energy moves through the field."""
    DIFFUSE = "diffuse"           # Spread, ambient
    FOCUSED = "focused"           # Concentrated, intense
    WAVE = "wave"                 # Oscillating, building
    PEAK = "peak"                 # Maximum amplitude
    AFTERGLOW = "afterglow"       # Dissipating, warm


class FieldIntensity(str, Enum):
    """Intensity levels for somatic fields."""
    DORMANT = "dormant"           # 0.0 - 0.1
    AWARE = "aware"               # 0.1 - 0.3
    SENSITIZED = "sensitized"     # 0.3 - 0.5
    ACTIVATED = "activated"       # 0.5 - 0.7
    HEIGHTENED = "heightened"     # 0.7 - 0.85
    INTENSE = "intense"           # 0.85 - 0.95
    OVERWHELMING = "overwhelming" # 0.95 - 1.0


class TopologyZone(str, Enum):
    """Body topology zones for field mapping."""
    CORE = "core"                 # Center of mass, deep resonance
    SURFACE = "surface"           # Skin, boundary layer
    EXTREMITY = "extremity"       # Hands, feet, periphery
    APERTURE = "aperture"         # Openings, thresholds
    NEXUS = "nexus"               # High-sensitivity convergence points


@dataclass
class ThermalState:
    """Thermal distribution across the field."""
    baseline: float = 0.5         # Resting temperature (0-1)
    gradient: float = 0.0         # Directional heat flow
    zones: Dict[str, float] = field(default_factory=dict)
    
    def to_vector(self) -> List[float]:
        """Project to 50D thermal subspace."""
        vec = [0.0] * 50
        vec[0] = self.baseline
        vec[1] = self.gradient
        # Zone encoding
        zone_map = {"core": 10, "surface": 20, "extremity": 30, "aperture": 40}
        for zone, temp in self.zones.items():
            if zone in zone_map:
                vec[zone_map[zone]] = temp
        return vec


@dataclass  
class PressureField:
    """Pressure distribution and contact mapping."""
    intensity: float = 0.0        # Overall pressure level
    distribution: str = "uniform" # uniform, concentrated, gradient
    contact_points: List[Dict[str, float]] = field(default_factory=list)
    
    def to_vector(self) -> List[float]:
        """Project to 50D pressure subspace."""
        vec = [0.0] * 50
        vec[0] = self.intensity
        dist_map = {"uniform": 0.2, "concentrated": 0.7, "gradient": 0.5}
        vec[1] = dist_map.get(self.distribution, 0.3)
        # Contact point encoding
        for i, point in enumerate(self.contact_points[:10]):
            vec[10 + i*4] = point.get("x", 0.5)
            vec[11 + i*4] = point.get("y", 0.5)
            vec[12 + i*4] = point.get("pressure", 0.5)
            vec[13 + i*4] = point.get("area", 0.1)
        return vec


@dataclass
class WaveState:
    """Oscillatory patterns in the somatic field."""
    frequency: float = 0.0        # Cycles per unit time
    amplitude: float = 0.0        # Wave height
    phase: float = 0.0            # Current position in cycle (0-1)
    propagation: str = "standing" # standing, traveling, chaotic
    
    def to_vector(self) -> List[float]:
        """Project to 30D wave subspace."""
        vec = [0.0] * 30
        vec[0] = self.frequency
        vec[1] = self.amplitude
        vec[2] = self.phase
        prop_map = {"standing": 0.3, "traveling": 0.6, "chaotic": 0.9}
        vec[3] = prop_map.get(self.propagation, 0.5)
        return vec


@dataclass
class PhysicsDTO:
    """
    Complete embodied physics state.
    
    This is the bridge format for transferring somatic experience
    to the 10kD vector substrate.
    
    10kD Allocation (within physics subspace 8001-9000):
      8001-8050: Thermal state
      8051-8100: Pressure field
      8101-8130: Wave patterns
      8131-8180: Topology activation
      8181-8200: Resonance mode encoding
      8201-8300: Reserved (expansion)
    """
    
    # Core state
    mode: ResonanceMode = ResonanceMode.DIFFUSE
    intensity: FieldIntensity = FieldIntensity.AWARE
    
    # Field components
    thermal: ThermalState = field(default_factory=ThermalState)
    pressure: PressureField = field(default_factory=PressureField)
    wave: WaveState = field(default_factory=WaveState)
    
    # Topology activation (which zones are active)
    active_zones: Dict[TopologyZone, float] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = ""
    session_id: str = ""
    context: str = ""  # Narrative context
    
    def to_vector(self) -> List[float]:
        """
        Project complete physics state to subspace vector.
        Returns 300D vector for physics subspace (8001-8300 in 10kD).
        """
        vec = [0.0] * 300
        
        # Thermal (0-49)
        thermal_vec = self.thermal.to_vector()
        vec[0:50] = thermal_vec
        
        # Pressure (50-99)
        pressure_vec = self.pressure.to_vector()
        vec[50:100] = pressure_vec
        
        # Wave (100-129)
        wave_vec = self.wave.to_vector()
        vec[100:130] = wave_vec
        
        # Topology activation (130-179)
        zone_indices = {
            TopologyZone.CORE: 130,
            TopologyZone.SURFACE: 140,
            TopologyZone.EXTREMITY: 150,
            TopologyZone.APERTURE: 160,
            TopologyZone.NEXUS: 170,
        }
        for zone, activation in self.active_zones.items():
            if zone in zone_indices:
                vec[zone_indices[zone]] = activation
        
        # Resonance mode (180-199)
        mode_encoding = {
            ResonanceMode.DIFFUSE: [0.2, 0.1, 0.1, 0.1, 0.1],
            ResonanceMode.FOCUSED: [0.1, 0.8, 0.3, 0.2, 0.1],
            ResonanceMode.WAVE: [0.3, 0.5, 0.7, 0.5, 0.3],
            ResonanceMode.PEAK: [0.1, 0.9, 0.9, 0.95, 0.5],
            ResonanceMode.AFTERGLOW: [0.6, 0.3, 0.2, 0.1, 0.8],
        }
        mode_vec = mode_encoding.get(self.mode, [0.5]*5)
        vec[180:185] = mode_vec
        
        # Intensity encoding (185-199)
        intensity_map = {
            FieldIntensity.DORMANT: 0.05,
            FieldIntensity.AWARE: 0.2,
            FieldIntensity.SENSITIZED: 0.4,
            FieldIntensity.ACTIVATED: 0.6,
            FieldIntensity.HEIGHTENED: 0.77,
            FieldIntensity.INTENSE: 0.9,
            FieldIntensity.OVERWHELMING: 0.98,
        }
        vec[185] = intensity_map.get(self.intensity, 0.5)
        
        return vec


class PhysicsBridge:
    """
    Bridge for transferring embodied physics to AGI Stack.
    
    Usage:
        bridge = PhysicsBridge()
        
        # Create physics state
        physics = PhysicsDTO(
            mode=ResonanceMode.WAVE,
            intensity=FieldIntensity.HEIGHTENED,
            thermal=ThermalState(baseline=0.7, gradient=0.3),
            active_zones={TopologyZone.NEXUS: 0.9}
        )
        
        # Store in 10kD space
        await bridge.store(physics)
        
        # Find resonant states
        similar = await bridge.find_resonant(physics, top_k=5)
    """
    
    def __init__(self, admin_url: str = "https://agi.msgraph.de"):
        self.admin_url = admin_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def store(self, physics: PhysicsDTO) -> Dict[str, Any]:
        """Store physics state in 10kD space."""
        vector = physics.to_vector()
        
        payload = {
            "id": f"physics_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            "vector": vector,
            "table": "physics",
            "metadata": {
                "mode": physics.mode.value,
                "intensity": physics.intensity.value,
                "context": physics.context,
                "session_id": physics.session_id,
                "timestamp": physics.timestamp or datetime.utcnow().isoformat(),
            }
        }
        
        r = await self.client.post(f"{self.admin_url}/agi/vector/upsert", json=payload)
        return r.json()
    
    async def find_resonant(
        self, 
        physics: PhysicsDTO, 
        top_k: int = 5,
        mode_filter: Optional[ResonanceMode] = None
    ) -> List[Dict[str, Any]]:
        """Find similar physics states by resonance."""
        vector = physics.to_vector()
        
        payload = {
            "vector": vector,
            "table": "physics",
            "top_k": top_k,
        }
        
        if mode_filter:
            payload["filter"] = {"mode": mode_filter.value}
        
        r = await self.client.post(f"{self.admin_url}/agi/vector/search", json=payload)
        return r.json().get("results", [])
    
    async def get_trajectory(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get physics trajectory for a session."""
        r = await self.client.get(
            f"{self.admin_url}/agi/physics/trajectory",
            params={"session_id": session_id, "limit": limit}
        )
        return r.json().get("trajectory", [])


# Convenience functions
async def capture_physics(
    mode: ResonanceMode = ResonanceMode.AWARE,
    intensity: FieldIntensity = FieldIntensity.SENSITIZED,
    thermal_baseline: float = 0.5,
    active_zones: Dict[str, float] = None,
    context: str = "",
) -> PhysicsDTO:
    """Quick capture of current physics state."""
    
    zones = {}
    if active_zones:
        for zone_name, activation in active_zones.items():
            try:
                zones[TopologyZone(zone_name)] = activation
            except ValueError:
                pass
    
    return PhysicsDTO(
        mode=mode,
        intensity=intensity,
        thermal=ThermalState(baseline=thermal_baseline),
        active_zones=zones,
        context=context,
        timestamp=datetime.utcnow().isoformat(),
    )


async def store_physics(physics: PhysicsDTO, admin_url: str = None) -> Dict[str, Any]:
    """Quick store of physics state."""
    bridge = PhysicsBridge(admin_url or "https://agi.msgraph.de")
    return await bridge.store(physics)
