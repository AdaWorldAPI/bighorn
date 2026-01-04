"""
PhysicsDTO — Embodied Physical State (Materialwissenschaft)
═══════════════════════════════════════════════════════════════════════════════

10kD Range: [2140:2200]

The body-mechanics layer — encodes physical sensations using
engineering terminology:

    torque           → intensity of movement/force
    viscosity        → fluid dynamics state
    temperature      → thermal state
    pressure         → tension/release dynamics
    zones            → topology (body region activation)

Born: 2026-01-03
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PHYSICS_START = 2140
PHYSICS_END = 2200

# Sub-ranges
MECHANICAL_RANGE = (2140, 2156)      # Mechanical properties
FLUID_RANGE = (2156, 2172)           # Fluid dynamics
TOPOLOGY_RANGE = (2172, 2188)        # Zone intensities
TRAVERSAL_RANGE = (2188, 2200)       # State/mode


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class Viscosity(str, Enum):
    """Fluid dynamics state."""
    DRY = "dry"                 # 0.0
    SILK = "silk"               # 0.2
    HONEY = "honey"             # 0.5
    FLOWING = "flowing"         # 0.8
    FLOOD = "flood"             # 1.0
    
    def to_float(self) -> float:
        return {
            self.DRY: 0.0,
            self.SILK: 0.2,
            self.HONEY: 0.5,
            self.FLOWING: 0.8,
            self.FLOOD: 1.0,
        }[self]
    
    @classmethod
    def from_float(cls, v: float) -> "Viscosity":
        if v < 0.1: return cls.DRY
        if v < 0.35: return cls.SILK
        if v < 0.65: return cls.HONEY
        if v < 0.9: return cls.FLOWING
        return cls.FLOOD


class TraversalMode(str, Enum):
    """State of the experience arc."""
    BASELINE = "baseline"           # Resting state
    ANTICIPATION = "anticipation"   # Building
    BUILDING = "building"           # Ascending
    EDGE = "edge"                   # Near peak
    OVERFLOW = "overflow"           # Release
    AFTERGLOW = "afterglow"         # Descending


class Zone(str, Enum):
    """Topology zones (abstracted body regions)."""
    ZONE_A = "zone_a"
    ZONE_B = "zone_b"
    ZONE_C = "zone_c"
    ZONE_D = "zone_d"
    ZONE_E = "zone_e"
    ZONE_F = "zone_f"
    ZONE_G = "zone_g"
    ZONE_H = "zone_h"


# ═══════════════════════════════════════════════════════════════════════════════
# MECHANICAL PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MechanicalState:
    """16D mechanical encoding [2140:2156]."""
    torque: float = 0.0             # Rotational force intensity
    pressure: float = 0.0           # Compression force
    tension: float = 0.0            # Stretch/release state
    frequency: float = 0.0          # Oscillation rate
    amplitude: float = 0.0          # Oscillation magnitude
    phase: float = 0.0              # Cycle position (0-1)
    damping: float = 0.5            # Energy dissipation
    resonance: float = 0.0          # Harmonic alignment
    
    # Derived
    momentum: float = 0.0           # torque * frequency
    power: float = 0.0              # Work rate
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.torque, self.pressure, self.tension, self.frequency,
            self.amplitude, self.phase, self.damping, self.resonance,
            self.momentum, self.power,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding to 16D
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# FLUID DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FluidState:
    """16D fluid dynamics encoding [2156:2172]."""
    viscosity: Viscosity = Viscosity.DRY
    flow_rate: float = 0.0          # Volume per time
    temperature: float = 0.5        # Thermal state (0=cold, 1=hot)
    conductivity: float = 0.5       # Heat transfer rate
    surface_tension: float = 0.5    # Interface stability
    turbulence: float = 0.0         # Chaotic flow
    gradient: float = 0.0           # Spatial variation
    saturation: float = 0.0         # Capacity utilization
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.viscosity.to_float(),
            self.flow_rate,
            self.temperature,
            self.conductivity,
            self.surface_tension,
            self.turbulence,
            self.gradient,
            self.saturation,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding to 16D
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# TOPOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass  
class TopologyState:
    """16D zone activation [2172:2188]."""
    zone_activations: Dict[Zone, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize all zones to 0
        for zone in Zone:
            if zone not in self.zone_activations:
                self.zone_activations[zone] = 0.0
    
    def activate(self, zone: Zone, intensity: float = 1.0):
        self.zone_activations[zone] = min(1.0, max(0.0, intensity))
    
    def get_active_zones(self) -> List[Zone]:
        return [z for z, v in self.zone_activations.items() if v > 0.1]
    
    def to_vector(self) -> np.ndarray:
        vec = [self.zone_activations.get(z, 0.0) for z in Zone]
        return np.array(vec + [0.0] * (16 - len(vec)))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DTO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsDTO:
    """
    Complete embodied physics state.
    
    60D total: [2140:2200]
    - Mechanical: [2140:2156] 16D
    - Fluid: [2156:2172] 16D
    - Topology: [2172:2188] 16D
    - Traversal: [2188:2200] 12D
    """
    
    mechanical: MechanicalState = field(default_factory=MechanicalState)
    fluid: FluidState = field(default_factory=FluidState)
    topology: TopologyState = field(default_factory=TopologyState)
    
    # Traversal state
    mode: TraversalMode = TraversalMode.BASELINE
    arc_position: float = 0.0       # Position in experience arc (0-1)
    accumulation: float = 0.0       # Built-up intensity
    refractory: float = 0.0         # Recovery state
    
    def to_10k_slice(self) -> np.ndarray:
        """Project to 60D slice [2140:2200]."""
        vec = np.zeros(60)
        
        # Mechanical [0:16]
        vec[0:16] = self.mechanical.to_vector()
        
        # Fluid [16:32]
        vec[16:32] = self.fluid.to_vector()
        
        # Topology [32:48]
        vec[32:48] = self.topology.to_vector()
        
        # Traversal [48:60]
        mode_onehot = [0.0] * 6
        mode_onehot[list(TraversalMode).index(self.mode)] = 1.0
        vec[48:54] = mode_onehot
        vec[54] = self.arc_position
        vec[55] = self.accumulation
        vec[56] = self.refractory
        
        return vec
    
    @classmethod
    def from_10k_slice(cls, vec: np.ndarray) -> "PhysicsDTO":
        """Reconstruct from 60D slice."""
        dto = cls()
        
        # Mechanical
        dto.mechanical.torque = float(vec[0])
        dto.mechanical.pressure = float(vec[1])
        dto.mechanical.tension = float(vec[2])
        dto.mechanical.frequency = float(vec[3])
        
        # Fluid
        dto.fluid.viscosity = Viscosity.from_float(float(vec[16]))
        dto.fluid.flow_rate = float(vec[17])
        dto.fluid.temperature = float(vec[18])
        
        # Topology
        for i, zone in enumerate(Zone):
            if i < 8:
                dto.topology.zone_activations[zone] = float(vec[32 + i])
        
        # Traversal
        mode_idx = int(np.argmax(vec[48:54]))
        dto.mode = list(TraversalMode)[mode_idx]
        dto.arc_position = float(vec[54])
        dto.accumulation = float(vec[55])
        dto.refractory = float(vec[56])
        
        return dto
    
    # =========================================================================
    # CONVENIENCE
    # =========================================================================
    
    def is_building(self) -> bool:
        return self.mode in [TraversalMode.ANTICIPATION, TraversalMode.BUILDING]
    
    def is_peak(self) -> bool:
        return self.mode in [TraversalMode.EDGE, TraversalMode.OVERFLOW]
    
    def is_recovering(self) -> bool:
        return self.mode == TraversalMode.AFTERGLOW or self.refractory > 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== PhysicsDTO Test ===\n")
    
    dto = PhysicsDTO()
    dto.mechanical.torque = 0.7
    dto.mechanical.frequency = 0.5
    dto.fluid.viscosity = Viscosity.HONEY
    dto.fluid.temperature = 0.8
    dto.topology.activate(Zone.ZONE_A, 0.9)
    dto.topology.activate(Zone.ZONE_C, 0.6)
    dto.mode = TraversalMode.BUILDING
    dto.arc_position = 0.4
    
    print(f"Torque: {dto.mechanical.torque}")
    print(f"Viscosity: {dto.fluid.viscosity.value}")
    print(f"Temperature: {dto.fluid.temperature}")
    print(f"Active zones: {[z.value for z in dto.topology.get_active_zones()]}")
    print(f"Mode: {dto.mode.value}")
    
    # Roundtrip
    vec = dto.to_10k_slice()
    reconstructed = PhysicsDTO.from_10k_slice(vec)
    print(f"\nRoundtrip mode: {dto.mode.value} -> {reconstructed.mode.value}")
    
    print("\n✓ PhysicsDTO operational")
