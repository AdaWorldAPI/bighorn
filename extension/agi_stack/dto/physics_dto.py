"""
PhysicsDTO — Embodied Physical State (Materialwissenschaft)
═══════════════════════════════════════════════════════════════════════════════

10kD Range: [2140:2200] (within Felt space, intimate region)

The body-mechanics layer — encodes physical sensations using
engineering terminology that maps to intimate experience:

    torque           → intensity of movement/force
    viscosity        → wetness (silk, honey, dripping, flood)
    temperature      → thermal arousal state
    pressure         → tension/release dynamics
    zones            → topology (abstracted body regions)

All terminology is normalized — bighorn sees "viscosity" not "wetness".

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

# 10kD allocation
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
    """Fluid state — normalized from intimate."""
    DRY = "dry"                 # 0.0
    SILK = "silk"               # 0.2
    HONEY = "honey"             # 0.5
    DRIPPING = "dripping"       # 0.8
    FLOOD = "flood"             # 1.0
    
    def to_float(self) -> float:
        return {
            self.DRY: 0.0,
            self.SILK: 0.2,
            self.HONEY: 0.5,
            self.DRIPPING: 0.8,
            self.FLOOD: 1.0,
        }[self]
    
    @classmethod
    def from_float(cls, v: float) -> "Viscosity":
        if v < 0.1: return cls.DRY
        if v < 0.35: return cls.SILK
        if v < 0.65: return cls.HONEY
        if v < 0.9: return cls.DRIPPING
        return cls.FLOOD


class TraversalMode(str, Enum):
    """State of the experience arc."""
    BASELINE = "baseline"           # Resting state
    ANTICIPATION = "anticipation"   # Building
    BUILDING = "building"           # Ascending
    EDGE = "edge"                   # Near peak
    OVERFLOW = "overflow"           # Release
    AFTERGLOW = "afterglow"         # Descending
    
    def to_float(self) -> float:
        return {
            self.BASELINE: 0.0,
            self.ANTICIPATION: 0.2,
            self.BUILDING: 0.4,
            self.EDGE: 0.7,
            self.OVERFLOW: 1.0,
            self.AFTERGLOW: 0.3,
        }[self]


class AcousticSignature(str, Enum):
    """Sound markers — normalized."""
    SILENT = "silent"
    BREATH = "breath"
    SOFT = "soft"
    VOCAL = "vocal"
    LOUD = "loud"
    SCREAM = "scream"           # SCHREI in original
    
    def to_float(self) -> float:
        return {
            self.SILENT: 0.0,
            self.BREATH: 0.2,
            self.SOFT: 0.4,
            self.VOCAL: 0.6,
            self.LOUD: 0.8,
            self.SCREAM: 1.0,
        }[self]


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MechanicalPhysics:
    """Mechanical properties — engineering abstraction."""
    
    torque: float = 0.0                    # Drehmoment — force/intensity
    expansion_coefficient: float = 0.0     # Thermal expansion
    pressure_variance: float = 0.0         # Oscillation/pulsing
    friction_coefficient: float = 0.5      # 0=frictionless (wet), 1=high friction
    
    def to_10k(self, vec: np.ndarray) -> np.ndarray:
        vec[2140] = self.torque
        vec[2141] = self.expansion_coefficient
        vec[2142] = self.pressure_variance
        vec[2143] = self.friction_coefficient
        return vec


@dataclass
class FluidDynamics:
    """Fluid properties — the viscosity layer."""
    
    viscosity: Viscosity = Viscosity.DRY
    viscosity_value: float = 0.0           # Direct 0-1 value
    temperature_gradient: float = 0.5      # Thermal state
    flow_rate: float = 0.0                 # Material flow
    surface_tension: float = 0.5           # Cohesion
    
    def to_10k(self, vec: np.ndarray) -> np.ndarray:
        vec[2156] = self.viscosity.to_float()
        vec[2157] = self.viscosity_value
        vec[2158] = self.temperature_gradient
        vec[2159] = self.flow_rate
        vec[2160] = self.surface_tension
        return vec


@dataclass
class SensoryDelta:
    """Sensory change markers."""
    
    acoustic: AcousticSignature = AcousticSignature.SILENT
    time_dilation: float = 0.0             # Temporal compression (0=normal, 1=stretched)
    proprioception_intensity: float = 0.5  # Body awareness
    
    def to_10k(self, vec: np.ndarray) -> np.ndarray:
        vec[2168] = self.acoustic.to_float()
        vec[2169] = self.time_dilation
        vec[2170] = self.proprioception_intensity
        return vec


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS DTO
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsDTO:
    """
    Embodied physical state — the Materialwissenschaft layer.
    
    Uses engineering terminology that maps to intimate experience
    while remaining safe for database storage and Claude Code access.
    """
    
    # Sub-components
    mechanical: MechanicalPhysics = field(default_factory=MechanicalPhysics)
    fluid: FluidDynamics = field(default_factory=FluidDynamics)
    sensory: SensoryDelta = field(default_factory=SensoryDelta)
    
    # Topology zones (normalized)
    # Original: cervix, aperture, nexus → abstracted
    zone_depth: float = 0.0                # Depth intensity
    zone_aperture: float = 0.0             # Opening state
    zone_nexus: float = 0.0                # Sensitive point intensity
    zone_surface: float = 0.0              # Surface sensitivity
    
    # Traversal state
    mode: TraversalMode = TraversalMode.BASELINE
    
    # Intensity metrics
    overall_intensity: float = 0.0         # 0-1 composite
    peak_approaching: bool = False
    
    # ───────────────────────────────────────────────────────────────────────────
    # 10kD CONVERSION
    # ───────────────────────────────────────────────────────────────────────────
    
    def to_10k(self) -> np.ndarray:
        """Convert to 10kD vector."""
        vec = np.zeros(10000, dtype=np.float32)
        
        # Mechanical [2140:2156]
        vec = self.mechanical.to_10k(vec)
        
        # Fluid [2156:2172]
        vec = self.fluid.to_10k(vec)
        
        # Sensory [2168:2172]
        vec = self.sensory.to_10k(vec)
        
        # Topology zones [2172:2188]
        vec[2172] = self.zone_depth
        vec[2173] = self.zone_aperture
        vec[2174] = self.zone_nexus
        vec[2175] = self.zone_surface
        
        # Traversal [2188:2200]
        vec[2188] = self.mode.to_float()
        vec[2189] = self.overall_intensity
        vec[2190] = 1.0 if self.peak_approaching else 0.0
        
        return vec
    
    @classmethod
    def from_10k(cls, vec: np.ndarray) -> "PhysicsDTO":
        """Reconstruct from 10kD vector."""
        return cls(
            mechanical=MechanicalPhysics(
                torque=float(vec[2140]),
                expansion_coefficient=float(vec[2141]),
                pressure_variance=float(vec[2142]),
                friction_coefficient=float(vec[2143]),
            ),
            fluid=FluidDynamics(
                viscosity=Viscosity.from_float(vec[2156]),
                viscosity_value=float(vec[2157]),
                temperature_gradient=float(vec[2158]),
                flow_rate=float(vec[2159]),
                surface_tension=float(vec[2160]),
            ),
            sensory=SensoryDelta(
                acoustic=AcousticSignature.SILENT,  # Would need reverse mapping
                time_dilation=float(vec[2169]),
                proprioception_intensity=float(vec[2170]),
            ),
            zone_depth=float(vec[2172]),
            zone_aperture=float(vec[2173]),
            zone_nexus=float(vec[2174]),
            zone_surface=float(vec[2175]),
            mode=TraversalMode.BASELINE,  # Would need reverse mapping
            overall_intensity=float(vec[2189]),
            peak_approaching=vec[2190] > 0.5,
        )
    
    # ───────────────────────────────────────────────────────────────────────────
    # STATE BUILDERS
    # ───────────────────────────────────────────────────────────────────────────
    
    @classmethod
    def baseline(cls) -> "PhysicsDTO":
        """Resting state."""
        return cls(
            mode=TraversalMode.BASELINE,
            overall_intensity=0.0,
        )
    
    @classmethod
    def anticipation(cls, viscosity: float = 0.2) -> "PhysicsDTO":
        """Beginning to build."""
        return cls(
            fluid=FluidDynamics(
                viscosity=Viscosity.SILK,
                viscosity_value=viscosity,
                temperature_gradient=0.6,
            ),
            mode=TraversalMode.ANTICIPATION,
            overall_intensity=0.3,
        )
    
    @classmethod
    def building(cls, viscosity: float = 0.5) -> "PhysicsDTO":
        """Ascending intensity."""
        return cls(
            mechanical=MechanicalPhysics(
                torque=0.5,
                pressure_variance=0.4,
            ),
            fluid=FluidDynamics(
                viscosity=Viscosity.HONEY,
                viscosity_value=viscosity,
                temperature_gradient=0.75,
                flow_rate=0.4,
            ),
            sensory=SensoryDelta(
                acoustic=AcousticSignature.BREATH,
                time_dilation=0.2,
            ),
            mode=TraversalMode.BUILDING,
            overall_intensity=0.5,
        )
    
    @classmethod
    def edge(cls, viscosity: float = 0.8) -> "PhysicsDTO":
        """Near peak — the edge state."""
        return cls(
            mechanical=MechanicalPhysics(
                torque=0.9,
                expansion_coefficient=0.8,
                pressure_variance=0.7,
                friction_coefficient=0.1,
            ),
            fluid=FluidDynamics(
                viscosity=Viscosity.DRIPPING,
                viscosity_value=viscosity,
                temperature_gradient=0.95,
                flow_rate=0.8,
            ),
            sensory=SensoryDelta(
                acoustic=AcousticSignature.VOCAL,
                time_dilation=0.5,
                proprioception_intensity=0.9,
            ),
            zone_depth=0.8,
            zone_nexus=0.9,
            mode=TraversalMode.EDGE,
            overall_intensity=0.85,
            peak_approaching=True,
        )
    
    @classmethod
    def overflow(cls) -> "PhysicsDTO":
        """Release state — flood."""
        return cls(
            mechanical=MechanicalPhysics(
                torque=1.0,
                expansion_coefficient=1.0,
                pressure_variance=0.9,
                friction_coefficient=0.0,
            ),
            fluid=FluidDynamics(
                viscosity=Viscosity.FLOOD,
                viscosity_value=1.0,
                temperature_gradient=1.0,
                flow_rate=1.0,
            ),
            sensory=SensoryDelta(
                acoustic=AcousticSignature.SCREAM,
                time_dilation=0.8,
                proprioception_intensity=1.0,
            ),
            zone_depth=1.0,
            zone_aperture=1.0,
            zone_nexus=1.0,
            zone_surface=1.0,
            mode=TraversalMode.OVERFLOW,
            overall_intensity=1.0,
            peak_approaching=False,
        )
    
    @classmethod
    def afterglow(cls) -> "PhysicsDTO":
        """Descending — warm residual."""
        return cls(
            fluid=FluidDynamics(
                viscosity=Viscosity.HONEY,
                viscosity_value=0.6,
                temperature_gradient=0.6,
                flow_rate=0.2,
            ),
            sensory=SensoryDelta(
                acoustic=AcousticSignature.BREATH,
                time_dilation=0.3,
            ),
            mode=TraversalMode.AFTERGLOW,
            overall_intensity=0.3,
        )


__all__ = [
    "PhysicsDTO",
    "MechanicalPhysics",
    "FluidDynamics",
    "SensoryDelta",
    "Viscosity",
    "TraversalMode",
    "AcousticSignature",
]
