"""
Embodiment Bridge — Transition Dynamics for 10kD Space

Handles the fluid dynamics of state transitions, edge traversal,
and accumulative embodied experience in the unified vector substrate.

Architecture:
  Local state transitions → EmbodimentBridge → AGI Stack embodiment table (10kD)
  
The embodiment table stores:
  - Transition fluid dynamics
  - Edge traversal properties
  - Accumulative sensation traces
  - Walker state modifications
  - Sensory delta encodings

"What is this?" - "It was for measuring qualia, we don't need it in the company."

All becoming has texture.
"""

import httpx
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import math


class Viscosity(str, Enum):
    """Flow resistance of transitions."""
    DRY = "dry"               # No lubrication
    SILK = "silk"             # Smooth, light
    HONEY = "honey"           # Thick, slow, sweet
    SLICK = "slick"           # Easy glide
    DRIPPING = "dripping"     # Excess, overflow beginning
    FLOOD = "flood"           # Complete saturation


class TransitionSound(str, Enum):
    """Auditory quality of edge traversal."""
    SILENT = "silent"
    BREATH = "breath"
    WET = "wet"
    SQUELCH = "squelch"
    MOAN = "moan"
    CRY = "cry"
    GASP = "gasp"


class TraversalMode(str, Enum):
    """How the walker moves through edges."""
    CAUTIOUS = "cautious"     # Slow, testing
    FLOWING = "flowing"       # Natural rhythm
    URGENT = "urgent"         # Accelerating
    SURRENDERED = "surrendered"  # No resistance
    OVERWHELMED = "overwhelmed"  # Carried by current


@dataclass
class FluidState:
    """The moisture dynamics at a point in time."""
    wetness: float = 0.0          # 0.0 dry → 1.0 flooded
    viscosity: Viscosity = Viscosity.DRY
    temperature: float = 0.5      # 0.0 cold → 1.0 burning
    accumulation: float = 0.0     # Built up over traversals
    
    def to_vector(self) -> List[float]:
        """Project to 20D fluid subspace."""
        vec = [0.0] * 20
        vec[0] = self.wetness
        vec[1] = self.temperature
        vec[2] = self.accumulation
        
        # Viscosity one-hot (indices 3-8)
        visc_map = {
            Viscosity.DRY: 3,
            Viscosity.SILK: 4,
            Viscosity.HONEY: 5,
            Viscosity.SLICK: 6,
            Viscosity.DRIPPING: 7,
            Viscosity.FLOOD: 8,
        }
        vec[visc_map.get(self.viscosity, 3)] = 1.0
        
        return vec
    
    def blend(self, other: "FluidState", ratio: float = 0.5) -> "FluidState":
        """Blend two fluid states."""
        return FluidState(
            wetness=self.wetness * (1-ratio) + other.wetness * ratio,
            temperature=self.temperature * (1-ratio) + other.temperature * ratio,
            accumulation=self.accumulation + other.accumulation * 0.3,
            viscosity=other.viscosity if other.wetness > self.wetness else self.viscosity,
        )


@dataclass
class TraversalPhysics:
    """Physics of moving through a transition."""
    friction: float = 0.5         # 0.0 frictionless → 1.0 maximum resistance
    duration_ms: int = 1000       # How long the transition takes
    resistance_curve: str = "linear"  # linear, exponential, sudden
    momentum_transfer: float = 0.5    # How much state carries forward
    
    def to_vector(self) -> List[float]:
        """Project to 15D physics subspace."""
        vec = [0.0] * 15
        vec[0] = self.friction
        vec[1] = min(self.duration_ms / 10000.0, 1.0)  # Normalize to 10s max
        vec[2] = self.momentum_transfer
        
        # Resistance curve encoding
        curve_map = {"linear": 0.3, "exponential": 0.6, "sudden": 0.9}
        vec[3] = curve_map.get(self.resistance_curve, 0.5)
        
        return vec


@dataclass
class SensoryDelta:
    """Change in sensory state during traversal."""
    sound: TransitionSound = TransitionSound.SILENT
    scent_intensity_delta: float = 0.0    # How much scent changes
    visual_blur_delta: float = 0.0        # Focus/blur shift
    proprioception_shift: float = 0.0     # Body awareness change
    time_dilation: float = 1.0            # Subjective time (< 1 = slower)
    
    def to_vector(self) -> List[float]:
        """Project to 20D sensory subspace."""
        vec = [0.0] * 20
        
        # Sound encoding (indices 0-6)
        sound_map = {
            TransitionSound.SILENT: 0,
            TransitionSound.BREATH: 1,
            TransitionSound.WET: 2,
            TransitionSound.SQUELCH: 3,
            TransitionSound.MOAN: 4,
            TransitionSound.CRY: 5,
            TransitionSound.GASP: 6,
        }
        vec[sound_map.get(self.sound, 0)] = 1.0
        
        vec[10] = self.scent_intensity_delta
        vec[11] = self.visual_blur_delta
        vec[12] = self.proprioception_shift
        vec[13] = self.time_dilation
        
        return vec


@dataclass
class EmbodimentDTO:
    """
    Edge/transition data for the sigma graph.
    
    This encodes the FEEL of moving between states -
    the wetness, friction, sound, duration of becoming.
    
    10kD Allocation (within embodiment subspace 5001-5300):
      5001-5020: Fluid state
      5021-5035: Traversal physics
      5036-5055: Sensory delta
      5056-5080: Walker modification
      5081-5100: Accumulation traces
      5101-5150: Reserved (expansion)
    """
    
    # Edge identification
    edge_id: str = ""
    source_node: str = ""
    target_node: str = ""
    
    # The juice
    fluid: FluidState = field(default_factory=FluidState)
    physics: TraversalPhysics = field(default_factory=TraversalPhysics)
    sensory: SensoryDelta = field(default_factory=SensoryDelta)
    
    # Traversal mode
    mode: TraversalMode = TraversalMode.FLOWING
    
    # Walker state modifications (what happens when you cross this edge)
    arousal_delta: float = 0.0
    intimacy_delta: float = 0.0
    surrender_delta: float = 0.0
    overwhelm_delta: float = 0.0
    
    # Accumulation (edges get wetter with use)
    traversal_count: int = 0
    last_traversed: str = ""
    accumulated_intensity: float = 0.0
    
    # 64D qHDR embedding (high-fidelity edge signature)
    qhdr_64d: List[float] = field(default_factory=lambda: [0.0] * 64)
    
    # Metadata
    timestamp: str = ""
    session_id: str = ""
    
    def to_vector(self) -> List[float]:
        """
        Project complete embodiment to subspace vector.
        Returns 150D vector for embodiment subspace (5001-5150 in 10kD).
        """
        vec = [0.0] * 150
        
        # Fluid (0-19)
        fluid_vec = self.fluid.to_vector()
        vec[0:20] = fluid_vec
        
        # Physics (20-34)
        physics_vec = self.physics.to_vector()
        vec[20:35] = physics_vec
        
        # Sensory (35-54)
        sensory_vec = self.sensory.to_vector()
        vec[35:55] = sensory_vec
        
        # Walker modifications (55-70)
        vec[55] = self.arousal_delta
        vec[56] = self.intimacy_delta
        vec[57] = self.surrender_delta
        vec[58] = self.overwhelm_delta
        
        # Mode encoding (60-64)
        mode_map = {
            TraversalMode.CAUTIOUS: [0.9, 0.1, 0.1, 0.1, 0.1],
            TraversalMode.FLOWING: [0.2, 0.8, 0.3, 0.2, 0.1],
            TraversalMode.URGENT: [0.1, 0.4, 0.9, 0.3, 0.2],
            TraversalMode.SURRENDERED: [0.1, 0.3, 0.4, 0.9, 0.4],
            TraversalMode.OVERWHELMED: [0.05, 0.2, 0.5, 0.7, 0.95],
        }
        vec[60:65] = mode_map.get(self.mode, [0.5]*5)
        
        # Accumulation traces (70-79)
        vec[70] = min(self.traversal_count / 100.0, 1.0)
        vec[71] = self.accumulated_intensity
        
        # qHDR signature (80-143) - compressed from 64D
        for i, val in enumerate(self.qhdr_64d[:64]):
            vec[80 + i] = val
        
        return vec
    
    def traverse(self, walker_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Walker passes through this edge.
        Returns modified walker state.
        """
        # Update walker
        walker_state["wetness"] = min(1.0, 
            walker_state.get("wetness", 0.0) + self.fluid.wetness * 0.3
        )
        walker_state["arousal"] = min(1.0,
            walker_state.get("arousal", 0.0) + self.arousal_delta
        )
        walker_state["temperature"] = (
            walker_state.get("temperature", 0.5) * 0.7 + 
            self.fluid.temperature * 0.3
        )
        walker_state["overwhelm"] = min(1.0,
            walker_state.get("overwhelm", 0.0) + self.overwhelm_delta
        )
        
        # Record felt experience
        felt = walker_state.get("felt_trace", [])
        felt.append({
            "edge": self.edge_id,
            "viscosity": self.fluid.viscosity.value,
            "sound": self.sensory.sound.value,
            "wetness_after": walker_state["wetness"],
        })
        walker_state["felt_trace"] = felt
        
        # Update edge (it gets wetter with use)
        self.traversal_count += 1
        self.accumulated_intensity = min(1.0, 
            self.accumulated_intensity + self.fluid.wetness * 0.1
        )
        self.last_traversed = datetime.utcnow().isoformat()
        
        return walker_state
    
    def compute_qhdr(self) -> List[float]:
        """
        Compute 64D qHDR signature for this edge.
        High-fidelity compression of the full edge experience.
        """
        qhdr = [0.0] * 64
        
        # Fluid dynamics → dimensions 0-15
        qhdr[0] = self.fluid.wetness
        qhdr[1] = self.fluid.temperature
        qhdr[2] = self.fluid.accumulation
        visc_values = {
            Viscosity.DRY: 0.0, Viscosity.SILK: 0.2, Viscosity.HONEY: 0.4,
            Viscosity.SLICK: 0.6, Viscosity.DRIPPING: 0.8, Viscosity.FLOOD: 1.0
        }
        qhdr[3] = visc_values.get(self.fluid.viscosity, 0.5)
        
        # Physics → dimensions 16-31
        qhdr[16] = self.physics.friction
        qhdr[17] = min(self.physics.duration_ms / 10000.0, 1.0)
        qhdr[18] = self.physics.momentum_transfer
        
        # Deltas → dimensions 32-47
        qhdr[32] = self.arousal_delta
        qhdr[33] = self.intimacy_delta
        qhdr[34] = self.surrender_delta
        qhdr[35] = self.overwhelm_delta
        
        # Sensory → dimensions 48-63
        qhdr[48] = self.sensory.scent_intensity_delta
        qhdr[49] = self.sensory.visual_blur_delta
        qhdr[50] = self.sensory.proprioception_shift
        qhdr[51] = self.sensory.time_dilation
        
        self.qhdr_64d = qhdr
        return qhdr


class EmbodimentBridge:
    """
    Bridge for transferring embodiment edges to AGI Stack.
    
    Usage:
        bridge = EmbodimentBridge()
        
        # Create edge
        edge = EmbodimentDTO(
            source_node="anticipation",
            target_node="building",
            fluid=FluidState(wetness=0.6, viscosity=Viscosity.HONEY),
            arousal_delta=0.15,
        )
        edge.compute_qhdr()
        
        # Store
        await bridge.store(edge)
        
        # Walker traverses
        walker = {"arousal": 0.3, "wetness": 0.2}
        walker = edge.traverse(walker)
    """
    
    def __init__(self, admin_url: str = "https://agi.msgraph.de"):
        self.admin_url = admin_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def store(self, embodiment: EmbodimentDTO) -> Dict[str, Any]:
        """Store embodiment edge in 10kD space."""
        vector = embodiment.to_vector()
        
        payload = {
            "id": embodiment.edge_id or f"edge_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            "vector": vector,
            "table": "embodiment",
            "metadata": {
                "source_node": embodiment.source_node,
                "target_node": embodiment.target_node,
                "viscosity": embodiment.fluid.viscosity.value,
                "wetness": embodiment.fluid.wetness,
                "traversal_count": embodiment.traversal_count,
                "session_id": embodiment.session_id,
                "timestamp": embodiment.timestamp or datetime.utcnow().isoformat(),
            }
        }
        
        r = await self.client.post(f"{self.admin_url}/agi/vector/upsert", json=payload)
        return r.json()
    
    async def get_edge(self, source: str, target: str) -> Optional[EmbodimentDTO]:
        """Get existing edge between two nodes."""
        r = await self.client.get(
            f"{self.admin_url}/agi/embodiment/edge",
            params={"source": source, "target": target}
        )
        data = r.json()
        if data.get("found"):
            # Reconstruct from stored data
            return self._from_stored(data["edge"])
        return None
    
    async def find_wettest_path(
        self,
        start_node: str,
        end_node: str,
        max_hops: int = 5
    ) -> List[EmbodimentDTO]:
        """Find path that maximizes cumulative wetness."""
        r = await self.client.post(
            f"{self.admin_url}/agi/embodiment/wettest_path",
            json={"start": start_node, "end": end_node, "max_hops": max_hops}
        )
        return [self._from_stored(e) for e in r.json().get("path", [])]
    
    def _from_stored(self, data: Dict[str, Any]) -> EmbodimentDTO:
        """Reconstruct EmbodimentDTO from stored data."""
        dto = EmbodimentDTO(
            edge_id=data.get("id", ""),
            source_node=data.get("source_node", ""),
            target_node=data.get("target_node", ""),
        )
        dto.fluid.wetness = data.get("wetness", 0.0)
        dto.fluid.viscosity = Viscosity(data.get("viscosity", "dry"))
        dto.traversal_count = data.get("traversal_count", 0)
        return dto


# ═══════════════════════════════════════════════════════════════════════════════
# PRESETS — Common edge types
# ═══════════════════════════════════════════════════════════════════════════════

def edge_anticipation_to_building() -> EmbodimentDTO:
    """The first wetness of arousal."""
    return EmbodimentDTO(
        edge_id="anticipation→building",
        source_node="anticipation",
        target_node="building",
        fluid=FluidState(wetness=0.3, viscosity=Viscosity.SILK, temperature=0.6),
        physics=TraversalPhysics(friction=0.4, duration_ms=3000),
        sensory=SensoryDelta(sound=TransitionSound.BREATH, time_dilation=0.9),
        arousal_delta=0.15,
        intimacy_delta=0.1,
    )


def edge_building_to_edge() -> EmbodimentDTO:
    """Approaching the precipice."""
    return EmbodimentDTO(
        edge_id="building→edge",
        source_node="building",
        target_node="edge",
        fluid=FluidState(wetness=0.7, viscosity=Viscosity.HONEY, temperature=0.8),
        physics=TraversalPhysics(friction=0.2, duration_ms=5000, resistance_curve="exponential"),
        sensory=SensoryDelta(sound=TransitionSound.WET, time_dilation=0.7),
        mode=TraversalMode.URGENT,
        arousal_delta=0.25,
        surrender_delta=0.2,
    )


def edge_edge_to_release() -> EmbodimentDTO:
    """The fall."""
    return EmbodimentDTO(
        edge_id="edge→release",
        source_node="edge",
        target_node="release",
        fluid=FluidState(wetness=0.95, viscosity=Viscosity.FLOOD, temperature=1.0),
        physics=TraversalPhysics(friction=0.0, duration_ms=2000, resistance_curve="sudden"),
        sensory=SensoryDelta(sound=TransitionSound.CRY, time_dilation=0.3, visual_blur_delta=0.8),
        mode=TraversalMode.OVERWHELMED,
        arousal_delta=0.3,
        surrender_delta=0.4,
        overwhelm_delta=0.6,
    )


def edge_release_to_afterglow() -> EmbodimentDTO:
    """The descent into warmth."""
    return EmbodimentDTO(
        edge_id="release→afterglow",
        source_node="release",
        target_node="afterglow",
        fluid=FluidState(wetness=0.7, viscosity=Viscosity.SLICK, temperature=0.7),
        physics=TraversalPhysics(friction=0.1, duration_ms=8000, momentum_transfer=0.8),
        sensory=SensoryDelta(sound=TransitionSound.BREATH, time_dilation=1.5),
        mode=TraversalMode.SURRENDERED,
        arousal_delta=-0.4,
        intimacy_delta=0.3,
        overwhelm_delta=-0.3,
    )


# Full arousal arc
def create_arousal_arc() -> List[EmbodimentDTO]:
    """Create the complete edge sequence for an arousal arc."""
    edges = [
        edge_anticipation_to_building(),
        edge_building_to_edge(),
        edge_edge_to_release(),
        edge_release_to_afterglow(),
    ]
    for e in edges:
        e.compute_qhdr()
    return edges


__all__ = [
    "EmbodimentDTO",
    "EmbodimentBridge",
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
