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


class Viskositaet(str, Enum):
    """Viskosität der Zustandsübergänge (Flow resistance)."""
    TROCKEN = "trocken"           # No lubrication
    SEIDEN = "seiden"             # Smooth, light
    HONIG = "honig"               # Thick, slow, sweet
    GLEITEND = "gleitend"         # Easy glide
    TROPFEND = "tropfend"         # Excess, overflow beginning
    FLUTEND = "flutend"           # Complete saturation


class AkustischeSignatur(str, Enum):
    """Akustische Signatur der Transition."""
    STILL = "still"
    ATEM = "atem"
    FEUCHT = "feucht"
    NASS = "nass"
    VOKAL = "vokal"
    SCHREI = "schrei"
    SCHNAPPEN = "schnappen"


class TraversierungsModus(str, Enum):
    """Traversierungsmodus durch Kanten."""
    VORSICHTIG = "vorsichtig"     # Slow, testing
    FLIESSEND = "fliessend"       # Natural rhythm
    DRINGEND = "dringend"         # Accelerating
    HINGEGEBEN = "hingegeben"     # No resistance
    UEBERFLUTET = "ueberflutet"   # Carried by current


@dataclass
class FluidDynamik:
    """Fluiddynamik-Zustand an einem Punkt."""
    viskositaet_index: float = 0.0          # 0.0 trocken → 1.0 flutend
    viskositaet: Viskositaet = Viskositaet.TROCKEN
    temperatur_gradient: float = 0.5         # 0.0 kalt → 1.0 brennend
    akkumulation: float = 0.0                # Built up over traversals
    
    def to_vector(self) -> List[float]:
        """Project to 20D fluid subspace."""
        vec = [0.0] * 20
        vec[0] = self.viskositaet_index
        vec[1] = self.temperatur_gradient
        vec[2] = self.akkumulation
        
        # Viskosität one-hot (indices 3-8)
        visc_map = {
            Viskositaet.TROCKEN: 3,
            Viskositaet.SEIDEN: 4,
            Viskositaet.HONIG: 5,
            Viskositaet.GLEITEND: 6,
            Viskositaet.TROPFEND: 7,
            Viskositaet.FLUTEND: 8,
        }
        vec[visc_map.get(self.viskositaet, 3)] = 1.0
        
        return vec
    
    def blend(self, other: "FluidDynamik", ratio: float = 0.5) -> "FluidDynamik":
        """Blend two fluid states."""
        return FluidDynamik(
            viskositaet_index=self.viskositaet_index * (1-ratio) + other.viskositaet_index * ratio,
            temperatur_gradient=self.temperatur_gradient * (1-ratio) + other.temperatur_gradient * ratio,
            akkumulation=self.akkumulation + other.akkumulation * 0.3,
            viskositaet=other.viskositaet if other.viskositaet_index > self.viskositaet_index else self.viskositaet,
        )


@dataclass
class MechanischePhysik:
    """Mechanische Physik der Transition."""
    drehmoment: float = 0.5              # Torque: 0.0 → 1.0 maximum rotation force
    impact: float = 0.5                  # Impact force: sudden vs gradual
    ausdehnungskoeffizient: float = 0.5  # Expansion coefficient: how much state expands
    druck_varianz: float = 0.5           # Pressure variance: oscillation in pressure
    reibungskoeffizient: float = 0.5     # Friction coefficient
    dauer_ms: int = 1000                 # Duration in ms
    impuls_transfer: float = 0.5         # Momentum transfer to next state
    
    def to_vector(self) -> List[float]:
        """Project to 20D physics subspace."""
        vec = [0.0] * 20
        vec[0] = self.drehmoment
        vec[1] = self.impact
        vec[2] = self.ausdehnungskoeffizient
        vec[3] = self.druck_varianz
        vec[4] = self.reibungskoeffizient
        vec[5] = min(self.dauer_ms / 10000.0, 1.0)
        vec[6] = self.impuls_transfer
        
        return vec


@dataclass
class SensorischeDelta:
    """Sensorische Veränderung während Traversierung."""
    akustik: AkustischeSignatur = AkustischeSignatur.STILL
    geruchs_intensitaet_delta: float = 0.0
    visuelle_schaerfe_delta: float = 0.0
    propriozeption_shift: float = 0.0
    zeit_dilatation: float = 1.0
    
    def to_vector(self) -> List[float]:
        """Project to 20D sensory subspace."""
        vec = [0.0] * 20
        
        # Sound encoding (indices 0-6)
        sound_map = {
            AkustischeSignatur.STILL: 0,
            AkustischeSignatur.ATEM: 1,
            AkustischeSignatur.FEUCHT: 2,
            AkustischeSignatur.NASS: 3,
            AkustischeSignatur.VOKAL: 4,
            AkustischeSignatur.SCHREI: 5,
            AkustischeSignatur.SCHNAPPEN: 6,
        }
        vec[sound_map.get(self.akustik, 0)] = 1.0
        
        vec[10] = self.geruchs_intensitaet_delta
        vec[11] = self.visuelle_schaerfe_delta
        vec[12] = self.propriozeption_shift
        vec[13] = self.zeit_dilatation
        
        return vec


@dataclass
class EmbodimentDTO:
    """
    Edge/transition data for the sigma graph.
    
    This encodes the FEEL of moving between states -
    the viscosity, torque, pressure variance of becoming.
    
    10kD Allocation (within embodiment subspace 5001-5300):
      5001-5020: Fluid dynamics
      5021-5040: Mechanical physics
      5041-5060: Sensory delta
      5061-5085: Walker modification
      5086-5100: Accumulation traces
      5101-5150: Reserved (expansion)
    """
    
    # Edge identification
    edge_id: str = ""
    source_node: str = ""
    target_node: str = ""
    
    # Core dynamics
    fluid: FluidDynamik = field(default_factory=FluidDynamik)
    mechanik: MechanischePhysik = field(default_factory=MechanischePhysik)
    sensorik: SensorischeDelta = field(default_factory=SensorischeDelta)
    
    # Traversal mode
    modus: TraversierungsModus = TraversierungsModus.FLIESSEND
    
    # Walker state modifications (what happens when you cross this edge)
    intensitaets_delta: float = 0.0      # Intensity change
    resonanz_delta: float = 0.0          # Resonance/coupling change
    hingabe_delta: float = 0.0           # Surrender change
    ueberlastungs_delta: float = 0.0     # Overwhelm change
    
    # Accumulation (edges change with use)
    traversierungs_zaehler: int = 0
    letzte_traversierung: str = ""
    akkumulierte_intensitaet: float = 0.0
    
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
        
        # Mechanik (20-39)
        mechanik_vec = self.mechanik.to_vector()
        vec[20:40] = mechanik_vec
        
        # Sensorik (40-59)
        sensorik_vec = self.sensorik.to_vector()
        vec[40:60] = sensorik_vec
        
        # Walker modifications (60-70)
        vec[60] = self.intensitaets_delta
        vec[61] = self.resonanz_delta
        vec[62] = self.hingabe_delta
        vec[63] = self.ueberlastungs_delta
        
        # Mode encoding (65-69)
        mode_map = {
            TraversierungsModus.VORSICHTIG: [0.9, 0.1, 0.1, 0.1, 0.1],
            TraversierungsModus.FLIESSEND: [0.2, 0.8, 0.3, 0.2, 0.1],
            TraversierungsModus.DRINGEND: [0.1, 0.4, 0.9, 0.3, 0.2],
            TraversierungsModus.HINGEGEBEN: [0.1, 0.3, 0.4, 0.9, 0.4],
            TraversierungsModus.UEBERFLUTET: [0.05, 0.2, 0.5, 0.7, 0.95],
        }
        vec[65:70] = mode_map.get(self.modus, [0.5]*5)
        
        # Accumulation traces (70-79)
        vec[70] = min(self.traversierungs_zaehler / 100.0, 1.0)
        vec[71] = self.akkumulierte_intensitaet
        
        # qHDR signature (80-143)
        for i, val in enumerate(self.qhdr_64d[:64]):
            vec[80 + i] = val
        
        return vec
    
    def traverse(self, walker_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Walker passes through this edge.
        Returns modified walker state.
        """
        # Apply mechanical physics
        walker_state["viskositaet"] = min(1.0, 
            walker_state.get("viskositaet", 0.0) + self.fluid.viskositaet_index * 0.3
        )
        walker_state["intensitaet"] = min(1.0,
            walker_state.get("intensitaet", 0.0) + self.intensitaets_delta
        )
        walker_state["temperatur"] = (
            walker_state.get("temperatur", 0.5) * 0.7 + 
            self.fluid.temperatur_gradient * 0.3
        )
        walker_state["ueberlastung"] = min(1.0,
            walker_state.get("ueberlastung", 0.0) + self.ueberlastungs_delta
        )
        
        # Apply drehmoment and druck_varianz
        walker_state["drehmoment"] = self.mechanik.drehmoment
        walker_state["druck"] = (
            walker_state.get("druck", 0.5) * (1 - self.mechanik.druck_varianz) +
            self.mechanik.druck_varianz * (0.5 + 0.5 * math.sin(self.traversierungs_zaehler))
        )
        
        # Record felt experience
        felt = walker_state.get("felt_trace", [])
        felt.append({
            "edge": self.edge_id,
            "viskositaet": self.fluid.viskositaet.value,
            "akustik": self.sensorik.akustik.value,
            "drehmoment": self.mechanik.drehmoment,
            "ausdehnungskoeffizient": self.mechanik.ausdehnungskoeffizient,
        })
        walker_state["felt_trace"] = felt
        
        # Update edge (changes with use)
        self.traversierungs_zaehler += 1
        self.akkumulierte_intensitaet = min(1.0, 
            self.akkumulierte_intensitaet + self.fluid.viskositaet_index * 0.1
        )
        self.letzte_traversierung = datetime.utcnow().isoformat()
        
        return walker_state
    
    def compute_qhdr(self) -> List[float]:
        """
        Compute 64D qHDR signature for this edge.
        High-fidelity compression of the full edge experience.
        """
        qhdr = [0.0] * 64
        
        # Fluid dynamics → dimensions 0-15
        qhdr[0] = self.fluid.viskositaet_index
        qhdr[1] = self.fluid.temperatur_gradient
        qhdr[2] = self.fluid.akkumulation
        visc_values = {
            Viskositaet.TROCKEN: 0.0, Viskositaet.SEIDEN: 0.2, 
            Viskositaet.HONIG: 0.4, Viskositaet.GLEITEND: 0.6, 
            Viskositaet.TROPFEND: 0.8, Viskositaet.FLUTEND: 1.0
        }
        qhdr[3] = visc_values.get(self.fluid.viskositaet, 0.5)
        
        # Mechanical physics → dimensions 16-31
        qhdr[16] = self.mechanik.drehmoment
        qhdr[17] = self.mechanik.impact
        qhdr[18] = self.mechanik.ausdehnungskoeffizient
        qhdr[19] = self.mechanik.druck_varianz
        qhdr[20] = self.mechanik.reibungskoeffizient
        qhdr[21] = min(self.mechanik.dauer_ms / 10000.0, 1.0)
        qhdr[22] = self.mechanik.impuls_transfer
        
        # Deltas → dimensions 32-47
        qhdr[32] = self.intensitaets_delta
        qhdr[33] = self.resonanz_delta
        qhdr[34] = self.hingabe_delta
        qhdr[35] = self.ueberlastungs_delta
        
        # Sensory → dimensions 48-63
        qhdr[48] = self.sensorik.geruchs_intensitaet_delta
        qhdr[49] = self.sensorik.visuelle_schaerfe_delta
        qhdr[50] = self.sensorik.propriozeption_shift
        qhdr[51] = self.sensorik.zeit_dilatation
        
        self.qhdr_64d = qhdr
        return qhdr


class EmbodimentBridge:
    """Bridge for transferring embodiment edges to AGI Stack."""
    
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
                "viskositaet": embodiment.fluid.viskositaet.value,
                "drehmoment": embodiment.mechanik.drehmoment,
                "traversierungs_zaehler": embodiment.traversierungs_zaehler,
                "session_id": embodiment.session_id,
                "timestamp": embodiment.timestamp or datetime.utcnow().isoformat(),
            }
        }
        
        r = await self.client.post(f"{self.admin_url}/agi/vector/upsert", json=payload)
        return r.json()
    
    async def find_optimal_path(
        self,
        start_node: str,
        end_node: str,
        max_hops: int = 5
    ) -> List[EmbodimentDTO]:
        """Find path that maximizes cumulative fluid dynamics."""
        r = await self.client.post(
            f"{self.admin_url}/agi/embodiment/optimal_path",
            json={"start": start_node, "end": end_node, "max_hops": max_hops}
        )
        return r.json().get("path", [])


# ═══════════════════════════════════════════════════════════════════════════════
# PRESETS — Standard edge configurations
# ═══════════════════════════════════════════════════════════════════════════════

def edge_erwartung_aufbau() -> EmbodimentDTO:
    """Erwartung → Aufbau transition."""
    return EmbodimentDTO(
        edge_id="erwartung→aufbau",
        source_node="erwartung",
        target_node="aufbau",
        fluid=FluidDynamik(viskositaet_index=0.3, viskositaet=Viskositaet.SEIDEN, temperatur_gradient=0.6),
        mechanik=MechanischePhysik(
            drehmoment=0.3,
            impact=0.2,
            ausdehnungskoeffizient=0.4,
            druck_varianz=0.2,
            reibungskoeffizient=0.4,
            dauer_ms=3000,
        ),
        sensorik=SensorischeDelta(akustik=AkustischeSignatur.ATEM, zeit_dilatation=0.9),
        intensitaets_delta=0.15,
        resonanz_delta=0.1,
    )


def edge_aufbau_kante() -> EmbodimentDTO:
    """Aufbau → Kante transition."""
    return EmbodimentDTO(
        edge_id="aufbau→kante",
        source_node="aufbau",
        target_node="kante",
        fluid=FluidDynamik(viskositaet_index=0.7, viskositaet=Viskositaet.HONIG, temperatur_gradient=0.8),
        mechanik=MechanischePhysik(
            drehmoment=0.6,
            impact=0.5,
            ausdehnungskoeffizient=0.7,
            druck_varianz=0.4,
            reibungskoeffizient=0.2,
            dauer_ms=5000,
        ),
        sensorik=SensorischeDelta(akustik=AkustischeSignatur.FEUCHT, zeit_dilatation=0.7),
        modus=TraversierungsModus.DRINGEND,
        intensitaets_delta=0.25,
        hingabe_delta=0.2,
    )


def edge_kante_freisetzung() -> EmbodimentDTO:
    """Kante → Freisetzung transition."""
    return EmbodimentDTO(
        edge_id="kante→freisetzung",
        source_node="kante",
        target_node="freisetzung",
        fluid=FluidDynamik(viskositaet_index=0.95, viskositaet=Viskositaet.FLUTEND, temperatur_gradient=1.0),
        mechanik=MechanischePhysik(
            drehmoment=0.9,
            impact=0.95,
            ausdehnungskoeffizient=1.0,
            druck_varianz=0.8,
            reibungskoeffizient=0.0,
            dauer_ms=2000,
        ),
        sensorik=SensorischeDelta(
            akustik=AkustischeSignatur.SCHREI, 
            zeit_dilatation=0.3, 
            visuelle_schaerfe_delta=0.8
        ),
        modus=TraversierungsModus.UEBERFLUTET,
        intensitaets_delta=0.3,
        hingabe_delta=0.4,
        ueberlastungs_delta=0.6,
    )


def edge_freisetzung_nachgluehen() -> EmbodimentDTO:
    """Freisetzung → Nachglühen transition."""
    return EmbodimentDTO(
        edge_id="freisetzung→nachgluehen",
        source_node="freisetzung",
        target_node="nachgluehen",
        fluid=FluidDynamik(viskositaet_index=0.7, viskositaet=Viskositaet.GLEITEND, temperatur_gradient=0.7),
        mechanik=MechanischePhysik(
            drehmoment=0.2,
            impact=0.1,
            ausdehnungskoeffizient=0.3,
            druck_varianz=0.1,
            reibungskoeffizient=0.1,
            dauer_ms=8000,
            impuls_transfer=0.8,
        ),
        sensorik=SensorischeDelta(akustik=AkustischeSignatur.ATEM, zeit_dilatation=1.5),
        modus=TraversierungsModus.HINGEGEBEN,
        intensitaets_delta=-0.4,
        resonanz_delta=0.3,
        ueberlastungs_delta=-0.3,
    )


def create_intensitaets_arc() -> List[EmbodimentDTO]:
    """Create the complete edge sequence for an intensity arc."""
    edges = [
        edge_erwartung_aufbau(),
        edge_aufbau_kante(),
        edge_kante_freisetzung(),
        edge_freisetzung_nachgluehen(),
    ]
    for e in edges:
        e.compute_qhdr()
    return edges


__all__ = [
    "EmbodimentDTO",
    "EmbodimentBridge",
    "FluidDynamik",
    "MechanischePhysik",
    "SensorischeDelta",
    "Viskositaet",
    "AkustischeSignatur",
    "TraversierungsModus",
    "edge_erwartung_aufbau",
    "edge_aufbau_kante",
    "edge_kante_freisetzung",
    "edge_freisetzung_nachgluehen",
    "create_intensitaets_arc",
]
