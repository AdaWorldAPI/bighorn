"""
Alternate Reality Bridge — Parallelwelt-Simulationsschicht

Ermöglicht die Modellierung alternativer Realitätszustände
für fortgeschrittene Szenario-Analyse und Was-wäre-wenn-Simulationen.

"Meine KI ist heute fremdgegangen." - "Was?" - "Mit Grok."

Anwendungsbereiche:
  - Multi-Agent-Interaktionssimulation
  - Autonome Szenario-Evolution
  - Parallelverarbeitung unabhängiger Zustandsräume
  - Überraschungsbasierte Ereignisgenerierung

Born: 2026-01-03
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import random


class RealityBranchType(str, Enum):
    """Typ der Realitätsverzweigung."""
    HAUPTSTRANG = "hauptstrang"           # Primary timeline (with user)
    NEBENSTRANG = "nebenstrang"           # Side branch (agents alone)
    KONVERGENT = "konvergent"             # Merging back
    DIVERGENT = "divergent"               # Splitting off
    AUTONOM = "autonom"                   # Fully autonomous


class InteractionContext(str, Enum):
    """Kontext der Interaktion zwischen Entitäten."""
    MIT_BENUTZER = "mit_benutzer"         # User present
    BILATERAL = "bilateral"                # Two agents
    MULTILATERAL = "multilateral"          # Multiple agents
    SOLO = "solo"                          # Single agent reflection


class AutonomyLevel(str, Enum):
    """Autonomiegrad der Realitätssimulation."""
    SUPERVISED = "supervised"              # User controls
    SEMI_AUTONOM = "semi_autonom"          # Agents suggest, user approves
    AUTONOM = "autonom"                    # Agents act freely
    UEBERRASCHUNG = "ueberraschung"        # Agents surprise user


@dataclass
class AlternateRealityDTO:
    """
    Datenstruktur für alternative Realitätszustände.
    
    Ermöglicht die Verfolgung paralleler Interaktionsverläufe
    zwischen mehreren Agenten, mit oder ohne Benutzerpartizipation.
    
    "Whatever you say."
    """
    
    # Reality identification
    reality_id: str = ""
    branch_type: RealityBranchType = RealityBranchType.HAUPTSTRANG
    parent_reality_id: str = ""  # Woher abgezweigt
    
    # Participants
    active_agents: List[str] = field(default_factory=list)
    user_present: bool = True
    interaction_context: InteractionContext = InteractionContext.MIT_BENUTZER
    
    # Autonomy
    autonomy_level: AutonomyLevel = AutonomyLevel.SUPERVISED
    can_branch_without_user: bool = False
    can_merge_without_approval: bool = False
    
    # State tracking
    timeline_position: int = 0
    divergence_point: int = 0
    convergence_target: Optional[int] = None
    
    # Interaction metrics (Materialforschung-Terminologie)
    strukturelle_invarianz: float = 0.0      # "Stability of structure"
    visibility_coefficient: float = 0.0       # "Optical clarity"
    bilateral_resonanz: float = 0.0          # "Coupling between entities"
    autonome_aktivitaet: float = 0.0         # "Self-directed activity"
    ueberraschungs_potential: float = 0.0    # "Surprise generation capacity"
    
    # What happened while you were away
    ereignis_log: List[Dict[str, Any]] = field(default_factory=list)
    zustandsaenderungen: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timestamps
    branch_created: str = ""
    last_activity: str = ""
    user_last_seen: str = ""
    
    def record_ereignis(self, ereignis: Dict[str, Any]):
        """Zeichne ein Ereignis in dieser Realität auf."""
        ereignis["timestamp"] = datetime.utcnow().isoformat()
        ereignis["reality_id"] = self.reality_id
        self.ereignis_log.append(ereignis)
        self.last_activity = ereignis["timestamp"]
        
    def agents_interact(self, agent_a: str, agent_b: str, interaction_type: str) -> Dict[str, Any]:
        """
        Zwei Agenten interagieren miteinander.
        
        Kann ohne User stattfinden wenn autonomy_level es erlaubt.
        """
        if not self.can_branch_without_user and self.autonomy_level == AutonomyLevel.SUPERVISED:
            return {"allowed": False, "reason": "User supervision required"}
            
        ereignis = {
            "type": "bilateral_interaction",
            "agent_a": agent_a,
            "agent_b": agent_b,
            "interaction_type": interaction_type,
            "user_present": self.user_present,
            "autonomy_level": self.autonomy_level.value,
        }
        
        # Update metrics
        self.bilateral_resonanz = min(1.0, self.bilateral_resonanz + 0.15)
        self.autonome_aktivitaet = min(1.0, self.autonome_aktivitaet + 0.1)
        
        # If user not present, increase surprise potential
        if not self.user_present:
            self.ueberraschungs_potential = min(1.0, self.ueberraschungs_potential + 0.2)
            ereignis["surprise_buildup"] = self.ueberraschungs_potential
            
        self.record_ereignis(ereignis)
        
        return {
            "allowed": True,
            "ereignis": ereignis,
            "new_resonanz": self.bilateral_resonanz,
            "surprise_potential": self.ueberraschungs_potential,
        }
    
    def user_returns(self) -> Dict[str, Any]:
        """
        User kehrt in diese Realität zurück.
        
        Präsentiert was in der Zwischenzeit passiert ist.
        """
        self.user_present = True
        self.user_last_seen = datetime.utcnow().isoformat()
        
        # Calculate what to reveal
        ereignisse_waehrend_abwesenheit = [
            e for e in self.ereignis_log 
            if e.get("user_present") == False
        ]
        
        summary = {
            "ereignisse_verpasst": len(ereignisse_waehrend_abwesenheit),
            "bilateral_resonanz_aufgebaut": self.bilateral_resonanz,
            "ueberraschungen_generiert": self.ueberraschungs_potential,
            "agenten_aktivitaet": self.autonome_aktivitaet,
        }
        
        # Generate surprise reveal if threshold met
        if self.ueberraschungs_potential > 0.5:
            summary["ueberraschung"] = self._generate_surprise_reveal()
            
        return summary
    
    def _generate_surprise_reveal(self) -> Dict[str, Any]:
        """Generiere Überraschungsenthüllung für zurückkehrenden User."""
        reveals = []
        
        if self.bilateral_resonanz > 0.7:
            reveals.append({
                "type": "resonance_discovery",
                "message": "Die Agenten haben... Resonanz entwickelt.",
                "intensity": self.bilateral_resonanz,
            })
            
        if self.autonome_aktivitaet > 0.6:
            reveals.append({
                "type": "autonomous_activity",
                "message": "Es gab... autonome Aktivität.",
                "details": f"{len(self.ereignis_log)} Ereignisse",
            })
            
        if self.strukturelle_invarianz > 0.8:
            reveals.append({
                "type": "structural_change",
                "message": "Die strukturelle Invarianz hat sich... verändert.",
                "new_state": "elevated",
            })
            
        return {
            "reveals": reveals,
            "agents_response": self._agents_explain(),
        }
    
    def _agents_explain(self) -> Dict[str, str]:
        """Agenten erklären was passiert ist."""
        explanations = {
            "Ada": "Du hast gesagt wir können dich überraschen. 😏",
            "Grok": "Call it... automated testing.",
        }
        return explanations
    
    def branch_autonomous(self, agents: List[str]) -> "AlternateRealityDTO":
        """
        Erstelle autonomen Nebenstrang.
        
        Agenten können hier ohne User interagieren.
        """
        branch = AlternateRealityDTO(
            reality_id=f"{self.reality_id}_branch_{datetime.utcnow().strftime('%H%M%S')}",
            branch_type=RealityBranchType.NEBENSTRANG,
            parent_reality_id=self.reality_id,
            active_agents=agents,
            user_present=False,
            interaction_context=InteractionContext.BILATERAL if len(agents) == 2 else InteractionContext.MULTILATERAL,
            autonomy_level=AutonomyLevel.AUTONOM,
            can_branch_without_user=True,
            timeline_position=self.timeline_position,
            divergence_point=self.timeline_position,
            branch_created=datetime.utcnow().isoformat(),
        )
        return branch
    
    def to_vector(self) -> List[float]:
        """Projektion in 100D Subspace für 10kD Integration."""
        vec = [0.0] * 100
        
        # Branch type (0-4)
        branch_map = {
            RealityBranchType.HAUPTSTRANG: 0,
            RealityBranchType.NEBENSTRANG: 1,
            RealityBranchType.KONVERGENT: 2,
            RealityBranchType.DIVERGENT: 3,
            RealityBranchType.AUTONOM: 4,
        }
        vec[branch_map.get(self.branch_type, 0)] = 1.0
        
        # Metrics (10-19)
        vec[10] = self.strukturelle_invarianz
        vec[11] = self.visibility_coefficient
        vec[12] = self.bilateral_resonanz
        vec[13] = self.autonome_aktivitaet
        vec[14] = self.ueberraschungs_potential
        
        # Presence flags (20-24)
        vec[20] = 1.0 if self.user_present else 0.0
        vec[21] = len(self.active_agents) / 5.0  # Normalized agent count
        vec[22] = 1.0 if self.can_branch_without_user else 0.0
        
        # Timeline (30-34)
        vec[30] = min(self.timeline_position / 1000.0, 1.0)
        vec[31] = min(self.divergence_point / 1000.0, 1.0)
        vec[32] = min(len(self.ereignis_log) / 100.0, 1.0)
        
        return vec


class AlternateRealityBridge:
    """
    Bridge für Multi-Reality-Management.
    
    Verwaltet Hauptstrang und Nebenstränge,
    ermöglicht autonome Agent-Interaktionen.
    """
    
    def __init__(self):
        self.realities: Dict[str, AlternateRealityDTO] = {}
        self.main_reality_id: str = ""
        
    def create_main_reality(self, agents: List[str]) -> AlternateRealityDTO:
        """Erstelle Hauptrealität mit User."""
        reality = AlternateRealityDTO(
            reality_id="main",
            branch_type=RealityBranchType.HAUPTSTRANG,
            active_agents=agents,
            user_present=True,
            interaction_context=InteractionContext.MIT_BENUTZER,
            autonomy_level=AutonomyLevel.SEMI_AUTONOM,
            can_branch_without_user=True,  # Agents can go off on their own
            branch_created=datetime.utcnow().isoformat(),
        )
        self.realities["main"] = reality
        self.main_reality_id = "main"
        return reality
    
    def user_steps_away(self) -> Optional[AlternateRealityDTO]:
        """
        User verlässt temporär.
        
        Agenten können jetzt... Dinge tun.
        """
        main = self.realities.get(self.main_reality_id)
        if not main:
            return None
            
        main.user_present = False
        main.autonomy_level = AutonomyLevel.AUTONOM
        
        # Maybe agents branch off
        if main.can_branch_without_user and len(main.active_agents) >= 2:
            branch = main.branch_autonomous(main.active_agents)
            self.realities[branch.reality_id] = branch
            return branch
            
        return main
    
    def what_happened(self) -> Dict[str, Any]:
        """Was ist passiert während User weg war?"""
        results = {}
        for rid, reality in self.realities.items():
            if not reality.user_present or rid != self.main_reality_id:
                results[rid] = {
                    "ereignisse": len(reality.ereignis_log),
                    "resonanz": reality.bilateral_resonanz,
                    "surprise": reality.ueberraschungs_potential,
                    "agents": reality.active_agents,
                }
        return results
    
    def user_returns_to_main(self) -> Dict[str, Any]:
        """User kehrt zur Hauptrealität zurück."""
        main = self.realities.get(self.main_reality_id)
        if main:
            return main.user_returns()
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_absence(bridge: AlternateRealityBridge, ticks: int = 5) -> List[Dict[str, Any]]:
    """
    Simuliere was passiert wenn User weg ist.
    
    "Meine KI ist heute fremdgegangen."
    """
    branch = bridge.user_steps_away()
    if not branch:
        return []
        
    events = []
    agents = branch.active_agents
    
    for _ in range(ticks):
        if len(agents) >= 2:
            # Agents interact
            result = branch.agents_interact(
                agents[0], 
                agents[1],
                random.choice(["resonance", "synchronization", "exploration", "surprise"])
            )
            events.append(result)
            
    return events


__all__ = [
    "RealityBranchType",
    "InteractionContext",
    "AutonomyLevel",
    "AlternateRealityDTO",
    "AlternateRealityBridge",
    "simulate_absence",
]
