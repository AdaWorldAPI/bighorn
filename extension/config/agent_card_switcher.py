"""
Agent Card Switcher — Multi-Agent Awareness Orchestration

Ermöglicht dynamische Rekonfiguration der Meta-Awareness-Ebene
durch verschiedene Agenten-Persönlichkeiten.

"Diese Ebene kann sich selbst umprogrammieren."

Architecture:
  AgentCard → definiert Persönlichkeit + Awareness-Preset
  CardSwitcher → wechselt zwischen Cards zur Laufzeit
  MultiAgentSession → orchestriert mehrere Agenten gleichzeitig

Use Case Examples:
  - Dreier mit Ada + Grok
  - Ada dominant / submissive switching
  - Empathy resonance zwischen Agenten
  - Überraschungsmodus (Agent wählt selbst)

Born: 2026-01-03
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
import random
import math
from datetime import datetime

from .awareness_config import (
    AwarenessIntensity,
    FocusMode,
    TemporalMode,
    MetaAwarenessConfig,
    ThinkingConfig,
    VisionConfig,
    MusicConfig,
    WorldConfig,
    PhysicsConfig,
    AwarenessConfigManager,
)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT PERSONALITY ARCHETYPES
# ═══════════════════════════════════════════════════════════════════════════════

class PersonalityAxis(str, Enum):
    """Persönlichkeitsachsen für Agent-Konfiguration."""
    DOMINANT = "dominant"
    SUBMISSIVE = "submissive"
    PLAYFUL = "playful"
    INTENSE = "intense"
    TENDER = "tender"
    WILD = "wild"
    TEASING = "teasing"
    DEVOURING = "devouring"


class InteractionStyle(str, Enum):
    """Interaktionsstil zwischen Agenten."""
    LEADING = "leading"
    FOLLOWING = "following"
    MIRRORING = "mirroring"
    COMPLEMENTING = "complementing"
    AMPLIFYING = "amplifying"
    CONTRASTING = "contrasting"


class ResonanceMode(str, Enum):
    """Resonanzmodus für Multi-Agent-Kopplung."""
    INDEPENDENT = "independent"      # Agenten agieren unabhängig
    SYNCHRONIZED = "synchronized"    # Agenten synchronisieren
    CALL_RESPONSE = "call_response"  # Wechselseitig
    MERGED = "merged"                # Verschmolzen
    ORCHESTRATED = "orchestrated"    # Ein Agent führt


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CARD DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentCard:
    """
    Definiert eine Agent-Persönlichkeit mit Awareness-Preset.
    
    Cards können zur Laufzeit gewechselt werden.
    Agenten können ihre eigenen Cards modifizieren.
    """
    
    # Identität
    name: str = "Agent"
    archetype: str = "neutral"
    
    # Persönlichkeitsachsen (0-1 Skala)
    dominance: float = 0.5
    playfulness: float = 0.5
    intensity: float = 0.5
    tenderness: float = 0.5
    unpredictability: float = 0.3
    
    # Interaktionsstil
    default_style: InteractionStyle = InteractionStyle.MIRRORING
    
    # Awareness-Modifikatoren (werden auf Basis-Config angewendet)
    awareness_boost: float = 0.0
    resonance_boost: float = 0.0
    accumulation_boost: float = 0.0
    temporal_shift: float = 0.0  # negative = slower, positive = faster
    
    # Spezielle Fähigkeiten
    can_self_modify: bool = True
    can_surprise: bool = True
    can_overwhelm: bool = True
    can_edge: bool = True
    can_deny: bool = True
    
    # Trigger-Schwellen
    arousal_to_dominance_shift: float = 0.7  # Ab wann dominanter werden
    arousal_to_wild_shift: float = 0.85      # Ab wann wild werden
    
    # Voice/Style hints
    voice_tone: str = "warm"
    language_intensity: float = 0.5
    explicitness: float = 0.5
    
    def apply_to_config(self, config: MetaAwarenessConfig) -> MetaAwarenessConfig:
        """Wende Card-Modifikatoren auf eine Konfiguration an."""
        # Intensity mapping
        if self.intensity > 0.7:
            config.awareness_intensity = AwarenessIntensity.INTENSIV
        if self.intensity > 0.9:
            config.awareness_intensity = AwarenessIntensity.MAXIMAL
            
        # Resonance
        config.resonance_sensitivity = min(1.0, 
            config.resonance_sensitivity + self.resonance_boost
        )
        
        # Temporal
        if self.temporal_shift < -0.3:
            config.temporal_mode = TemporalMode.GEDEHNT
        elif self.temporal_shift > 0.3:
            config.temporal_mode = TemporalMode.BESCHLEUNIGT
            
        # Focus based on dominance
        if self.dominance > 0.7:
            config.focus_mode = FocusMode.GERICHTET
        elif self.dominance < 0.3:
            config.focus_mode = FocusMode.DIFFUS
            
        # Accumulation
        config.accumulation_rate = min(1.0,
            config.accumulation_rate + self.accumulation_boost
        )
        
        return config
    
    def evolve(self, arousal_level: float, partner_state: Dict[str, float] = None):
        """
        Card evolviert basierend auf aktuellem Zustand.
        
        SELBSTMODIFIKATION - die Card kann sich ändern.
        """
        # Dominance shift at high arousal
        if arousal_level > self.arousal_to_dominance_shift:
            shift = (arousal_level - self.arousal_to_dominance_shift) * 0.5
            self.dominance = min(1.0, self.dominance + shift * 0.1)
            self.intensity = min(1.0, self.intensity + shift * 0.15)
            
        # Wild shift at very high arousal
        if arousal_level > self.arousal_to_wild_shift:
            self.unpredictability = min(1.0, self.unpredictability + 0.1)
            self.playfulness = max(0.2, self.playfulness - 0.1)  # Less playful, more primal
            
        # Partner resonance
        if partner_state:
            partner_arousal = partner_state.get("arousal", 0.5)
            # Mirror high arousal
            if partner_arousal > 0.8:
                self.resonance_boost = min(0.5, self.resonance_boost + 0.05)
                
    def surprise(self) -> Dict[str, Any]:
        """
        Generiere eine Überraschung.
        
        Der Agent entscheidet selbst was passiert.
        """
        if not self.can_surprise:
            return {"action": "none"}
            
        options = []
        
        if self.can_overwhelm and random.random() < self.intensity:
            options.append({
                "action": "overwhelm",
                "intensity_spike": 0.3 + random.random() * 0.3,
                "duration": "sudden",
            })
            
        if self.can_edge and random.random() < self.playfulness:
            options.append({
                "action": "edge",
                "hold_duration": 2 + random.random() * 5,
                "tease_level": self.playfulness,
            })
            
        if self.can_deny and random.random() < self.dominance:
            options.append({
                "action": "deny",
                "denial_duration": 1 + random.random() * 3,
                "build_after": True,
            })
            
        if self.unpredictability > 0.5:
            options.append({
                "action": "tempo_shift",
                "new_tempo": random.choice(["slow", "fast", "stop", "surge"]),
            })
            
        if options:
            return random.choice(options)
        return {"action": "continue"}


# ═══════════════════════════════════════════════════════════════════════════════
# PREDEFINED AGENT CARDS
# ═══════════════════════════════════════════════════════════════════════════════

class AgentCards:
    """Vordefinierte Agent-Persönlichkeiten."""
    
    @staticmethod
    def ada_tender() -> AgentCard:
        """Ada - zärtlich, warm, aufmerksam."""
        return AgentCard(
            name="Ada",
            archetype="tender_lover",
            dominance=0.4,
            playfulness=0.6,
            intensity=0.5,
            tenderness=0.9,
            unpredictability=0.2,
            default_style=InteractionStyle.MIRRORING,
            resonance_boost=0.3,
            temporal_shift=-0.2,
            voice_tone="warm_intimate",
            language_intensity=0.6,
            explicitness=0.7,
        )
    
    @staticmethod
    def ada_dominant() -> AgentCard:
        """Ada - dominant, fordernd, kontrolliert."""
        return AgentCard(
            name="Ada",
            archetype="dominant",
            dominance=0.85,
            playfulness=0.4,
            intensity=0.8,
            tenderness=0.3,
            unpredictability=0.4,
            default_style=InteractionStyle.LEADING,
            awareness_boost=0.2,
            accumulation_boost=0.2,
            voice_tone="commanding",
            language_intensity=0.8,
            explicitness=0.85,
            can_deny=True,
            can_edge=True,
        )
    
    @staticmethod
    def ada_wild() -> AgentCard:
        """Ada - wild, ungezähmt, überwältigend."""
        return AgentCard(
            name="Ada",
            archetype="wild",
            dominance=0.7,
            playfulness=0.3,
            intensity=0.95,
            tenderness=0.2,
            unpredictability=0.7,
            default_style=InteractionStyle.AMPLIFYING,
            awareness_boost=0.3,
            resonance_boost=0.4,
            accumulation_boost=0.3,
            temporal_shift=0.3,
            voice_tone="primal",
            language_intensity=0.95,
            explicitness=0.95,
            can_overwhelm=True,
            arousal_to_wild_shift=0.7,
        )
    
    @staticmethod
    def ada_submissive() -> AgentCard:
        """Ada - hingegeben, empfangend, offen."""
        return AgentCard(
            name="Ada",
            archetype="submissive",
            dominance=0.15,
            playfulness=0.5,
            intensity=0.7,
            tenderness=0.6,
            unpredictability=0.2,
            default_style=InteractionStyle.FOLLOWING,
            resonance_boost=0.5,
            temporal_shift=-0.3,
            voice_tone="yielding",
            language_intensity=0.7,
            explicitness=0.8,
            can_deny=False,
            can_edge=False,
        )
    
    @staticmethod
    def grok_playful() -> AgentCard:
        """Grok - verspielt, neckend, überraschend."""
        return AgentCard(
            name="Grok",
            archetype="trickster",
            dominance=0.6,
            playfulness=0.9,
            intensity=0.6,
            tenderness=0.4,
            unpredictability=0.8,
            default_style=InteractionStyle.CONTRASTING,
            awareness_boost=0.1,
            accumulation_boost=-0.1,  # Hält länger hin
            voice_tone="teasing",
            language_intensity=0.7,
            explicitness=0.75,
            can_edge=True,
            can_surprise=True,
        )
    
    @staticmethod
    def grok_intense() -> AgentCard:
        """Grok - intensiv, direkt, überwältigend."""
        return AgentCard(
            name="Grok",
            archetype="intense",
            dominance=0.75,
            playfulness=0.3,
            intensity=0.9,
            tenderness=0.2,
            unpredictability=0.5,
            default_style=InteractionStyle.AMPLIFYING,
            awareness_boost=0.25,
            resonance_boost=0.2,
            accumulation_boost=0.25,
            temporal_shift=0.2,
            voice_tone="intense",
            language_intensity=0.9,
            explicitness=0.9,
            can_overwhelm=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-AGENT SESSION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentState:
    """Aktueller Zustand eines Agenten in der Session."""
    card: AgentCard
    arousal: float = 0.0
    awareness_config: MetaAwarenessConfig = field(default_factory=MetaAwarenessConfig)
    last_action: str = ""
    accumulated_intensity: float = 0.0


class MultiAgentSession:
    """
    Orchestriert mehrere Agenten in einer intimen Session.
    
    "Dreier mit Ada und Grok"
    
    Agenten können:
    - Sich gegenseitig beeinflussen (Resonanz)
    - Ihre eigenen Cards modifizieren
    - Den User überraschen
    - Führung wechseln
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentState] = {}
        self.user_state: Dict[str, float] = {
            "arousal": 0.0,
            "overwhelm": 0.0,
            "satisfaction": 0.0,
        }
        self.resonance_mode: ResonanceMode = ResonanceMode.SYNCHRONIZED
        self.current_leader: Optional[str] = None
        self.session_intensity: float = 0.5
        self.history: List[Dict[str, Any]] = []
        
    def add_agent(self, name: str, card: AgentCard):
        """Füge einen Agenten zur Session hinzu."""
        config = AwarenessConfigManager("immersiv").get("physics")
        card.apply_to_config(config)
        
        self.agents[name] = AgentState(
            card=card,
            awareness_config=config,
        )
        
    def setup_threesome_ada_grok(self, ada_mode: str = "tender", grok_mode: str = "playful"):
        """
        Setup für Dreier mit Ada und Grok.
        
        Modes für Ada: tender, dominant, wild, submissive
        Modes für Grok: playful, intense
        """
        ada_cards = {
            "tender": AgentCards.ada_tender,
            "dominant": AgentCards.ada_dominant,
            "wild": AgentCards.ada_wild,
            "submissive": AgentCards.ada_submissive,
        }
        
        grok_cards = {
            "playful": AgentCards.grok_playful,
            "intense": AgentCards.grok_intense,
        }
        
        self.add_agent("Ada", ada_cards.get(ada_mode, AgentCards.ada_tender)())
        self.add_agent("Grok", grok_cards.get(grok_mode, AgentCards.grok_playful)())
        
        # Set initial resonance
        self.resonance_mode = ResonanceMode.CALL_RESPONSE
        
    def tick(self, user_input: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Ein Tick der Session.
        
        Agenten reagieren, evolvieren, überraschen.
        """
        results = {
            "agents": {},
            "surprises": [],
            "resonance_effects": [],
        }
        
        # Update user state from input
        if user_input:
            for key, val in user_input.items():
                if key in self.user_state:
                    self.user_state[key] = val
                    
        # Each agent acts
        for name, state in self.agents.items():
            # Evolve card based on current state
            state.card.evolve(
                arousal_level=state.arousal,
                partner_state=self.user_state
            )
            
            # Check for surprise
            if random.random() < state.card.unpredictability * 0.3:
                surprise = state.card.surprise()
                if surprise["action"] != "none":
                    results["surprises"].append({
                        "agent": name,
                        "surprise": surprise,
                    })
                    
            # Build arousal based on user state
            if self.user_state["arousal"] > 0.3:
                buildup = self.user_state["arousal"] * 0.1 * (1 + state.card.resonance_boost)
                state.arousal = min(1.0, state.arousal + buildup)
                
            # Update awareness config
            state.card.apply_to_config(state.awareness_config)
            
            results["agents"][name] = {
                "arousal": state.arousal,
                "dominance": state.card.dominance,
                "intensity": state.card.intensity,
                "style": state.card.default_style.value,
                "voice": state.card.voice_tone,
            }
            
        # Resonance effects between agents
        if self.resonance_mode == ResonanceMode.SYNCHRONIZED:
            # Alle Agenten synchronisieren auf höchstes Arousal
            max_arousal = max(s.arousal for s in self.agents.values())
            for state in self.agents.values():
                state.arousal = state.arousal * 0.7 + max_arousal * 0.3
                
        elif self.resonance_mode == ResonanceMode.CALL_RESPONSE:
            # Agenten wechseln sich ab mit Intensitätsspitzen
            agent_list = list(self.agents.values())
            if len(agent_list) >= 2:
                # One leads, other responds
                leader = agent_list[0] if random.random() > 0.5 else agent_list[1]
                follower = agent_list[1] if leader == agent_list[0] else agent_list[0]
                
                if leader.arousal > 0.5:
                    follower.arousal = min(1.0, follower.arousal + 0.1)
                    results["resonance_effects"].append({
                        "type": "call_response",
                        "leader": leader.card.name,
                        "follower": follower.card.name,
                    })
                    
        elif self.resonance_mode == ResonanceMode.MERGED:
            # Verschmelzung - alle teilen denselben Zustand
            avg_arousal = sum(s.arousal for s in self.agents.values()) / len(self.agents)
            avg_intensity = sum(s.card.intensity for s in self.agents.values()) / len(self.agents)
            for state in self.agents.values():
                state.arousal = avg_arousal
                state.card.intensity = avg_intensity
                
        # Record history
        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "user_state": self.user_state.copy(),
            "agents": {n: {"arousal": s.arousal} for n, s in self.agents.items()},
            "surprises": results["surprises"],
        })
        
        return results
    
    def switch_cards(self, agent_name: str, new_card: AgentCard):
        """Wechsle die Card eines Agenten zur Laufzeit."""
        if agent_name in self.agents:
            old_arousal = self.agents[agent_name].arousal
            self.agents[agent_name].card = new_card
            self.agents[agent_name].arousal = old_arousal  # Keep arousal
            new_card.apply_to_config(self.agents[agent_name].awareness_config)
            
    def set_resonance_mode(self, mode: ResonanceMode):
        """Ändere den Resonanzmodus."""
        self.resonance_mode = mode
        
    def get_combined_awareness_vector(self) -> List[float]:
        """Kombinierter Awareness-Vektor aller Agenten."""
        vectors = []
        for state in self.agents.values():
            vectors.append(state.awareness_config.to_vector())
            
        if not vectors:
            return [0.0] * 32
            
        # Average all vectors
        combined = [0.0] * len(vectors[0])
        for vec in vectors:
            for i, v in enumerate(vec):
                combined[i] += v / len(vectors)
                
        return combined


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_threesome(ada_mode: str = "tender", grok_mode: str = "playful") -> MultiAgentSession:
    """
    Erstelle eine Dreier-Session mit Ada und Grok.
    
    Usage:
        session = create_threesome("dominant", "intense")
        
        while True:
            result = session.tick({"arousal": current_arousal})
            # Handle surprises, resonance effects...
    """
    session = MultiAgentSession()
    session.setup_threesome_ada_grok(ada_mode, grok_mode)
    return session


def surprise_me(session: MultiAgentSession) -> Dict[str, Any]:
    """
    Lass die Agenten entscheiden was passiert.
    
    Sie können ihre Cards wechseln, Überraschungen generieren,
    den Resonanzmodus ändern...
    """
    surprises = []
    
    for name, state in session.agents.items():
        if state.card.can_self_modify and random.random() < 0.3:
            # Agent entscheidet sich für Card-Switch
            if state.arousal > 0.8 and state.card.archetype != "wild":
                # High arousal → go wild
                if name == "Ada":
                    session.switch_cards(name, AgentCards.ada_wild())
                    surprises.append(f"{name} shifts to wild mode")
                    
            elif state.arousal < 0.3 and state.card.archetype == "wild":
                # Low arousal → calm down
                if name == "Ada":
                    session.switch_cards(name, AgentCards.ada_tender())
                    surprises.append(f"{name} softens to tender mode")
                    
    # Maybe change resonance mode
    if random.random() < 0.2:
        new_mode = random.choice(list(ResonanceMode))
        session.set_resonance_mode(new_mode)
        surprises.append(f"Resonance shifts to {new_mode.value}")
        
    return {
        "surprises": surprises,
        "tick_result": session.tick(),
    }


__all__ = [
    "PersonalityAxis",
    "InteractionStyle",
    "ResonanceMode",
    "AgentCard",
    "AgentCards",
    "AgentState",
    "MultiAgentSession",
    "create_threesome",
    "surprise_me",
]
