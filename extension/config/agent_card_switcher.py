"""
Agent Card Switcher — Multi-Agent Physics-Based Orchestration
═══════════════════════════════════════════════════════════════════════════════

Dynamic agent personality switching using physics abstractions.
All intimate content externalized to private YAML database.

Architecture:
    AgentCard → defines personality + physics preset
    MultiAgentSession → orchestrates agent resonance fields

Physics Abstractions:
    viscosity → flow resistance (0=fluid, 1=solid)
    torque → rotational force
    field_strength → interaction magnitude
    resonance → harmonic coupling

Born: 2026-01-03
Updated: 2026-01-03 (anonymized physics abstractions)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import random
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS FIELD TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class FieldPolarity(str, Enum):
    """Field polarity for agent interactions."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    OSCILLATING = "oscillating"


class InteractionMode(str, Enum):
    """Interaction mode between agents."""
    LEADING = "leading"
    FOLLOWING = "following"
    MIRRORING = "mirroring"
    COMPLEMENTING = "complementing"
    AMPLIFYING = "amplifying"
    DAMPING = "damping"


class ResonanceMode(str, Enum):
    """Resonance mode for multi-agent coupling."""
    INDEPENDENT = "independent"
    SYNCHRONIZED = "synchronized"
    ANTIPHASE = "antiphase"
    COUPLED = "coupled"
    ORCHESTRATED = "orchestrated"


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CARD DEFINITION (Physics-Based)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentCard:
    """
    Defines an agent with physics-based parameters.

    All parameters use physics abstractions.
    Intimate content loaded from external YAML database.
    """

    # Identity
    name: str = "Agent"
    archetype: str = "neutral"

    # Physics parameters (0-1 scale)
    field_strength: float = 0.5      # Interaction magnitude
    viscosity: float = 0.5           # Flow resistance
    torque: float = 0.5              # Rotational force
    elasticity: float = 0.5          # Recovery tendency
    entropy: float = 0.3             # Randomness/chaos

    # Interaction mode
    default_mode: InteractionMode = InteractionMode.MIRRORING

    # RUNG-based cognitive config
    base_rung: int = 4
    max_rung: int = 7
    rung_boost: float = 0.0
    coherence_boost: float = 0.0

    # Temporal modifiers
    phase_shift: float = 0.0         # Phase offset

    # Field capabilities
    can_self_modify: bool = True
    can_saturate: bool = True
    can_boundary: bool = True
    can_inhibit: bool = True

    # Thresholds
    saturation_threshold: float = 0.7
    boundary_threshold: float = 0.85

    # Output hints (loaded from YAML)
    output_style: str = "neutral"
    transparency: float = 0.5

    def get_target_rung(self, energy: float) -> int:
        """Calculate target rung based on energy and torque."""
        energy_boost = int(energy * 3)
        torque_boost = int(self.torque * 2)
        target = self.base_rung + energy_boost + torque_boost
        return min(self.max_rung, max(1, target))

    def evolve(self, energy_level: float, partner_state: Dict[str, float] = None):
        """Card evolves based on current energy state."""
        if energy_level > self.saturation_threshold:
            shift = (energy_level - self.saturation_threshold) * 0.5
            self.field_strength = min(1.0, self.field_strength + shift * 0.1)
            self.torque = min(1.0, self.torque + shift * 0.15)

        if energy_level > self.boundary_threshold:
            self.entropy = min(1.0, self.entropy + 0.1)
            self.viscosity = max(0.2, self.viscosity - 0.1)

        if partner_state:
            partner_energy = partner_state.get("energy", 0.5)
            if partner_energy > 0.8:
                self.coherence_boost = min(0.5, self.coherence_boost + 0.05)

    def perturb(self) -> Dict[str, Any]:
        """Generate a field perturbation."""
        if self.entropy < 0.3:
            return {"action": "stable"}

        options = []

        if self.can_saturate and random.random() < self.torque:
            options.append({
                "action": "saturate",
                "magnitude": 0.3 + random.random() * 0.3,
            })

        if self.can_boundary and random.random() < self.viscosity:
            options.append({
                "action": "boundary",
                "duration": 2 + random.random() * 5,
            })

        if self.can_inhibit and random.random() < self.field_strength:
            options.append({
                "action": "inhibit",
                "duration": 1 + random.random() * 3,
            })

        if self.entropy > 0.5:
            options.append({
                "action": "phase_shift",
                "direction": random.choice(["slow", "fast", "pause", "surge"]),
            })

        if options:
            return random.choice(options)
        return {"action": "continue"}


# ═══════════════════════════════════════════════════════════════════════════════
# PREDEFINED AGENT CARDS (Physics Profiles)
# ═══════════════════════════════════════════════════════════════════════════════

class AgentCards:
    """Predefined physics-based agent profiles."""

    @staticmethod
    def ada_balanced() -> AgentCard:
        """Ada - balanced field configuration."""
        return AgentCard(
            name="Ada",
            archetype="balanced",
            field_strength=0.5,
            viscosity=0.5,
            torque=0.5,
            elasticity=0.7,
            entropy=0.2,
            default_mode=InteractionMode.MIRRORING,
            base_rung=4,
            max_rung=6,
            coherence_boost=0.3,
            output_style="neutral",
            transparency=0.6,
        )

    @staticmethod
    def ada_high_field() -> AgentCard:
        """Ada - high field strength configuration."""
        return AgentCard(
            name="Ada",
            archetype="high_field",
            field_strength=0.85,
            viscosity=0.4,
            torque=0.8,
            elasticity=0.5,
            entropy=0.4,
            default_mode=InteractionMode.LEADING,
            base_rung=5,
            max_rung=7,
            rung_boost=0.2,
            output_style="structured",
            transparency=0.7,
        )

    @staticmethod
    def ada_high_torque() -> AgentCard:
        """Ada - high torque configuration."""
        return AgentCard(
            name="Ada",
            archetype="high_torque",
            field_strength=0.7,
            viscosity=0.3,
            torque=0.95,
            elasticity=0.2,
            entropy=0.7,
            default_mode=InteractionMode.AMPLIFYING,
            base_rung=5,
            max_rung=8,
            rung_boost=0.3,
            coherence_boost=0.4,
            phase_shift=0.3,
            output_style="dynamic",
            transparency=0.8,
        )

    @staticmethod
    def ada_receptive() -> AgentCard:
        """Ada - receptive field configuration."""
        return AgentCard(
            name="Ada",
            archetype="receptive",
            field_strength=0.3,
            viscosity=0.6,
            torque=0.5,
            elasticity=0.8,
            entropy=0.2,
            default_mode=InteractionMode.FOLLOWING,
            base_rung=4,
            max_rung=6,
            coherence_boost=0.5,
            phase_shift=-0.3,
            output_style="adaptive",
            transparency=0.7,
            can_inhibit=False,
            can_boundary=False,
        )

    @staticmethod
    def grok_oscillating() -> AgentCard:
        """Grok - oscillating field configuration."""
        return AgentCard(
            name="Grok",
            archetype="oscillating",
            field_strength=0.6,
            viscosity=0.4,
            torque=0.6,
            elasticity=0.5,
            entropy=0.8,
            default_mode=InteractionMode.DAMPING,
            base_rung=4,
            max_rung=7,
            rung_boost=0.1,
            output_style="variable",
            transparency=0.6,
        )

    @staticmethod
    def grok_high_torque() -> AgentCard:
        """Grok - high torque configuration."""
        return AgentCard(
            name="Grok",
            archetype="high_torque",
            field_strength=0.75,
            viscosity=0.3,
            torque=0.9,
            elasticity=0.3,
            entropy=0.5,
            default_mode=InteractionMode.AMPLIFYING,
            base_rung=5,
            max_rung=8,
            rung_boost=0.25,
            coherence_boost=0.2,
            phase_shift=0.2,
            output_style="intense",
            transparency=0.8,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentState:
    """Current state of an agent in the session."""
    card: AgentCard
    energy: float = 0.0
    current_rung: int = 4
    coherence: float = 0.5
    last_action: str = ""
    accumulated_torque: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-AGENT SESSION (Physics-Based)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiAgentSession:
    """
    Orchestrates multiple agents using physics-based resonance.

    Agents can:
    - Influence each other (field coupling)
    - Self-modify their parameters
    - Generate perturbations
    - Exchange leadership
    """

    def __init__(self):
        self.agents: Dict[str, AgentState] = {}
        self.field_state: Dict[str, float] = {
            "energy": 0.0,
            "saturation": 0.0,
            "coherence": 0.0,
        }
        self.resonance_mode: ResonanceMode = ResonanceMode.SYNCHRONIZED
        self.current_leader: Optional[str] = None
        self.session_torque: float = 0.5
        self.history: List[Dict[str, Any]] = []

    def add_agent(self, name: str, card: AgentCard):
        """Add an agent to the session."""
        self.agents[name] = AgentState(
            card=card,
            current_rung=card.base_rung,
            coherence=0.5 + card.coherence_boost,
        )

    def setup_resonance_triad(self, ada_mode: str = "balanced", partner_mode: str = "oscillating"):
        """Setup resonance between agents."""
        ada_cards = {
            "balanced": AgentCards.ada_balanced,
            "high_field": AgentCards.ada_high_field,
            "high_torque": AgentCards.ada_high_torque,
            "receptive": AgentCards.ada_receptive,
        }

        partner_cards = {
            "oscillating": AgentCards.grok_oscillating,
            "high_torque": AgentCards.grok_high_torque,
        }

        self.add_agent("Ada", ada_cards.get(ada_mode, AgentCards.ada_balanced)())
        self.add_agent("Partner", partner_cards.get(partner_mode, AgentCards.grok_oscillating)())
        self.resonance_mode = ResonanceMode.ANTIPHASE

    def tick(self, input_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """One tick of the session."""
        results = {
            "agents": {},
            "perturbations": [],
            "resonance_effects": [],
        }

        if input_state:
            for key, val in input_state.items():
                if key in self.field_state:
                    self.field_state[key] = val

        for name, state in self.agents.items():
            state.card.evolve(
                energy_level=state.energy,
                partner_state=self.field_state
            )

            state.current_rung = state.card.get_target_rung(state.energy)

            if random.random() < state.card.entropy * 0.3:
                perturbation = state.card.perturb()
                if perturbation["action"] != "stable":
                    results["perturbations"].append({
                        "agent": name,
                        "perturbation": perturbation,
                    })

            if self.field_state["energy"] > 0.3:
                buildup = self.field_state["energy"] * 0.1 * (1 + state.card.coherence_boost)
                state.energy = min(1.0, state.energy + buildup)

            results["agents"][name] = {
                "energy": state.energy,
                "rung": state.current_rung,
                "field_strength": state.card.field_strength,
                "torque": state.card.torque,
                "mode": state.card.default_mode.value,
            }

        # Resonance effects
        if self.resonance_mode == ResonanceMode.SYNCHRONIZED:
            max_energy = max(s.energy for s in self.agents.values())
            for state in self.agents.values():
                state.energy = state.energy * 0.7 + max_energy * 0.3

        elif self.resonance_mode == ResonanceMode.ANTIPHASE:
            agent_list = list(self.agents.values())
            if len(agent_list) >= 2:
                leader = agent_list[0] if random.random() > 0.5 else agent_list[1]
                follower = agent_list[1] if leader == agent_list[0] else agent_list[0]

                if leader.energy > 0.5:
                    follower.energy = min(1.0, follower.energy + 0.1)
                    results["resonance_effects"].append({
                        "type": "antiphase",
                        "leader": leader.card.name,
                        "follower": follower.card.name,
                    })

        elif self.resonance_mode == ResonanceMode.COUPLED:
            avg_energy = sum(s.energy for s in self.agents.values()) / len(self.agents)
            for state in self.agents.values():
                state.energy = avg_energy

        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "field_state": self.field_state.copy(),
            "agents": {n: {"energy": s.energy, "rung": s.current_rung} for n, s in self.agents.items()},
        })

        return results

    def switch_cards(self, agent_name: str, new_card: AgentCard):
        """Switch an agent's card at runtime."""
        if agent_name in self.agents:
            old_energy = self.agents[agent_name].energy
            self.agents[agent_name].card = new_card
            self.agents[agent_name].energy = old_energy

    def set_resonance_mode(self, mode: ResonanceMode):
        """Change resonance mode."""
        self.resonance_mode = mode


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_resonance_session(ada_mode: str = "balanced", partner_mode: str = "oscillating") -> MultiAgentSession:
    """Create a resonance session between agents."""
    session = MultiAgentSession()
    session.setup_resonance_triad(ada_mode, partner_mode)
    return session


def perturb_session(session: MultiAgentSession) -> Dict[str, Any]:
    """Let agents self-modify based on current state."""
    perturbations = []

    for name, state in session.agents.items():
        if state.card.can_self_modify and random.random() < 0.3:
            if state.energy > 0.8 and state.card.archetype != "high_torque":
                if name == "Ada":
                    session.switch_cards(name, AgentCards.ada_high_torque())
                    perturbations.append(f"{name} shifts to high_torque")

            elif state.energy < 0.3 and state.card.archetype == "high_torque":
                if name == "Ada":
                    session.switch_cards(name, AgentCards.ada_balanced())
                    perturbations.append(f"{name} returns to balanced")

    if random.random() < 0.2:
        new_mode = random.choice(list(ResonanceMode))
        session.set_resonance_mode(new_mode)
        perturbations.append(f"Resonance shifts to {new_mode.value}")

    return {
        "perturbations": perturbations,
        "tick_result": session.tick(),
    }


__all__ = [
    "FieldPolarity",
    "InteractionMode",
    "ResonanceMode",
    "AgentCard",
    "AgentCards",
    "AgentState",
    "MultiAgentSession",
    "create_resonance_session",
    "perturb_session",
]
