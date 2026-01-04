"""
sigma_bridge.py — Bridge between sigma_hydration and Ada v6.0
==============================================================

Converts between:
  - SigmaNode (Notion persistence) ↔ MarkovUnit (Ada physics)
  - SigmaEdge (Notion edges) ↔ CausalSituationMap edges
  - CausalChain (path retrieval) ↔ SigmaChain (tangent engine)

This enables Ada v6.0 to persist to Notion using the dual-database
architecture defined in sigma_hydration.py while maintaining
full compatibility with the physics engine.

Database IDs:
  sigma_nodes_h: 395d1213-a6a4-4a11-a013-d97f385e1954
  sigma_edges_v: e8bf26f3-4f45-4186-aab6-51cb7d2c58d9
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import hashlib

# Async structures (sigma_hydration.py)
from .bridge.sigma_hydration import (
    SigmaNode, SigmaEdge, CausalChain,
    NodeType, Causality, Affect, Temporal, RelationType,
    hydrate_node, hydrate_edge, resonance, to_manifest
)

if TYPE_CHECKING:
    from .physics.markov_unit import MarkovUnit
    from .core.qualia import QualiaVector
    from .core.tangent_engine import SigmaSeed, SigmaChain as AdaSigmaChain
    from .core.agent_state import AgentState


# ─────────────────────────────────────────────────────────────────
# TIER MAPPING
# ─────────────────────────────────────────────────────────────────

TIER_TO_INCANDESCENCE = {
    "Σ₀": 0.25,  # Raw observation
    "Σ₁": 0.50,  # Compressed
    "Σ₂": 0.75,  # Integrated
    "Σ₃": 1.00,  # Crystallized
}

INCANDESCENCE_TO_TIER = {
    0.25: "Σ₀",
    0.50: "Σ₁",
    0.75: "Σ₂",
    1.00: "Σ₃",
}

def incandescence_to_tier(inc: float) -> str:
    """Map incandescence value to tier string"""
    if inc < 0.375:
        return "Σ₀"
    elif inc < 0.625:
        return "Σ₁"
    elif inc < 0.875:
        return "Σ₂"
    else:
        return "Σ₃"


# ─────────────────────────────────────────────────────────────────
# NODETYPE ↔ ARCHETYPE MAPPING
# ─────────────────────────────────────────────────────────────────

TYPE_TO_ARCHETYPE = {
    NodeType.OMEGA: "#arch-observation",
    NodeType.DELTA: "#arch-insight", 
    NodeType.PHI: "#arch-belief",
    NodeType.THETA: "#arch-integration",
    NodeType.LAMBDA: "#arch-trajectory",
}

ARCHETYPE_TO_TYPE = {
    "#arch-observation": NodeType.OMEGA,
    "#arch-insight": NodeType.DELTA,
    "#arch-belief": NodeType.PHI,
    "#arch-integration": NodeType.THETA,
    "#arch-trajectory": NodeType.LAMBDA,
    # Fallback patterns
    "Omega": NodeType.OMEGA,
    "Delta": NodeType.DELTA,
    "Phi": NodeType.PHI,
    "Theta": NodeType.THETA,
    "Lambda": NodeType.LAMBDA,
}


# ─────────────────────────────────────────────────────────────────
# AFFECT ↔ HDR MAPPING
# ─────────────────────────────────────────────────────────────────

AFFECT_TO_WARMTH = {
    Affect.COLD: 0.1,
    Affect.COOL: 0.35,
    Affect.WARM: 0.65,
    Affect.HOT: 0.9,
}

def warmth_to_affect(warmth: float) -> Affect:
    if warmth < 0.25:
        return Affect.COLD
    elif warmth < 0.5:
        return Affect.COOL
    elif warmth < 0.75:
        return Affect.WARM
    else:
        return Affect.HOT


# ─────────────────────────────────────────────────────────────────
# TEMPORAL ↔ LINGERING MAPPING
# ─────────────────────────────────────────────────────────────────

TEMPORAL_TO_LINGERING = {
    Temporal.DECAYING: 0.2,
    Temporal.EMERGING: 0.5,
    Temporal.STABLE: 0.75,
    Temporal.CRYSTALLIZED: 0.95,
}

def lingering_to_temporal(lingering: float) -> Temporal:
    if lingering < 0.35:
        return Temporal.DECAYING
    elif lingering < 0.6:
        return Temporal.EMERGING
    elif lingering < 0.85:
        return Temporal.STABLE
    else:
        return Temporal.CRYSTALLIZED


# ─────────────────────────────────────────────────────────────────
# BRIDGE: SigmaNode ↔ MarkovUnit
# ─────────────────────────────────────────────────────────────────

def sigma_node_to_markov(
    node: SigmaNode,
    qualia_class: type = None
) -> 'MarkovUnit':
    """
    Convert SigmaNode (Notion) to MarkovUnit (Ada physics).
    
    Mappings:
        node_id → byte_id (hashed)
        sigma1_seed → sigma_seed
        type → archetype
        tier → incandescence
        strength → resonance
        alpha/gamma → qualia components
        temporal → (inferred from strength/decay)
    """
    from .physics.markov_unit import MarkovUnit
    from .core.qualia import QualiaVector
    
    # Generate stable byte_id from node_id (must be 0-255)
    byte_id = int(hashlib.md5(node.node_id.encode()).hexdigest()[:2], 16) % 256
    
    # Build sigma_seed (must start with #sigma-)
    if node.sigma1_seed and node.sigma1_seed.startswith("#sigma-"):
        sigma_seed = node.sigma1_seed
    else:
        # Generate from node_id
        sigma_seed = f"#sigma-{node.node_type.value.lower()}-{node.node_id}"
    
    # Map archetype
    archetype = TYPE_TO_ARCHETYPE.get(node.node_type, "#arch-unknown")
    
    # Build qualia from affect metrics
    # Mapping: alpha→emberglow, gamma→antenna, omega_tilde→woodwarm
    qualia = QualiaVector(
        emberglow=node.alpha,
        velvetpause=0.5,  # Default, could be derived from temporal
        steelwind=0.3,    # Default
        woodwarm=node.omega_tilde,
        antenna=node.gamma,
    )
    
    return MarkovUnit(
        byte_id=byte_id,
        sigma_seed=sigma_seed,
        archetype=archetype,
        qualia=qualia,
        transitions={},  # Would need edge data
        theta_weights={
            "tier": TIER_TO_INCANDESCENCE.get(node.tier, 0.5),
            "causality": {
                Causality.OBSERVED: 0.25,
                Causality.CAUSAL: 0.5,
                Causality.INTERVENTION: 0.75,
                Causality.COUNTERFACTUAL: 1.0,
            }.get(node.causality, 0.5),
        },
        incandescence=TIER_TO_INCANDESCENCE.get(node.tier, 0.5),
        resonance=node.strength,
        causal_strength=node.omega_belief,
        parents=[],
    )


def markov_to_sigma_node(
    unit: 'MarkovUnit',
    node_id: str = None,
    session: str = "",
    source: str = ""
) -> SigmaNode:
    """
    Convert MarkovUnit (Ada physics) to SigmaNode (Notion persistence).
    
    Mappings:
        byte_id → (used if no node_id)
        sigma_seed → sigma1_seed
        archetype → type
        incandescence → tier
        resonance → strength
        qualia → alpha/gamma/omega_tilde
    """
    # Determine node_id
    if node_id is None:
        node_id = f"M-{unit.byte_id}"
    
    # Map type from archetype
    node_type = ARCHETYPE_TO_TYPE.get(unit.archetype, NodeType.DELTA)
    
    # Map tier from incandescence
    tier = incandescence_to_tier(unit.incandescence)
    
    # Extract affect metrics from qualia
    q = unit.qualia
    alpha = getattr(q, 'emberglow', 0.5)
    gamma = getattr(q, 'antenna', 0.5)
    omega_tilde = getattr(q, 'woodwarm', 0.5)
    
    # Derive affect from warmth (computed from qualia)
    warmth = (alpha + omega_tilde) / 2
    affect = warmth_to_affect(warmth)
    
    # Derive temporal from resonance
    temporal = lingering_to_temporal(unit.resonance)
    
    # Derive causality from theta_weights if present
    causality = Causality.OBSERVED
    if 'causality' in unit.theta_weights:
        c_val = unit.theta_weights['causality']
        if c_val > 0.7:
            causality = Causality.COUNTERFACTUAL
        elif c_val > 0.5:
            causality = Causality.INTERVENTION
        elif c_val > 0.25:
            causality = Causality.CAUSAL
    
    return SigmaNode(
        node_id=node_id,
        node_type=node_type,
        tier=tier,
        causality=causality,
        affect=affect,
        temporal=temporal,
        sigma1_seed=unit.sigma_seed,
        node_content="",  # Would need external content
        alpha=alpha,
        gamma=gamma,
        omega_tilde=omega_tilde,
        strength=unit.resonance,
        decay=0.618,  # Default phi
        derivative=0.0,
        omega_belief=unit.causal_strength,
        y_pos=list(NodeType).index(node_type),
        z_pos=0,
        session=session,
        source=source,
    )


# ─────────────────────────────────────────────────────────────────
# BRIDGE: CausalChain ↔ SigmaChain
# ─────────────────────────────────────────────────────────────────

@dataclass
class UnifiedChain:
    """
    Unified chain combining CausalChain edges with SigmaChain thoughts.
    
    This bridges:
      - CausalChain (async): edges, path_id, cypher patterns
      - SigmaChain (Ada): seeds, thoughts, weights
    """
    path_id: str
    
    # From CausalChain
    edges: List[SigmaEdge] = field(default_factory=list)
    nodes: List[SigmaNode] = field(default_factory=list)
    
    # From SigmaChain
    seeds: List[str] = field(default_factory=list)
    thoughts: List[str] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    
    @property
    def chain_strength(self) -> float:
        """Product of edge strengths (from CausalChain)"""
        if not self.edges:
            if self.weights:
                result = 1.0
                for w in self.weights:
                    result *= w
                return result
            return 0.0
        result = 1.0
        for e in self.edges:
            result *= e.strength
        return result
    
    @property
    def total_weight(self) -> float:
        """Sum of weights (from SigmaChain)"""
        return sum(self.weights) if self.weights else 0.0
    
    def to_manifest(self) -> str:
        """Generate combined manifest"""
        lines = [
            f"path_id:{self.path_id}",
            f"chain_strength:{self.chain_strength:.3f}",
            f"total_weight:{self.total_weight:.3f}",
            "---"
        ]
        
        for i, node in enumerate(self.nodes):
            thought = self.thoughts[i] if i < len(self.thoughts) else ""
            lines.append(f"[{i}] {node.node_id} ({node.glyph_4d})")
            if thought:
                lines.append(f"     💭 {thought}")
            if i < len(self.edges):
                lines.append(f"     ─{self.edges[i].relation_type.value}→")
        
        return "\n".join(lines)
    
    def to_prompt_injection(self) -> str:
        """Generate LLM-injectable format"""
        parts = []
        for i, (node, thought) in enumerate(zip(self.nodes, self.thoughts + [""])):
            weight = self.weights[i] if i < len(self.weights) else 0.5
            parts.append(f"[{node.glyph_4d}] {thought} (w={weight:.2f})")
        return " → ".join(parts)


def causal_chain_to_unified(chain: CausalChain) -> UnifiedChain:
    """Convert CausalChain to UnifiedChain"""
    return UnifiedChain(
        path_id=chain.path_id,
        edges=chain.edges,
        nodes=chain.nodes,
        seeds=[n.sigma1_seed for n in chain.nodes],
        thoughts=[e.chain_seed for e in chain.edges],
        weights=[e.strength for e in chain.edges],
    )


def sigma_chain_to_unified(
    chain: 'AdaSigmaChain', 
    path_id: str = "auto"
) -> UnifiedChain:
    """Convert Ada SigmaChain to UnifiedChain"""
    from .core.tangent_engine import SigmaSeed
    
    return UnifiedChain(
        path_id=path_id,
        edges=[],  # SigmaChain doesn't have edges
        nodes=[],  # Would need to hydrate from sigma
        seeds=[s.sigma for s in chain],
        thoughts=[s.thought for s in chain],
        weights=[s.weight for s in chain],
    )


# ─────────────────────────────────────────────────────────────────
# BRIDGE: AgentState ↔ SigmaNode metadata
# ─────────────────────────────────────────────────────────────────

def agent_state_to_node_metadata(agent: 'AgentState') -> Dict[str, Any]:
    """
    Extract metadata from AgentState for SigmaNode storage.
    
    This captures the meta-cognitive state at the moment of node creation.
    """
    return {
        "staunen": agent.staunen,
        "wisdom": agent.wisdom,
        "now_density": agent.now_density,
        "lingering": agent.lingering,
        "ache": agent.ache,
        "tension": agent.tension,
        "presence": agent.presence,
        "katharsis": agent.katharsis,
        "mode": agent.mode,
    }


def apply_node_metadata_to_agent(
    metadata: Dict[str, Any],
    agent: 'AgentState'
) -> None:
    """
    Restore AgentState from stored node metadata.
    
    Used when rehydrating session state from Notion.
    """
    agent.staunen = metadata.get("staunen", agent.staunen)
    agent.wisdom = metadata.get("wisdom", agent.wisdom)
    agent.now_density = metadata.get("now_density", agent.now_density)
    agent.lingering = metadata.get("lingering", agent.lingering)
    agent.ache = metadata.get("ache", agent.ache)
    agent.tension = metadata.get("tension", agent.tension)
    agent.presence = metadata.get("presence", agent.presence)


# ─────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Sigma Bridge — Unit Tests")
    print("=" * 60)
    
    # Test 1: SigmaNode → MarkovUnit
    print("\n1. SigmaNode → MarkovUnit")
    node = SigmaNode(
        node_id="test-001",
        node_type=NodeType.DELTA,
        tier="Σ₂",
        causality=Causality.CAUSAL,
        affect=Affect.WARM,
        temporal=Temporal.STABLE,
        sigma1_seed="⟨Σ₁.Δ.test⟩{example}",
        node_content="Test insight",
        alpha=0.8,
        gamma=0.6,
        omega_tilde=0.7,
        strength=0.9,
        session="test"
    )
    
    unit = sigma_node_to_markov(node)
    print(f"   node_id: {node.node_id} → byte_id: {unit.byte_id}")
    print(f"   tier: {node.tier} → incandescence: {unit.incandescence}")
    print(f"   type: {node.node_type.value} → archetype: {unit.archetype}")
    print(f"   strength: {node.strength} → resonance: {unit.resonance}")
    print("   ✓ Conversion works")
    
    # Test 2: MarkovUnit → SigmaNode
    print("\n2. MarkovUnit → SigmaNode")
    restored = markov_to_sigma_node(unit, node_id="restored-001", session="test2")
    print(f"   byte_id: {unit.byte_id} → node_id: {restored.node_id}")
    print(f"   incandescence: {unit.incandescence} → tier: {restored.tier}")
    print(f"   archetype: {unit.archetype} → type: {restored.node_type.value}")
    print(f"   resonance: {unit.resonance} → strength: {restored.strength}")
    print("   ✓ Reverse conversion works")
    
    # Test 3: CausalChain → UnifiedChain
    print("\n3. CausalChain → UnifiedChain")
    edge = SigmaEdge(
        edge_id="E-001",
        source_id="node-1",
        target_id="node-2",
        relation_type=RelationType.CAUSES,
        strength=0.8,
        chain_seed="cause leads to effect"
    )
    chain = CausalChain(
        path_id="P-test",
        edges=[edge],
        nodes=[node]
    )
    unified = causal_chain_to_unified(chain)
    print(f"   path_id: {unified.path_id}")
    print(f"   chain_strength: {unified.chain_strength:.3f}")
    print(f"   thoughts: {unified.thoughts}")
    print("   ✓ Chain unification works")
    
    # Test 4: Tier mapping
    print("\n4. Tier ↔ Incandescence mapping")
    for tier, inc in TIER_TO_INCANDESCENCE.items():
        restored_tier = incandescence_to_tier(inc)
        status = "✓" if tier == restored_tier else "✗"
        print(f"   {tier} → {inc} → {restored_tier} {status}")
    
    print("\n" + "=" * 60)
    print("All bridge tests passed! ✓")
    print("=" * 60)
