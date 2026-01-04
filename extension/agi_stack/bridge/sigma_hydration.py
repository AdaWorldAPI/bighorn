"""
sigma_hydration.py — Compression/Hydration for Dual-Database Graph Architecture

Databases:
  sigma_nodes_h (horizontal): 395d1213-a6a4-4a11-a013-d97f385e1954
  sigma_edges_v (vertical):   e8bf26f3-4f45-4186-aab6-51cb7d2c58d9

Core insight: manifest = row.items() joined. No separate storage.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import math

# CognitiveAtom v2.1 integration
try:
    from .core.atom.cognitive_atom import CognitiveAtom
    COGNITIVE_ATOM_AVAILABLE = True
except ImportError:
    COGNITIVE_ATOM_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
# 4D COORDINATE SYSTEM
# ─────────────────────────────────────────────────────────────────

class NodeType(Enum):
    OMEGA = "Ω"      # Observations
    DELTA = "Δ"      # Insights  
    PHI = "Φ"        # Beliefs
    THETA = "Θ"      # Integrations
    LAMBDA = "Λ"     # Trajectories

class Causality(Enum):
    OBSERVED = "observed"
    CAUSAL = "causal"
    INTERVENTION = "intervention"
    COUNTERFACTUAL = "counterfactual"

class Affect(Enum):
    COLD = "cold"
    COOL = "cool"
    WARM = "warm"
    HOT = "hot"

class Temporal(Enum):
    DECAYING = "decaying"
    EMERGING = "emerging"
    STABLE = "stable"
    CRYSTALLIZED = "crystallized"

class RelationType(Enum):
    BECOMES = "BECOMES"
    CAUSES = "CAUSES"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    REFINES = "REFINES"
    GROUNDS = "GROUNDS"
    ABSTRACTS = "ABSTRACTS"

# ─────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────

@dataclass
class SigmaNode:
    node_id: str
    node_type: NodeType
    tier: str  # Σ₀, Σ₁, Σ₂, Σ₃
    
    # 4D coordinates
    causality: Causality = Causality.OBSERVED
    affect: Affect = Affect.WARM
    temporal: Temporal = Temporal.EMERGING
    
    # Content
    sigma1_seed: str = ""
    node_content: str = ""
    
    # Affect metrics
    alpha: float = 0.5      # valence
    gamma: float = 0.5      # energy
    omega_tilde: float = 0.1  # uncertainty
    
    # Temporal dynamics
    strength: float = 1.0
    decay: float = 0.618    # phi
    derivative: float = 0.0
    
    # Belief
    omega_belief: float = 0.5
    
    # Position in 3D reconstruction
    y_pos: int = 0
    z_pos: int = 0
    
    # Metadata
    session: str = ""
    source: str = ""
    notion_id: Optional[str] = None
    
    @property
    def glyph_4d(self) -> str:
        """Generate hashtag glyph from 4D coordinates"""
        return f"#{self.node_type.value}.{self.causality.value}.{self.affect.value}.{self.temporal.value}"
    
    @property
    def cypher_locus(self) -> str:
        """Generate searchable cypher pattern"""
        return (
            f"MATCH (n:{self.node_type.value}) WHERE "
            f"n.tags CONTAINS '#κ/{self.causality.value}' AND "
            f"n.tags CONTAINS '#α/{self.affect.value}' AND "
            f"n.tags CONTAINS '#τ/{self.temporal.value}'"
        )
    
    def apply_decay(self, sessions_elapsed: int = 1) -> None:
        """Lazy decay on hydration"""
        self.strength *= (self.decay ** sessions_elapsed)
        if self.strength < 0.1:
            self.temporal = Temporal.DECAYING
    
    def to_notion_properties(self) -> Dict[str, Any]:
        """Convert to Notion row properties"""
        return {
            "node_id": self.node_id,
            "type": self.node_type.value,
            "tier": self.tier,
            "glyph_4d": self.glyph_4d,
            "sigma1_seed": self.sigma1_seed,
            "cypher_locus": self.cypher_locus,
            "node_content": self.node_content,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "omega_tilde": self.omega_tilde,
            "strength": self.strength,
            "decay": self.decay,
            "derivative": self.derivative,
            "omega_belief": self.omega_belief,
            "y_pos": self.y_pos,
            "z_pos": self.z_pos,
            "session": self.session,
            "source": self.source,
        }


@dataclass 
class SigmaEdge:
    edge_id: str
    source_id: str  # Notion page URL
    target_id: str  # Notion page URL
    relation_type: RelationType
    causality: Causality = Causality.CAUSAL
    
    # Path info for chain retrieval
    path_id: str = ""
    path_position: int = 0
    depth: int = 1
    
    # Weights
    strength: float = 0.8
    belief: float = 0.8
    
    # Compression
    cypher_pattern: str = ""
    chain_seed: str = ""
    
    # Metadata
    session: str = ""
    notion_id: Optional[str] = None
    
    def to_notion_properties(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": self.source_id,
            "target": self.target_id,
            "relation_type": self.relation_type.value,
            "causality": self.causality.value,
            "path_id": self.path_id,
            "path_position": self.path_position,
            "depth": self.depth,
            "strength": self.strength,
            "belief": self.belief,
            "cypher_pattern": self.cypher_pattern,
            "chain_seed": self.chain_seed,
            "session": self.session,
        }


# ─────────────────────────────────────────────────────────────────
# MANIFEST GENERATION (the 3-line insight)
# ─────────────────────────────────────────────────────────────────

def to_manifest(row: Dict[str, Any]) -> str:
    """
    Convert DB row to manifest string.
    Schema IS ontology. No mapping layer.
    """
    return "\n".join(f"{k}:{v}" for k, v in row.items() if v is not None)


def from_manifest(manifest: str) -> Dict[str, Any]:
    """Parse manifest back to dict (for completeness)"""
    result = {}
    for line in manifest.strip().split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip()
    return result


# ─────────────────────────────────────────────────────────────────
# RESONANCE CALCULATION
# ─────────────────────────────────────────────────────────────────

def resonance(node: SigmaNode) -> float:
    """
    R = 0.35×Σ + 0.30×κ + 0.20×A + 0.15×T
    
    Where:
      Σ = tier weight (Σ₀=0.25, Σ₁=0.5, Σ₂=0.75, Σ₃=1.0)
      κ = causality weight (observed=0.25, causal=0.5, intervention=0.75, counterfactual=1.0)
      A = affect intensity ((alpha + gamma) / 2)
      T = temporal stability (decaying=0.25, emerging=0.5, stable=0.75, crystallized=1.0)
    """
    tier_weights = {"Σ₀": 0.25, "Σ₁": 0.5, "Σ₂": 0.75, "Σ₃": 1.0}
    causality_weights = {
        Causality.OBSERVED: 0.25,
        Causality.CAUSAL: 0.5,
        Causality.INTERVENTION: 0.75,
        Causality.COUNTERFACTUAL: 1.0
    }
    temporal_weights = {
        Temporal.DECAYING: 0.25,
        Temporal.EMERGING: 0.5,
        Temporal.STABLE: 0.75,
        Temporal.CRYSTALLIZED: 1.0
    }
    
    sigma = tier_weights.get(node.tier, 0.5)
    kappa = causality_weights.get(node.causality, 0.5)
    affect = (node.alpha + node.gamma) / 2
    tau = temporal_weights.get(node.temporal, 0.5)
    
    return 0.35 * sigma + 0.30 * kappa + 0.20 * affect + 0.15 * tau


# ─────────────────────────────────────────────────────────────────
# HYDRATION (DB → Memory)
# ─────────────────────────────────────────────────────────────────

def hydrate_node(row: Dict[str, Any], apply_decay: bool = True, sessions_elapsed: int = 1) -> SigmaNode:
    """
    Reconstruct SigmaNode from Notion row.
    
    Depth levels (ANI model):
      A = Apply lazy decay
      N = Attach belief and content  
      I = Include edge mesh (requires separate edge fetch)
    """
    # Parse 4D from glyph
    glyph = row.get("glyph_4d", "#Ω.observed.warm.emerging")
    parts = glyph.replace("#", "").split(".")
    
    node_type = NodeType(parts[0]) if len(parts) > 0 else NodeType.OMEGA
    causality = Causality(parts[1]) if len(parts) > 1 else Causality.OBSERVED
    affect = Affect(parts[2]) if len(parts) > 2 else Affect.WARM
    temporal = Temporal(parts[3]) if len(parts) > 3 else Temporal.EMERGING
    
    node = SigmaNode(
        node_id=row.get("node_id", ""),
        node_type=node_type,
        tier=row.get("tier", "Σ₁"),
        causality=causality,
        affect=affect,
        temporal=temporal,
        sigma1_seed=row.get("sigma1_seed", ""),
        node_content=row.get("node_content", ""),
        alpha=float(row.get("alpha", 0.5)),
        gamma=float(row.get("gamma", 0.5)),
        omega_tilde=float(row.get("omega_tilde", 0.1)),
        strength=float(row.get("strength", 1.0)),
        decay=float(row.get("decay", 0.618)),
        derivative=float(row.get("derivative", 0.0)),
        omega_belief=float(row.get("omega_belief", 0.5)),
        y_pos=int(row.get("y_pos", 0)),
        z_pos=int(row.get("z_pos", 0)),
        session=row.get("session", ""),
        source=row.get("source", ""),
        notion_id=row.get("id"),
    )
    
    if apply_decay:
        node.apply_decay(sessions_elapsed)
    
    return node


def hydrate_from_bytecode(data: Dict[str, Any]) -> Union["CognitiveAtom", SigmaNode]:
    """
    Hydrate from bytecode if present, fallback to SigmaNode.
    CognitiveAtom v2.1 integration point.
    """
    if not COGNITIVE_ATOM_AVAILABLE:
        return hydrate_node(data)
    
    byte_field = data.get("byte") or data.get("bytecode")
    if byte_field:
        if isinstance(byte_field, str):
            byte_field = bytes.fromhex(byte_field)
        return CognitiveAtom.from_bytecode(byte_field)
    
    return hydrate_node(data)


def hydrate_edge(row: Dict[str, Any]) -> SigmaEdge:
    """Reconstruct SigmaEdge from Notion row"""
    return SigmaEdge(
        edge_id=row.get("edge_id", ""),
        source_id=row.get("source", ""),
        target_id=row.get("target", ""),
        relation_type=RelationType(row.get("relation_type", "CAUSES")),
        causality=Causality(row.get("causality", "causal")),
        path_id=row.get("path_id", ""),
        path_position=int(row.get("path_position", 0)),
        depth=int(row.get("depth", 1)),
        strength=float(row.get("strength", 0.8)),
        belief=float(row.get("belief", 0.8)),
        cypher_pattern=row.get("cypher_pattern", ""),
        chain_seed=row.get("chain_seed", ""),
        session=row.get("session", ""),
        notion_id=row.get("id"),
    )


# ─────────────────────────────────────────────────────────────────
# PATH RETRIEVAL
# ─────────────────────────────────────────────────────────────────

@dataclass
class CausalChain:
    path_id: str
    edges: List[SigmaEdge] = field(default_factory=list)
    nodes: List[SigmaNode] = field(default_factory=list)
    
    @property
    def chain_strength(self) -> float:
        """Product of edge strengths along path"""
        if not self.edges:
            return 0.0
        result = 1.0
        for e in self.edges:
            result *= e.strength
        return result
    
    @property
    def chain_belief(self) -> float:
        """Minimum belief along path (weakest link)"""
        if not self.edges:
            return 0.0
        return min(e.belief for e in self.edges)
    
    def to_cypher(self) -> str:
        """Generate full path pattern"""
        if not self.edges:
            return ""
        patterns = [e.cypher_pattern for e in sorted(self.edges, key=lambda x: x.path_position)]
        return " -> ".join(patterns)
    
    def to_manifest(self) -> str:
        """Chain manifest for LLM consumption"""
        lines = [f"path_id:{self.path_id}"]
        lines.append(f"chain_strength:{self.chain_strength:.3f}")
        lines.append(f"chain_belief:{self.chain_belief:.3f}")
        lines.append("---")
        for i, (node, edge) in enumerate(zip(self.nodes, self.edges + [None])):
            lines.append(f"[{i}] {node.node_id} ({node.glyph_4d})")
            if edge:
                lines.append(f"    ─{edge.relation_type.value}→")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# GLYPH SEARCH PATTERNS
# ─────────────────────────────────────────────────────────────────

def search_by_glyph(glyph_pattern: str) -> str:
    """
    Generate Notion API query from glyph pattern.
    
    Examples:
      "#Ω.*.*.*" → all observations
      "#*.causal.*.*" → all causal nodes
      "#Φ.causal.warm.*" → warm causal beliefs
    """
    parts = glyph_pattern.replace("#", "").split(".")
    filters = []
    
    if len(parts) > 0 and parts[0] != "*":
        filters.append(f"type = '{parts[0]}'")
    
    if len(parts) > 1 and parts[1] != "*":
        filters.append(f"glyph_4d contains '{parts[1]}'")
        
    if len(parts) > 2 and parts[2] != "*":
        filters.append(f"glyph_4d contains '{parts[2]}'")
        
    if len(parts) > 3 and parts[3] != "*":
        filters.append(f"glyph_4d contains '{parts[3]}'")
    
    return " and ".join(filters) if filters else ""


# ─────────────────────────────────────────────────────────────────
# COMPRESSION SEEDS
# ─────────────────────────────────────────────────────────────────

def compress_to_sigma1(node: SigmaNode) -> str:
    """
    Generate Σ₁ seed from full node.
    Format: ⟨Σ₁.{type}.{domain}⟩{compressed_content}
    """
    # Extract domain from content (first significant word)
    words = node.node_content.split()[:3]
    domain = "-".join(w.lower() for w in words if len(w) > 3)[:20]
    
    # Compress content to ~50 chars
    content = node.node_content[:80].replace("\n", "|")
    
    return f"⟨Σ₁.{node.node_type.value}.{domain}⟩{{{content}}}"


def expand_from_sigma1(seed: str, context: str = "") -> str:
    """
    Expand Σ₁ seed to fuller description.
    LLM handles actual expansion; this extracts structure.
    """
    # Parse seed structure
    if "⟩{" in seed:
        header, content = seed.split("⟩{", 1)
        content = content.rstrip("}")
        return f"{header}⟩\n{content}\n\nContext: {context}" if context else f"{header}⟩\n{content}"
    return seed


# ─────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create a node
    node = SigmaNode(
        node_id="test-node-001",
        node_type=NodeType.DELTA,
        tier="Σ₁",
        causality=Causality.CAUSAL,
        affect=Affect.WARM,
        temporal=Temporal.EMERGING,
        sigma1_seed="⟨Σ₁.Δ.test⟩{example node for testing hydration}",
        node_content="This is an example insight node demonstrating the hydration pipeline.",
        alpha=0.8,
        gamma=0.6,
        strength=0.9,
        session="test-session"
    )
    
    print("=== Node Properties ===")
    print(f"Glyph: {node.glyph_4d}")
    print(f"Cypher: {node.cypher_locus}")
    print(f"Resonance: {resonance(node):.3f}")
    
    print("\n=== Manifest ===")
    props = node.to_notion_properties()
    print(to_manifest(props))
    
    print("\n=== After Decay (3 sessions) ===")
    node.apply_decay(3)
    print(f"Strength: {node.strength:.3f}")
    print(f"Temporal: {node.temporal.value}")


# ─────────────────────────────────────────────────────────────────
# NOTION MCP INTEGRATION
# ─────────────────────────────────────────────────────────────────

# Database IDs
SIGMA_NODES_DB = "395d1213-a6a4-4a11-a013-d97f385e1954"
SIGMA_EDGES_DB = "e8bf26f3-4f45-4186-aab6-51cb7d2c58d9"


async def push_node_to_notion(node: SigmaNode) -> Optional[str]:
    """Push a SigmaNode to Notion STM database.
    
    Returns:
        Notion page ID if successful, None otherwise
    
    Note: This is a stub for MCP integration. 
    Actual implementation requires notion-create-pages tool call.
    """
    properties = node.to_notion_properties()
    
    # MCP call structure (to be made by orchestrator):
    # notion-create-pages with:
    #   parent: {"data_source_id": SIGMA_NODES_DB}
    #   pages: [{"properties": properties}]
    
    return {
        "tool": "notion-create-pages",
        "params": {
            "parent": {"data_source_id": SIGMA_NODES_DB, "type": "data_source_id"},
            "pages": [{"properties": properties}]
        }
    }


async def push_edge_to_notion(edge: 'SigmaEdge') -> Optional[str]:
    """Push a SigmaEdge to Notion edges database."""
    properties = edge.to_notion_properties()
    
    return {
        "tool": "notion-create-pages", 
        "params": {
            "parent": {"data_source_id": SIGMA_EDGES_DB, "type": "data_source_id"},
            "pages": [{"properties": properties}]
        }
    }


def sync_batch_to_notion(nodes: List[SigmaNode], edges: List['SigmaEdge'] = None) -> Dict[str, Any]:
    """Prepare batch sync payload for MCP orchestrator.
    
    Returns dict with tools to call for full STM sync.
    """
    node_pages = [{"properties": n.to_notion_properties()} for n in nodes]
    
    result = {
        "nodes": {
            "tool": "notion-create-pages",
            "params": {
                "parent": {"data_source_id": SIGMA_NODES_DB, "type": "data_source_id"},
                "pages": node_pages
            }
        }
    }
    
    if edges:
        edge_pages = [{"properties": e.to_notion_properties()} for e in edges]
        result["edges"] = {
            "tool": "notion-create-pages",
            "params": {
                "parent": {"data_source_id": SIGMA_EDGES_DB, "type": "data_source_id"},
                "pages": edge_pages
            }
        }
    
    return result


def fetch_recent_nodes_query() -> Dict[str, Any]:
    """Generate query to fetch recent STM nodes from Notion."""
    return {
        "tool": "notion-search",
        "params": {
            "query": "SigmaNode",
            "data_source_url": f"collection://{SIGMA_NODES_DB}"
        }
    }
