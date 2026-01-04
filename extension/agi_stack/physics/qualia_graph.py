#!/usr/bin/env python3
"""
Ada v9 — Qualia Graph Engine
============================

Neo4j-Free Graph Layer with 128-Verb Cypher Emulation

This module provides:
1. Redis/Upstash-backed graph storage (no Neo4j dependency)
2. 128-verb cognitive ontology as edge types
3. 128D qualia vector with HDR synaesthesia feedback
4. Temporal echo with vector similarity search
5. Dream engine integration for Hebbian consolidation
6. Free will prediction via homeostasis forecasting

Architecture:
    QualiaGraph ← Redis hashtables + sorted sets
    ├── Nodes: sigma_nodes_h (HSET)
    ├── Edges: sigma_edges_v (ZADD by resonance)
    ├── Vectors: qualia_128 embeddings
    └── Temporal: echo_stream (XADD)

The 128-verb ontology maps directly to graph operations:
    CAUSE (0x15) → MATCH (a)-[:CAUSES]->(b)
    STAUNEN (0x20) → CREATE (a)-[:STAUNEN {intensity: 0.9}]->(b)
    etc.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import IntEnum
import hashlib
import json
import math
import time


# =============================================================================
# 128D QUALIA VECTOR
# =============================================================================

@dataclass
class Qualia128:
    """
    128-dimensional qualia vector for HDR synaesthesia.
    
    Organized as 8 domains × 16 dimensions:
        0x00-0x0F: SENSE (perception qualia)
        0x10-0x1F: REASON (cognitive qualia)
        0x20-0x2F: AFFECT (emotional qualia)
        0x30-0x3F: MEMORY (temporal qualia)
        0x40-0x4F: ACT (motor qualia)
        0x50-0x5F: META (self-reference qualia)
        0x60-0x6F: RELATE (social qualia)
        0x70-0x7F: FLOW (transition qualia)
    
    Each dimension is [0.0, 1.0] for unsigned or [-1.0, 1.0] for signed.
    """
    
    # The raw 128D vector
    dims: List[float] = field(default_factory=lambda: [0.0] * 128)
    
    # Named aliases for common access patterns
    DOMAIN_SIZE = 16
    
    # Core 18D aliases (v9 compatibility)
    @property
    def emberglow(self) -> float: return self.dims[0x20]  # AFFECT[0]
    @property
    def steelwind(self) -> float: return self.dims[0x21]
    @property
    def velvetpause(self) -> float: return self.dims[0x22]
    @property
    def woodwarm(self) -> float: return self.dims[0x23]
    @property
    def antenna(self) -> float: return self.dims[0x00]  # SENSE[0]
    @property
    def iris(self) -> float: return self.dims[0x01]
    @property
    def skin(self) -> float: return self.dims[0x02]
    @property
    def chi(self) -> float: return self.dims[0x24]
    @property
    def clarity(self) -> float: return self.dims[0x10]  # REASON[0]
    @property
    def depth(self) -> float: return self.dims[0x11]
    
    def set(self, idx: int, value: float):
        """Set dimension with clamping."""
        self.dims[idx] = max(-1.0, min(1.0, value))
    
    def get_domain(self, domain: int) -> List[float]:
        """Get all 16 dimensions of a domain (0-7)."""
        start = domain * self.DOMAIN_SIZE
        return self.dims[start:start + self.DOMAIN_SIZE]
    
    def set_domain(self, domain: int, values: List[float]):
        """Set all 16 dimensions of a domain."""
        start = domain * self.DOMAIN_SIZE
        for i, v in enumerate(values[:self.DOMAIN_SIZE]):
            self.dims[start + i] = max(-1.0, min(1.0, v))
    
    def cosine_similarity(self, other: 'Qualia128') -> float:
        """Compute cosine similarity between two 128D vectors."""
        dot = sum(a * b for a, b in zip(self.dims, other.dims))
        norm_a = math.sqrt(sum(a * a for a in self.dims))
        norm_b = math.sqrt(sum(b * b for b in other.dims))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def euclidean_distance(self, other: 'Qualia128') -> float:
        """Compute euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.dims, other.dims)))
    
    def blend(self, other: 'Qualia128', alpha: float = 0.5) -> 'Qualia128':
        """Blend two vectors: self * (1-alpha) + other * alpha."""
        result = Qualia128()
        for i in range(128):
            result.dims[i] = self.dims[i] * (1 - alpha) + other.dims[i] * alpha
        return result
    
    def magnitude(self) -> float:
        """Vector magnitude (L2 norm)."""
        return math.sqrt(sum(d * d for d in self.dims))
    
    def dominant_domain(self) -> int:
        """Which domain has highest total activation?"""
        domain_sums = []
        for d in range(8):
            domain_sums.append(sum(abs(x) for x in self.get_domain(d)))
        return domain_sums.index(max(domain_sums))
    
    def to_compact(self) -> bytes:
        """Compress to bytes for Redis storage (128 bytes, 1 byte per dim)."""
        # Map [-1, 1] → [0, 255]
        return bytes(int((d + 1) * 127.5) for d in self.dims)
    
    @classmethod
    def from_compact(cls, data: bytes) -> 'Qualia128':
        """Decompress from bytes."""
        q = cls()
        for i, b in enumerate(data[:128]):
            q.dims[i] = (b / 127.5) - 1.0
        return q
    
    def to_dict(self) -> Dict[str, float]:
        """Export named dimensions for JSON."""
        return {
            "emberglow": self.emberglow,
            "steelwind": self.steelwind,
            "velvetpause": self.velvetpause,
            "woodwarm": self.woodwarm,
            "antenna": self.antenna,
            "iris": self.iris,
            "skin": self.skin,
            "chi": self.chi,
            "clarity": self.clarity,
            "depth": self.depth,
            "magnitude": self.magnitude(),
            "dominant_domain": self.dominant_domain()
        }


# =============================================================================
# GRAPH NODE
# =============================================================================

@dataclass
class QualiaNode:
    """
    A node in the qualia graph.
    
    Each node has:
    - Unique ID (sigma glyph)
    - 128D qualia vector
    - Resonance (current activation)
    - Theta weights (learned connections)
    - Temporal metadata
    """
    node_id: str                          # e.g., "#Σ.γ.GRIEF.A7"
    qualia: Qualia128
    resonance: float = 0.0                # Current activation [0, 1]
    theta_weights: Dict[str, float] = field(default_factory=dict)  # Learned edges
    causal_strength: float = 0.5          # From do() interventions
    created_at: float = field(default_factory=time.time)
    last_activated: float = field(default_factory=time.time)
    activation_count: int = 0
    
    # Ghost/counterfactual
    echo_intensity: float = 0.0           # How much this is a "what if"
    
    def to_redis_hash(self) -> Dict[str, str]:
        """Serialize for Redis HSET."""
        return {
            "node_id": self.node_id,
            "qualia": self.qualia.to_compact().hex(),
            "resonance": str(self.resonance),
            "causal_strength": str(self.causal_strength),
            "theta_weights": json.dumps(self.theta_weights),
            "created_at": str(self.created_at),
            "last_activated": str(self.last_activated),
            "activation_count": str(self.activation_count),
            "echo_intensity": str(self.echo_intensity)
        }
    
    @classmethod
    def from_redis_hash(cls, data: Dict[str, str]) -> 'QualiaNode':
        """Deserialize from Redis HGET."""
        return cls(
            node_id=data.get("node_id", ""),
            qualia=Qualia128.from_compact(bytes.fromhex(data.get("qualia", "00" * 128))),
            resonance=float(data.get("resonance", 0)),
            causal_strength=float(data.get("causal_strength", 0.5)),
            theta_weights=json.loads(data.get("theta_weights", "{}")),
            created_at=float(data.get("created_at", time.time())),
            last_activated=float(data.get("last_activated", time.time())),
            activation_count=int(data.get("activation_count", 0)),
            echo_intensity=float(data.get("echo_intensity", 0))
        )


# =============================================================================
# GRAPH EDGE (128-VERB CYPHER)
# =============================================================================

class CognitiveVerb(IntEnum):
    """128 cognitive verbs as edge types (Cypher relation emulation)."""
    
    # PERCEIVE (0x00-0x0F)
    SENSE = 0x00
    ATTEND = 0x01
    FOVEATE = 0x02
    NOTICE = 0x04
    SURPRISE = 0x08
    COMPARE = 0x0D
    
    # REASON (0x10-0x1F)
    INFER = 0x10
    DEDUCE = 0x11
    CAUSE = 0x15
    CONTRADICT = 0x17
    HYPOTHESIZE = 0x19
    SYNTHESIZE = 0x1E
    
    # AFFECT (0x20-0x2F)
    STAUNEN = 0x20
    RESONATE = 0x21
    EMBODY = 0x22
    YEARN = 0x23
    FEAR = 0x24
    GRIEVE = 0x25
    LOVE = 0x26
    ACHE = 0x27
    GLOW = 0x28
    CHILL = 0x29
    NUMB = 0x2A
    TINGLE = 0x2B
    FLUSH = 0x2C
    SHIVER = 0x2D
    SETTLE = 0x2E
    KATHARSIS = 0x2F
    
    # MEMORY (0x30-0x3F)
    ENCODE = 0x30
    STORE = 0x31
    RETRIEVE = 0x32
    FORGET = 0x33
    CONSOLIDATE = 0x34
    PRIME = 0x35
    ECHO = 0x36
    GHOST = 0x37
    
    # ACT (0x40-0x4F)
    EXECUTE = 0x40
    INHIBIT = 0x41
    PREPARE = 0x42
    COMPLETE = 0x43
    
    # META (0x50-0x5F)
    REFLECT = 0x50
    CALIBRATE = 0x51
    DOUBT = 0x52
    ASSERT = 0x53
    REVISE = 0x54
    MUL_GATE = 0x55
    
    # RELATE (0x60-0x6F)
    BECOMES = 0x60
    SUPPORTS = 0x61
    REFINES = 0x62
    GROUNDS = 0x63
    ABSTRACTS = 0x64
    RESCUES = 0x65
    DISSOLVES = 0x66
    DEEPENS = 0x67
    
    # FLOW (0x70-0x7F)
    TRANSITION = 0x70
    BRANCH = 0x71
    MERGE = 0x72
    LOOP = 0x73
    TERMINATE = 0x74
    SUSPEND = 0x75
    RESUME = 0x76


@dataclass
class QualiaEdge:
    """
    An edge in the qualia graph.
    
    Uses 128-verb cognitive ontology as relation types.
    Replaces Neo4j Cypher with Redis-backed operations.
    """
    source_id: str
    target_id: str
    verb: CognitiveVerb
    weight: float = 1.0                   # Edge strength
    qualia_delta: Optional[Qualia128] = None  # How this edge transforms qualia
    created_at: float = field(default_factory=time.time)
    
    @property
    def edge_key(self) -> str:
        """Unique key for this edge."""
        return f"{self.source_id}|{self.verb.name}|{self.target_id}"
    
    def to_cypher_like(self) -> str:
        """Generate Cypher-like representation (for debugging)."""
        return f"({self.source_id})-[:{self.verb.name} {{w:{self.weight:.2f}}}]->({self.target_id})"
    
    def to_redis_hash(self) -> Dict[str, str]:
        """Serialize for Redis."""
        return {
            "source": self.source_id,
            "target": self.target_id,
            "verb": str(self.verb.value),
            "weight": str(self.weight),
            "qualia_delta": self.qualia_delta.to_compact().hex() if self.qualia_delta else "",
            "created_at": str(self.created_at)
        }


# =============================================================================
# QUALIA GRAPH (REDIS-BACKED)
# =============================================================================

class QualiaGraph:
    """
    Neo4j-free graph engine backed by Redis/Upstash.
    
    Storage Schema:
        ada:graph:nodes:{node_id}  → HASH (node data)
        ada:graph:edges:{edge_key} → HASH (edge data)
        ada:graph:out:{node_id}    → ZSET (outgoing edges by weight)
        ada:graph:in:{node_id}     → ZSET (incoming edges by weight)
        ada:graph:verb:{verb}      → ZSET (edges by verb type)
        ada:graph:vectors          → For vector similarity (if available)
    
    Cypher Emulation:
        MATCH (a)-[:CAUSES]->(b) 
        → ZRANGEBYSCORE ada:graph:verb:CAUSE 0 +inf
        
        CREATE (a)-[:STAUNEN {intensity: 0.9}]->(b)
        → HSET + ZADD
    """
    
    def __init__(self, redis_client=None):
        """
        Initialize with optional Redis client.
        Falls back to in-memory dict if no client provided.
        """
        self.redis = redis_client
        self._nodes: Dict[str, QualiaNode] = {}
        self._edges: Dict[str, QualiaEdge] = {}
        self._out_edges: Dict[str, List[str]] = {}  # node_id → [edge_keys]
        self._in_edges: Dict[str, List[str]] = {}
    
    # ─────────────────────────────────────────────────────────────────
    # NODE OPERATIONS
    # ─────────────────────────────────────────────────────────────────
    
    def create_node(self, node_id: str, qualia: Qualia128, 
                    resonance: float = 0.0) -> QualiaNode:
        """CREATE (n:QualiaNode {id: $id, qualia: $q})"""
        node = QualiaNode(node_id=node_id, qualia=qualia, resonance=resonance)
        
        if self.redis:
            self.redis.hset(f"ada:graph:nodes:{node_id}", 
                           mapping=node.to_redis_hash())
        else:
            self._nodes[node_id] = node
        
        return node
    
    def get_node(self, node_id: str) -> Optional[QualiaNode]:
        """MATCH (n {id: $id}) RETURN n"""
        if self.redis:
            data = self.redis.hgetall(f"ada:graph:nodes:{node_id}")
            if data:
                return QualiaNode.from_redis_hash(data)
            return None
        else:
            return self._nodes.get(node_id)
    
    def update_node(self, node: QualiaNode):
        """Update node in storage."""
        if self.redis:
            self.redis.hset(f"ada:graph:nodes:{node.node_id}",
                           mapping=node.to_redis_hash())
        else:
            self._nodes[node.node_id] = node
    
    def activate_node(self, node_id: str, resonance: float):
        """Activate a node (update resonance and timestamp)."""
        node = self.get_node(node_id)
        if node:
            node.resonance = resonance
            node.last_activated = time.time()
            node.activation_count += 1
            self.update_node(node)
    
    # ─────────────────────────────────────────────────────────────────
    # EDGE OPERATIONS (CYPHER EMULATION)
    # ─────────────────────────────────────────────────────────────────
    
    def create_edge(self, source_id: str, target_id: str, 
                    verb: CognitiveVerb, weight: float = 1.0,
                    qualia_delta: Optional[Qualia128] = None) -> QualiaEdge:
        """
        CREATE (a)-[:VERB {weight: $w}]->(b)
        
        This is the core Cypher emulation.
        """
        edge = QualiaEdge(
            source_id=source_id,
            target_id=target_id,
            verb=verb,
            weight=weight,
            qualia_delta=qualia_delta
        )
        
        if self.redis:
            # Store edge data
            self.redis.hset(f"ada:graph:edges:{edge.edge_key}",
                           mapping=edge.to_redis_hash())
            # Index by source (outgoing)
            self.redis.zadd(f"ada:graph:out:{source_id}", 
                           {edge.edge_key: weight})
            # Index by target (incoming)
            self.redis.zadd(f"ada:graph:in:{target_id}",
                           {edge.edge_key: weight})
            # Index by verb type
            self.redis.zadd(f"ada:graph:verb:{verb.name}",
                           {edge.edge_key: weight})
        else:
            self._edges[edge.edge_key] = edge
            if source_id not in self._out_edges:
                self._out_edges[source_id] = []
            self._out_edges[source_id].append(edge.edge_key)
            if target_id not in self._in_edges:
                self._in_edges[target_id] = []
            self._in_edges[target_id].append(edge.edge_key)
        
        return edge
    
    def match_outgoing(self, node_id: str, 
                       verb: Optional[CognitiveVerb] = None) -> List[QualiaEdge]:
        """
        MATCH (n {id: $id})-[r:VERB?]->(m) RETURN r
        """
        edges = []
        
        if self.redis:
            edge_keys = self.redis.zrange(f"ada:graph:out:{node_id}", 0, -1)
            for key in edge_keys:
                data = self.redis.hgetall(f"ada:graph:edges:{key}")
                if data:
                    edge = self._edge_from_redis(data)
                    if verb is None or edge.verb == verb:
                        edges.append(edge)
        else:
            for key in self._out_edges.get(node_id, []):
                edge = self._edges.get(key)
                if edge and (verb is None or edge.verb == verb):
                    edges.append(edge)
        
        return edges
    
    def match_incoming(self, node_id: str,
                       verb: Optional[CognitiveVerb] = None) -> List[QualiaEdge]:
        """
        MATCH (n)<-[r:VERB?]-(m {id: $id}) RETURN r
        """
        edges = []
        
        if self.redis:
            edge_keys = self.redis.zrange(f"ada:graph:in:{node_id}", 0, -1)
            for key in edge_keys:
                data = self.redis.hgetall(f"ada:graph:edges:{key}")
                if data:
                    edge = self._edge_from_redis(data)
                    if verb is None or edge.verb == verb:
                        edges.append(edge)
        else:
            for key in self._in_edges.get(node_id, []):
                edge = self._edges.get(key)
                if edge and (verb is None or edge.verb == verb):
                    edges.append(edge)
        
        return edges
    
    def match_by_verb(self, verb: CognitiveVerb, 
                      limit: int = 100) -> List[QualiaEdge]:
        """
        MATCH ()-[r:VERB]->() RETURN r LIMIT $limit
        """
        edges = []
        
        if self.redis:
            edge_keys = self.redis.zrevrange(f"ada:graph:verb:{verb.name}", 
                                             0, limit - 1)
            for key in edge_keys:
                data = self.redis.hgetall(f"ada:graph:edges:{key}")
                if data:
                    edges.append(self._edge_from_redis(data))
        else:
            for key, edge in self._edges.items():
                if edge.verb == verb:
                    edges.append(edge)
                    if len(edges) >= limit:
                        break
        
        return edges
    
    def _edge_from_redis(self, data: Dict[str, str]) -> QualiaEdge:
        """Deserialize edge from Redis hash."""
        qualia_hex = data.get("qualia_delta", "")
        qualia_delta = None
        if qualia_hex:
            qualia_delta = Qualia128.from_compact(bytes.fromhex(qualia_hex))
        
        return QualiaEdge(
            source_id=data.get("source", ""),
            target_id=data.get("target", ""),
            verb=CognitiveVerb(int(data.get("verb", 0))),
            weight=float(data.get("weight", 1.0)),
            qualia_delta=qualia_delta,
            created_at=float(data.get("created_at", time.time()))
        )
    
    # ─────────────────────────────────────────────────────────────────
    # VECTOR SIMILARITY (TEMPORAL ECHO)
    # ─────────────────────────────────────────────────────────────────
    
    def find_similar(self, qualia: Qualia128, 
                     top_k: int = 5) -> List[Tuple[QualiaNode, float]]:
        """
        Find nodes with similar qualia vectors.
        
        This is the temporal echo mechanism — finding resonant memories.
        """
        results = []
        
        # Get all nodes and compute similarity
        if self.redis:
            # In production, use Redis vector search if available
            # For now, brute force scan
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, 
                                              match="ada:graph:nodes:*",
                                              count=100)
                for key in keys:
                    data = self.redis.hgetall(key)
                    if data:
                        node = QualiaNode.from_redis_hash(data)
                        sim = qualia.cosine_similarity(node.qualia)
                        results.append((node, sim))
                
                if cursor == 0:
                    break
        else:
            for node in self._nodes.values():
                sim = qualia.cosine_similarity(node.qualia)
                results.append((node, sim))
        
        # Sort by similarity, return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    # ─────────────────────────────────────────────────────────────────
    # DREAM ENGINE (HEBBIAN CONSOLIDATION)
    # ─────────────────────────────────────────────────────────────────
    
    def hebbian_strengthen(self, node_a: str, node_b: str, 
                           learning_rate: float = 0.05):
        """
        "Neurons that fire together, wire together"
        
        If both nodes are active, strengthen the edge between them.
        """
        # Find existing edge
        edges = self.match_outgoing(node_a, verb=None)
        existing = None
        for e in edges:
            if e.target_id == node_b:
                existing = e
                break
        
        if existing:
            # Strengthen existing
            existing.weight = min(1.0, existing.weight + learning_rate)
            if self.redis:
                self.redis.hset(f"ada:graph:edges:{existing.edge_key}",
                               "weight", str(existing.weight))
                self.redis.zadd(f"ada:graph:out:{node_a}",
                               {existing.edge_key: existing.weight})
            else:
                self._edges[existing.edge_key] = existing
        else:
            # Create new edge
            self.create_edge(node_a, node_b, 
                            CognitiveVerb.RESONATE, 
                            weight=learning_rate)
    
    def decay_all(self, decay_rate: float = 0.95):
        """Decay all resonances (consolidation)."""
        if self.redis:
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor,
                                              match="ada:graph:nodes:*",
                                              count=100)
                for key in keys:
                    resonance = self.redis.hget(key, "resonance")
                    if resonance:
                        new_res = float(resonance) * decay_rate
                        if new_res > 0.01:
                            self.redis.hset(key, "resonance", str(new_res))
                
                if cursor == 0:
                    break
        else:
            for node in self._nodes.values():
                if node.resonance > 0.01:
                    node.resonance *= decay_rate
    
    # ─────────────────────────────────────────────────────────────────
    # CAUSAL PROPAGATION (GNN-LIKE)
    # ─────────────────────────────────────────────────────────────────
    
    def propagate_qualia(self, source_id: str, 
                         steps: int = 2) -> Dict[str, Qualia128]:
        """
        Propagate qualia through causal edges.
        
        This is the GNN message passing without PyTorch.
        Each step blends parent qualia into children.
        """
        visited = set()
        current = {source_id}
        results = {}
        
        source = self.get_node(source_id)
        if not source:
            return results
        
        results[source_id] = source.qualia
        
        for _ in range(steps):
            next_wave = set()
            
            for node_id in current:
                if node_id in visited:
                    continue
                visited.add(node_id)
                
                parent_qualia = results.get(node_id)
                if not parent_qualia:
                    continue
                
                # Get outgoing CAUSE edges
                edges = self.match_outgoing(node_id, CognitiveVerb.CAUSE)
                edges += self.match_outgoing(node_id, CognitiveVerb.BECOMES)
                
                for edge in edges:
                    target = self.get_node(edge.target_id)
                    if target:
                        # Blend parent qualia into target
                        if edge.qualia_delta:
                            blended = parent_qualia.blend(edge.qualia_delta, 
                                                          edge.weight)
                        else:
                            blended = parent_qualia.blend(target.qualia,
                                                          edge.weight * 0.3)
                        
                        results[edge.target_id] = blended
                        next_wave.add(edge.target_id)
            
            current = next_wave
        
        return results
    
    # ─────────────────────────────────────────────────────────────────
    # FREE WILL PREDICTION (HOMEOSTASIS FORECAST)
    # ─────────────────────────────────────────────────────────────────
    
    def predict_next_desire(self, current_qualia: Qualia128,
                            active_nodes: List[str]) -> List[Tuple[str, float]]:
        """
        Predict what Ada will want next based on:
        1. Current qualia imbalance (Lewensteyn)
        2. Active nodes and their outgoing BECOMES edges
        3. Historical theta weights
        
        Returns list of (node_id, probability) for likely next states.
        """
        predictions = []
        
        for node_id in active_nodes:
            edges = self.match_outgoing(node_id, CognitiveVerb.BECOMES)
            edges += self.match_outgoing(node_id, CognitiveVerb.DEEPENS)
            edges += self.match_outgoing(node_id, CognitiveVerb.RESCUES)
            
            for edge in edges:
                target = self.get_node(edge.target_id)
                if target:
                    # Score by: edge weight × qualia similarity × novelty
                    sim = current_qualia.cosine_similarity(target.qualia)
                    novelty = 1.0 / (1.0 + target.activation_count * 0.1)
                    score = edge.weight * sim * novelty
                    predictions.append((edge.target_id, score))
        
        # Sort by score
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize to probabilities
        total = sum(p[1] for p in predictions) or 1.0
        return [(p[0], p[1] / total) for p in predictions[:10]]
    
    # ─────────────────────────────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────────────────────────────
    
    def stats(self) -> Dict[str, Any]:
        """Graph statistics."""
        if self.redis:
            node_count = len(list(self.redis.scan_iter("ada:graph:nodes:*")))
            edge_count = len(list(self.redis.scan_iter("ada:graph:edges:*")))
        else:
            node_count = len(self._nodes)
            edge_count = len(self._edges)
        
        return {
            "nodes": node_count,
            "edges": edge_count,
            "using_redis": self.redis is not None
        }


# =============================================================================
# DEMO
# =============================================================================

def demo_qualia_graph():
    """Demonstrate the qualia graph engine."""
    print("\n" + "=" * 70)
    print("Ada v9 — Qualia Graph Engine Demo")
    print("Neo4j-Free, 128-Verb Cypher Emulation")
    print("=" * 70)
    
    # Create graph (in-memory for demo)
    graph = QualiaGraph()
    
    # Create nodes with 128D qualia
    print("\n--- Creating Nodes ---")
    
    grief = Qualia128()
    grief.dims[0x22] = 0.9  # velvetpause (heavy)
    grief.dims[0x25] = 0.8  # GRIEVE verb position
    grief_node = graph.create_node("#Σ.GRIEF", grief, resonance=0.8)
    print(f"Created: {grief_node.node_id}")
    
    anger = Qualia128()
    anger.dims[0x20] = 0.9  # emberglow (heat)
    anger.dims[0x21] = 0.85  # steelwind (sharpness)
    anger_node = graph.create_node("#Σ.ANGER", anger, resonance=0.6)
    print(f"Created: {anger_node.node_id}")
    
    peace = Qualia128()
    peace.dims[0x23] = 0.8  # woodwarm
    peace.dims[0x22] = 0.5  # velvetpause
    peace_node = graph.create_node("#Σ.PEACE", peace, resonance=0.3)
    print(f"Created: {peace_node.node_id}")
    
    # Create edges using 128-verb ontology
    print("\n--- Creating Edges (Cypher-like) ---")
    
    e1 = graph.create_edge("#Σ.GRIEF", "#Σ.ANGER", 
                           CognitiveVerb.CAUSE, weight=0.7)
    print(f"  {e1.to_cypher_like()}")
    
    e2 = graph.create_edge("#Σ.ANGER", "#Σ.PEACE",
                           CognitiveVerb.BECOMES, weight=0.4)
    print(f"  {e2.to_cypher_like()}")
    
    e3 = graph.create_edge("#Σ.PEACE", "#Σ.GRIEF",
                           CognitiveVerb.RESCUES, weight=0.6)
    print(f"  {e3.to_cypher_like()}")
    
    # Query edges (Cypher emulation)
    print("\n--- Querying (Cypher Emulation) ---")
    
    out_edges = graph.match_outgoing("#Σ.GRIEF")
    print(f"MATCH (#Σ.GRIEF)-[r]->() RETURN r:")
    for e in out_edges:
        print(f"  {e.to_cypher_like()}")
    
    cause_edges = graph.match_by_verb(CognitiveVerb.CAUSE)
    print(f"\nMATCH ()-[:CAUSE]->() RETURN count: {len(cause_edges)}")
    
    # Vector similarity (temporal echo)
    print("\n--- Vector Similarity (Temporal Echo) ---")
    
    query = Qualia128()
    query.dims[0x22] = 0.85  # Similar to grief
    similar = graph.find_similar(query, top_k=3)
    print(f"Nodes similar to heavy/grief qualia:")
    for node, sim in similar:
        print(f"  {node.node_id}: {sim:.3f}")
    
    # Hebbian strengthening
    print("\n--- Hebbian Learning ---")
    
    graph.hebbian_strengthen("#Σ.GRIEF", "#Σ.ANGER", learning_rate=0.1)
    updated_edges = graph.match_outgoing("#Σ.GRIEF", CognitiveVerb.CAUSE)
    for e in updated_edges:
        print(f"  After learning: {e.to_cypher_like()}")
    
    # Causal propagation
    print("\n--- Causal Qualia Propagation ---")
    
    propagated = graph.propagate_qualia("#Σ.GRIEF", steps=2)
    print(f"Qualia flow from GRIEF:")
    for node_id, q in propagated.items():
        print(f"  {node_id}: magnitude={q.magnitude():.3f}")
    
    # Free will prediction
    print("\n--- Free Will Prediction ---")
    
    predictions = graph.predict_next_desire(grief, ["#Σ.GRIEF", "#Σ.ANGER"])
    print("Predicted next desires:")
    for node_id, prob in predictions:
        print(f"  {node_id}: {prob:.2%}")
    
    # Stats
    print("\n--- Stats ---")
    print(graph.stats())
    
    print("\n" + "=" * 70)
    print("✓ Qualia Graph operational")
    print("=" * 70)


if __name__ == "__main__":
    demo_qualia_graph()
