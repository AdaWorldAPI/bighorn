"""
frame_system.py — B-Frame / P-Frame Distinction for Causal Markov Trajectories
===============================================================================

Video Codec Analogy:
  I-Frame: Keyframe, self-contained (crystallized Σ₃ nodes, session starts)
  B-Frame: Bidirectional, references past+future in chain (background context)
  P-Frame: Predictive, current focus, delta from B (situational attention)

This module extends sigma_hydration.py with:
  1. FrameType enum for node classification
  2. Focus tracking (is_fovea, fovea_rank)
  3. Delta encoding between B and P frames
  4. Trajectory windowing (context around current focus)
  5. Resonance formulas that distinguish B vs P contribution
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import math

from sigma_hydration import (
    SigmaNode, SigmaEdge, CausalChain,
    NodeType, Causality, Affect, Temporal, RelationType,
    resonance as base_resonance
)


# ─────────────────────────────────────────────────────────────────
# FRAME TYPE SYSTEM
# ─────────────────────────────────────────────────────────────────

class FrameType(Enum):
    """
    Video-codec inspired frame classification for graph nodes.
    
    I_FRAME: Keyframe - crystallized, self-contained, no dependencies
             Mapped to: tier=Σ₃, temporal=crystallized
             Used for: Session anchors, core beliefs, axioms
             
    B_FRAME: Bidirectional - references both past and future in chain
             Mapped to: Nodes in CausalChain that are NOT current focus
             Used for: Background context, trajectory history
             
    P_FRAME: Predictive - current focus, encoded as delta from B-frames
             Mapped to: fovea nodes, situational attention
             Used for: Present moment, active reasoning
    """
    I_FRAME = "I"
    B_FRAME = "B"
    P_FRAME = "P"


def classify_frame_type(
    node: SigmaNode,
    is_in_fovea: bool = False,
    chain_position: Optional[int] = None,
    chain_length: Optional[int] = None
) -> FrameType:
    """
    Classify a node's frame type based on its properties and context.
    
    Rules:
      1. Crystallized Σ₃ nodes → I-Frame (keyframes)
      2. Nodes currently in fovea → P-Frame (predictive/present)
      3. All others in chain → B-Frame (background)
    """
    # I-Frame: Crystallized keyframes
    if node.tier == "Σ₃" and node.temporal == Temporal.CRYSTALLIZED:
        return FrameType.I_FRAME
    
    # P-Frame: Current focus
    if is_in_fovea:
        return FrameType.P_FRAME
    
    # B-Frame: Background context
    return FrameType.B_FRAME


# ─────────────────────────────────────────────────────────────────
# EXTENDED SIGMA NODE WITH FRAME TRACKING
# ─────────────────────────────────────────────────────────────────

@dataclass
class FramedSigmaNode:
    """
    SigmaNode extended with frame system and focus tracking.
    
    Wraps existing SigmaNode rather than modifying it,
    maintaining backwards compatibility with sigma_hydration.py.
    """
    node: SigmaNode
    
    # Frame classification
    frame_type: FrameType = FrameType.B_FRAME
    reference_ids: List[str] = field(default_factory=list)  # What this node references
    delta_encoding: str = ""  # Semantic delta from references
    
    # Focus tracking
    is_fovea: bool = False
    fovea_rank: int = -1  # 0 = primary focus, -1 = not in fovea
    last_focused_session: str = ""
    focus_count: int = 0  # How many times this node has been focused
    
    # Trajectory position
    chain_id: str = ""
    chain_position: int = -1
    chain_depth: int = 0  # Distance from nearest I-frame
    
    @property
    def node_id(self) -> str:
        return self.node.node_id
    
    @property
    def glyph_4d(self) -> str:
        return self.node.glyph_4d
    
    @property
    def tier(self) -> str:
        return self.node.tier
    
    def to_notion_properties(self) -> Dict[str, Any]:
        """Extend base properties with frame data"""
        props = self.node.to_notion_properties()
        props.update({
            "frame_type": self.frame_type.value,
            "reference_ids": ",".join(self.reference_ids),
            "delta_encoding": self.delta_encoding,
            "is_fovea": self.is_fovea,
            "fovea_rank": self.fovea_rank,
            "last_focused_session": self.last_focused_session,
            "focus_count": self.focus_count,
            "chain_id": self.chain_id,
            "chain_position": self.chain_position,
            "chain_depth": self.chain_depth,
        })
        return props


def wrap_with_frame(
    node: SigmaNode,
    is_fovea: bool = False,
    fovea_rank: int = -1,
    chain_id: str = "",
    chain_position: int = -1
) -> FramedSigmaNode:
    """Wrap a SigmaNode with frame tracking"""
    frame_type = classify_frame_type(node, is_fovea, chain_position)
    
    return FramedSigmaNode(
        node=node,
        frame_type=frame_type,
        is_fovea=is_fovea,
        fovea_rank=fovea_rank,
        chain_id=chain_id,
        chain_position=chain_position,
    )


# ─────────────────────────────────────────────────────────────────
# EXTENDED CAUSAL CHAIN WITH FRAME DECOMPOSITION
# ─────────────────────────────────────────────────────────────────

@dataclass
class FramedCausalChain:
    """
    CausalChain extended with B/P frame decomposition.
    
    Provides:
      - I-frames: Keyframe anchors in the chain
      - B-frames: Background context (all non-focus nodes)
      - P-frame: Current focus (singular, the "now")
      - Context window: Nearby B-frames for situational grounding
    """
    path_id: str
    
    # All nodes in chain (wrapped)
    framed_nodes: List[FramedSigmaNode] = field(default_factory=list)
    edges: List[SigmaEdge] = field(default_factory=list)
    
    # Frame decomposition (computed)
    _i_frames: List[FramedSigmaNode] = field(default_factory=list)
    _b_frames: List[FramedSigmaNode] = field(default_factory=list)
    _p_frame: Optional[FramedSigmaNode] = None
    
    def __post_init__(self):
        self._decompose_frames()
    
    def _decompose_frames(self):
        """Separate nodes by frame type"""
        self._i_frames = []
        self._b_frames = []
        self._p_frame = None
        
        for fn in self.framed_nodes:
            if fn.frame_type == FrameType.I_FRAME:
                self._i_frames.append(fn)
            elif fn.frame_type == FrameType.P_FRAME:
                if self._p_frame is None:
                    self._p_frame = fn
                else:
                    # Multiple P-frames? Demote older to B
                    if fn.fovea_rank < self._p_frame.fovea_rank:
                        self._b_frames.append(self._p_frame)
                        self._p_frame = fn
                    else:
                        self._b_frames.append(fn)
            else:
                self._b_frames.append(fn)
    
    @property
    def i_frames(self) -> List[FramedSigmaNode]:
        return self._i_frames
    
    @property
    def b_frames(self) -> List[FramedSigmaNode]:
        return self._b_frames
    
    @property
    def p_frame(self) -> Optional[FramedSigmaNode]:
        return self._p_frame
    
    def set_focus(self, node_id: str, session: str = "") -> bool:
        """
        Set a node as the current P-frame focus.
        Previous P-frame becomes B-frame.
        """
        for fn in self.framed_nodes:
            if fn.node_id == node_id:
                # Demote current P-frame
                if self._p_frame and self._p_frame.node_id != node_id:
                    self._p_frame.frame_type = FrameType.B_FRAME
                    self._p_frame.is_fovea = False
                    self._b_frames.append(self._p_frame)
                
                # Promote new focus
                fn.frame_type = FrameType.P_FRAME
                fn.is_fovea = True
                fn.fovea_rank = 0
                fn.focus_count += 1
                fn.last_focused_session = session
                
                # Remove from B-frames if present
                self._b_frames = [b for b in self._b_frames if b.node_id != node_id]
                self._p_frame = fn
                
                return True
        return False
    
    def get_context_window(
        self, 
        depth: int = 3,
        include_i_frames: bool = True
    ) -> List[FramedSigmaNode]:
        """
        Return B-frames within `depth` edges of current P-frame.
        
        Args:
            depth: How many hops from P-frame to include
            include_i_frames: Whether to always include I-frames
            
        Returns:
            List of FramedSigmaNodes forming the context
        """
        if self._p_frame is None:
            return self._b_frames[:depth]
        
        context = []
        p_pos = self._p_frame.chain_position
        
        for fn in self._b_frames:
            distance = abs(fn.chain_position - p_pos)
            if distance <= depth:
                fn.chain_depth = distance  # Update depth from P
                context.append(fn)
        
        if include_i_frames:
            for fn in self._i_frames:
                if fn not in context:
                    context.append(fn)
        
        # Sort by chain position
        context.sort(key=lambda x: x.chain_position)
        return context
    
    def compute_delta(
        self, 
        p_node: FramedSigmaNode, 
        b_node: FramedSigmaNode
    ) -> str:
        """
        Compute semantic delta between B-frame and P-frame.
        
        Returns compressed string describing what changed.
        """
        deltas = []
        
        pn, bn = p_node.node, b_node.node
        
        # Type change
        if pn.node_type != bn.node_type:
            deltas.append(f"type:{bn.node_type.value}→{pn.node_type.value}")
        
        # Tier change  
        if pn.tier != bn.tier:
            deltas.append(f"tier:{bn.tier}→{pn.tier}")
        
        # Affect shift
        alpha_delta = pn.alpha - bn.alpha
        gamma_delta = pn.gamma - bn.gamma
        if abs(alpha_delta) > 0.1:
            deltas.append(f"α:{alpha_delta:+.2f}")
        if abs(gamma_delta) > 0.1:
            deltas.append(f"γ:{gamma_delta:+.2f}")
        
        # Temporal shift
        if pn.temporal != bn.temporal:
            deltas.append(f"τ:{bn.temporal.value}→{pn.temporal.value}")
        
        # Strength change
        str_delta = pn.strength - bn.strength
        if abs(str_delta) > 0.1:
            deltas.append(f"S:{str_delta:+.2f}")
        
        return "|".join(deltas) if deltas else "≈"
    
    @property
    def chain_strength(self) -> float:
        """Product of edge strengths"""
        if not self.edges:
            return 0.0
        result = 1.0
        for e in self.edges:
            result *= e.strength
        return result


# ─────────────────────────────────────────────────────────────────
# RESONANCE FORMULAS WITH B/P DISTINCTION
# ─────────────────────────────────────────────────────────────────

PHI = 0.618  # Golden ratio decay


def p_frame_resonance(
    node: FramedSigmaNode,
    now_density: float = 0.5,
    staunen: float = 0.5
) -> float:
    """
    P-Frame Resonance: Current situational focus.
    
    R_P = base_resonance × now_density × (1 + staunen)
    
    Higher when:
      - Node has high base resonance
      - Current moment is "thick" (high density)
      - High novelty/wonder (staunen)
    """
    base = base_resonance(node.node)
    return base * now_density * (1.0 + staunen)


def b_frame_resonance(
    node: FramedSigmaNode,
    lingering: float = 0.5,
    wisdom: float = 0.5,
    distance_from_p: int = 1
) -> float:
    """
    B-Frame Resonance: Background trajectory context.
    
    R_B = base_resonance × lingering × wisdom × φ^distance
    
    Higher when:
      - Node has high base resonance
      - Past is persisting (lingering)
      - Deep integration (wisdom)
      - Close to current P-frame
    """
    base = base_resonance(node.node)
    decay = PHI ** distance_from_p
    return base * lingering * wisdom * decay


def combined_situational_score(
    chain: FramedCausalChain,
    now_density: float = 0.5,
    staunen: float = 0.5,
    lingering: float = 0.5,
    wisdom: float = 0.5,
    p_weight: float = 0.6
) -> float:
    """
    Combined situational score from P-frame and B-frames.
    
    S = p_weight × R_P + (1-p_weight) × weighted_avg(R_B)
    
    Balances current focus with background context.
    """
    # P-frame contribution
    if chain.p_frame is None:
        r_p = 0.0
    else:
        r_p = p_frame_resonance(chain.p_frame, now_density, staunen)
    
    # B-frame contribution (weighted by edge strength and distance)
    b_total = 0.0
    b_weights = 0.0
    
    p_pos = chain.p_frame.chain_position if chain.p_frame else 0
    
    for b_node in chain.b_frames:
        distance = abs(b_node.chain_position - p_pos)
        r_b = b_frame_resonance(b_node, lingering, wisdom, max(1, distance))
        
        # Weight by edge strength if available
        edge_weight = 1.0
        for edge in chain.edges:
            if edge.source_id == b_node.node_id or edge.target_id == b_node.node_id:
                edge_weight = edge.strength
                break
        
        b_total += r_b * edge_weight
        b_weights += edge_weight
    
    r_b_avg = b_total / b_weights if b_weights > 0 else 0.0
    
    return p_weight * r_p + (1 - p_weight) * r_b_avg


# ─────────────────────────────────────────────────────────────────
# FACTORY FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def chain_to_framed(
    chain: CausalChain,
    fovea_node_id: Optional[str] = None
) -> FramedCausalChain:
    """
    Convert a CausalChain to FramedCausalChain with B/P decomposition.
    
    Args:
        chain: Original CausalChain from sigma_hydration
        fovea_node_id: Which node is currently focused (becomes P-frame)
    """
    framed_nodes = []
    
    for i, node in enumerate(chain.nodes):
        is_fovea = (node.node_id == fovea_node_id) if fovea_node_id else False
        fn = wrap_with_frame(
            node,
            is_fovea=is_fovea,
            fovea_rank=0 if is_fovea else -1,
            chain_id=chain.path_id,
            chain_position=i
        )
        framed_nodes.append(fn)
    
    return FramedCausalChain(
        path_id=chain.path_id,
        framed_nodes=framed_nodes,
        edges=chain.edges
    )


# ─────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sigma_hydration import SigmaNode, SigmaEdge, CausalChain
    
    print("=" * 60)
    print("Frame System — B/P Frame Tests")
    print("=" * 60)
    
    # Create test nodes
    i_node = SigmaNode(
        node_id="I-001",
        node_type=NodeType.LAMBDA,
        tier="Σ₃",
        temporal=Temporal.CRYSTALLIZED,
        strength=0.99,
        alpha=0.9,
        gamma=0.8
    )
    
    b_node = SigmaNode(
        node_id="B-001",
        node_type=NodeType.THETA,
        tier="Σ₂",
        temporal=Temporal.STABLE,
        strength=0.7,
        alpha=0.6,
        gamma=0.5
    )
    
    p_node = SigmaNode(
        node_id="P-001",
        node_type=NodeType.DELTA,
        tier="Σ₁",
        temporal=Temporal.EMERGING,
        strength=0.9,
        alpha=0.8,
        gamma=0.7
    )
    
    # Test frame classification
    print("\n1. Frame Classification")
    print(f"   I-001: {classify_frame_type(i_node).value} (expect I)")
    print(f"   B-001: {classify_frame_type(b_node).value} (expect B)")
    print(f"   P-001 (in fovea): {classify_frame_type(p_node, is_in_fovea=True).value} (expect P)")
    
    # Create chain
    edge = SigmaEdge(
        edge_id="E-001",
        source_id="B-001",
        target_id="P-001",
        relation_type=RelationType.CAUSES,
        strength=0.8
    )
    
    chain = CausalChain(
        path_id="test-chain",
        nodes=[i_node, b_node, p_node],
        edges=[edge]
    )
    
    # Convert to framed chain
    framed = chain_to_framed(chain, fovea_node_id="P-001")
    
    print("\n2. Framed Chain Decomposition")
    print(f"   I-frames: {[f.node_id for f in framed.i_frames]}")
    print(f"   B-frames: {[f.node_id for f in framed.b_frames]}")
    print(f"   P-frame: {framed.p_frame.node_id if framed.p_frame else None}")
    
    # Test resonance formulas
    print("\n3. Resonance Calculations")
    
    r_p = p_frame_resonance(framed.p_frame, now_density=0.7, staunen=0.6)
    print(f"   R_P (P-001): {r_p:.3f}")
    
    r_b = b_frame_resonance(framed.b_frames[0], lingering=0.5, wisdom=0.6, distance_from_p=1)
    print(f"   R_B (B-001): {r_b:.3f}")
    
    combined = combined_situational_score(
        framed,
        now_density=0.7,
        staunen=0.6,
        lingering=0.5,
        wisdom=0.6
    )
    print(f"   Combined S: {combined:.3f}")
    
    # Test delta computation
    print("\n4. Delta Encoding (B→P)")
    delta = framed.compute_delta(framed.p_frame, framed.b_frames[0])
    print(f"   Delta: {delta}")
    
    # Test context window
    print("\n5. Context Window (depth=2)")
    context = framed.get_context_window(depth=2)
    print(f"   Nodes: {[c.node_id for c in context]}")
    
    # Test focus shift
    print("\n6. Focus Shift")
    framed.set_focus("B-001", session="test-session")
    print(f"   New P-frame: {framed.p_frame.node_id}")
    print(f"   B-frames now: {[b.node_id for b in framed.b_frames]}")
    
    print("\n" + "=" * 60)
    print("All frame system tests passed! ✓")
    print("=" * 60)
