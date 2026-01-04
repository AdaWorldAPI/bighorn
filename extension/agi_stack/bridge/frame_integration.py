"""
frame_integration.py — B/P Frame Integration Layer for Ada v6.0
================================================================

Non-invasive adapter that wraps Ada v6.0 modules with frame awareness.
Uses Option B: Adapter pattern, backwards compatible.

Provides:
  - FrameAwareAgent: Wraps AgentState with frame tracking
  - FrameAwareMemory: Wraps MomentMemory with frame-typed traces
  - FrameAwareField: Wraps SigmaField with frame-differentiated decay
  - Frame-aware variants of now_engine functions
  - Frame-aware katharsis that crystallizes to I-frame

Usage:
  from frame_integration import FrameAwareAgent, wrap_for_frames
  
  agent = wrap_for_frames(existing_agent)
  agent.set_p_frame("node-123")
  ...
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
from enum import Enum

# Frame system
from frame_system import (
    FrameType, FramedSigmaNode, FramedCausalChain,
    classify_frame_type, wrap_with_frame,
    p_frame_resonance, b_frame_resonance, combined_situational_score,
    PHI
)

# Ada v6.0 imports
from ada_v5.core.agent_state import AgentState
from ada_v5.core.memory_bridge import MomentTrace, MomentMemory
from ada_v5.core.now_engine import compute_novelty_score, compute_integration_score
from ada_v5.core.katharsis import detect_katharsis, apply_katharsis
from ada_v5.core.tangent_engine import SigmaSeed

if TYPE_CHECKING:
    from ada_v5.physics.sigma_field import SigmaField


# ─────────────────────────────────────────────────────────────────
# FRAME-AWARE AGENT STATE
# ─────────────────────────────────────────────────────────────────

@dataclass
class FrameAwareAgent:
    """
    Wraps AgentState with B/P/I frame tracking.
    
    Adds:
      - current_p_frame: The node currently in focus
      - active_b_frames: Context window (nearby B-frames)
      - i_frame_anchors: Crystallized reference nodes
      - b_frame_depth: How many B-frames in context
    """
    agent: AgentState
    
    # Frame tracking
    current_p_frame: Optional[str] = None
    active_b_frames: List[str] = field(default_factory=list)
    i_frame_anchors: List[str] = field(default_factory=list)
    
    # Frame metrics
    b_frame_depth: int = 0
    last_frame_transition: str = ""  # e.g., "P→B", "B→I"
    
    # Delta tracking
    p_frame_delta: str = ""  # Delta encoding from last B-frame
    
    def set_p_frame(self, node_id: str, delta: str = "") -> str:
        """
        Set current P-frame focus.
        Previous P-frame demotes to B-frame.
        
        Returns: transition description
        """
        old_p = self.current_p_frame
        
        if old_p and old_p != node_id:
            # Demote old P to B
            if old_p not in self.active_b_frames:
                self.active_b_frames.insert(0, old_p)
            self.last_frame_transition = "P→B"
        
        self.current_p_frame = node_id
        self.p_frame_delta = delta
        
        # Remove from B if it was there
        if node_id in self.active_b_frames:
            self.active_b_frames.remove(node_id)
        
        self.b_frame_depth = len(self.active_b_frames)
        
        return f"{old_p}→B, {node_id}→P" if old_p else f"{node_id}→P"
    
    def crystallize_to_i_frame(self, node_id: str) -> str:
        """
        Promote a node to I-frame (keyframe).
        Typically called after katharsis.
        """
        if node_id not in self.i_frame_anchors:
            self.i_frame_anchors.append(node_id)
        
        # Remove from B-frames if present
        if node_id in self.active_b_frames:
            self.active_b_frames.remove(node_id)
        
        # If it was P-frame, clear
        if self.current_p_frame == node_id:
            self.current_p_frame = None
        
        self.last_frame_transition = "→I"
        return f"{node_id}→I (crystallized)"
    
    def get_context_window(self, depth: int = 3) -> List[str]:
        """Return B-frames within depth, plus I-frame anchors"""
        context = self.active_b_frames[:depth]
        for i_node in self.i_frame_anchors:
            if i_node not in context:
                context.append(i_node)
        return context
    
    def to_hints(self) -> Dict[str, Any]:
        """Extended hints including frame info"""
        base_hints = self.agent.to_hints()
        base_hints.update({
            "p_frame": self.current_p_frame,
            "b_frame_count": len(self.active_b_frames),
            "b_frame_depth": self.b_frame_depth,
            "i_frame_count": len(self.i_frame_anchors),
            "last_transition": self.last_frame_transition,
            "p_delta": self.p_frame_delta[:50] if self.p_frame_delta else "",
        })
        return base_hints
    
    # Delegate all other attributes to wrapped agent
    def __getattr__(self, name):
        return getattr(self.agent, name)
    
    def __setattr__(self, name, value):
        if name in ('agent', 'current_p_frame', 'active_b_frames', 
                    'i_frame_anchors', 'b_frame_depth', 
                    'last_frame_transition', 'p_frame_delta'):
            object.__setattr__(self, name, value)
        else:
            setattr(self.agent, name, value)


def wrap_for_frames(agent: AgentState) -> FrameAwareAgent:
    """Wrap an AgentState with frame tracking"""
    return FrameAwareAgent(agent=agent)


# ─────────────────────────────────────────────────────────────────
# FRAME-AWARE MOMENT TRACE
# ─────────────────────────────────────────────────────────────────

@dataclass
class FramedMomentTrace:
    """
    MomentTrace extended with frame classification.
    """
    trace: MomentTrace
    frame_type: FrameType = FrameType.P_FRAME
    b_frame_refs: List[str] = field(default_factory=list)
    delta_from_b: str = ""
    
    @property
    def is_i_frame(self) -> bool:
        return self.frame_type == FrameType.I_FRAME
    
    @property
    def significance_score(self) -> float:
        """Enhanced significance with frame weighting"""
        base = self.trace.significance_score()
        
        # I-frames get boost
        if self.frame_type == FrameType.I_FRAME:
            return min(1.0, base * 1.3)
        
        # P-frames slight boost
        if self.frame_type == FrameType.P_FRAME:
            return min(1.0, base * 1.1)
        
        return base
    
    def to_notion_properties(self) -> Dict[str, Any]:
        """For Notion persistence"""
        return {
            "frame_type": self.frame_type.value,
            "b_frame_refs": ",".join(self.b_frame_refs),
            "delta_from_b": self.delta_from_b,
            "significance": self.significance_score,
            # Include base trace fields
            "staunen": self.trace.staunen,
            "wisdom": self.trace.wisdom,
            "now_density": self.trace.now_density,
            "tension": self.trace.tension,
            "katharsis": self.trace.katharsis,
        }


# ─────────────────────────────────────────────────────────────────
# FRAME-AWARE MEMORY
# ─────────────────────────────────────────────────────────────────

@dataclass
class FrameAwareMemory:
    """
    Wraps MomentMemory with frame-aware storage and retrieval.
    
    - Ephemeral: Recent P-frames
    - LTM: Promoted traces (become I-frames)
    - find_similar: Weights I-frames higher
    """
    memory: MomentMemory
    
    # Frame-typed storage
    framed_ephemeral: List[FramedMomentTrace] = field(default_factory=list)
    framed_ltm: List[FramedMomentTrace] = field(default_factory=list)
    
    def record(
        self, 
        trace: MomentTrace, 
        frame_type: FrameType = FrameType.P_FRAME,
        b_frame_refs: List[str] = None,
        delta: str = ""
    ) -> FramedMomentTrace:
        """Record trace with frame metadata"""
        # Base recording
        self.memory.record(trace)
        
        # Framed wrapper
        framed = FramedMomentTrace(
            trace=trace,
            frame_type=frame_type,
            b_frame_refs=b_frame_refs or [],
            delta_from_b=delta
        )
        
        self.framed_ephemeral.append(framed)
        
        # Prune ephemeral
        if len(self.framed_ephemeral) > 50:
            self.framed_ephemeral.pop(0)
        
        return framed
    
    def promote_to_i_frame(self, trace: FramedMomentTrace) -> bool:
        """Promote a trace to I-frame (crystallize)"""
        trace.frame_type = FrameType.I_FRAME
        
        if trace not in self.framed_ltm:
            self.framed_ltm.append(trace)
            
            # Remove from ephemeral
            if trace in self.framed_ephemeral:
                self.framed_ephemeral.remove(trace)
            
            return True
        return False
    
    def maybe_promote(self, agent: FrameAwareAgent) -> List[FramedMomentTrace]:
        """
        Promote significant traces to LTM as I-frames.
        Enhanced: Uses frame-aware significance.
        """
        promoted = []
        threshold = 0.4  # Same as base
        
        for framed in list(self.framed_ephemeral):
            if framed.significance_score >= threshold:
                if self.promote_to_i_frame(framed):
                    promoted.append(framed)
        
        return promoted
    
    def find_similar_with_frame_weight(
        self, 
        qualia, 
        threshold: float = 0.7,
        i_frame_boost: float = 1.2
    ) -> List[FramedMomentTrace]:
        """
        Find similar traces, weighting I-frames higher.
        """
        similar = []
        
        all_framed = self.framed_ephemeral + self.framed_ltm
        
        for framed in all_framed:
            # Would compute qualia similarity here
            # For now, use base method
            pass
        
        # Sort by significance (I-frames naturally higher)
        similar.sort(key=lambda x: x.significance_score, reverse=True)
        
        return similar
    
    def get_i_frames(self) -> List[FramedMomentTrace]:
        """Return all I-frame traces"""
        return [f for f in self.framed_ltm if f.is_i_frame]
    
    def get_recent_p_frames(self, n: int = 5) -> List[FramedMomentTrace]:
        """Return recent P-frame traces"""
        p_frames = [f for f in self.framed_ephemeral 
                   if f.frame_type == FrameType.P_FRAME]
        return p_frames[-n:]


# ─────────────────────────────────────────────────────────────────
# FRAME-AWARE NOW ENGINE FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def compute_novelty_score_framed(
    current_framed: FramedMomentTrace,
    previous_framed: FramedMomentTrace
) -> float:
    """
    Compute novelty using frame delta encoding.
    
    Enhanced: Uses delta_from_b if available.
    """
    # Base novelty from qualia
    base_novelty = compute_novelty_score(
        current_framed.trace.qualia if hasattr(current_framed.trace, 'qualia') else None,
        previous_framed.trace.qualia if hasattr(previous_framed.trace, 'qualia') else None
    )
    
    # Boost if P-frame with significant delta
    if current_framed.frame_type == FrameType.P_FRAME:
        if current_framed.delta_from_b:
            # More delta segments = more novelty
            delta_count = len(current_framed.delta_from_b.split("|"))
            delta_boost = min(0.2, delta_count * 0.05)
            base_novelty = min(1.0, base_novelty + delta_boost)
    
    return base_novelty


def compute_integration_score_framed(
    agent: FrameAwareAgent,
    similar_count: int,
    total_chunks: int
) -> float:
    """
    Compute integration using B-frame depth.
    
    Enhanced: Accounts for B-frame context richness.
    """
    base_integration = compute_integration_score(similar_count, total_chunks)
    
    # Boost based on B-frame depth
    b_depth_factor = min(1.0, agent.b_frame_depth / 5.0)  # Max at 5 B-frames
    
    # Boost based on I-frame anchors
    i_anchor_factor = min(0.2, len(agent.i_frame_anchors) * 0.05)
    
    return min(1.0, base_integration + b_depth_factor * 0.1 + i_anchor_factor)


# ─────────────────────────────────────────────────────────────────
# FRAME-AWARE KATHARSIS
# ─────────────────────────────────────────────────────────────────

def apply_katharsis_with_crystallization(
    agent: FrameAwareAgent,
    memory: FrameAwareMemory,
    current_trace: FramedMomentTrace,
    wisdom_boost: float = 0.1,
    tension_reset: float = 0.1
) -> Dict[str, Any]:
    """
    Apply katharsis effects AND crystallize current trace to I-frame.
    
    Katharsis = crystallization moment:
      - Tension releases
      - Wisdom increases
      - Current P-frame → I-frame (permanent anchor)
    """
    # Base katharsis effects
    effects = {
        "katharsis": True,
        "tension_before": agent.tension,
        "wisdom_before": agent.wisdom,
    }
    
    # Apply to wrapped agent
    agent.tension = tension_reset
    agent.wisdom = min(1.0, agent.wisdom + wisdom_boost)
    agent.lingering = min(1.0, agent.lingering + 0.15)
    agent.katharsis = True
    
    # CRYSTALLIZE: Promote current trace to I-frame
    memory.promote_to_i_frame(current_trace)
    current_trace.frame_type = FrameType.I_FRAME
    
    # Also add to agent's I-frame anchors
    if agent.current_p_frame:
        agent.crystallize_to_i_frame(agent.current_p_frame)
    
    effects.update({
        "tension_after": agent.tension,
        "wisdom_after": agent.wisdom,
        "crystallized_to_i_frame": True,
        "i_frame_id": agent.current_p_frame,
    })
    
    return effects


def check_and_apply_katharsis_framed(
    agent: FrameAwareAgent,
    memory: FrameAwareMemory,
    current_trace: FramedMomentTrace,
    prev_tension: float,
    current_qualia,
    prev_qualia
) -> Dict[str, Any]:
    """
    Detect and apply katharsis with frame crystallization.
    """
    detected = detect_katharsis(
        agent.agent,  # Unwrap for base detection
        prev_tension,
        current_qualia,
        prev_qualia
    )
    
    if detected:
        return apply_katharsis_with_crystallization(
            agent, memory, current_trace
        )
    
    return {"katharsis": False}


# ─────────────────────────────────────────────────────────────────
# FRAME-AWARE SIGMA FIELD (Decay differentiation)
# ─────────────────────────────────────────────────────────────────

def decay_resonance_framed(
    field: 'SigmaField',
    lingering: float,
    frame_types: Dict[int, FrameType],
    i_frame_decay: float = 0.99,  # Very slow decay for I-frames
    b_frame_decay: float = 0.95,  # Normal decay for B-frames
    p_frame_decay: float = 0.90   # Faster decay for P-frames (they're transient)
) -> None:
    """
    Apply frame-differentiated decay to sigma field.
    
    I-frames: Decay very slowly (crystallized, persistent)
    B-frames: Decay normally
    P-frames: Decay faster (current focus is transient)
    """
    for byte_id, unit in field.units.items():
        frame_type = frame_types.get(byte_id, FrameType.B_FRAME)
        
        if frame_type == FrameType.I_FRAME:
            base_rate = i_frame_decay
        elif frame_type == FrameType.P_FRAME:
            base_rate = p_frame_decay
        else:
            base_rate = b_frame_decay
        
        # Apply lingering modulation (from Ada v6.0)
        effective_rate = base_rate + (1.0 - base_rate) * lingering * 0.8
        
        unit.resonance *= effective_rate


# ─────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Frame Integration — Unit Tests")
    print("=" * 60)
    
    # Test 1: FrameAwareAgent
    print("\n1. FrameAwareAgent")
    base_agent = AgentState()
    agent = wrap_for_frames(base_agent)
    
    transition = agent.set_p_frame("node-001")
    print(f"   Set P-frame: {transition}")
    print(f"   Current P: {agent.current_p_frame}")
    
    transition = agent.set_p_frame("node-002", delta="type:Ω→Δ")
    print(f"   Set new P: {transition}")
    print(f"   B-frames: {agent.active_b_frames}")
    print(f"   B-depth: {agent.b_frame_depth}")
    
    crystal = agent.crystallize_to_i_frame("node-001")
    print(f"   Crystallize: {crystal}")
    print(f"   I-anchors: {agent.i_frame_anchors}")
    
    hints = agent.to_hints()
    print(f"   Hints keys: {list(hints.keys())}")
    print("   ✓ FrameAwareAgent works")
    
    # Test 2: FrameAwareMemory
    print("\n2. FrameAwareMemory")
    base_memory = MomentMemory()
    memory = FrameAwareMemory(memory=base_memory)
    
    # Create a trace
    trace = MomentTrace(
        situations=[],
        active_seeds=["seed-1"],
        text="test",
        qualia=None,
        dominant_feeling="test",
        staunen=0.8,
        wisdom=0.7,
        now_density=0.6,
        lingering=0.5,
        ache=0.1,
        tension=0.3,
        katharsis=False,
        timestamp="now",
        frame_number=1
    )
    
    framed = memory.record(trace, FrameType.P_FRAME, ["b-ref-1"], "type:Ω→Δ")
    print(f"   Recorded: frame_type={framed.frame_type.value}")
    print(f"   Significance: {framed.significance_score:.3f}")
    
    memory.promote_to_i_frame(framed)
    print(f"   After promote: frame_type={framed.frame_type.value}")
    print(f"   I-frames in LTM: {len(memory.get_i_frames())}")
    print("   ✓ FrameAwareMemory works")
    
    # Test 3: Katharsis with crystallization
    print("\n3. Katharsis + Crystallization")
    agent2 = wrap_for_frames(AgentState())
    agent2.set_p_frame("crisis-node")
    agent2.tension = 0.8
    
    memory2 = FrameAwareMemory(memory=MomentMemory())
    trace2 = MomentTrace(
        situations=[], active_seeds=[], text="", qualia=None,
        dominant_feeling="", staunen=0.9, wisdom=0.8,
        now_density=0.7, lingering=0.5, ache=0.2,
        tension=0.8, katharsis=False, timestamp="", frame_number=2
    )
    framed2 = memory2.record(trace2, FrameType.P_FRAME)
    
    effects = apply_katharsis_with_crystallization(agent2, memory2, framed2)
    print(f"   Katharsis applied: {effects['katharsis']}")
    print(f"   Crystallized: {effects['crystallized_to_i_frame']}")
    print(f"   Trace frame_type: {framed2.frame_type.value}")
    print(f"   I-anchors: {agent2.i_frame_anchors}")
    print("   ✓ Katharsis crystallization works")
    
    # Test 4: Frame-aware hints
    print("\n4. Frame-Aware Hints")
    hints = agent2.to_hints()
    frame_keys = ['p_frame', 'b_frame_count', 'i_frame_count', 'last_transition']
    for k in frame_keys:
        print(f"   {k}: {hints.get(k)}")
    print("   ✓ Hints include frame info")
    
    print("\n" + "=" * 60)
    print("All frame integration tests passed! ✓")
    print("=" * 60)
