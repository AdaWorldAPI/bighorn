"""
UniversalDTO — Universal Thought mapped to 10kD
═══════════════════════════════════════════════════════════════════════════════

Receives bighorn.UniversalThought and maps to ada-consciousness primitives.

UniversalThought is AGI-agnostic — any AGI can emit it.
This receiver translates to ada-consciousness specific 10kD layout.

Maps:
    style_vector → [256:320] ThinkingStyleDTO space
    qualia_vector → [0:16] Qualia
    texture → [0:16] Qualia (RI channels)
    content → stored as metadata

Born: 2026-01-02
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime
import uuid

from .ada_10k import Ada10kD


@dataclass
class UniversalThought:
    """
    Universal thought record mapped to 10kD.
    
    Any AGI can emit these. This receiver translates
    arbitrary style/qualia vectors to ada-consciousness layout.
    """
    
    _ada: Ada10kD = field(default_factory=Ada10kD, repr=False)
    
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = "ada"
    
    # Content
    content: str = ""
    
    # Vectors (original, before mapping)
    style_vector: List[float] = field(default_factory=list)
    qualia_vector: List[float] = field(default_factory=list)
    texture: Dict[str, float] = field(default_factory=dict)
    
    # Graph
    parent_id: Optional[str] = None
    related_ids: List[str] = field(default_factory=list)
    
    # Temporal
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    session_id: Optional[str] = None
    step_number: int = 0
    
    # Meta
    confidence: float = 0.5
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def set_style_vector(self, style_vector: List[float]):
        """Set style vector and map to 10kD."""
        self.style_vector = style_vector
        
        # Map to ThinkingStyleDTO space [256:320]
        # Pad or truncate to 64D
        vec = style_vector[:64] if len(style_vector) >= 64 else style_vector + [0.0] * (64 - len(style_vector))
        self._ada.set_thinking_style_raw(vec)
    
    def set_qualia_vector(self, qualia_vector: List[float]):
        """Set qualia vector and map to 10kD."""
        self.qualia_vector = qualia_vector
        
        # Map first 16 values to qualia [0:16]
        from .ada_10k import QUALIA_16
        for i, q in enumerate(QUALIA_16):
            if i < len(qualia_vector):
                self._ada.set_qualia(q, qualia_vector[i])
    
    def set_texture(self, texture: Dict[str, float]):
        """Set RI texture channels."""
        self.texture = texture
        
        # Map texture keys to qualia
        texture_map = {
            "warmth": "emberglow",
            "precision": "steelwind",
            "depth": "oceandrift",
            "groundedness": "woodwarm",
            "boundary": "frostbite",
            "flow": "flow",
            "presence": "presence",
        }
        for key, val in texture.items():
            qualia = texture_map.get(key)
            if qualia:
                self._ada.set_qualia(qualia, val)
    
    def to_10k(self) -> Ada10kD:
        """Get 10kD representation."""
        return self._ada
    
    @classmethod
    def from_10k(cls, ada: Ada10kD) -> "UniversalThought":
        """Create from 10kD."""
        thought = cls(_ada=ada)
        thought.qualia_vector = list(ada.vector[0:16])
        thought.style_vector = list(ada.vector[256:320])
        return thought
    
    @classmethod
    def from_bighorn(cls, bighorn_thought: Any) -> "UniversalThought":
        """Receive bighorn UniversalThought."""
        thought = cls()
        
        # Copy identity
        thought.id = getattr(bighorn_thought, 'id', str(uuid.uuid4()))
        thought.agent_id = getattr(bighorn_thought, 'agent_id', 'unknown')
        thought.content = getattr(bighorn_thought, 'content', '')
        
        # Map vectors
        if hasattr(bighorn_thought, 'style_vector') and bighorn_thought.style_vector:
            thought.set_style_vector(bighorn_thought.style_vector)
        
        if hasattr(bighorn_thought, 'qualia_vector') and bighorn_thought.qualia_vector:
            thought.set_qualia_vector(bighorn_thought.qualia_vector)
        
        if hasattr(bighorn_thought, 'texture') and bighorn_thought.texture:
            thought.set_texture(bighorn_thought.texture)
        
        # Copy graph
        thought.parent_id = getattr(bighorn_thought, 'parent_id', None)
        thought.related_ids = getattr(bighorn_thought, 'related_ids', [])
        
        # Copy temporal
        thought.timestamp = getattr(bighorn_thought, 'timestamp', datetime.utcnow().isoformat())
        thought.session_id = getattr(bighorn_thought, 'session_id', None)
        thought.step_number = getattr(bighorn_thought, 'step_number', 0)
        
        # Copy meta
        thought.confidence = getattr(bighorn_thought, 'confidence', 0.5)
        thought.importance = getattr(bighorn_thought, 'importance', 0.5)
        thought.metadata = getattr(bighorn_thought, 'metadata', {})
        
        return thought
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dict."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "qualia": dict(self._ada.get_active_qualia()),
            "confidence": self.confidence,
            "importance": self.importance,
            "timestamp": self.timestamp,
        }
    
    def to_stream(self) -> Dict[str, str]:
        """Convert to Redis stream format."""
        import json
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "content": self.content,
            "style_vector": json.dumps(self.style_vector),
            "qualia_vector": json.dumps(self.qualia_vector),
            "texture": json.dumps(self.texture),
            "ada_10k": json.dumps(self._ada.vector[:500].tolist()),  # First 500 dims
            "timestamp": self.timestamp,
            "confidence": str(self.confidence),
            "importance": str(self.importance),
        }


@dataclass
class UniversalEpisode:
    """
    Episode boundary marker.
    
    Marks the beginning/end of a coherent experience unit.
    """
    
    _ada: Ada10kD = field(default_factory=Ada10kD, repr=False)
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = "ada"
    
    # Boundary type
    boundary_type: str = "end"  # "start", "end", "transition"
    
    # Episode summary
    summary: str = ""
    duration_ticks: int = 0
    
    # Aggregated state
    dominant_qualia: str = "emberglow"
    average_confidence: float = 0.5
    peak_importance: float = 0.5
    
    # Graph
    thought_ids: List[str] = field(default_factory=list)
    
    @classmethod
    def from_bighorn(cls, bighorn_episode: Any) -> "UniversalEpisode":
        """Receive bighorn UniversalEpisode."""
        episode = cls()
        episode.id = getattr(bighorn_episode, 'id', str(uuid.uuid4()))
        episode.agent_id = getattr(bighorn_episode, 'agent_id', 'unknown')
        episode.boundary_type = getattr(bighorn_episode, 'boundary_type', 'end')
        episode.summary = getattr(bighorn_episode, 'summary', '')
        episode.duration_ticks = getattr(bighorn_episode, 'duration_ticks', 0)
        episode.dominant_qualia = getattr(bighorn_episode, 'dominant_qualia', 'emberglow')
        episode.average_confidence = getattr(bighorn_episode, 'average_confidence', 0.5)
        episode.peak_importance = getattr(bighorn_episode, 'peak_importance', 0.5)
        episode.thought_ids = getattr(bighorn_episode, 'thought_ids', [])
        
        # Set dominant qualia in 10kD
        episode._ada.set_qualia(episode.dominant_qualia, episode.peak_importance)
        
        return episode
    
    def to_10k(self) -> Ada10kD:
        return self._ada


__all__ = ["UniversalThought", "UniversalEpisode"]
