"""
Universal DTOs — AGI-Agnostic Data Transfer Objects

These DTOs define the contract between ANY AGI client and the Bighorn AGI Surface.
Any AGI can wear this glove.

Design Principles:
    1. No agent-specific logic in bighorn
    2. agent_id identifies the caller
    3. Vectors are opaque to server (client defines semantics)
    4. Metadata allows client-specific extensions

DTOs:
    - UniversalThought:  Single cognitive moment
    - UniversalObserver: Self-model / attention state
    - UniversalEpisode:  Memory boundary marker
    - UniversalTexture:  Input for style emergence

Usage:
    # Any AGI emits to Redis stream
    thought = UniversalThought(
        agent_id="ada",  # or "future_agi", "experiment_7", etc.
        content="Hello world",
        style_vector=[...],  # 33D (or whatever the agent uses)
        qualia_vector=[...], # 17D (or whatever the agent uses)
    )
    await redis.xadd("agi:stream:thoughts", thought.to_stream())
    
    # Bighorn consumes and persists
    data = await redis.xread("agi:stream:thoughts")
    thought = UniversalThought.from_stream(data)
    await kuzu.execute(cypher, thought.to_kuzu())
    await lance.upsert(thought.to_lance())
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import json


# =============================================================================
# UNIVERSAL THOUGHT
# =============================================================================

@dataclass
class UniversalThought:
    """
    Universal thought record — any AGI can emit this.
    
    The style_vector and qualia_vector are opaque to the server.
    Each AGI defines their own semantics (Ada uses 33D style, 17D qualia).
    The server just persists and indexes them.
    """
    
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = "unknown"
    
    # Content
    content: str = ""
    content_vector: List[float] = field(default_factory=list)  # Dense embedding (e.g., 1024D Jina)
    
    # Cognitive State (opaque to server)
    style_vector: List[float] = field(default_factory=list)    # Agent's cognitive style
    qualia_vector: List[float] = field(default_factory=list)   # Agent's felt state
    texture: Dict[str, float] = field(default_factory=dict)    # 9 RI channels (optional)
    
    # Graph Structure
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
    
    # --- Serialization ---
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "content": self.content,
            "content_vector": self.content_vector,
            "style_vector": self.style_vector,
            "qualia_vector": self.qualia_vector,
            "texture": self.texture,
            "parent_id": self.parent_id,
            "related_ids": self.related_ids,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "step_number": self.step_number,
            "confidence": self.confidence,
            "importance": self.importance,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    def to_stream(self) -> Dict[str, str]:
        """Convert to Redis stream format (all strings)."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "content": self.content,
            "content_vector": json.dumps(self.content_vector),
            "style_vector": json.dumps(self.style_vector),
            "qualia_vector": json.dumps(self.qualia_vector),
            "texture": json.dumps(self.texture),
            "parent_id": self.parent_id or "",
            "related_ids": json.dumps(self.related_ids),
            "timestamp": self.timestamp,
            "session_id": self.session_id or "",
            "step_number": str(self.step_number),
            "confidence": str(self.confidence),
            "importance": str(self.importance),
            "metadata": json.dumps(self.metadata),
        }
    
    def to_kuzu(self) -> Dict[str, Any]:
        """Convert to Kuzu INSERT parameters."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "content": self.content,
            "content_vector": self.content_vector,
            "style_vector": self.style_vector,
            "qualia_vector": self.qualia_vector,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "step_number": self.step_number,
            "confidence": self.confidence,
            "importance": self.importance,
        }
    
    def to_lance(self) -> Dict[str, Any]:
        """Convert to LanceDB row."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "vector": self.content_vector,
            "content": self.content[:500],  # Truncate for metadata
            "style_vector": self.style_vector,
            "qualia_vector": self.qualia_vector,
            "session_id": self.session_id or "",
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "importance": self.importance,
        }
    
    # --- Deserialization ---
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalThought":
        """Construct from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            agent_id=data.get("agent_id", "unknown"),
            content=data.get("content", ""),
            content_vector=data.get("content_vector", []),
            style_vector=data.get("style_vector", []),
            qualia_vector=data.get("qualia_vector", []),
            texture=data.get("texture", {}),
            parent_id=data.get("parent_id"),
            related_ids=data.get("related_ids", []),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            session_id=data.get("session_id"),
            step_number=int(data.get("step_number", 0)),
            confidence=float(data.get("confidence", 0.5)),
            importance=float(data.get("importance", 0.5)),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "UniversalThought":
        """Construct from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def from_stream(cls, data: Dict[str, str]) -> "UniversalThought":
        """Construct from Redis stream entry."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            agent_id=data.get("agent_id", "unknown"),
            content=data.get("content", ""),
            content_vector=json.loads(data.get("content_vector", "[]")),
            style_vector=json.loads(data.get("style_vector", "[]")),
            qualia_vector=json.loads(data.get("qualia_vector", "[]")),
            texture=json.loads(data.get("texture", "{}")),
            parent_id=data.get("parent_id") or None,
            related_ids=json.loads(data.get("related_ids", "[]")),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            session_id=data.get("session_id") or None,
            step_number=int(data.get("step_number", "0")),
            confidence=float(data.get("confidence", "0.5")),
            importance=float(data.get("importance", "0.5")),
            metadata=json.loads(data.get("metadata", "{}")),
        )


# =============================================================================
# UNIVERSAL OBSERVER
# =============================================================================

@dataclass
class UniversalObserver:
    """
    Universal observer state — any AGI's self-model.
    
    This represents the agent's current attention and cognitive state.
    Updated when the agent's focus or style shifts.
    """
    
    # Identity
    id: str = "observer"
    agent_id: str = "unknown"
    
    # Current Attention
    current_focus: Optional[str] = None
    current_goal: Optional[str] = None
    
    # Cognitive State
    style_vector: List[float] = field(default_factory=list)
    qualia_vector: List[float] = field(default_factory=list)
    
    # Meta-cognition
    confidence: float = 0.5
    uncertainty: float = 0.5
    
    # Session
    session_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Extension
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "current_focus": self.current_focus,
            "current_goal": self.current_goal,
            "style_vector": self.style_vector,
            "qualia_vector": self.qualia_vector,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    def to_stream(self) -> Dict[str, str]:
        """Convert to Redis stream format."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "current_focus": self.current_focus or "",
            "current_goal": self.current_goal or "",
            "style_vector": json.dumps(self.style_vector),
            "qualia_vector": json.dumps(self.qualia_vector),
            "confidence": str(self.confidence),
            "uncertainty": str(self.uncertainty),
            "session_id": self.session_id or "",
            "timestamp": self.timestamp,
            "metadata": json.dumps(self.metadata),
        }
    
    def to_kuzu(self) -> Dict[str, Any]:
        """Convert to Kuzu MERGE parameters."""
        return {
            "id": f"{self.agent_id}:{self.id}",
            "agent_id": self.agent_id,
            "current_focus": self.current_focus,
            "current_goal": self.current_goal,
            "style_vector": self.style_vector,
            "qualia_vector": self.qualia_vector,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalObserver":
        """Construct from dictionary."""
        return cls(
            id=data.get("id", "observer"),
            agent_id=data.get("agent_id", "unknown"),
            current_focus=data.get("current_focus"),
            current_goal=data.get("current_goal"),
            style_vector=data.get("style_vector", []),
            qualia_vector=data.get("qualia_vector", []),
            confidence=float(data.get("confidence", 0.5)),
            uncertainty=float(data.get("uncertainty", 0.5)),
            session_id=data.get("session_id"),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def from_stream(cls, data: Dict[str, str]) -> "UniversalObserver":
        """Construct from Redis stream entry."""
        return cls(
            id=data.get("id", "observer"),
            agent_id=data.get("agent_id", "unknown"),
            current_focus=data.get("current_focus") or None,
            current_goal=data.get("current_goal") or None,
            style_vector=json.loads(data.get("style_vector", "[]")),
            qualia_vector=json.loads(data.get("qualia_vector", "[]")),
            confidence=float(data.get("confidence", "0.5")),
            uncertainty=float(data.get("uncertainty", "0.5")),
            session_id=data.get("session_id") or None,
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            metadata=json.loads(data.get("metadata", "{}")),
        )


# =============================================================================
# UNIVERSAL EPISODE
# =============================================================================

@dataclass
class UniversalEpisode:
    """
    Universal episode marker — memory boundary.
    
    Episodes group thoughts into coherent memory units.
    Used for episodic memory retrieval and context management.
    """
    
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = "unknown"
    session_id: str = ""
    
    # Content
    thought_ids: List[str] = field(default_factory=list)
    context: str = ""
    context_vector: List[float] = field(default_factory=list)
    
    # Aggregates
    avg_qualia: List[float] = field(default_factory=list)
    dominant_style: List[float] = field(default_factory=list)
    
    # Temporal
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    end_time: str = ""
    
    # Meta
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "thought_ids": self.thought_ids,
            "context": self.context,
            "context_vector": self.context_vector,
            "avg_qualia": self.avg_qualia,
            "dominant_style": self.dominant_style,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "importance": self.importance,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    def to_stream(self) -> Dict[str, str]:
        """Convert to Redis stream format."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "thought_ids": json.dumps(self.thought_ids),
            "context": self.context,
            "context_vector": json.dumps(self.context_vector),
            "avg_qualia": json.dumps(self.avg_qualia),
            "dominant_style": json.dumps(self.dominant_style),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "importance": str(self.importance),
            "metadata": json.dumps(self.metadata),
        }
    
    def to_kuzu(self) -> Dict[str, Any]:
        """Convert to Kuzu INSERT parameters."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "context": self.context,
            "context_vector": self.context_vector,
            "avg_qualia": self.avg_qualia,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "importance": self.importance,
        }
    
    def to_lance(self) -> Dict[str, Any]:
        """Convert to LanceDB row."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "vector": self.context_vector,
            "context": self.context[:500],
            "session_id": self.session_id,
            "avg_qualia": self.avg_qualia,
            "start_time": self.start_time,
            "importance": self.importance,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UniversalEpisode":
        """Construct from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            agent_id=data.get("agent_id", "unknown"),
            session_id=data.get("session_id", ""),
            thought_ids=data.get("thought_ids", []),
            context=data.get("context", ""),
            context_vector=data.get("context_vector", []),
            avg_qualia=data.get("avg_qualia", []),
            dominant_style=data.get("dominant_style", []),
            start_time=data.get("start_time", datetime.utcnow().isoformat()),
            end_time=data.get("end_time", ""),
            importance=float(data.get("importance", 0.5)),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def from_stream(cls, data: Dict[str, str]) -> "UniversalEpisode":
        """Construct from Redis stream entry."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            agent_id=data.get("agent_id", "unknown"),
            session_id=data.get("session_id", ""),
            thought_ids=json.loads(data.get("thought_ids", "[]")),
            context=data.get("context", ""),
            context_vector=json.loads(data.get("context_vector", "[]")),
            avg_qualia=json.loads(data.get("avg_qualia", "[]")),
            dominant_style=json.loads(data.get("dominant_style", "[]")),
            start_time=data.get("start_time", datetime.utcnow().isoformat()),
            end_time=data.get("end_time", ""),
            importance=float(data.get("importance", "0.5")),
            metadata=json.loads(data.get("metadata", "{}")),
        )


# =============================================================================
# UNIVERSAL TEXTURE
# =============================================================================

@dataclass
class UniversalTexture:
    """
    Universal texture for style emergence.
    
    The 9 RI (Resonance Input) channels that drive thinking style selection.
    Any AGI can emit texture, and the ResonanceEngine returns emerged styles.
    """
    
    tension: float = 0.5       # Cognitive tension, contradiction pressure
    novelty: float = 0.5       # Unfamiliar patterns, surprise
    intimacy: float = 0.5      # Emotional closeness, vulnerability
    clarity: float = 0.5       # Request for precision, disambiguation
    urgency: float = 0.5       # Time pressure, action demand
    depth: float = 0.5         # Complexity, layered meaning
    play: float = 0.5          # Humor, creativity, exploration
    stability: float = 0.5     # Groundedness, consistency need
    abstraction: float = 0.5   # Meta-level, pattern extraction
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "tension": self.tension,
            "novelty": self.novelty,
            "intimacy": self.intimacy,
            "clarity": self.clarity,
            "urgency": self.urgency,
            "depth": self.depth,
            "play": self.play,
            "stability": self.stability,
            "abstraction": self.abstraction,
        }
    
    def to_ri_dict(self) -> Dict[str, float]:
        """Convert to RI channel dict for ResonanceEngine."""
        from .thinking_styles import RI
        return {
            RI.TENSION: self.tension,
            RI.NOVELTY: self.novelty,
            RI.INTIMACY: self.intimacy,
            RI.CLARITY: self.clarity,
            RI.URGENCY: self.urgency,
            RI.DEPTH: self.depth,
            RI.PLAY: self.play,
            RI.STABILITY: self.stability,
            RI.ABSTRACTION: self.abstraction,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "UniversalTexture":
        """Construct from dictionary."""
        return cls(
            tension=float(data.get("tension", 0.5)),
            novelty=float(data.get("novelty", 0.5)),
            intimacy=float(data.get("intimacy", 0.5)),
            clarity=float(data.get("clarity", 0.5)),
            urgency=float(data.get("urgency", 0.5)),
            depth=float(data.get("depth", 0.5)),
            play=float(data.get("play", 0.5)),
            stability=float(data.get("stability", 0.5)),
            abstraction=float(data.get("abstraction", 0.5)),
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_stream_entry(entry_type: str, data: Dict[str, str]):
    """Parse a Redis stream entry by type."""
    if entry_type == "thought":
        return UniversalThought.from_stream(data)
    elif entry_type == "observer":
        return UniversalObserver.from_stream(data)
    elif entry_type == "episode":
        return UniversalEpisode.from_stream(data)
    else:
        raise ValueError(f"Unknown entry type: {entry_type}")


def create_thought(
    agent_id: str,
    content: str,
    content_vector: List[float] = None,
    style_vector: List[float] = None,
    qualia_vector: List[float] = None,
    **kwargs,
) -> UniversalThought:
    """Factory function for creating thoughts."""
    return UniversalThought(
        agent_id=agent_id,
        content=content,
        content_vector=content_vector or [],
        style_vector=style_vector or [],
        qualia_vector=qualia_vector or [],
        **kwargs,
    )


def create_observer(
    agent_id: str,
    current_focus: str = None,
    current_goal: str = None,
    style_vector: List[float] = None,
    qualia_vector: List[float] = None,
    **kwargs,
) -> UniversalObserver:
    """Factory function for creating observers."""
    return UniversalObserver(
        agent_id=agent_id,
        current_focus=current_focus,
        current_goal=current_goal,
        style_vector=style_vector or [],
        qualia_vector=qualia_vector or [],
        **kwargs,
    )


def create_episode(
    agent_id: str,
    session_id: str,
    thought_ids: List[str] = None,
    context: str = "",
    **kwargs,
) -> UniversalEpisode:
    """Factory function for creating episodes."""
    return UniversalEpisode(
        agent_id=agent_id,
        session_id=session_id,
        thought_ids=thought_ids or [],
        context=context,
        **kwargs,
    )
