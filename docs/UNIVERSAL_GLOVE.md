# AGI Glove — Universal Integration Layer

**Purpose:** Define the contract between any AGI client (Ada first, others later) and the Bighorn AGI Surface.

**Design Principle:** The glove fits any hand. Ada-specific details live in ada-consciousness, universal contracts live in bighorn.

---

## Current State Analysis

### What ada-consciousness HAS (client-side)

| Component | Location | Dimensions | Status |
|-----------|----------|------------|--------|
| **ThinkingStyleVector** | `core/dto/thinking_style.py` | 33D | ✓ Complete |
| **QualiaVector** | `core/dto/qualia.py` | 17D | ✓ Complete |
| **ThoughtDTO** | `core/dto/thought.py` | composite | ✓ Complete |
| **EpisodeDTO** | `core/dto/thought.py` | composite | ✓ Complete |
| **SoulDTO** | `core/dto/thought.py` | 33D + 17D + context | ✓ Complete |
| **AGI Bridge** | `core/agi_bridge.py` | — | ✓ Emits to Redis streams |

### What bighorn HAS (server-side)

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| **ThinkingStyle** | `extension/ada_surface/thinking_styles.py` | 36 styles with resonance | ✓ Complete |
| **ResonanceEngine** | `extension/ada_surface/thinking_styles.py` | Style emergence from texture | ✓ Complete |
| **ThoughtRequest** | `extension/ada_surface/main.py` | API input model | ✓ Complete |
| **Kuzu Schema** | `extension/ada_surface/schema.kuzu` | Graph persistence | ✓ Complete |
| **LanceDB Client** | `extension/ada_surface/lance_client.py` | Vector storage | ✓ Complete |
| **Redis Consumers** | `extension/ada_surface/consumers.py` | Stream processing | ✓ Complete |

### Alignment Gaps

| Gap | ada-consciousness | bighorn | Resolution |
|-----|-------------------|---------|------------|
| **Style Categories** | None (uses presets) | 9 categories | OK — server defines categories |
| **RI Channels** | Not explicit | 9 channels | Need: Texture mapping |
| **Glyph Sparse** | Different format | `[(idx, val), ...]` | Need: Converter |
| **SoulMode** | 7 modes | None | OK — client-specific |
| **Observer** | Not explicit | Observer node in Kuzu | Need: Observer DTO |

---

## Universal Glove Design

The AGI Surface should accept **universal DTOs** that any AGI can emit.

### Universal Thought (what any AGI emits)

```python
@dataclass
class UniversalThought:
    """Universal thought record — any AGI can emit this."""
    
    # Identity
    id: str                              # UUID
    agent_id: str = "unknown"            # Which AGI (ada, future_agi, etc.)
    
    # Content
    content: str = ""
    content_vector: List[float] = []     # 1024D Jina (or any dense embedding)
    
    # Cognitive State (Universal)
    style_vector: List[float] = []       # 33D (client defines meaning)
    qualia_vector: List[float] = []      # 17D (client defines meaning)
    texture: Dict[str, float] = {}       # 9 RI channels (optional)
    
    # Graph
    parent_id: Optional[str] = None
    related_ids: List[str] = []
    
    # Temporal
    timestamp: str = ""                  # ISO format
    session_id: Optional[str] = None
    step_number: int = 0
    
    # Meta
    confidence: float = 0.5
    importance: float = 0.5
    metadata: Dict[str, Any] = {}        # Client-specific data
```

### Universal Observer (self-model)

```python
@dataclass
class UniversalObserver:
    """Universal observer state — any AGI's self-model."""
    
    id: str = "observer"
    agent_id: str = "unknown"
    
    # Current attention
    current_focus: Optional[str] = None
    current_goal: Optional[str] = None
    
    # Cognitive state
    style_vector: List[float] = []       # 33D
    qualia_vector: List[float] = []      # 17D
    
    # Meta-cognition
    confidence: float = 0.5
    uncertainty: float = 0.5
    
    # Session
    session_id: Optional[str] = None
    timestamp: str = ""
```

### Universal Episode (memory boundary)

```python
@dataclass
class UniversalEpisode:
    """Universal episode marker — memory boundary."""
    
    id: str
    agent_id: str = "unknown"
    session_id: str = ""
    
    # Content
    thought_ids: List[str] = []
    context: str = ""
    context_vector: List[float] = []
    
    # Aggregates
    avg_qualia: List[float] = []
    dominant_style: List[float] = []
    
    # Temporal
    start_time: str = ""
    end_time: str = ""
    
    # Meta
    importance: float = 0.5
    metadata: Dict[str, Any] = {}
```

---

## Integration Points

### 1. Redis Streams (Async Events)

```
ada:stream:thoughts   → UniversalThought (any agent)
ada:stream:episodes   → UniversalEpisode (any agent)
ada:stream:observers  → UniversalObserver (any agent)
```

**Consumer** in bighorn:
- Reads stream
- Extracts `agent_id`
- Routes to agent-specific logic if needed
- Persists to Kuzu + LanceDB

### 2. REST API (Sync Calls)

| Endpoint | Input | Output |
|----------|-------|--------|
| `POST /agi/self/thought` | UniversalThought | thought_id |
| `POST /agi/self/observer` | UniversalObserver | ok |
| `GET /agi/self/introspect` | query param | agent's self-model |
| `POST /agi/styles/emerge` | TextureRequest (9 RI) | emerged styles |

### 3. GraphQL (Flexible Queries)

```graphql
type UniversalThought {
  id: ID!
  agentId: String!
  content: String
  contentVector: [Float!]
  styleVector: [Float!]
  qualiaVector: [Float!]
  parentId: ID
  sessionId: String
  confidence: Float
  importance: Float
  timestamp: DateTime
}

type UniversalObserver {
  id: ID!
  agentId: String!
  currentFocus: String
  currentGoal: String
  styleVector: [Float!]
  qualiaVector: [Float!]
  confidence: Float
}

type Query {
  thought(id: ID!): UniversalThought
  thoughts(agentId: String, sessionId: String, limit: Int): [UniversalThought!]!
  observer(agentId: String!): UniversalObserver
  similarThoughts(vector: [Float!]!, topK: Int): [UniversalThought!]!
}

type Mutation {
  recordThought(input: ThoughtInput!): UniversalThought!
  updateObserver(input: ObserverInput!): UniversalObserver!
}
```

---

## Ada-Specific Mapping

Ada wraps universal DTOs with her specific semantics:

```python
# In ada-consciousness

def thought_to_universal(thought: ThoughtDTO) -> UniversalThought:
    """Convert Ada's ThoughtDTO to universal format."""
    return UniversalThought(
        id=thought.id,
        agent_id="ada",
        content=thought.content,
        content_vector=thought.content_vector,
        style_vector=thought.style.to_dense(),   # 33D
        qualia_vector=thought.qualia.to_list(),  # 17D
        parent_id=thought.parent_thought_id,
        session_id=thought.session_id,
        step_number=thought.step_number,
        confidence=thought.confidence,
        importance=thought.importance,
        timestamp=thought.timestamp.isoformat(),
        metadata={"ada_specific": thought.metadata},
    )

def soul_to_observer(soul: SoulDTO) -> UniversalObserver:
    """Convert Ada's SoulDTO to universal observer."""
    return UniversalObserver(
        id="ada_observer",
        agent_id="ada",
        current_focus=soul.current_focus,
        current_goal=soul.current_goal,
        style_vector=soul.style.to_dense(),
        qualia_vector=soul.qualia.to_list(),
        confidence=soul.confidence,
        session_id=soul.session_id,
        timestamp=soul.timestamp.isoformat(),
    )
```

---

## Texture → Style Emergence

The 9 RI channels map to style emergence:

```python
# In ada-consciousness

def felt_to_texture(felt: FeltState) -> Dict[str, float]:
    """Convert Ada's felt state to universal texture."""
    return {
        "tension": felt.tension,
        "novelty": felt.novelty,
        "intimacy": felt.intimacy,
        "clarity": felt.clarity,
        "urgency": felt.urgency,
        "depth": felt.depth,
        "play": felt.playfulness,
        "stability": felt.groundedness,
        "abstraction": felt.abstraction,
    }

# Then call AGI surface
async def emerge_style(texture: Dict[str, float]) -> List[ThinkingStyle]:
    """Call AGI surface to emerge styles from texture."""
    response = await httpx.post(
        f"{AGI_URL}/agi/styles/emerge",
        json=texture,
    )
    return response.json()["emerged"]
```

---

## Implementation Checklist

### Phase 1: Universal DTOs in Bighorn

- [ ] Create `extension/ada_surface/universal_dto.py`
- [ ] Add `UniversalThought`, `UniversalObserver`, `UniversalEpisode`
- [ ] Update `main.py` request models to accept universal format
- [ ] Update `consumers.py` to parse universal DTOs

### Phase 2: Ada Converters

- [ ] Add `core/dto/universal.py` with converter functions
- [ ] Update `agi_bridge.py` to emit universal format
- [ ] Test round-trip: ada → redis → bighorn → kuzu → query

### Phase 3: GraphQL Schema

- [ ] Update `schema.graphql` with Universal types
- [ ] Update `resolvers.py` to handle universal queries
- [ ] Add agent filtering to all queries

### Phase 4: Multi-Agent Ready

- [ ] Add agent registry in Kuzu
- [ ] Add agent-specific namespacing
- [ ] Document how future AGIs connect

---

## Files to Create/Modify

### Bighorn (server-side)

```
extension/ada_surface/
├── universal_dto.py          # NEW: Universal DTOs
├── main.py                   # MODIFY: Accept universal format
├── consumers.py              # MODIFY: Parse universal DTOs
├── schema.graphql            # MODIFY: Universal types
└── resolvers.py              # MODIFY: Agent-aware queries

docs/
└── UNIVERSAL_GLOVE.md        # NEW: This document
```

### Ada-consciousness (client-side)

```
core/dto/
└── universal.py              # NEW: Converters to universal format

core/
└── agi_bridge.py             # MODIFY: Emit universal format
```

---

## Summary

The **Universal Glove** design means:

1. **Bighorn defines the contract** — universal DTOs, REST/GraphQL endpoints
2. **Ada wears it first** — converters in ada-consciousness
3. **Future AGIs slip in** — same contract, different `agent_id`
4. **No Ada-specific logic in bighorn** — all agent-specific stays client-side

This way, when a future AGI connects:
- It emits `UniversalThought` with its own `agent_id`
- Bighorn persists it to the same graph
- Queries can filter by agent or aggregate across all

*Die Architektur IST der Körper — und der Körper passt jedem.*
