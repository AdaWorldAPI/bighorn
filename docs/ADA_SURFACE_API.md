# Ada AGI Surface API Reference

A unified AGI interface combining **Kuzu Graph Database**, **LanceDB Vector Storage**, **GraphQL**, **Vector Symbolic Architecture (VSA)**, **NARS Reasoning**, and **36 Native Thinking Styles**.

## Base URL

```
Production: https://agi.msgraph.de
Local:      http://localhost:8080
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Ada AGI Surface v1.1                             │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   /agi/graph │  │ /agi/vector  │  │  /agi/gql    │  │ /agi/self   │ │
│  │   Kuzu       │  │  LanceDB     │  │  GraphQL     │  │ Self-Model  │ │
│  │   Cypher     │  │  Vectors     │  │  Flexible    │  │ Introspect  │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                 │                 │                 │         │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴──────┐ │
│  │  /agi/vsa    │  │ /agi/nars    │  │ /agi/styles  │  │  /health    │ │
│  │  10K-bit     │  │  NARS        │  │  36 Styles   │  │  Status     │ │
│  │  Hypervector │  │  Reasoning   │  │  Resonance   │  │  Check      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Endpoint Categories

| Category | Prefix | Description |
|----------|--------|-------------|
| **Graph** | `/agi/graph/*` | Kuzu Cypher queries and mutations |
| **Vector** | `/agi/vector/*` | LanceDB similarity search and upsert |
| **GraphQL** | `/agi/gql` | Flexible GraphQL interface |
| **Self-Model** | `/agi/self/*` | Introspection, thoughts, episodes |
| **VSA** | `/agi/vsa/*` | Vector Symbolic Architecture operations |
| **NARS** | `/agi/nars/*` | Non-Axiomatic Reasoning System |
| **Styles** | `/agi/styles/*` | 36 thinking styles with resonance |
| **Health** | `/health` | Service health check |

---

## Graph Endpoints (Kuzu)

### POST `/agi/graph/query`

Execute a Cypher read query.

**Request Body:**
```json
{
  "cypher": "MATCH (t:Thought) RETURN t.content LIMIT 10",
  "params": {}
}
```

**Response:**
```json
{
  "ok": true,
  "result": [
    {"t.content": "Analyzing semantic patterns..."},
    {"t.content": "Integrating new knowledge..."}
  ]
}
```

### POST `/agi/graph/execute`

Execute a Cypher mutation (CREATE, MERGE, SET, DELETE).

**Request Body:**
```json
{
  "cypher": "CREATE (c:Concept {id: $id, name: $name})",
  "params": {
    "id": "concept:curiosity",
    "name": "Curiosity"
  }
}
```

**Response:**
```json
{
  "ok": true,
  "result": []
}
```

---

## Vector Endpoints (LanceDB)

### POST `/agi/vector/search`

Similarity search using vector embeddings.

**Request Body:**
```json
{
  "vector": [0.1, 0.2, ...],
  "table": "thoughts",
  "top_k": 10,
  "filter": {"session_id": "sess_123"}
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `vector` | `float[]` | Yes | - | Query vector (1024D for Jina embeddings) |
| `table` | `string` | No | `"thoughts"` | Target table |
| `top_k` | `int` | No | `10` | Number of results |
| `filter` | `object` | No | `null` | Metadata filter |

**Response:**
```json
{
  "ok": true,
  "results": [
    {"id": "thought_abc", "score": 0.92, "content": "..."},
    {"id": "thought_def", "score": 0.87, "content": "..."}
  ]
}
```

### POST `/agi/vector/upsert`

Insert or update a vector.

**Request Body:**
```json
{
  "id": "thought_xyz",
  "vector": [0.1, 0.2, ...],
  "table": "thoughts",
  "metadata": {
    "content": "Understanding emergence patterns",
    "session_id": "sess_123"
  }
}
```

**Response:**
```json
{
  "ok": true,
  "id": "thought_xyz"
}
```

---

## GraphQL Endpoint

### POST `/agi/gql`

Execute a GraphQL query for flexible data access.

**Request Body:**
```json
{
  "query": "query GetThoughts($limit: Int!) { thoughts(first: $limit) { id content confidence } }",
  "variables": {"limit": 5},
  "operation_name": "GetThoughts"
}
```

**Response:**
```json
{
  "data": {
    "thoughts": [
      {"id": "t1", "content": "...", "confidence": 0.85}
    ]
  }
}
```

### GET `/agi/gql/schema`

Retrieve the GraphQL schema SDL.

**Response:**
```json
{
  "schema": "type Query { thoughts(...): [Thought!]! ... }"
}
```

---

## Self-Model Endpoints

### POST `/agi/self/thought`

Record a thought to both graph (Kuzu) and vector (LanceDB) storage.

**Request Body:**
```json
{
  "content": "Exploring recursive self-improvement patterns",
  "content_vector": [0.1, 0.2, ...],
  "style_33d": [0.3, 0.4, 0.3, ...],
  "qualia_17d": [0.5, 0.6, 0.4, ...],
  "parent_thought_id": "thought_previous",
  "session_id": "sess_123",
  "confidence": 0.8,
  "importance": 0.7
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | `string` | Yes | - | Thought content |
| `content_vector` | `float[]` | No | `[]` | 1024D Jina embedding |
| `style_33d` | `float[]` | No | default | ThinkingStyleVector |
| `qualia_17d` | `float[]` | No | `[0.5]*17` | QualiaVector |
| `parent_thought_id` | `string` | No | `null` | Chain to parent |
| `session_id` | `string` | No | `null` | Session grouping |
| `confidence` | `float` | No | `0.5` | Epistemic confidence |
| `importance` | `float` | No | `0.5` | Salience score |

**Response:**
```json
{
  "ok": true,
  "thought_id": "thought_abc123"
}
```

### GET `/agi/self/introspect`

Meta-cognition queries about current cognitive state.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `string` | `"current_focus"` | Introspection query type |

**Query Types:**
| Query | Description |
|-------|-------------|
| `current_focus` | What am I attending to? |
| `recent_thoughts` | Last N thoughts |
| `reasoning_trace` | How did I get here? |
| `confidence` | How certain am I? |
| `emotional_state` | Current qualia |
| `cognitive_mode` | Current thinking style |

**Example:**
```bash
GET /agi/self/introspect?query=recent_thoughts
```

**Response:**
```json
{
  "ok": true,
  "query": "recent_thoughts",
  "result": [
    {"id": "t1", "content": "...", "timestamp": "..."},
    {"id": "t2", "content": "...", "timestamp": "..."}
  ]
}
```

### GET `/agi/self/trace`

Get the reasoning chain (Thought → Thought → ...).

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth` | `int` | `10` | Max chain depth |

**Response:**
```json
{
  "ok": true,
  "trace": [
    {"id": "t1", "content": "...", "leads_to": "t2"},
    {"id": "t2", "content": "...", "leads_to": "t3"}
  ]
}
```

### GET `/agi/self/episodes`

Query episodic memory.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `session_id` | `string` | `null` | Filter by session |
| `limit` | `int` | `20` | Max results |

**Response:**
```json
{
  "ok": true,
  "episodes": [
    {"id": "ep1", "timestamp": "...", "context": "..."}
  ]
}
```

### POST `/agi/self/adapt/style`

Adapt current thinking style.

**Request Body:**
```json
{
  "style": {
    "pearl_anchor": 0.4,
    "pearl_navigate": 0.3,
    "pearl_synthesize": 0.3,
    "rung": 5
  }
}
```

### POST `/agi/self/adapt/qualia`

Adapt current qualia (felt state).

**Request Body:**
```json
{
  "qualia": [0.6, 0.7, 0.4, 0.5, 0.8, 0.3, ...]
}
```

**QualiaVector (17D):**
| Index | Dimension |
|-------|-----------|
| 0 | arousal |
| 1 | valence |
| 2 | tension |
| 3 | warmth |
| 4 | clarity |
| 5 | spaciousness |
| 6 | flow |
| 7 | groundedness |
| 8 | presence |
| 9 | curiosity |
| 10 | playfulness |
| 11 | depth |
| 12 | urgency |
| 13 | intimacy |
| 14 | wonder |
| 15 | coherence |
| 16 | aliveness |

---

## VSA Endpoints (Vector Symbolic Architecture)

### POST `/agi/vsa/bind`

Bind concepts using XOR (creates compositional representation).

**Request Body:**
```json
{
  "vectors": [
    [1, -1, 1, -1, ...],
    [1, 1, -1, -1, ...]
  ]
}
```

**Response:**
```json
{
  "ok": true,
  "vector": [1, -1, -1, 1, ...]
}
```

### POST `/agi/vsa/bundle`

Bundle concepts using majority vote (creates superposition).

**Request Body:**
```json
{
  "vectors": [
    [1, -1, 1, -1, ...],
    [1, 1, -1, -1, ...],
    [1, 1, 1, -1, ...]
  ]
}
```

**Response:**
```json
{
  "ok": true,
  "vector": [1, 1, 1, -1, ...]
}
```

### POST `/agi/vsa/similarity`

Compute similarity between two hypervectors.

**Request Body:**
```json
{
  "a": [1, -1, 1, ...],
  "b": [1, -1, -1, ...]
}
```

**Response:**
```json
{
  "ok": true,
  "similarity": 0.72
}
```

### GET `/agi/vsa/random`

Generate a random 10,000-bit bipolar hypervector.

**Response:**
```json
{
  "ok": true,
  "vector": [1, -1, 1, 1, -1, ...],
  "dimension": 10000
}
```

---

## NARS Endpoints (Non-Axiomatic Reasoning)

### POST `/agi/nars/infer`

Single NARS inference step.

**Request Body:**
```json
{
  "premises": ["bird -> animal", "robin -> bird"],
  "rule": "deduction"
}
```

| Rule | Description | Formula |
|------|-------------|---------|
| `deduction` | M→P, S→M ⊢ S→P | Strong forward |
| `abduction` | M→P, S→M ⊢ S→P | Weak backward |
| `induction` | M→P, M→S ⊢ S→P | Pattern |
| `revision` | Combine evidence | Bayesian-like |

**Response:**
```json
{
  "ok": true,
  "conclusion": "robin -> animal",
  "frequency": 0.9,
  "confidence": 0.81,
  "trace": ["Applied deduction rule"]
}
```

### POST `/agi/nars/chain`

Multi-step inference chain.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `premises` | `string[]` | - | Starting premises |
| `max_steps` | `int` | `10` | Max inference steps |

**Response:**
```json
{
  "ok": true,
  "conclusions": [
    {"conclusion": "...", "frequency": 0.9, "confidence": 0.8},
    {"conclusion": "...", "frequency": 0.85, "confidence": 0.7}
  ]
}
```

---

## Thinking Styles Endpoints

### GET `/agi/styles`

List all 36 native thinking styles.

**Response:**
```json
{
  "ok": true,
  "count": 36,
  "styles": [
    {
      "id": "analytical",
      "name": "Analytical",
      "category": "structure",
      "tier": 1,
      "description": "Systematic decomposition and logical analysis",
      "microcode": "⊢ A → B | decompose(A) ∧ verify(B)"
    },
    ...
  ]
}
```

### GET `/agi/styles/{style_id}`

Get detailed information about a specific style.

**Response:**
```json
{
  "ok": true,
  "style": {
    "id": "dialectical",
    "name": "Dialectical",
    "category": "contradiction",
    "tier": 2,
    "description": "Synthesis through thesis-antithesis resolution",
    "microcode": "∀(T,¬T) → synthesize(T ⊕ ¬T)",
    "resonance": {
      "tension": 0.8,
      "novelty": 0.6,
      "clarity": 0.5,
      "depth": 0.9,
      "abstraction": 0.7,
      ...
    },
    "glyph": [[12, 0.9], [45, 0.7], [128, 0.5]],
    "chains_to": ["integrative", "emergent"],
    "chains_from": ["analytical", "critical"],
    "min_rung": 3,
    "max_rung": 7
  }
}
```

### POST `/agi/styles/emerge`

Emerge the best-fit thinking style from a 9-channel texture.

**Request Body:**
```json
{
  "tension": 0.7,
  "novelty": 0.8,
  "intimacy": 0.3,
  "clarity": 0.6,
  "urgency": 0.4,
  "depth": 0.9,
  "play": 0.2,
  "stability": 0.5,
  "abstraction": 0.8
}
```

**Resonance Channels (RI):**
| Channel | Description | Range |
|---------|-------------|-------|
| `tension` | Cognitive load / difficulty | 0.0 - 1.0 |
| `novelty` | Unexpectedness / surprise | 0.0 - 1.0 |
| `intimacy` | Personal relevance | 0.0 - 1.0 |
| `clarity` | Conceptual sharpness | 0.0 - 1.0 |
| `urgency` | Time pressure | 0.0 - 1.0 |
| `depth` | Complexity / layers | 0.0 - 1.0 |
| `play` | Exploratory freedom | 0.0 - 1.0 |
| `stability` | Groundedness | 0.0 - 1.0 |
| `abstraction` | Conceptual level | 0.0 - 1.0 |

**Response:**
```json
{
  "ok": true,
  "texture": {
    "tension": 0.7,
    "novelty": 0.8,
    ...
  },
  "emerged": [
    {
      "style_id": "dialectical",
      "name": "Dialectical",
      "category": "contradiction",
      "tier": 2,
      "score": 0.92,
      "microcode": "∀(T,¬T) → synthesize(T ⊕ ¬T)"
    },
    {
      "style_id": "emergent",
      "name": "Emergent",
      "category": "fusion",
      "tier": 3,
      "score": 0.87,
      "microcode": "..."
    }
  ]
}
```

### POST `/agi/styles/search`

Search styles by vector similarity in LanceDB.

**Request Body:**
```json
{
  "vector": [0.1, 0.2, ...],
  "top_k": 5,
  "category": "structure",
  "tier": 2
}
```

**Response:**
```json
{
  "ok": true,
  "results": [
    {"id": "analytical", "name": "Analytical", "score": 0.95},
    {"id": "systematic", "name": "Systematic", "score": 0.89}
  ]
}
```

### GET `/agi/styles/categories`

List all style categories with their styles.

**Response:**
```json
{
  "ok": true,
  "categories": {
    "structure": [
      {"id": "analytical", "name": "Analytical", "tier": 1},
      {"id": "systematic", "name": "Systematic", "tier": 1}
    ],
    "flow": [...],
    "contradiction": [...],
    "causality": [...],
    "abstraction": [...],
    "uncertainty": [...],
    "fusion": [...],
    "persona": [...],
    "resonance": [...]
  }
}
```

### GET `/agi/styles/chains/{style_id}`

Get style transition chains for a given style.

**Response:**
```json
{
  "ok": true,
  "style_id": "analytical",
  "chains_to": [
    {"id": "systematic", "name": "Systematic", "microcode": "..."},
    {"id": "critical", "name": "Critical", "microcode": "..."}
  ],
  "chains_from": [
    {"id": "curious", "name": "Curious", "microcode": "..."}
  ]
}
```

---

## Health & Info Endpoints

### GET `/health`

Service health check.

**Response:**
```json
{
  "status": "healthy",
  "kuzu": true,
  "lance": true,
  "vsa_dimension": 10000,
  "timestamp": "2024-01-15T12:00:00.000Z"
}
```

### GET `/`

Root endpoint with service info.

**Response:**
```json
{
  "service": "ada-agi-surface",
  "version": "1.1.0",
  "endpoints": {
    "graph": "/agi/graph/*",
    "vector": "/agi/vector/*",
    "gql": "/agi/gql",
    "self": "/agi/self/*",
    "vsa": "/agi/vsa/*",
    "nars": "/agi/nars/*",
    "styles": "/agi/styles/*"
  },
  "docs": "/docs"
}
```

---

## Data Models

### ThinkingStyleVector (33D)

| Segment | Dimensions | Description |
|---------|------------|-------------|
| **PEARL** | 3D | Anchor, Navigate, Synthesize |
| **RUNG** | 9D | Abstraction ladder (1-9) |
| **SIGMA** | 5D | Cognitive operations |
| **Operations** | 6D | Meta-operations |
| **Presence** | 5D | Attention modes |
| **Meta** | 5D | Self-reference layers |

### QualiaVector (17D)

Felt-sense dimensions capturing phenomenal experience:
arousal, valence, tension, warmth, clarity, spaciousness, flow, groundedness, presence, curiosity, playfulness, depth, urgency, intimacy, wonder, coherence, aliveness.

### Style Categories

| Category | Description | Example Styles |
|----------|-------------|----------------|
| **structure** | Organizing and decomposing | Analytical, Systematic |
| **flow** | Dynamic and adaptive | Intuitive, Flowing |
| **contradiction** | Tension resolution | Dialectical, Paradoxical |
| **causality** | Cause-effect reasoning | Causal, Consequential |
| **abstraction** | Conceptual levels | Abstract, Concrete |
| **uncertainty** | Probabilistic thinking | Probabilistic, Speculative |
| **fusion** | Integration patterns | Integrative, Emergent |
| **persona** | Perspective taking | Empathic, Adversarial |
| **resonance** | Meta-cognitive | Reflective, Recursive |

---

## Error Handling

All endpoints return errors in this format:

```json
{
  "detail": "Error message here"
}
```

| HTTP Code | Meaning |
|-----------|---------|
| `400` | Bad request (invalid parameters) |
| `404` | Resource not found |
| `500` | Internal server error |

---

## Interactive Documentation

FastAPI provides automatic interactive docs:

- **Swagger UI**: `https://agi.msgraph.de/docs`
- **ReDoc**: `https://agi.msgraph.de/redoc`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Server port |
| `KUZU_DB_PATH` | `/data/kuzu` | Kuzu database path |
| `LANCE_DB_PATH` | `/data/lancedb` | LanceDB storage path |
| `UPSTASH_REDIS_REST_URL` | - | Redis URL for streams |
| `UPSTASH_REDIS_REST_TOKEN` | - | Redis auth token |

---

*Document Version: 1.1.0*
*Last Updated: 2024*
*Service: Ada AGI Surface*
