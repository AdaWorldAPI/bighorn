# Multi-Repository Integration Architecture

**Version:** 1.0
**Scope:** Full system integration across ada-consciousness, bighorn, and adarail_mcp

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TENANT (hive.msgraph.de)                           │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      ada-consciousness                                  │ │
│  │                                                                         │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │ │
│  │  │  Cognitive  │───►│   dto_vsa   │───►│  Outbound DTO (10K VSA)     │ │ │
│  │  │   State     │    │  (Encoder)  │    │  - Thought Hypervector      │ │ │
│  │  └─────────────┘    └─────────────┘    │  - Style Vector (33D)       │ │ │
│  │                                        │  - Qualia Vector (17D)      │ │ │
│  │                                        │  - Context Binding          │ │ │
│  │                                        └──────────────┬──────────────┘ │ │
│  └───────────────────────────────────────────────────────┼────────────────┘ │
└──────────────────────────────────────────────────────────┼──────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ROUTER (adarail_mcp)                                │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     MCP Protocol Router                                 │ │
│  │                                                                         │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │ │
│  │  │   REST      │    │    MCP      │    │     Request Routing         │ │ │
│  │  │  Endpoints  │───►│  Protocol   │───►│  - DTO Validation           │ │ │
│  │  └─────────────┘    └─────────────┘    │  - Session Management       │ │ │
│  │                                        │  - Auth / Rate Limiting     │ │ │
│  │                                        └──────────────┬──────────────┘ │ │
│  └───────────────────────────────────────────────────────┼────────────────┘ │
└──────────────────────────────────────────────────────────┼──────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGI STACK (agi.msgraph.de)                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         bighorn/extension                            │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │    │
│  │  │  graphql_agi    │  │                 │  │      agi_stack      │  │    │
│  │  │                 │  │                 │  │                     │  │    │
│  │  │  - Ladybug      │  │                 │  │  - 36 Styles        │  │    │
│  │  │  - Reasoning    │  │                 │  │  - Resonance        │  │    │
│  │  │  - GraphQL      │  │                 │  │  - VSA/NARS         │  │    │
│  │  │  - Planning     │  │                 │  │  - Kuzu/Lance       │  │    │
│  │  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘  │    │
│  │           │                    │                      │              │    │
│  │           └────────────────────┼──────────────────────┘              │    │
│  │                                ▼                                     │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │    │
│  │  │                        Kuzu Core                                 │ │    │
│  │  │            Graph Database + Vector Extensions                    │ │    │
│  │  └─────────────────────────────────────────────────────────────────┘ │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Inventory

| Repository | Location | Purpose | Status |
|------------|----------|---------|--------|
| **ada-consciousness** | `AdaWorldAPI/ada-consciousness` | Tenant cognitive layer, outbound VSA DTOs | Private |
| **bighorn** | `AdaWorldAPI/bighorn` | AGI stack, Kuzu extensions | Active |
| **adarail_mcp** | `AdaWorldAPI/adarail_mcp` | MCP router between tenant and AGI | Private |

### bighorn Extensions

| Extension | Path | Purpose | Status |
|-----------|------|---------|--------|
| **graphql_agi** | `extension/graphql_agi/` | GraphQL + Ladybug + Reasoning | Exists |
| **agi_stack** | `extension/agi_stack/` | Styles + Resonance + NARS + DTOs | Exists |

---

## Component Capabilities

### 1. graphql_agi (Existing)

**Files:** ~2,500 lines across C++ and Python

| Component | Capability |
|-----------|------------|
| **Ladybug Debugger** | Query analysis, complexity scoring, anti-pattern detection, slow query logging, HTML reports |
| **Reasoning Engine** | CoT, ToT, ReAct, Self-Consistency, Reflexion, Plan-and-Execute, Multi-hop reasoning |
| **GraphQL Schema** | Auto-generation from Kuzu catalog, Relay pagination, introspection |
| **LanceDB Vector Store** | Semantic search, hybrid search, embedding storage, multi-provider (OpenAI/Voyage/Cohere) |

**Key Integrations:**
- Kuzu graph database (C++ extension)
- LanceDB vector storage
- LLM providers (Claude, OpenAI)

### 2. agi_stack (Current)

**Files:** ~4,500 lines Python

| Component | Capability |
|-----------|------------|
| **36 Thinking Styles** | 9 categories × 4 styles, resonance profiles, sparse glyphs, style chains |
| **Resonance Engine** | 9 RI channels, texture→style emergence, rung pressure |
| **VSA Module** | 10K hypervectors, bind/bundle/permute, cognitive primitives |
| **NARS Module** | Truth values (f,c), deduction/induction/abduction, chain inference |
| **Kuzu Client** | Cypher queries, Observer/Thought/Episode/Concept schema |
| **LanceDB Client** | Vector search, style indexing, hybrid search |
| **GraphQL Resolvers** | Full schema for ThinkingStyle (33D), Qualia (17D) |
| **Redis Consumers** | Async thought/episode/adaptation processing |

### 3. agi_stack DTOs (Planned Extensions)

**Purpose:** Receive inbound DTOs from tenant, coordinate with other extensions

| Component | Capability |
|-----------|------------|
| **Inbound DTO** | Parse VSA-encoded thought vectors from tenant |
| **Core Logic** | State machine for cognitive transitions |
| **Ladybug Gate** | Governance layer for all transitions |
| **Global Workspace** | Competition and broadcast mechanism |

---

## DTO Flow Architecture

### Outbound (Tenant → AGI)

```python
# ada-consciousness/dto_vsa

@dataclass
class OutboundDTO:
    """Tenant sends this to AGI stack."""

    # Identity
    session_id: str
    thought_id: str
    timestamp: float

    # Content (1024D Jina embedding)
    content_vector: List[float]

    # Cognitive state
    style_33d: List[float]     # ThinkingStyleVector
    qualia_17d: List[float]    # QualiaVector

    # VSA encoding (10K hypervector)
    vsa_binding: List[int]     # H = content ⊗ style ⊗ qualia

    # Metadata
    parent_thought_id: Optional[str]
    step_number: int
    confidence: float
```

### Inbound (AGI → Tenant)

```python
# bighorn/extension/agi_stack/dto

@dataclass
class InboundDTO:
    """AGI stack returns this to tenant."""

    # Response
    thought_id: str
    response_content: str

    # Style adaptation
    suggested_style: Optional[str]
    style_delta: Optional[List[float]]

    # Qualia modulation
    qualia_delta: Optional[List[float]]

    # Reasoning trace
    reasoning_steps: List[Dict]
    confidence: float

    # Graph updates
    new_concepts: List[str]
    new_relationships: List[Tuple[str, str, str]]
```

---

## Integration Points

### 1. Ladybug ↔ Resonance Engine

**Current State:** Both exist independently
**Integration Goal:** Ladybug governs style transitions from Resonance

```python
# Integration: ladybug_resonance_bridge.py

class LadybugResonanceBridge:
    """Bridge between graphql_agi.Ladybug and agi_stack.ResonanceEngine"""

    def __init__(self, ladybug: LadybugDebugger, resonance: ResonanceEngine):
        self.ladybug = ladybug
        self.resonance = resonance

    async def request_style_transition(
        self,
        texture: Dict[RI, float],
        current_style: Optional[str] = None,
    ) -> Optional[ThinkingStyle]:
        """
        Request style transition through Ladybug governance.

        1. Resonance computes emerged styles
        2. Ladybug evaluates transition safety
        3. Approved transition returns new style
        """
        # Get candidates from resonance
        emerged = self.resonance.emerge_styles(texture, top_k=5)

        if not emerged:
            return None

        # Evaluate each candidate through Ladybug
        for style, score in emerged:
            # Create debug session for audit trail
            session = self.ladybug.start_session(
                f"style_transition:{current_style}→{style.id}"
            )

            # Check complexity (layer boundary)
            analysis = self.ladybug.analyze_query(style.microcode)

            if analysis.valid and analysis.complexity < 100:
                self.ladybug.end_session(session, success=True)
                return style
            else:
                self.ladybug.log_event(
                    session,
                    TraceSeverity.WARNING,
                    "governance",
                    f"Style transition blocked: {analysis.warnings}"
                )
                self.ladybug.end_session(session, success=False)

        return None
```

**Blockers:**
1. Ladybug currently analyzes GraphQL/Cypher queries, not style microcodes
2. Need to define complexity metrics for styles

### 2. Reasoning Engine ↔ NARS

**Current State:** graphql_agi has CoT/ToT, agi_stack has NARS
**Integration Goal:** NARS provides logical backbone for reasoning

```python
# Integration: reasoning_nars_bridge.py

class ReasoningNARSBridge:
    """Combine LLM-based reasoning with NARS logic."""

    def __init__(self, reasoning: ReasoningEngine, nars: NARSReasoner):
        self.reasoning = reasoning
        self.nars = nars

    async def hybrid_reason(
        self,
        question: str,
        use_nars_validation: bool = True,
    ) -> ReasoningResult:
        """
        1. LLM generates reasoning chain
        2. NARS validates logical consistency
        3. Combined confidence score
        """
        # Get LLM reasoning
        result = self.reasoning.reason(
            question=question,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )

        if not use_nars_validation:
            return result

        # Extract propositions from reasoning steps
        propositions = self._extract_propositions(result.steps)

        # Validate with NARS
        for prop in propositions:
            self.nars.add_belief(prop)

        # Check for contradictions
        conclusions = self.nars.chain_inference(
            [str(p) for p in propositions],
            max_steps=5
        )

        # Adjust confidence based on NARS consistency
        nars_confidence = self._compute_consistency(conclusions)
        result.confidence = (result.confidence + nars_confidence) / 2

        return result
```

**Blockers:**
1. Proposition extraction from natural language reasoning
2. Mapping NARS truth values to LLM confidence scores

### 3. VSA ↔ LanceDB

**Current State:** Both store vectors independently
**Integration Goal:** VSA hypervectors indexed in LanceDB for cognitive similarity search

```python
# Integration: vsa_lance_bridge.py

class VSALanceBridge:
    """Store and search VSA hypervectors in LanceDB."""

    def __init__(self, vsa: HypervectorSpace, lance: LanceClient):
        self.vsa = vsa
        self.lance = lance
        self._init_vsa_table()

    def _init_vsa_table(self):
        """Create table for 10K hypervectors (stored as quantized 1K)."""
        # Quantize 10K → 1K for storage efficiency
        schema = pa.schema([
            ("id", pa.string()),
            ("vector", pa.list_(pa.float32(), 1000)),  # Quantized
            ("full_hash", pa.string()),  # Hash of full 10K
            ("type", pa.string()),  # thought/concept/binding
            ("metadata", pa.string()),
        ])
        # ...

    async def store_binding(
        self,
        binding_id: str,
        hypervector: np.ndarray,
        binding_type: str,
        metadata: Dict,
    ):
        """Store a VSA binding with quantization."""
        quantized = self._quantize_10k_to_1k(hypervector)
        # ...

    async def find_similar_bindings(
        self,
        query_hv: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find similar cognitive bindings."""
        # ...
```

**Blockers:**
1. Quantization quality (10K → 1K)
2. Similarity preservation after quantization
3. Index configuration for hypervector search

### 4. graphql_agi GraphQL ↔ agi_stack REST

**Current State:** Two separate API surfaces
**Integration Goal:** Unified API gateway

```yaml
# Unified gateway configuration

routes:
  # GraphQL AGI (existing)
  - path: /graphql
    upstream: graphql_agi
  - path: /reason
    upstream: graphql_agi
  - path: /search
    upstream: graphql_agi
  - path: /debug/*
    upstream: graphql_agi

  # AGI Stack (existing)
  - path: /agi/self/*
    upstream: agi_stack
  - path: /agi/graph/*
    upstream: agi_stack
  - path: /agi/vector/*
    upstream: agi_stack
  - path: /agi/styles/*
    upstream: agi_stack
  - path: /agi/vsa/*
    upstream: agi_stack
  - path: /agi/nars/*
    upstream: agi_stack

  # Unified (new)
  - path: /v2/gql
    upstream: merged_graphql  # Merged schema
  - path: /v2/reason
    upstream: hybrid_reasoner  # LLM + NARS
```

**Blockers:**
1. Schema merging strategy
2. Resolver conflict resolution
3. Shared state management (Kuzu connection)

---

## adarail_mcp Integration

### MCP Protocol Handlers

```python
# adarail_mcp expected interface

class MCPHandler:
    """Handle MCP protocol requests from tenant."""

    async def handle_thought(self, dto: OutboundDTO) -> InboundDTO:
        """
        Process thought from tenant.

        1. Validate DTO
        2. Route to appropriate AGI extension
        3. Aggregate response
        """
        # Validate
        if not self._validate_dto(dto):
            raise MCPError("Invalid DTO")

        # Store in graph (agi_stack)
        thought_id = await self.agi_stack.create_thought(
            content=dto.content,
            style_33d=dto.style_33d,
            qualia_17d=dto.qualia_17d,
        )

        # Compute style emergence (agi_stack)
        texture = self._extract_texture(dto.qualia_17d)
        emerged = await self.agi_stack.emerge_style(texture)

        # Reason about content (graphql_agi)
        reasoning = await self.graphql_agi.reason(
            question=f"What is the significance of: {dto.content[:200]}",
            strategy="chain_of_thought"
        )

        # Build response
        return InboundDTO(
            thought_id=thought_id,
            response_content=reasoning.answer,
            suggested_style=emerged[0].id if emerged else None,
            confidence=reasoning.confidence,
            reasoning_steps=reasoning.steps,
        )
```

---

## Aspirations (Full Vision)

### 1. Unified Cognitive Graph

```
Observer ──THINKS──► Thought ──USES_STYLE──► Style
    │                   │                       │
    │                   ▼                       │
    │            ──ABOUT──► Concept             │
    │                   │                       │
    │                   ▼                       │
    └──REMEMBERS──► Episode ◄──EMERGED_FROM─────┘
```

### 2. Real-Time Style Subscriptions

```graphql
subscription StyleTransitions {
  styleTransition {
    fromStyle { id name }
    toStyle { id name }
    trigger { texture rungPressure }
    ladybugDecision { approved reason }
  }
}
```

### 3. Cross-Repository Observability

```
┌─────────────────────────────────────────────────────────────┐
│                    Ladybug Observatory                       │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Tenant    │  │   Router    │  │      AGI Stack      │ │
│  │   Traces    │  │   Traces    │  │      Traces         │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         └────────────────┼─────────────────────┘            │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Unified Trace Correlation                 │  │
│  │                                                        │  │
│  │  request_id → [tenant_span, router_span, agi_span]    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 4. Self-Modifying Style Evolution

```python
# Future: Style breeding and evolution

class StyleEvolution:
    """Evolve styles based on success metrics."""

    async def breed(self, parent_a: str, parent_b: str) -> ThinkingStyle:
        """Create child style from successful parents."""
        pass

    async def mutate(self, style_id: str, pressure: Dict[RI, float]) -> ThinkingStyle:
        """Mutate style based on resonance pressure."""
        pass

    async def prune(self, style_id: str):
        """Remove unsuccessful derived styles."""
        pass
```

---

## Blockers Summary

### Critical Path Blockers

| Blocker | Impact | Owner | ETA |
|---------|--------|-------|-----|
| agi_stack/dto not created | Cannot receive tenant DTOs | TBD | P0 |
| Ladybug style analysis | No governance for transitions | TBD | P1 |
| Schema merge strategy | Split-brain API | TBD | P1 |
| VSA quantization | Cannot index hypervectors efficiently | TBD | P2 |

### Integration Blockers

| Blocker | Components | Resolution |
|---------|------------|------------|
| Shared Kuzu connection | graphql_agi ↔ agi_stack | Connection pool or unified client |
| Reasoning ↔ NARS | graphql_agi ↔ agi_stack | Proposition extraction layer |
| Style ↔ GraphQL | agi_stack → graphql_agi | Extend graphql_agi schema |
| LanceDB table collision | Both use `thoughts` table | Namespacing or merge |

### External Blockers

| Blocker | Dependency | Status |
|---------|------------|--------|
| ada-consciousness DTO spec | Private repo | Need access |
| adarail_mcp interface | Private repo | Need access |
| Upstash Redis streams | Already wired | ✅ |

---

## Phase Roadmap

### Phase 1: agi_stack Foundation
1. Create `extension/agi_stack/dto/` with inbound/outbound types
2. Create `extension/agi_stack/core/` with state machine
3. Wire to adarail_mcp

### Phase 2: Extension Integration
1. Ladybug ↔ Resonance bridge
2. Reasoning ↔ NARS bridge
3. Unified Kuzu connection

### Phase 3: API Unification
1. Merge GraphQL schemas
2. Unified gateway routing
3. Cross-repo tracing

### Phase 4: Evolution
1. VSA ↔ LanceDB bridge
2. Style breeding
3. Global Workspace

---

*Document Version: 1.0*
*Status: Integration Planning*
