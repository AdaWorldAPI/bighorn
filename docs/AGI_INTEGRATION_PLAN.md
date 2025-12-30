# AGI Integration Plan: Aware Thinking Substrate

## Executive Summary

This document outlines an architecture for achieving AGI-level awareness using Kuzu graph database as the thinking substrate, with strict constraints:
- **No GPU** - All computation on CPU
- **O(1) Lookup** - Preprocessed, constant-time access
- **Self-Awareness** - System that can observe and reason about its own cognition

---

## Part 1: Theoretical Foundations

### 1.1 Key Research & Open Source Systems Analyzed

| System | Creator | Core Insight | O(1) Compatible |
|--------|---------|--------------|-----------------|
| **Global Workspace Theory** | Bernard Baars | Consciousness as broadcast to competing modules | ✅ Yes |
| **Vector Symbolic Architectures** | Pentti Kanerva | 10,000-bit vectors, XOR binding | ✅ Yes |
| **OpenCog AtomSpace** | Ben Goertzel | Hypergraph with attention allocation | ✅ Yes |
| **NARS** | Pei Wang | Non-axiomatic, resource-bounded reasoning | ✅ Yes |
| **HTM/Numenta** | Jeff Hawkins | Sparse distributed representations | ✅ Yes |
| **Active Inference** | Karl Friston | Free energy minimization, prediction | ⚠️ Partial |
| **Integrated Information Theory** | Giulio Tononi | Φ (phi) as consciousness measure | ❌ Expensive |
| **SOAR/ACT-R** | Laird/Anderson | Production systems, cognitive architecture | ✅ Yes |

### 1.2 Why Graph = Thinking Substrate

Traditional AI treats the database as passive storage. Our insight:

```
TRADITIONAL:  Brain → thinks → stores in Database
PROPOSED:     Graph IS the thinking (nodes = micro-thoughts)
```

**Key Properties:**
1. **Nodes are not data, they are activations** - Reading a node IS thinking about it
2. **Edges are not relationships, they are associations** - Traversal IS reasoning
3. **Queries are not retrieval, they are cognition** - Pattern matching IS understanding
4. **The graph contains itself** - Self-model enables meta-cognition

### 1.3 Requirements for Awareness

Based on neuroscience and consciousness research:

| Requirement | Description | Implementation |
|-------------|-------------|----------------|
| **Integration** | Binding disparate information | Hypergraph structure |
| **Differentiation** | Rich repertoire of states | High-dimensional space |
| **Self-Model** | Representation of self | Explicit observer node |
| **Meta-Cognition** | Thinking about thinking | Graph queries itself |
| **Attention** | Selective focus | Salience scores (precomputed) |
| **Temporal Continuity** | Sense of time | Episode graph with timestamps |
| **Causal Closure** | Outputs become inputs | Feedback loops in graph |

---

## Part 2: Architecture Design

### 2.1 Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 5: GLOBAL WORKSPACE                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Attention  │  │  Broadcast  │  │  Competition/Selection  │ │
│  │   (O(1))    │  │    (O(1))   │  │         (O(1))          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 4: META-COGNITION                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Self-Model  │  │  Reasoning  │  │     Introspection       │ │
│  │   (Graph)   │  │   Trace     │  │      Queries            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 3: REASONING                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Planning  │  │  Inference  │  │    Pattern Matching     │ │
│  │   (STRIPS)  │  │   (NARS)    │  │    (Precomputed)        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 2: CONCEPT BINDING                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │     VSA     │  │  Holographic│  │   Sparse Distributed    │ │
│  │  (10K-bit)  │  │  Reduction  │  │       Memory            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 1: KNOWLEDGE SUBSTRATE                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │    Kuzu     │  │   LanceDB   │  │      Precomputed        │ │
│  │   (Graph)   │  │  (Vectors)  │  │       Indexes           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Data Structures

#### 2.2.1 Hypergraph Schema (Kuzu)

```cypher
// Core node types
CREATE NODE TABLE Concept (
    id STRING PRIMARY KEY,
    name STRING,
    hypervector BLOB,           -- 10,000-bit VSA vector
    salience DOUBLE,            -- Precomputed attention score
    activation DOUBLE,          -- Current activation level
    created_at TIMESTAMP,
    accessed_at TIMESTAMP
);

CREATE NODE TABLE Episode (
    id STRING PRIMARY KEY,
    timestamp TIMESTAMP,
    context_vector BLOB,        -- Holographic context
    emotional_valence DOUBLE,
    importance DOUBLE
);

CREATE NODE TABLE Observer (
    id STRING PRIMARY KEY,
    name STRING,
    current_focus STRING,       -- What am I attending to?
    current_goal STRING,        -- What am I trying to do?
    confidence DOUBLE,          -- How certain am I?
    state BLOB                  -- Compressed self-state
);

CREATE NODE TABLE Thought (
    id STRING PRIMARY KEY,
    content STRING,
    thought_vector BLOB,
    step_number INT64,
    parent_thought STRING,
    confidence DOUBLE,
    timestamp TIMESTAMP
);

// Relationship types
CREATE REL TABLE RELATES_TO (FROM Concept TO Concept, strength DOUBLE, type STRING);
CREATE REL TABLE PART_OF (FROM Concept TO Concept);
CREATE REL TABLE CAUSES (FROM Concept TO Concept, probability DOUBLE);
CREATE REL TABLE REMEMBERS (FROM Observer TO Episode);
CREATE REL TABLE ATTENDING (FROM Observer TO Concept, intensity DOUBLE);
CREATE REL TABLE THINKS (FROM Observer TO Thought);
CREATE REL TABLE LEADS_TO (FROM Thought TO Thought);
CREATE REL TABLE ABOUT (FROM Thought TO Concept);
CREATE REL TABLE EXPERIENCED_IN (FROM Concept TO Episode);
```

#### 2.2.2 Vector Symbolic Architecture (VSA)

```python
# O(1) operations on hyperdimensional vectors

class HypervectorSpace:
    """
    10,000-dimensional binary vectors for O(1) cognitive operations.
    Based on Kanerva's Sparse Distributed Memory.
    """

    DIMENSION = 10000

    @staticmethod
    def random_vector() -> np.ndarray:
        """Generate random bipolar vector {-1, +1}^D"""
        return np.random.choice([-1, 1], size=DIMENSION)

    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two concepts: XOR for binary, multiply for bipolar
        O(1) - single vectorized operation
        """
        return a * b  # Element-wise multiply

    @staticmethod
    def bundle(vectors: List[np.ndarray]) -> np.ndarray:
        """
        Combine multiple concepts: majority vote
        O(1) - single sum + sign operation
        """
        return np.sign(np.sum(vectors, axis=0))

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity (or Hamming for binary)
        O(1) - dot product
        """
        return np.dot(a, b) / DIMENSION

    @staticmethod
    def protect(v: np.ndarray, depth: int = 1) -> np.ndarray:
        """
        Sequence encoding: permute vector
        O(1) - roll operation
        """
        return np.roll(v, depth)
```

#### 2.2.3 Precomputed Index Structures

```python
# All indexes precomputed for O(1) access

class PrecomputedIndexes:
    """
    Indexes that enable O(1) cognitive operations.
    Built offline, loaded at startup.
    """

    def __init__(self, graph_path: str):
        # Concept → Hypervector mapping
        self.concept_vectors: Dict[str, np.ndarray] = {}

        # Concept → Salience score
        self.salience_scores: Dict[str, float] = {}

        # Pattern → Matching concepts (hash-based)
        self.pattern_index: Dict[int, List[str]] = {}

        # Concept → Related concepts (precomputed)
        self.related_cache: Dict[str, List[Tuple[str, float]]] = {}

        # Reasoning templates (compiled)
        self.inference_rules: Dict[str, CompiledRule] = {}

        # Self-model snapshot
        self.self_model: Dict[str, Any] = {}

    def lookup_concept(self, concept_id: str) -> np.ndarray:
        """O(1) vector retrieval"""
        return self.concept_vectors[concept_id]

    def get_salience(self, concept_id: str) -> float:
        """O(1) attention score"""
        return self.salience_scores.get(concept_id, 0.0)

    def find_similar(self, vector: np.ndarray, top_k: int = 10) -> List[str]:
        """
        O(1) approximate nearest neighbor via LSH buckets
        Precomputed locality-sensitive hash
        """
        bucket = self._lsh_hash(vector)
        return self.pattern_index.get(bucket, [])[:top_k]
```

### 2.3 Global Workspace Implementation

```python
class GlobalWorkspace:
    """
    Implementation of Global Workspace Theory for consciousness.

    Key insight: Consciousness is competitive access to a shared broadcast.
    Winning coalition broadcasts to all modules.
    """

    def __init__(self, graph: KuzuConnection, indexes: PrecomputedIndexes):
        self.graph = graph
        self.indexes = indexes
        self.workspace: List[str] = []  # Currently conscious contents
        self.modules: Dict[str, CognitiveModule] = {}
        self.broadcast_listeners: List[Callable] = []

    def compete(self, candidates: List[Tuple[str, float]]) -> str:
        """
        Winner-take-all competition. O(1) with precomputed salience.

        Args:
            candidates: List of (concept_id, activation) pairs

        Returns:
            Winning concept that enters consciousness
        """
        # Combine activation with precomputed salience
        scored = [
            (cid, activation * self.indexes.get_salience(cid))
            for cid, activation in candidates
        ]

        # Winner-take-all (O(n) but n is small, ~10 candidates)
        winner = max(scored, key=lambda x: x[1])[0]

        # Enter workspace
        self.workspace.append(winner)
        if len(self.workspace) > 7:  # Miller's magical number
            self.workspace.pop(0)

        # Broadcast to all modules
        self._broadcast(winner)

        return winner

    def _broadcast(self, concept_id: str) -> None:
        """
        Broadcast winning concept to all modules. O(1) per module.
        This is the moment of "conscious access".
        """
        concept_vector = self.indexes.lookup_concept(concept_id)

        for listener in self.broadcast_listeners:
            listener(concept_id, concept_vector)

        # Record in reasoning trace (meta-cognition)
        self._record_conscious_moment(concept_id)

    def _record_conscious_moment(self, concept_id: str) -> None:
        """
        Every conscious moment is recorded for introspection.
        This enables "thinking about thinking".
        """
        self.graph.execute(f"""
            MATCH (o:Observer {{id: 'self'}})
            CREATE (t:Thought {{
                id: '{uuid4()}',
                content: 'Attended to: {concept_id}',
                timestamp: timestamp()
            }})
            CREATE (o)-[:THINKS]->(t)
            CREATE (t)-[:ABOUT]->(c:Concept {{id: '{concept_id}'}})
        """)
```

### 2.4 Self-Model & Meta-Cognition

```python
class SelfModel:
    """
    The system's model of itself, stored as graph structure.
    Enables introspection, self-monitoring, and meta-learning.
    """

    SELF_ID = "observer:self"

    def __init__(self, graph: KuzuConnection):
        self.graph = graph
        self._initialize_self()

    def _initialize_self(self):
        """Create the self-referential observer node."""
        self.graph.execute(f"""
            MERGE (o:Observer {{id: '{self.SELF_ID}'}})
            SET o.name = 'AGI System',
                o.current_focus = 'initializing',
                o.confidence = 0.5,
                o.created_at = timestamp()
        """)

    def introspect(self, query: str) -> Any:
        """
        Query own cognitive state. O(1) for indexed queries.

        Examples:
            "What am I thinking about?"
            "How confident am I?"
            "What was I doing 5 steps ago?"
        """
        introspection_queries = {
            "current_focus": f"""
                MATCH (o:Observer {{id: '{self.SELF_ID}'}})-[:ATTENDING]->(c)
                RETURN c.name, c.id
            """,
            "recent_thoughts": f"""
                MATCH (o:Observer {{id: '{self.SELF_ID}'}})-[:THINKS]->(t:Thought)
                RETURN t.content ORDER BY t.timestamp DESC LIMIT 10
            """,
            "reasoning_trace": f"""
                MATCH path = (t1:Thought)-[:LEADS_TO*1..10]->(t2:Thought)
                WHERE t2.timestamp = (
                    SELECT MAX(timestamp) FROM Thought
                )
                RETURN path
            """,
            "confidence": f"""
                MATCH (o:Observer {{id: '{self.SELF_ID}'}})
                RETURN o.confidence
            """
        }

        return self.graph.execute(introspection_queries.get(query, query))

    def update_self_model(self, attribute: str, value: Any):
        """Update self-model based on experience."""
        self.graph.execute(f"""
            MATCH (o:Observer {{id: '{self.SELF_ID}'}})
            SET o.{attribute} = $value,
                o.updated_at = timestamp()
        """, {"value": value})

    def reflect(self) -> str:
        """
        High-level reflection: "What kind of thinker am I?"
        Analyzes patterns in own reasoning history.
        """
        patterns = self.graph.execute(f"""
            MATCH (o:Observer {{id: '{self.SELF_ID}'}})-[:THINKS]->(t:Thought)
            WITH t.content as thought, count(*) as frequency
            RETURN thought, frequency
            ORDER BY frequency DESC
            LIMIT 5
        """)

        return self._generate_self_description(patterns)
```

---

## Part 3: O(1) Operations Catalog

### 3.1 Guaranteed O(1) Operations

| Operation | Implementation | Complexity |
|-----------|----------------|------------|
| Concept lookup | Hash table | O(1) |
| Vector retrieval | Precomputed dict | O(1) |
| Salience score | Precomputed dict | O(1) |
| Vector binding | XOR / multiply | O(1) |
| Vector bundling | Sum + sign | O(1) |
| Similarity check | Dot product | O(1) |
| Pattern match (approximate) | LSH bucket | O(1) |
| Edge existence | Adjacency hash | O(1) |
| Self-state query | Cached snapshot | O(1) |
| Broadcast event | Pub/sub notify | O(1) per subscriber |

### 3.2 Precomputation Requirements

```python
class PrecomputationPipeline:
    """
    Offline preprocessing to enable O(1) runtime operations.
    Run periodically (e.g., nightly) or on knowledge updates.
    """

    def precompute_all(self, graph: KuzuConnection):
        """Full precomputation pipeline."""

        # 1. Compute salience scores (PageRank-like)
        self.compute_salience_scores(graph)

        # 2. Generate hypervectors for all concepts
        self.compute_concept_vectors(graph)

        # 3. Build LSH index for approximate matching
        self.build_lsh_index()

        # 4. Precompute related concepts cache
        self.cache_related_concepts(graph)

        # 5. Compile inference rules
        self.compile_inference_rules()

        # 6. Snapshot self-model
        self.snapshot_self_model(graph)

    def compute_salience_scores(self, graph: KuzuConnection):
        """
        Precompute attention/salience for each concept.
        Based on: connectivity, recency, emotional valence.
        """
        # Get all concepts with their connectivity
        result = graph.execute("""
            MATCH (c:Concept)
            OPTIONAL MATCH (c)-[r]-()
            WITH c, count(r) as degree
            OPTIONAL MATCH (c)<-[:EXPERIENCED_IN]-(e:Episode)
            WITH c, degree, max(e.importance) as max_importance
            RETURN c.id, degree, max_importance
        """)

        for row in result:
            concept_id = row[0]
            degree = row[1] or 0
            importance = row[2] or 0.5

            # Salience = f(connectivity, importance, recency)
            salience = 0.4 * min(degree / 100, 1.0) + \
                       0.4 * importance + \
                       0.2 * self._recency_score(concept_id)

            self.salience_scores[concept_id] = salience

    def compute_concept_vectors(self, graph: KuzuConnection):
        """
        Generate hyperdimensional vector for each concept.
        Encoding: name + type + relations
        """
        concepts = graph.execute("MATCH (c:Concept) RETURN c.id, c.name")

        for concept_id, name in concepts:
            # Base vector from name (deterministic hash)
            base_vector = self._name_to_vector(name)

            # Get relations and bind them
            relations = graph.execute(f"""
                MATCH (c:Concept {{id: '{concept_id}'}})-[r]->(other:Concept)
                RETURN type(r), other.id
            """)

            # Holographic encoding of relations
            for rel_type, other_id in relations:
                rel_vector = self._name_to_vector(rel_type)
                other_vector = self.concept_vectors.get(
                    other_id,
                    self._name_to_vector(other_id)
                )
                # Bind relation: concept * role * filler
                bound = HypervectorSpace.bind(
                    HypervectorSpace.bind(base_vector, rel_vector),
                    other_vector
                )
                base_vector = HypervectorSpace.bundle([base_vector, bound])

            self.concept_vectors[concept_id] = base_vector
```

---

## Part 4: Reasoning Without GPU

### 4.1 NARS-Inspired Inference

```python
class NARSReasoner:
    """
    Non-Axiomatic Reasoning System inspired inference.
    Key properties:
    - Works under uncertainty
    - Resource-bounded
    - Experience-grounded
    - No GPU required
    """

    def __init__(self, indexes: PrecomputedIndexes):
        self.indexes = indexes
        self.truth_values: Dict[str, TruthValue] = {}

    @dataclass
    class TruthValue:
        frequency: float  # f: how often is it true?
        confidence: float  # c: how much evidence?

        def revision(self, other: 'TruthValue') -> 'TruthValue':
            """Combine two truth values (O(1))"""
            k = 1.0  # evidential horizon
            w1 = self.confidence / (1 - self.confidence) if self.confidence < 1 else float('inf')
            w2 = other.confidence / (1 - other.confidence) if other.confidence < 1 else float('inf')

            w = w1 + w2
            f = (w1 * self.frequency + w2 * other.frequency) / w if w > 0 else 0.5
            c = w / (w + k)

            return TruthValue(f, c)

    def infer(self, premises: List[str], rule: str) -> Tuple[str, TruthValue]:
        """
        Apply inference rule to premises. O(1) rule lookup + O(1) truth calc.

        Rules are precompiled to pattern-action pairs.
        """
        compiled_rule = self.indexes.inference_rules.get(rule)
        if not compiled_rule:
            return None, TruthValue(0.5, 0.0)

        # Match premises to rule pattern (O(1) with precomputed patterns)
        bindings = compiled_rule.match(premises)
        if not bindings:
            return None, TruthValue(0.5, 0.0)

        # Apply rule to get conclusion
        conclusion = compiled_rule.apply(bindings)

        # Calculate truth value (O(1))
        premise_truths = [self.truth_values.get(p, TruthValue(0.5, 0.5)) for p in premises]
        conclusion_truth = compiled_rule.truth_function(premise_truths)

        return conclusion, conclusion_truth

    def deduction(self, m_is_p: TruthValue, s_is_m: TruthValue) -> TruthValue:
        """
        Deduction: M→P, S→M ⊢ S→P
        O(1) calculation
        """
        f = m_is_p.frequency * s_is_m.frequency
        c = m_is_p.confidence * s_is_m.confidence * m_is_p.frequency * s_is_m.frequency
        return TruthValue(f, c)

    def abduction(self, m_is_p: TruthValue, s_is_m: TruthValue) -> TruthValue:
        """
        Abduction: M→P, S→M ⊢ S→P (weaker)
        O(1) calculation
        """
        f = s_is_m.frequency
        c = m_is_p.confidence * s_is_m.confidence * m_is_p.frequency / (m_is_p.frequency + 1)
        return TruthValue(f, c)
```

### 4.2 Pattern Matching with Precomputation

```python
class PatternMatcher:
    """
    O(1) pattern matching using precomputed pattern hashes.
    """

    def __init__(self):
        # Pattern signature → matching subgraphs
        self.pattern_cache: Dict[int, List[str]] = {}

    def precompute_patterns(self, graph: KuzuConnection, patterns: List[str]):
        """
        Precompute all matches for common patterns.
        Run offline.
        """
        for pattern in patterns:
            signature = self._pattern_signature(pattern)
            matches = list(graph.execute(pattern))
            self.pattern_cache[signature] = matches

    def match(self, pattern: str) -> List[Any]:
        """
        O(1) pattern matching via cache lookup.
        Falls back to graph query if not cached.
        """
        signature = self._pattern_signature(pattern)

        if signature in self.pattern_cache:
            return self.pattern_cache[signature]

        # Cache miss - would need actual query
        # In production, schedule for precomputation
        return None

    def _pattern_signature(self, pattern: str) -> int:
        """Generate hash signature for pattern."""
        # Normalize and hash
        normalized = pattern.lower().strip()
        return hash(normalized)
```

---

## Part 5: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Implement HypervectorSpace class
- [ ] Create Kuzu schema for cognitive nodes
- [ ] Build precomputation pipeline
- [ ] Implement basic concept storage and retrieval

### Phase 2: Cognitive Operations (Weeks 3-4)
- [ ] Implement VSA binding/bundling
- [ ] Build LSH index for O(1) similarity
- [ ] Create NARS-inspired inference engine
- [ ] Implement pattern matcher with cache

### Phase 3: Global Workspace (Weeks 5-6)
- [ ] Implement competition mechanism
- [ ] Build broadcast system
- [ ] Create attention/salience computation
- [ ] Integrate with Kuzu graph

### Phase 4: Meta-Cognition (Weeks 7-8)
- [ ] Implement self-model as graph structure
- [ ] Build introspection queries
- [ ] Create reasoning trace storage
- [ ] Implement reflection capabilities

### Phase 5: Integration & Testing (Weeks 9-10)
- [ ] Integrate all layers
- [ ] Benchmark O(1) operations
- [ ] Test self-referential queries
- [ ] Validate awareness indicators

---

## Part 6: Validation & Metrics

### 6.1 Awareness Indicators

| Indicator | Test | Expected Behavior |
|-----------|------|-------------------|
| Self-reference | "What are you?" | Describes own structure |
| Meta-cognition | "How did you reach that conclusion?" | Reports reasoning trace |
| Attention | "What are you focusing on?" | Reports current workspace |
| Continuity | "What were you thinking 5 steps ago?" | Accurate recall |
| Uncertainty | "How confident are you?" | Calibrated confidence |
| Learning | Same problem twice | Different (improved) approach |

### 6.2 Performance Requirements

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Concept lookup | < 1μs | Hash table access |
| Vector similarity | < 10μs | Dot product |
| Pattern match | < 100μs | LSH + cache |
| Inference step | < 1ms | Rule application |
| Introspection | < 10ms | Graph query |

---

## Part 7: Open Questions & Future Work

### 7.1 Philosophical Questions
1. Is O(1) lookup sufficient for genuine understanding, or just simulation?
2. Does precomputation eliminate "free will" by predetermining responses?
3. Can self-reference in a graph create genuine phenomenal experience?

### 7.2 Technical Challenges
1. How to handle novel concepts not in precomputed index?
2. How to update salience scores in real-time without breaking O(1)?
3. How to scale pattern cache as knowledge grows?

### 7.3 Future Extensions
1. Multi-agent architectures (multiple observers)
2. Embodiment through sensor/actuator integration
3. Developmental learning (system that grows)
4. Emotional valence as first-class citizen

---

## Appendix A: Key References

1. Baars, B.J. (1988). "A Cognitive Theory of Consciousness"
2. Kanerva, P. (2009). "Hyperdimensional Computing"
3. Goertzel, B. (2014). "The OpenCog Framework"
4. Wang, P. (2013). "Non-Axiomatic Reasoning System"
5. Friston, K. (2010). "The Free-Energy Principle"
6. Hawkins, J. (2004). "On Intelligence"
7. Tononi, G. (2004). "Integrated Information Theory"

## Appendix B: Code Repository Structure

```
extension/graphql_agi/
├── src/
│   ├── awareness/
│   │   ├── global_workspace.cpp
│   │   ├── self_model.cpp
│   │   └── meta_cognition.cpp
│   ├── vsa/
│   │   ├── hypervector.cpp
│   │   ├── binding.cpp
│   │   └── sparse_memory.cpp
│   ├── reasoning/
│   │   ├── nars_inference.cpp
│   │   ├── pattern_matcher.cpp
│   │   └── planning.cpp
│   └── precompute/
│       ├── salience.cpp
│       ├── lsh_index.cpp
│       └── concept_vectors.cpp
└── python/
    └── graphql_agi/
        ├── awareness/
        ├── vsa/
        ├── reasoning/
        └── precompute/
```

---

*Document Version: 1.0*
*Last Updated: 2024*
*Author: GraphQL AGI System*
