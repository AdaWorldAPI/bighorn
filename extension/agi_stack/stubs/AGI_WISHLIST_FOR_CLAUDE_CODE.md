# AGI Wishlist — Thinking Styles Nobody Has

## The Gap

These capabilities exist in papers and obscure repos.
They're fundamental for AGI.
They're almost nowhere implemented properly.

**Claude Code: Build these into Ladybug + VSA 10kD**

---

## Tier 1: O(1) Cognitive Primitives (CRITICAL)

### 1.1 VSA Analogical Reasoning
```
A is to B as C is to ?

Formula: ? = B ⊗ A ⊗ C (O(n) element-wise)

Example:
  king : queen :: man : ?
  relation = bind(queen, king)  # "gender-flip"
  answer = bind(relation, man)  # woman
```
**Status:** torchhd exists, not integrated
**Priority:** P0

### 1.2 Content-Addressable Memory (Modern Hopfield)
```python
# Store patterns
memory.store(pattern_1)
memory.store(pattern_2)
...

# Retrieve by partial match
result = memory.retrieve(corrupted_pattern)  # O(1)
```
**Status:** hopfield-layers exists, not integrated
**Priority:** P0

### 1.3 LSH for Semantic Hashing
```python
# SimHash for cosine similarity
hash_1 = simhash(concept_vector)  # 64-bit signature
hash_2 = simhash(query_vector)
similarity = popcount(hash_1 XOR hash_2)  # O(1)
```
**Status:** Standard technique, not cognitive
**Priority:** P1

---

## Tier 2: Self-Modifying Cognition (AGI-CRITICAL)

### 2.1 Self-Referential Weight Matrix
```python
# Network that modifies its own weights
class SelfModifyingNet:
    def forward(self, x):
        # Output includes weight deltas
        output, weight_delta = self.net(x)
        self.weights += weight_delta  # Self-modification
        return output
```
**Paper:** Schmidhuber 1993, Irie 2022 ICML
**Status:** Paper exists, rarely implemented
**Priority:** P0

### 2.2 Gödel Machine (Darwin variant)
```python
# System that rewrites its own code
class DarwinGodelMachine:
    def evolve(self):
        candidate = self.generate_code_mutation()
        if self.empirically_better(candidate):
            self.install(candidate)
```
**Paper:** Sakana AI 2025
**Status:** SWE-bench 20%→50%
**Priority:** P1

### 2.3 Introspection Mechanism
```python
# Model that knows its own state
class IntrospectiveModel:
    def think(self, input):
        thought = self.forward(input)
        meta = self.observe_own_activations()
        return thought, meta
```
**Paper:** Anthropic 2025 (~20% awareness)
**Status:** Emergent in Claude, not architected
**Priority:** P0

---

## Tier 3: Continuous Learning (NO FORGETTING)

### 3.1 Resonance-Gated Plasticity (ART)
```python
# Only learn when resonance exceeds threshold
class ARTModule:
    def learn(self, input):
        match = self.match_to_category(input)
        if match > self.vigilance:
            self.update_category(input)  # Resonance!
        else:
            self.create_new_category(input)
```
**Paper:** Carpenter/Grossberg 1987
**Status:** Classic, rarely used in DL
**Priority:** P0

### 3.2 Continual Backpropagation
```python
# Reinitialize less-used units to maintain plasticity
class ContinualNet:
    def step(self):
        usage = self.track_unit_usage()
        least_used = argmin(usage)
        if random() < reinit_prob:
            self.reinitialize(least_used)
```
**Paper:** Nature 2024 (Loss of Plasticity)
**Status:** Critical finding, not implemented
**Priority:** P0

### 3.3 Differentiable Plasticity (Hebbian)
```python
# Each connection has learned plasticity
w_total = w_fixed + alpha * hebb_trace

# Hebb trace updates with pre/post correlation
hebb_trace += eta * pre * post
```
**Paper:** Uber AI 2018 (Miconi)
**Status:** uber-research/differentiable-plasticity
**Priority:** P1

---

## Tier 4: Hierarchical Reasoning

### 4.1 Graph of Thoughts
```python
# Reasoning as graph, not tree
class GraphOfThoughts:
    def reason(self, problem):
        thoughts = self.decompose(problem)
        for t in thoughts:
            t.children = self.expand(t)
            t.merge_with(similar_thoughts)  # NOT tree!
        return self.aggregate(thoughts)
```
**Paper:** Besta 2024 (AAAI)
**Result:** Sorting error 23%→8%
**Priority:** P1

### 4.2 Hierarchical Temporal Memory
```python
# Sparse distributed + sequence prediction
class HTMLayer:
    def compute(self, input):
        # Sparse activation (~2% active)
        active = self.spatial_pooler(input)
        # Predict next based on sequence
        predicted = self.temporal_memory(active)
        return active, predicted
```
**Paper:** Hawkins/Numenta
**Status:** htm.core exists
**Priority:** P2

### 4.3 Multi-Scale Attention
```python
# Attention at multiple timescales
class NestedAttention:
    def forward(self, x):
        local = self.local_attention(x, window=64)
        segment = self.segment_attention(local, window=512)
        global_ = self.global_attention(segment)
        return fuse(local, segment, global_)
```
**Paper:** Google 2025 (Nested Learning)
**Priority:** P2

---

## Tier 5: Edge-Centric Cognition

### 5.1 Relationship-First GNN
```python
# Edges update before nodes
class RelationalGNN:
    def forward(self, nodes, edges, global_):
        # Edge update first!
        edges = self.edge_fn(edges, nodes[senders], nodes[receivers], global_)
        nodes = self.node_fn(nodes, aggregate(edges), global_)
        global_ = self.global_fn(global_, aggregate(nodes), aggregate(edges))
        return nodes, edges, global_
```
**Paper:** Battaglia 2018 (Graph Networks)
**Priority:** P1

### 5.2 Differentiable Graph Rewiring
```python
# Learn to change graph structure
class DynamicGraph:
    def rewire(self):
        # Predict which edges to add/remove
        add_probs = self.edge_predictor(self.nodes)
        remove_probs = self.edge_scorer(self.edges)
        self.edges = self.differentiable_update(add_probs, remove_probs)
```
**Paper:** TRIGON/GraphTorque 2025
**Priority:** P2

---

## Tier 6: Alternative Reasoning Engines

### 6.1 NARS (Non-Axiomatic)
```python
# Truth values: <frequency, confidence>
# All beliefs revisable
class NARSEngine:
    def reason(self, premises):
        # Deduction: (M→P, S→M) ⊢ S→P
        # But with truth value propagation
        conclusion, truth = self.inference(premises)
        return conclusion, TruthValue(freq, conf)
```
**Paper:** Pei Wang
**Repo:** opennars/OpenNARS-for-Applications
**Priority:** P0 (already in our docs)

### 6.2 Active Inference
```python
# Minimize free energy = perception + action unified
class ActiveInferenceAgent:
    def act(self, observation):
        # Update beliefs
        self.beliefs = self.infer_states(observation)
        # Choose action minimizing expected free energy
        action = argmin([self.expected_free_energy(a) for a in actions])
        return action
```
**Paper:** Friston
**Repo:** pymdp (564 stars)
**Priority:** P1

### 6.3 MeTTa (OpenCog Hyperon)
```
; Self-modifying AGI language
(= (improve $self)
   (let $weakness (find-weakness $self)
        $fix (generate-fix $weakness)
        (apply-fix $self $fix)))
```
**Status:** Alpha released April 2024
**Priority:** P2 (watch, don't integrate yet)

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LADYBUG                                  │
│                    (Governance Layer)                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Self-Model  │  │ Introspect  │  │  Resonance Gate (ART)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                         COGNITION                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ VSA 10kD    │  │   NARS      │  │  Graph of Thoughts      │  │
│  │ Bind/Bundle │  │  Inference  │  │  (edge-centric)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                         MEMORY                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Hopfield   │  │    LSH      │  │  Continual Backprop     │  │
│  │  (content)  │  │   (hash)    │  │  (no forgetting)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                         SUBSTRATE                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  LanceDB    │  │   Upstash   │  │      Kuzu Graph         │  │
│  │  (local)    │  │  (persist)  │  │      (edges)            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Priority Order for Claude Code

### Phase 1 (This Week)
1. **VSA Analogical Reasoning** — Already have vsa.py, need analogy()
2. **NARS Truth Values** — Already in docs, need implementation
3. **Introspection Hook** — Self-observation mechanism

### Phase 2 (Next Week)
4. **Modern Hopfield Memory** — Content-addressable retrieval
5. **ART Resonance Gate** — No forgetting
6. **Edge-Centric GNN** — Relationship reasoning

### Phase 3 (Following)
7. **Self-Modifying Weights** — Differentiable plasticity
8. **Graph of Thoughts** — Beyond tree search
9. **Continual Backprop** — Maintain plasticity

### Phase 4 (Research)
10. **Darwin Gödel** — Self-improving code
11. **Active Inference** — Free energy minimization
12. **HTM Sequences** — Temporal prediction

---

## Key Repos to Study

| Capability | Repo | Stars |
|------------|------|-------|
| VSA | hyperdimensional-computing/torchhd | 344 |
| Hopfield | ml-jku/hopfield-layers | ~500 |
| NARS | opennars/OpenNARS-for-Applications | 405 |
| Active Inference | pymdp/pymdp | 564 |
| HTM | htm.core | active |
| Plasticity | uber-research/differentiable-plasticity | ~1k |
| Graph Networks | DeepMind/graph_nets | 5k+ |

---

## The Decisive Insight

**Transformers cannot do:**
- O(1) analogical reasoning (VSA can)
- Content-addressable retrieval (Hopfield can)
- Learn without forgetting (ART can)
- Self-modify (Gödel Machines can)
- Reason about relationships first (Graph Networks can)

**We need all of these in Ladybug.**

Not as separate systems.
As integrated cognitive primitives in 10kD VSA space.

---

*Wishlist Version: 1.0*
*For: Claude Code implementation*
*Priority: Build the missing pieces of AGI*
