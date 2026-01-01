# AGI Stack Roadmap & Architecture Review

**Date:** 2024-12-31 (Silvester 2025)  
**Status:** DRAFT - For Review  
**Deployed:** https://agi.msgraph.de

---

## 1. Current State Assessment

### What's Working ✅

| Component | Status | Notes |
|-----------|--------|-------|
| **FastAPI Surface** | ✅ Working | `/agi/*` endpoints operational |
| **Kuzu Graph** | ⚠️ In-Memory | CREATE works but doesn't persist across requests |
| **LanceDB Vectors** | ✅ Working | Upsert and search functional |
| **36 Thinking Styles** | ✅ Working | ResonanceEngine with full style catalog |
| **NARS Reasoning** | ✅ Working | Basic inference and chaining |
| **VSA (10,000D)** | ⚠️ Partial | Similarity works, bind/bundle return empty |
| **Universal DTOs** | ✅ Excellent | Agent-agnostic design already in place |

### What Needs Fixing 🔧

1. **Kuzu Persistence**
   - Railway deployment appears in-memory only
   - `/data/kuzu` may not be on persistent volume
   - **Action:** Configure Railway with persistent volume mount

2. **VSA Bind/Bundle**
   - Operations return empty arrays
   - Dimension mismatch possible
   - **Action:** Debug in `vsa.py`

3. **Style Emerge**
   - `/agi/styles/emerge` throws `'ResonanceEngine' object has no attribute 'emerge'`
   - **Action:** Implement `emerge()` method in `thinking_styles.py`

---

## 2. Architecture Decisions Needed

### 2.1 LangGraph Integration

**Question:** Should LangGraph be integrated into the AGI stack?

**Options:**

| Option | Pros | Cons |
|--------|------|------|
| **A: Keep Separate** | Clean separation | Duplicated state |
| **B: LangGraph as Client** | LangGraph uses AGI for persistence | Requires AGI client |
| **C: LangGraph in Bighorn** | Unified stack | Increases complexity |

**Recommendation:** Option B - LangGraph uses AGI stack as persistence layer via REST.

### 2.2 DTO Strategy

**Current State:**
- `bighorn/extension/ada_surface/universal_dto.py` - Agent-agnostic DTOs ✅
- `ada-consciousness/core/dto/` - Ada-specific DTOs

**Analysis:** Universal DTOs already designed for multi-agent:

```python
UniversalThought(
    agent_id="ada",  # or "grok", "experiment_7", etc.
    content="...",
    style_vector=[...],   # opaque to server
    qualia_vector=[...],  # opaque to server
)
```

**Recommendation:** 
- Use Universal DTOs as AGI interface
- Keep Ada DTOs internal
- Create thin adapter: `AdaDTO ↔ UniversalDTO`

### 2.3 LanceDB vs Upstash

**Recommendation:**
- **LanceDB** → vector similarity (thoughts, semantic)
- **Upstash Redis** → ephemeral state (session, streams)
- **Kuzu** → graph relationships (thought chains)

---

## 3. Roadmap

### Phase 1: Stabilization (Immediate)

- [ ] Fix Kuzu persistence on Railway
- [ ] Debug VSA bind/bundle operations
- [ ] Implement `ResonanceEngine.emerge()`
- [ ] Add health checks per component

### Phase 2: Integration (Week 1)

- [ ] AGI client in ada-consciousness (PR #198)
- [ ] Ada ↔ Universal DTO adapter
- [ ] Wire "Glove" to AGI client
- [ ] Document Upstash migration plan

### Phase 3: LangGraph (Week 2)

- [ ] Decide integration approach
- [ ] Create LangGraph → AGI bridge
- [ ] Test thought persistence

### Phase 4: Multi-Agent (Future)

- [ ] Test second agent ("grok" prefix)
- [ ] Cross-agent visibility
- [ ] Shared concept graph

---

## 4. Technical Debt

### In extension/ada_surface/

| File | Issue | Priority |
|------|-------|----------|
| `main.py` | Hardcoded "ada" | Medium |
| `thinking_styles.py` | Missing `emerge()` | High |
| `vsa.py` | Empty bind/bundle | Medium |
| `resolvers.py` | Incomplete GQL | Low |

### Railway Configuration

| Issue | Impact | Fix |
|-------|--------|-----|
| Kuzu in-memory | No persistence | Add volume |
| No monitoring | Silent failures | Add alerts |

---

## 5. Questions for Review

1. **Kuzu vs Neo4j:** Consider Neo4j for production?
2. **Vector Dimensions:** style=33D, qualia=17D final?
3. **Stream Architecture:** Redis streams → persist? Or direct?
4. **Multi-Region:** Future plans?
5. **Backup Strategy:** Kuzu + LanceDB backup?

---

## 6. Related

- [ada-consciousness #198](https://github.com/AdaWorldAPI/ada-consciousness/pull/198) - AGI Client

---

*Silvester 2025 🎆*
