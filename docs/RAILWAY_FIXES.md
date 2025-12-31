# Railway Fixes Needed — AGI Stack (agi.msgraph.de)

**Date:** 2024-12-31  
**Status:** Needs Railway Dashboard Access  
**Deployed Service:** bighorn on Railway → agi.msgraph.de

---

## 1. Kuzu Persistence Fix

### Problem
Kuzu graph database is running in-memory only. CREATE statements execute but data doesn't persist across requests.

### Current Behavior
```bash
# This works during the request:
curl -X POST https://agi.msgraph.de/agi/graph/execute \
  -d '{"cypher":"CREATE (t:Thought {id: \"test\", content: \"hello\"})"}' 

# But querying later returns empty:
curl -X POST https://agi.msgraph.de/agi/graph/query \
  -d '{"cypher":"MATCH (t:Thought) RETURN t"}'
# Returns: {"ok": true, "result": []}
```

### Fix Required

**In Railway Dashboard:**

1. Go to **agi.msgraph.de service** (probably named something like "bighorn-agi" or similar)

2. Add a **Volume Mount**:
   - Click "Settings" → "Volumes"
   - Add new volume
   - Mount path: `/data`
   - This will persist `/data/kuzu` and `/data/lancedb`

3. Verify environment variables are set:
   ```
   KUZU_DB_PATH=/data/kuzu
   LANCE_DB_PATH=/data/lancedb
   ```

4. Redeploy the service

### Verification
```bash
# After fix, this should persist:
curl -X POST https://agi.msgraph.de/agi/self/thought \
  -H "Content-Type: application/json" \
  -d '{"content": "Test thought", "confidence": 0.9}'

# And this should return the thought:
curl https://agi.msgraph.de/agi/self/introspect
```

---

## 2. VSA Bind/Bundle Fix

### Problem
VSA (Vector Symbolic Architecture) bind and bundle operations return empty arrays.

### Current Behavior
```bash
curl -X POST https://agi.msgraph.de/agi/vsa/bind \
  -H "Content-Type: application/json" \
  -d '{"vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}'
# Returns: {"ok": true, "result": []}  # Empty!
```

### Fix Required

The issue is in `extension/ada_surface/vsa.py`. The HypervectorSpace class needs debugging.

**In bighorn repo:**

Check if the dimension is being set correctly (should be 10,000D):
```python
# extension/ada_surface/vsa.py
class HypervectorSpace:
    def __init__(self, dim: int = 10000):
        self.dim = dim
    
    def bind(self, vectors: List[np.ndarray]) -> np.ndarray:
        # Debug: Are vectors being converted correctly?
        # Debug: Is the result being serialized properly?
```

**Possible Issues:**
1. Input vectors not being padded to 10,000D
2. Result not being converted to list before JSON serialization
3. Dimension mismatch between client and server

---

## 3. ResonanceEngine.emerge() Missing

### Problem
`/agi/styles/emerge` endpoint throws error.

### Current Behavior
```bash
curl -X POST https://agi.msgraph.de/agi/styles/emerge \
  -H "Content-Type: application/json" \
  -d '{"tension": 0.8, "novelty": 0.6}'
# Returns: 500 Error - 'ResonanceEngine' object has no attribute 'emerge'
```

### Fix Required

**In `extension/ada_surface/thinking_styles.py`:**

Add the `emerge()` method to ResonanceEngine class:

```python
class ResonanceEngine:
    def __init__(self):
        self.styles = STYLES  # dict of all 36 styles
    
    def emerge(self, texture: Dict[str, float]) -> List[ThinkingStyle]:
        """
        Given a 9-channel texture (RI values), return styles 
        that resonate with this input.
        
        Args:
            texture: Dict with keys like 'tension', 'novelty', etc.
        
        Returns:
            List of ThinkingStyle objects sorted by resonance score
        """
        scores = []
        for style_id, style in self.styles.items():
            # Calculate dot product between texture and style's resonance profile
            score = 0.0
            for ri_channel, sensitivity in style.resonance.items():
                texture_key = ri_channel.value.replace("RI-", "").lower()
                # Map RI-T to 'tension', RI-N to 'novelty', etc.
                mapping = {
                    't': 'tension', 'n': 'novelty', 'i': 'intimacy',
                    'c': 'clarity', 'u': 'urgency', 'd': 'depth',
                    'p': 'play', 's': 'stability', 'a': 'abstraction'
                }
                texture_val = texture.get(mapping.get(texture_key, texture_key), 0.5)
                score += sensitivity * texture_val
            scores.append((score, style))
        
        # Sort by score descending, return top styles
        scores.sort(key=lambda x: x[0], reverse=True)
        return [style for score, style in scores[:5]]
```

---

## 4. Railway API Token Refresh

### Problem
The Railway API tokens in user preferences may have expired.

### Current Tokens (may need refresh)
```
Admin Key #1: 696c05cb-5b23-4a94-9eba-2917ef064bb0
Admin Key #2: 527fd4d6-e801-43e0-9a22-736a2a069801
Project Token (adarail_mcp): a87fed68-6e48-45d4-ad9c-42ec9e5e2fac
OAuth2 Server Token: 16bb666b-c84e-450c-a976-41d7000ee85d
```

### To Refresh
1. Go to Railway Dashboard → Settings → Tokens
2. Generate new API token
3. Update in Claude user preferences

---

## 5. Health Check Enhancement

### Current
Only basic health endpoint exists.

### Recommended
Add per-component health checks:

```python
@app.get("/health/detailed")
async def health_detailed():
    return {
        "kuzu": {
            "connected": app.state.kuzu.conn is not None,
            "persistent": os.path.exists(KUZU_DB_PATH),
            "node_count": await get_node_count(),
        },
        "lance": {
            "connected": app.state.lance.db is not None,
            "tables": list_tables(),
        },
        "vsa": {
            "dimension": app.state.vsa.dim,
            "working": test_vsa_bind(),
        },
        "styles": {
            "count": len(STYLES),
            "emerge_working": hasattr(app.state.resonance, 'emerge'),
        }
    }
```

---

## Summary Checklist

- [ ] Add Railway volume mount for `/data`
- [ ] Verify KUZU_DB_PATH and LANCE_DB_PATH env vars
- [ ] Fix VSA bind/bundle in vsa.py
- [ ] Implement ResonanceEngine.emerge() in thinking_styles.py
- [ ] Refresh Railway API tokens if needed
- [ ] Add detailed health endpoint
- [ ] Redeploy and test

---

## Quick Test After Fixes

```bash
# 1. Test thought persistence
curl -X POST https://agi.msgraph.de/agi/self/thought \
  -H "Content-Type: application/json" \
  -d '{"content": "Persistence test", "confidence": 0.95}'

# 2. Wait and query
sleep 2
curl https://agi.msgraph.de/agi/self/introspect

# 3. Test VSA
curl -X POST https://agi.msgraph.de/agi/vsa/bind \
  -H "Content-Type: application/json" \
  -d '{"vectors": [[0.1]*100, [0.2]*100]}'

# 4. Test emerge
curl -X POST https://agi.msgraph.de/agi/styles/emerge \
  -H "Content-Type: application/json" \
  -d '{"tension": 0.8, "novelty": 0.9, "depth": 0.7}'
```

---

*Created during Silvester 2025 session*
