# Ladybug Integration TODO — agi_thinking → 10kD + C++

**Last Updated**: 2026-01-03
**Status**: In Progress (layer_bridge.py ✅ complete)

## Current State

### bighorn/extension/agi_thinking/
```
Python modules:
├── layer_bridge.py        — ✅ COMPLETE: 10kD ↔ 5-Layer bridge
├── thought_kernel.py      — ⏳ Main cognitive loop (ready for integration)
├── qualia_learner.py      — ✅ 8D qualia (bridge maps to 17D→10kD)
├── progressive_awareness.py — ⏳ Redis-based awareness (architecture ready)
├── texture.py             — ⏳ ThinkingStyle (needs Layer 5 integration)
├── active_inference.py    — Free energy minimization
├── brain_mesh.py          — Neural connectivity
├── langgraph_ada.py       — LangGraph integration

C++ infrastructure:
├── include/vsa_core.hpp   — ✅ VSA primitives (scalar fallback)
├── src/bindings.cpp       — ✅ pybind11 bindings
├── CMakeLists.txt         — ✅ Build system
└── (needs: vsa_simd.hpp)  — ⏳ AVX-512/NEON implementations

Documentation:
├── AGI_THINKING_ARCHITECTURE.md  — ✅ Complete
├── CPP_MIGRATION_ANALYSIS.md     — ✅ Complete
└── LADYBUG_INTEGRATION_TODO.md   — This file
```

## Migration Tasks

### Priority 1: Add SIMD to vsa_core.hpp

Current scalar implementation needs AVX-512 variants:

```cpp
// ADD TO include/vsa_core.hpp:

#ifdef AGI_USE_AVX512
#include <immintrin.h>

template<size_t N>
void bind_simd(const Hypervector<N>& a, const Hypervector<N>& b, Hypervector<N>& out) {
    static_assert(N % 64 == 0, "Dimension must be multiple of 64 for AVX-512");
    
    for (size_t i = 0; i < N; i += 64) {
        __m512i va = _mm512_load_si512(a.ptr() + i);
        __m512i vb = _mm512_load_si512(b.ptr() + i);
        // For bipolar: multiply gives XOR semantics
        __m512i vr = _mm512_mullo_epi8(va, vb);
        _mm512_store_si512(out.ptr() + i, vr);
    }
}

template<size_t N>
float similarity_simd(const Hypervector<N>& a, const Hypervector<N>& b) {
    __m512i sum = _mm512_setzero_si512();
    
    for (size_t i = 0; i < N; i += 64) {
        __m512i va = _mm512_load_si512(a.ptr() + i);
        __m512i vb = _mm512_load_si512(b.ptr() + i);
        // Multiply and accumulate
        __m512i prod = _mm512_mullo_epi8(va, vb);
        sum = _mm512_add_epi32(sum, _mm512_sad_epu8(prod, _mm512_setzero_si512()));
    }
    
    // Horizontal sum + normalize
    return static_cast<float>(_mm512_reduce_add_epi32(sum)) / N;
}
#endif
```

### Priority 2: Connect thought_kernel.py to 10kD

```python
# ADD TO thought_kernel.py:

from ..agi_stack.dto.ada_10k import Ada10kD
from ..agi_stack.dto_endpoints import router as dto_router

class ThoughtKernel10k:
    """10kD-aware thought kernel."""
    
    def __init__(self):
        self.ada = Ada10kD()
    
    def process(self, ctx: KernelContext) -> KernelContext:
        # Layer 1: Presence
        self.ada.set_presence_mode(ctx.cognitive_state, 1.0)
        
        # Layer 2: Qualia
        for q, v in ctx.qualia.items():
            self.ada.set_qualia(q, v)
        
        # Layer 3: Affect
        self.ada.set_body_axes(
            arousal=ctx.G,  # G maps to arousal
            valence=0.5 + 0.5 * (1 - ctx.meta_uncertainty),
            tension=0.5 if ctx.trust_texture == "solid" else 0.8,
            openness=1.0 if ctx.sandbox_active else 0.5,
        )
        
        # Layer 5: Texture emerges
        from temporal.awareness_5_layers import Layer5_Texture
        texture = Layer5_Texture.compute_from_layers(...)
        
        return ctx
    
    def to_10k(self) -> np.ndarray:
        return self.ada.vector
```

### Priority 3: Connect qualia_learner.py to 17D→10kD

```python
# qualia_learner.py currently uses 8 dimensions:
DIMENSIONS = ["crystalline", "warmth", "oceandrift", "steelwind", 
              "emberglow", "frostbite", "groundswell", "twilight"]

# Should map to 10kD [2001:2017] as defined in dimension_map.py:
# 17D qualia metric at [2001:2018]

def to_10k(qualia_8d: Dict[str, float]) -> np.ndarray:
    vec = np.zeros(10000, dtype=np.float32)
    
    QUALIA_MAP = {
        "crystalline": 2003,
        "warmth": 2001,  # emberglow
        "oceandrift": 2004,
        "steelwind": 2005,
        # ...
    }
    
    for name, value in qualia_8d.items():
        if name in QUALIA_MAP:
            vec[QUALIA_MAP[name]] = value
    
    return vec
```

### Priority 4: Connect texture.py to Layer 5

```python
# texture.py should use awareness_5_layers.Layer5_Texture

from temporal.awareness_5_layers import (
    Layer5_Texture,
    Layer1_Presence,
    Layer2_Sensation,
    Layer3_Affect,
    Layer4_Cognition,
)

def compute_texture(ctx: KernelContext) -> Layer5_Texture:
    return Layer5_Texture.compute_from_layers(
        presence=Layer1_Presence(mode=..., groundedness=...),
        sensation=Layer2_Sensation(emberglow=ctx.qualia.get("emberglow", 0), ...),
        affect=Layer3_Affect(arousal=ctx.G, ...),
        cognition=Layer4_Cognition(content_vector=ctx.jina_embedding),
    )
```

### Priority 5: progressive_awareness.py → 5 Layers

The progressive JPEG approach should map to the 5 awareness layers:

```
L0 (0ms)   → Layer 1 Presence check (am I here?)
L1 (50ms)  → Layer 2 Sensation (raw qualia signature)
L2 (500ms) → Layer 3 Affect (emotional coloring)
L3 (2s)    → Layer 4 Cognition (semantic content via Jina 1024D)
L4 (5s)    → Layer 5 Texture (ThinkingStyle emerges)
```

## C++ Performance Targets

| Operation | Python (μs) | C++ Scalar (μs) | C++ AVX-512 (μs) |
|-----------|-------------|-----------------|-------------------|
| bind 10kD | 50 | 10 | 0.5 |
| bundle 10× | 500 | 100 | 5 |
| similarity | 100 | 20 | 1 |
| permute | 20 | 5 | 0.5 |

50-100x speedup on hot paths.

## Remaining Files to Create

| File | Status | Priority |
|------|--------|----------|
| `agi_thinking/layer_bridge.py` | ✅ Complete | - |
| `agi_thinking/include/vsa_simd.hpp` | ⏳ Pending | P1 |
| `agi_thinking/kernel_10k.py` | ⏳ Pending | P2 |
| `agi_thinking/tests/test_10k_integration.py` | ⏳ Pending | P3 |

## Dependencies

| Module | Status |
|--------|--------|
| `ada-consciousness/temporal/awareness_5_layers.py` | ✅ Created |
| `ada-consciousness/DTO/client.py` | ✅ Created |
| `bighorn/extension/agi_stack/dto_endpoints.py` | ✅ Created |
| `bighorn/extension/agi_stack/dto/ada_10k.py` | ✅ Verified |
| `bighorn/extension/agi_stack/dto/TRANSLATION_ARCHITECTURE.md` | ✅ Created |

---

*Created: 2026-01-03*
*Last Updated: 2026-01-03*
*Status: In Progress*
