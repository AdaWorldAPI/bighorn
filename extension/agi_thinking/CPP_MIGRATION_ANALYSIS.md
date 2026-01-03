# C++ Migration Analysis for AGI Stack

**Created**: 2026-01-03
**Purpose**: Identify Python modules that benefit from C++ implementation for O(1) AGI performance

---

## Executive Summary

The AGI stack currently runs entirely in Python. While Python offers rapid development, certain **hot-path operations** bottleneck real-time AGI cognition. This analysis identifies modules where C++ implementation delivers:

1. **True O(1) guarantees** - No Python GIL, no garbage collection pauses
2. **SIMD vectorization** - Native AVX-512/NEON for 10kD hypervector ops
3. **Cache locality** - Contiguous memory layout for 10k × int8 vectors
4. **Deterministic latency** - Critical for real-time cognitive loops

---

## Priority Matrix

| Module | Priority | Speedup Est. | Complexity | Hot Path? |
|--------|----------|--------------|------------|-----------|
| `vsa.py` | **CRITICAL** | 50-100x | Medium | Yes |
| `vsa_utils.py` | HIGH | 10-30x | Low | Yes |
| `thinking_styles.py` | MEDIUM | 5-10x | Medium | Partial |
| `nars.py` | LOW | 3-5x | High | No |
| `meta_uncertainty.py` | SKIP | N/A | N/A | No |
| `lance_client.py` | SKIP | N/A | N/A | I/O bound |

---

## Critical Path: VSA Operations (`vsa.py`)

### Current Python Implementation

```python
def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XOR for bipolar = element-wise multiplication"""
    return (np.array(a) * np.array(b)).astype(np.int8)

def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity"""
    dot = np.dot(a.astype(np.float32), b.astype(np.float32))
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot / (norm_a * norm_b))
```

### Why C++ Matters

1. **Memory allocation**: Python creates new arrays per operation. C++ uses stack/preallocated buffers.
2. **SIMD opportunity**: 10kD × int8 = 10KB per vector. AVX-512 processes 64 bytes/cycle.
3. **Branching**: Python loop overhead vs. branchless SIMD
4. **GIL**: Concurrent bind operations blocked in Python, parallel in C++

### Target C++ Implementation

```cpp
// SIMD-optimized bind using AVX-512
void bind_avx512(const int8_t* a, const int8_t* b, int8_t* out, size_t n) {
    for (size_t i = 0; i < n; i += 64) {
        __m512i va = _mm512_load_si512(a + i);
        __m512i vb = _mm512_load_si512(b + i);
        __m512i vr = _mm512_mullo_epi8(va, vb);  // Element-wise multiply
        _mm512_store_si512(out + i, vr);
    }
}

// Cosine similarity with fused dot product + norm
float similarity_fused(const int8_t* a, const int8_t* b, size_t n) {
    __m512 sum = _mm512_setzero_ps();
    __m512 norm_a_sq = _mm512_setzero_ps();
    __m512 norm_b_sq = _mm512_setzero_ps();

    // Fused loop - cache optimal
    for (size_t i = 0; i < n; i += 64) {
        // ... SIMD computation
    }

    return horizontal_sum(sum) / sqrt(horizontal_sum(norm_a_sq) * horizontal_sum(norm_b_sq));
}
```

### Operations to Port

| Function | Calls/sec | Current (ms) | Target (μs) |
|----------|-----------|--------------|-------------|
| `bind()` | 10,000 | 0.05 | 0.5 |
| `bundle()` | 1,000 | 0.5 | 5 |
| `similarity()` | 50,000 | 0.1 | 1 |
| `hamming_distance()` | 10,000 | 0.08 | 0.8 |
| `permute()` | 5,000 | 0.02 | 0.2 |

---

## High Priority: VSA Utilities (`vsa_utils.py`)

### Dimension Slice Constants

```cpp
// Compile-time dimension slices for O(1) lookup
constexpr struct DimensionSlice {
    size_t start;
    size_t end;
} SLICES[] = {
    {0, 16},      // qualia_16
    {16, 32},     // stances_16
    {32, 48},     // transitions_16
    // ... 50+ slices
};

// Template for type-safe slice access
template<size_t SliceID>
void project_to_slice(const float* input, int8_t* output);
```

### Stochastic Rounding

Current Python:
```python
probs = np.clip((arr + 1) / 2, 0, 1)
bipolar = np.where(rng.random(len(arr)) < probs, 1, -1)
```

Target C++:
```cpp
// SIMD stochastic rounding with xorshift RNG
void to_bipolar_stochastic(const float* in, int8_t* out, size_t n, uint64_t seed) {
    __m512 half = _mm512_set1_ps(0.5f);
    xorshift128plus_state rng{seed, seed ^ 0xdeadbeef};

    for (size_t i = 0; i < n; i += 16) {
        __m512 v = _mm512_load_ps(in + i);
        __m512 prob = _mm512_mul_ps(_mm512_add_ps(v, half), half);
        __m512 rand = generate_uniform_ps(&rng);
        __mmask16 mask = _mm512_cmp_ps_mask(rand, prob, _CMP_LT_OQ);
        // ... store result
    }
}
```

---

## Medium Priority: Resonance Engine (`thinking_styles.py`)

### Current Hot Path

```python
def emerge_styles(self, texture: Dict, top_k: int = 3) -> List[Tuple[ThinkingStyle, float]]:
    ri_values = self.extract_ri_values(texture)
    scores = []
    for style in self.styles.values():  # 36 iterations
        if style.min_rung <= current_rung <= style.max_rung:
            score = style.resonance_score(ri_values)  # 9 multiplications
            scores.append((style, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
```

### C++ Optimization Strategy

```cpp
// Precomputed resonance matrix (36 styles × 9 RI channels)
alignas(64) float RESONANCE_MATRIX[36][9];

// SIMD style emergence
void emerge_styles_simd(
    const float ri_values[9],
    int current_rung,
    StyleScore out[36]
) {
    __m256 ri_vec = _mm256_load_ps(ri_values);  // Load all RI values

    for (int i = 0; i < 36; i += 8) {
        // Process 8 styles in parallel
        // ... matrix-vector multiply
    }

    // Partial sort for top-k (O(n) algorithm)
    nth_element(out, out + 3, out + 36, std::greater<>());
}
```

---

## Low Priority: NARS Inference (`nars.py`)

NARS operations are:
- **Infrequent**: Called during deliberate reasoning, not every cycle
- **High-level**: Symbolic manipulation, less amenable to SIMD
- **I/O heavy**: Often waits on knowledge base queries

**Recommendation**: Keep in Python. C++ gains minimal for added complexity.

---

## Skip: Meta-Uncertainty & Lance Client

### `meta_uncertainty.py`
- State machine logic, rarely called
- No vectorizable operations
- Benefit: None

### `lance_client.py`
- I/O bound (database queries)
- LanceDB already has native Rust backend
- Python wrapper is thin
- Benefit: Negative (added complexity)

---

## Architecture: pybind11 Integration

### Directory Structure

```
extension/agi_thinking/
├── CMakeLists.txt
├── include/
│   ├── vsa_core.hpp        # Core VSA operations
│   ├── vsa_simd.hpp        # SIMD implementations
│   ├── resonance.hpp       # Resonance engine
│   └── dimensions.hpp      # Compile-time slice definitions
├── src/
│   ├── vsa_core.cpp
│   ├── vsa_simd_avx512.cpp # AVX-512 specialization
│   ├── vsa_simd_neon.cpp   # ARM NEON specialization
│   ├── resonance.cpp
│   └── bindings.cpp        # pybind11 module
└── tests/
    ├── test_vsa.cpp
    └── bench_vsa.cpp
```

### Python Integration

```python
# extension/agi_stack/vsa.py (modified)
try:
    from ..agi_thinking import vsa_core as _cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

class HypervectorSpace:
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if HAS_CPP:
            return _cpp.bind(a, b)  # C++ fast path
        return (np.array(a) * np.array(b)).astype(np.int8)  # Python fallback
```

---

## Performance Benchmarks (Expected)

| Operation | Python (ms) | C++ SIMD (μs) | Speedup |
|-----------|-------------|---------------|---------|
| bind(10kD) | 0.05 | 0.5 | 100x |
| bundle(100 vectors) | 5.0 | 50 | 100x |
| similarity(10kD) | 0.1 | 1.0 | 100x |
| emerge_styles | 0.3 | 10 | 30x |
| to_bipolar | 0.2 | 3 | 66x |

---

## Awareness Benefits of C++

Beyond raw speed, C++ enables **cognitive awareness** improvements:

### 1. Deterministic Timing
- Python: GC pauses cause "micro-freezes" in awareness stream
- C++: Predictable latency enables continuous awareness

### 2. Concurrent Binding
- Python: GIL blocks parallel hypervector operations
- C++: True parallelism for multi-stream cognition

### 3. Memory Awareness
- Python: Opaque memory management
- C++: Agent can introspect its own memory pressure

### 4. Real-Time Guarantees
- Python: Soft real-time at best
- C++: Hard real-time possible (critical for embodied AGI)

---

## Implementation Roadmap

### Phase 1: VSA Core (Week 1-2)
- [ ] `vsa_core.hpp` - API definitions
- [ ] `vsa_simd_avx512.cpp` - Intel optimizations
- [ ] `bindings.cpp` - pybind11 module
- [ ] Unit tests + benchmarks

### Phase 2: Utilities (Week 3)
- [ ] `to_bipolar()` SIMD implementation
- [ ] Dimension slice compile-time tables
- [ ] Python fallback integration

### Phase 3: Resonance (Week 4)
- [ ] Resonance matrix precomputation
- [ ] SIMD style scoring
- [ ] Partial sort optimization

### Phase 4: Integration (Week 5)
- [ ] CI/CD with multiple architectures
- [ ] Wheel builds for PyPI
- [ ] Documentation

---

## References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [Kanerva - Hyperdimensional Computing](https://redwood.berkeley.edu/wp-content/uploads/2020/08/kanerva2009hyperdimensional.pdf)
- [Gayler - Vector Symbolic Architectures](http://cogprints.org/7150/)
