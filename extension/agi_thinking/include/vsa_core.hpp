/**
 * @file vsa_core.hpp
 * @brief Vector Symbolic Architecture (VSA) Core Operations
 *
 * O(1) hyperdimensional computing primitives for AGI cognition.
 * SIMD-optimized for 10,000-dimensional bipolar vectors.
 *
 * @author Ada World API / AGI Stack
 * @date 2026-01-03
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <array>
#include <vector>
#include <random>

namespace agi {
namespace vsa {

//=============================================================================
// Constants
//=============================================================================

/// Default hypervector dimension (10kD space)
constexpr size_t DEFAULT_DIMENSION = 10000;

/// Cache line size for alignment
constexpr size_t CACHE_LINE = 64;

/// SIMD register width (bytes) - AVX-512
constexpr size_t SIMD_WIDTH = 64;

//=============================================================================
// Hypervector Type
//=============================================================================

/**
 * @brief Bipolar hypervector: elements are {-1, +1}
 *
 * Uses int8_t for compact storage (10KB per vector).
 * Aligned for SIMD operations.
 */
template<size_t N = DEFAULT_DIMENSION>
struct alignas(CACHE_LINE) Hypervector {
    std::array<int8_t, N> data;

    /// Default constructor (zero-initialized)
    Hypervector() : data{} {}

    /// Element access
    int8_t& operator[](size_t i) { return data[i]; }
    const int8_t& operator[](size_t i) const { return data[i]; }

    /// Iterator support
    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }

    /// Size
    static constexpr size_t size() { return N; }

    /// Raw pointer access for SIMD
    int8_t* ptr() { return data.data(); }
    const int8_t* ptr() const { return data.data(); }
};

/// Standard 10kD hypervector
using HV = Hypervector<DEFAULT_DIMENSION>;

//=============================================================================
// Core Operations (Scalar Fallback)
//=============================================================================

/**
 * @brief Bind two hypervectors (XOR for bipolar = element-wise multiply)
 *
 * Properties:
 * - Commutative: A ⊗ B = B ⊗ A
 * - Associative: (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)
 * - Self-inverse: A ⊗ A = 1
 *
 * @param a First hypervector
 * @param b Second hypervector
 * @param out Output hypervector (can alias a or b)
 */
template<size_t N>
void bind(const Hypervector<N>& a, const Hypervector<N>& b, Hypervector<N>& out) {
    for (size_t i = 0; i < N; ++i) {
        out[i] = a[i] * b[i];
    }
}

/**
 * @brief Unbind (same as bind for bipolar XOR)
 */
template<size_t N>
inline void unbind(const Hypervector<N>& bound, const Hypervector<N>& key, Hypervector<N>& out) {
    bind(bound, key, out);
}

/**
 * @brief Bundle multiple hypervectors (majority vote)
 *
 * @param vectors Array of hypervector pointers
 * @param count Number of vectors to bundle
 * @param out Output bundled hypervector
 * @param rng Random generator for tie-breaking
 */
template<size_t N, typename RNG>
void bundle(const Hypervector<N>* const* vectors, size_t count, Hypervector<N>& out, RNG& rng) {
    std::array<int32_t, N> sum{};

    // Sum all vectors
    for (size_t v = 0; v < count; ++v) {
        for (size_t i = 0; i < N; ++i) {
            sum[i] += (*vectors[v])[i];
        }
    }

    // Threshold (majority vote)
    std::uniform_int_distribution<int> coin(0, 1);
    for (size_t i = 0; i < N; ++i) {
        if (sum[i] > 0) {
            out[i] = 1;
        } else if (sum[i] < 0) {
            out[i] = -1;
        } else {
            out[i] = coin(rng) ? 1 : -1;
        }
    }
}

/**
 * @brief Weighted bundle
 *
 * @param vectors Array of hypervector pointers
 * @param weights Array of weights
 * @param count Number of vectors
 * @param out Output hypervector
 * @param rng Random generator for tie-breaking
 */
template<size_t N, typename RNG>
void weighted_bundle(
    const Hypervector<N>* const* vectors,
    const float* weights,
    size_t count,
    Hypervector<N>& out,
    RNG& rng
) {
    std::array<float, N> weighted_sum{};

    for (size_t v = 0; v < count; ++v) {
        float w = weights[v];
        for (size_t i = 0; i < N; ++i) {
            weighted_sum[i] += (*vectors[v])[i] * w;
        }
    }

    std::uniform_int_distribution<int> coin(0, 1);
    for (size_t i = 0; i < N; ++i) {
        if (weighted_sum[i] > 0.0f) {
            out[i] = 1;
        } else if (weighted_sum[i] < 0.0f) {
            out[i] = -1;
        } else {
            out[i] = coin(rng) ? 1 : -1;
        }
    }
}

/**
 * @brief Permute hypervector by circular shift
 *
 * Used for encoding positional/sequential information.
 *
 * @param v Input hypervector
 * @param shifts Number of positions to shift (positive = right)
 * @param out Output hypervector
 */
template<size_t N>
void permute(const Hypervector<N>& v, int shifts, Hypervector<N>& out) {
    shifts = ((shifts % static_cast<int>(N)) + static_cast<int>(N)) % static_cast<int>(N);
    for (size_t i = 0; i < N; ++i) {
        out[(i + shifts) % N] = v[i];
    }
}

/**
 * @brief Inverse permute (shift opposite direction)
 */
template<size_t N>
inline void inverse_permute(const Hypervector<N>& v, int shifts, Hypervector<N>& out) {
    permute(v, -shifts, out);
}

//=============================================================================
// Similarity Metrics
//=============================================================================

/**
 * @brief Cosine similarity between two hypervectors
 *
 * @return Value in [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite
 */
template<size_t N>
float similarity(const Hypervector<N>& a, const Hypervector<N>& b) {
    int64_t dot = 0;
    int64_t norm_a_sq = 0;
    int64_t norm_b_sq = 0;

    for (size_t i = 0; i < N; ++i) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }

    if (norm_a_sq == 0 || norm_b_sq == 0) {
        return 0.0f;
    }

    return static_cast<float>(dot) / std::sqrt(static_cast<float>(norm_a_sq * norm_b_sq));
}

/**
 * @brief Hamming distance (number of differing elements)
 */
template<size_t N>
size_t hamming_distance(const Hypervector<N>& a, const Hypervector<N>& b) {
    size_t dist = 0;
    for (size_t i = 0; i < N; ++i) {
        if (a[i] != b[i]) {
            ++dist;
        }
    }
    return dist;
}

/**
 * @brief Normalized Hamming distance [0, 1]
 */
template<size_t N>
float normalized_hamming(const Hypervector<N>& a, const Hypervector<N>& b) {
    return static_cast<float>(hamming_distance(a, b)) / static_cast<float>(N);
}

//=============================================================================
// Random Generation
//=============================================================================

/**
 * @brief Generate random bipolar hypervector
 */
template<size_t N, typename RNG>
void random_hv(Hypervector<N>& out, RNG& rng) {
    std::uniform_int_distribution<int> dist(0, 1);
    for (size_t i = 0; i < N; ++i) {
        out[i] = dist(rng) ? 1 : -1;
    }
}

/**
 * @brief Generate deterministic hypervector from seed
 */
template<size_t N>
void seeded_hv(Hypervector<N>& out, uint64_t seed) {
    std::mt19937_64 rng(seed);
    random_hv(out, rng);
}

/**
 * @brief Generate hypervector from name (deterministic)
 */
template<size_t N>
void named_hv(Hypervector<N>& out, const char* name) {
    // Simple hash function
    uint64_t hash = 14695981039346656037ULL;
    for (const char* p = name; *p; ++p) {
        hash ^= static_cast<uint64_t>(*p);
        hash *= 1099511628211ULL;
    }
    seeded_hv(out, hash);
}

//=============================================================================
// Bipolar Conversion
//=============================================================================

/**
 * @brief Convert float array to bipolar with optional stochastic rounding
 *
 * @param input Float array (assumed range [-1, 1] or [0, 1])
 * @param n Input length
 * @param out Output hypervector
 * @param stochastic Use stochastic rounding (preserves gradient info)
 * @param rng Random generator
 */
template<size_t N, typename RNG>
void to_bipolar(
    const float* input,
    size_t n,
    Hypervector<N>& out,
    bool stochastic,
    RNG& rng
) {
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    size_t copy_len = std::min(n, N);

    if (stochastic) {
        for (size_t i = 0; i < copy_len; ++i) {
            // Map [-1, 1] to [0, 1] probability
            float prob = (input[i] + 1.0f) * 0.5f;
            prob = std::clamp(prob, 0.0f, 1.0f);
            out[i] = (uniform(rng) < prob) ? 1 : -1;
        }
    } else {
        for (size_t i = 0; i < copy_len; ++i) {
            out[i] = (input[i] >= 0.0f) ? 1 : -1;
        }
    }

    // Fill remaining with random
    for (size_t i = copy_len; i < N; ++i) {
        out[i] = (uniform(rng) < 0.5f) ? 1 : -1;
    }
}

//=============================================================================
// Sequence Encoding
//=============================================================================

/**
 * @brief Encode ordered sequence
 *
 * S = P^0(v0) + P^1(v1) + P^2(v2) + ...
 */
template<size_t N, typename RNG>
void encode_sequence(
    const Hypervector<N>* const* vectors,
    size_t count,
    Hypervector<N>& out,
    RNG& rng
) {
    std::vector<Hypervector<N>> permuted(count);
    std::vector<const Hypervector<N>*> permuted_ptrs(count);

    for (size_t i = 0; i < count; ++i) {
        permute(*vectors[i], static_cast<int>(i), permuted[i]);
        permuted_ptrs[i] = &permuted[i];
    }

    bundle(permuted_ptrs.data(), count, out, rng);
}

//=============================================================================
// Structure Encoding (Role-Filler)
//=============================================================================

/**
 * @brief Encode role-filler structure
 *
 * S = (role1 ⊗ filler1) + (role2 ⊗ filler2) + ...
 */
template<size_t N, typename RNG>
void encode_structure(
    const Hypervector<N>* const* roles,
    const Hypervector<N>* const* fillers,
    size_t count,
    Hypervector<N>& out,
    RNG& rng
) {
    std::vector<Hypervector<N>> bindings(count);
    std::vector<const Hypervector<N>*> binding_ptrs(count);

    for (size_t i = 0; i < count; ++i) {
        bind(*roles[i], *fillers[i], bindings[i]);
        binding_ptrs[i] = &bindings[i];
    }

    bundle(binding_ptrs.data(), count, out, rng);
}

/**
 * @brief Query structure for filler given role
 */
template<size_t N>
void query_structure(
    const Hypervector<N>& structure,
    const Hypervector<N>& role,
    Hypervector<N>& out
) {
    unbind(structure, role, out);
}

//=============================================================================
// Analogy
//=============================================================================

/**
 * @brief Compute analogical mapping: A is to B as C is to ?
 *
 * Uses: ? = B ⊗ A ⊗ C = (B ⊗ A) ⊗ C
 */
template<size_t N>
void analogy(
    const Hypervector<N>& a,
    const Hypervector<N>& b,
    const Hypervector<N>& c,
    Hypervector<N>& out
) {
    Hypervector<N> relation;
    bind(b, a, relation);  // R = B ⊗ A
    bind(relation, c, out);  // D = R ⊗ C
}

} // namespace vsa
} // namespace agi
