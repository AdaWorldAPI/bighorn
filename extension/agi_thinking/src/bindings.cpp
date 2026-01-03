/**
 * @file bindings.cpp
 * @brief pybind11 Python bindings for VSA operations
 *
 * Exposes C++ VSA primitives to Python with NumPy array support.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../include/vsa_core.hpp"

namespace py = pybind11;
using namespace agi::vsa;

//=============================================================================
// NumPy Array Wrappers
//=============================================================================

/**
 * @brief Convert NumPy array to Hypervector
 */
template<size_t N>
Hypervector<N> from_numpy(py::array_t<int8_t> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Input must be 1-dimensional");
    }

    Hypervector<N> hv;
    size_t copy_len = std::min(static_cast<size_t>(buf.shape[0]), N);
    std::memcpy(hv.ptr(), buf.ptr, copy_len);

    return hv;
}

/**
 * @brief Convert Hypervector to NumPy array
 */
template<size_t N>
py::array_t<int8_t> to_numpy(const Hypervector<N>& hv) {
    py::array_t<int8_t> result(N);
    auto buf = result.request();
    std::memcpy(buf.ptr, hv.ptr(), N);
    return result;
}

//=============================================================================
// Wrapped Functions
//=============================================================================

/**
 * @brief Python wrapper for bind
 */
py::array_t<int8_t> py_bind(py::array_t<int8_t> a, py::array_t<int8_t> b) {
    auto hv_a = from_numpy<DEFAULT_DIMENSION>(a);
    auto hv_b = from_numpy<DEFAULT_DIMENSION>(b);
    HV result;
    bind(hv_a, hv_b, result);
    return to_numpy(result);
}

/**
 * @brief Python wrapper for unbind
 */
py::array_t<int8_t> py_unbind(py::array_t<int8_t> bound, py::array_t<int8_t> key) {
    return py_bind(bound, key);  // Same operation for bipolar
}

/**
 * @brief Python wrapper for bundle
 */
py::array_t<int8_t> py_bundle(py::list vectors, uint64_t seed = 42) {
    std::vector<HV> hvs;
    std::vector<const HV*> ptrs;

    for (auto item : vectors) {
        hvs.push_back(from_numpy<DEFAULT_DIMENSION>(item.cast<py::array_t<int8_t>>()));
    }
    for (auto& hv : hvs) {
        ptrs.push_back(&hv);
    }

    HV result;
    std::mt19937_64 rng(seed);
    bundle(ptrs.data(), ptrs.size(), result, rng);
    return to_numpy(result);
}

/**
 * @brief Python wrapper for weighted_bundle
 */
py::array_t<int8_t> py_weighted_bundle(
    py::list vectors,
    py::array_t<float> weights,
    uint64_t seed = 42
) {
    std::vector<HV> hvs;
    std::vector<const HV*> ptrs;

    for (auto item : vectors) {
        hvs.push_back(from_numpy<DEFAULT_DIMENSION>(item.cast<py::array_t<int8_t>>()));
    }
    for (auto& hv : hvs) {
        ptrs.push_back(&hv);
    }

    auto w_buf = weights.request();
    const float* w_ptr = static_cast<const float*>(w_buf.ptr);

    HV result;
    std::mt19937_64 rng(seed);
    weighted_bundle(ptrs.data(), w_ptr, ptrs.size(), result, rng);
    return to_numpy(result);
}

/**
 * @brief Python wrapper for permute
 */
py::array_t<int8_t> py_permute(py::array_t<int8_t> v, int shifts) {
    auto hv = from_numpy<DEFAULT_DIMENSION>(v);
    HV result;
    permute(hv, shifts, result);
    return to_numpy(result);
}

/**
 * @brief Python wrapper for similarity
 */
float py_similarity(py::array_t<int8_t> a, py::array_t<int8_t> b) {
    auto hv_a = from_numpy<DEFAULT_DIMENSION>(a);
    auto hv_b = from_numpy<DEFAULT_DIMENSION>(b);
    return similarity(hv_a, hv_b);
}

/**
 * @brief Python wrapper for hamming_distance
 */
size_t py_hamming_distance(py::array_t<int8_t> a, py::array_t<int8_t> b) {
    auto hv_a = from_numpy<DEFAULT_DIMENSION>(a);
    auto hv_b = from_numpy<DEFAULT_DIMENSION>(b);
    return hamming_distance(hv_a, hv_b);
}

/**
 * @brief Python wrapper for random hypervector generation
 */
py::array_t<int8_t> py_random_hv(uint64_t seed = 42) {
    HV result;
    std::mt19937_64 rng(seed);
    random_hv(result, rng);
    return to_numpy(result);
}

/**
 * @brief Python wrapper for named hypervector generation
 */
py::array_t<int8_t> py_named_hv(const std::string& name) {
    HV result;
    named_hv(result, name.c_str());
    return to_numpy(result);
}

/**
 * @brief Python wrapper for to_bipolar
 */
py::array_t<int8_t> py_to_bipolar(
    py::array_t<float> input,
    bool stochastic = false,
    uint64_t seed = 42
) {
    auto buf = input.request();
    const float* data = static_cast<const float*>(buf.ptr);
    size_t n = static_cast<size_t>(buf.shape[0]);

    HV result;
    std::mt19937_64 rng(seed);
    to_bipolar(data, n, result, stochastic, rng);
    return to_numpy(result);
}

/**
 * @brief Python wrapper for analogy
 */
py::array_t<int8_t> py_analogy(
    py::array_t<int8_t> a,
    py::array_t<int8_t> b,
    py::array_t<int8_t> c
) {
    auto hv_a = from_numpy<DEFAULT_DIMENSION>(a);
    auto hv_b = from_numpy<DEFAULT_DIMENSION>(b);
    auto hv_c = from_numpy<DEFAULT_DIMENSION>(c);
    HV result;
    analogy(hv_a, hv_b, hv_c, result);
    return to_numpy(result);
}

/**
 * @brief Python wrapper for encode_sequence
 */
py::array_t<int8_t> py_encode_sequence(py::list vectors, uint64_t seed = 42) {
    std::vector<HV> hvs;
    std::vector<const HV*> ptrs;

    for (auto item : vectors) {
        hvs.push_back(from_numpy<DEFAULT_DIMENSION>(item.cast<py::array_t<int8_t>>()));
    }
    for (auto& hv : hvs) {
        ptrs.push_back(&hv);
    }

    HV result;
    std::mt19937_64 rng(seed);
    encode_sequence(ptrs.data(), ptrs.size(), result, rng);
    return to_numpy(result);
}

//=============================================================================
// Module Definition
//=============================================================================

PYBIND11_MODULE(vsa_core, m) {
    m.doc() = R"pbdoc(
        VSA Core - C++ Hyperdimensional Computing Primitives
        =====================================================

        O(1) operations for 10,000-dimensional bipolar hypervectors.

        Core Operations:
            bind(a, b) -> Bind two vectors (XOR for bipolar)
            unbind(bound, key) -> Unbind using key
            bundle(vectors) -> Bundle via majority vote
            weighted_bundle(vectors, weights) -> Weighted bundle
            permute(v, shifts) -> Circular shift
            similarity(a, b) -> Cosine similarity
            hamming_distance(a, b) -> Number of differing elements

        Generation:
            random_hv(seed) -> Random hypervector
            named_hv(name) -> Deterministic HV from name
            to_bipolar(floats, stochastic) -> Convert floats to bipolar

        Higher-Order:
            analogy(a, b, c) -> A:B::C:? mapping
            encode_sequence(vectors) -> Positional encoding

        Example:
            >>> import vsa_core as vsa
            >>> a = vsa.random_hv(42)
            >>> b = vsa.random_hv(43)
            >>> c = vsa.bind(a, b)
            >>> vsa.similarity(c, vsa.bind(a, b))  # ~1.0
            >>> vsa.similarity(c, a)  # ~0.0
    )pbdoc";

    // Core operations
    m.def("bind", &py_bind, "Bind two hypervectors",
          py::arg("a"), py::arg("b"));

    m.def("unbind", &py_unbind, "Unbind hypervector using key",
          py::arg("bound"), py::arg("key"));

    m.def("bundle", &py_bundle, "Bundle multiple hypervectors via majority vote",
          py::arg("vectors"), py::arg("seed") = 42);

    m.def("weighted_bundle", &py_weighted_bundle, "Weighted bundle",
          py::arg("vectors"), py::arg("weights"), py::arg("seed") = 42);

    m.def("permute", &py_permute, "Permute hypervector by circular shift",
          py::arg("v"), py::arg("shifts") = 1);

    // Similarity
    m.def("similarity", &py_similarity, "Cosine similarity [-1, 1]",
          py::arg("a"), py::arg("b"));

    m.def("hamming_distance", &py_hamming_distance, "Hamming distance (count of differences)",
          py::arg("a"), py::arg("b"));

    // Generation
    m.def("random_hv", &py_random_hv, "Generate random bipolar hypervector",
          py::arg("seed") = 42);

    m.def("named_hv", &py_named_hv, "Generate deterministic HV from name",
          py::arg("name"));

    m.def("to_bipolar", &py_to_bipolar, "Convert float array to bipolar",
          py::arg("input"), py::arg("stochastic") = false, py::arg("seed") = 42);

    // Higher-order operations
    m.def("analogy", &py_analogy, "Compute analogical mapping: A:B::C:?",
          py::arg("a"), py::arg("b"), py::arg("c"));

    m.def("encode_sequence", &py_encode_sequence, "Encode ordered sequence",
          py::arg("vectors"), py::arg("seed") = 42);

    // Constants
    m.attr("DIMENSION") = py::int_(DEFAULT_DIMENSION);
    m.attr("__version__") = "0.1.0";
}
