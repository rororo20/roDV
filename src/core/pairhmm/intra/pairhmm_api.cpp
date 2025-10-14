#include "pairhmm_api.h"
#include "pairhmm_impl.h"
#include "simd_traits.h"

namespace pairhmm {
namespace intra {

// ============================================================================
// AVX2 Float
// ============================================================================
float compute_pairhmm_avx2_float(const TestCase& tc) {
    return PairHMMComputer<AVX2FloatTraits>::compute(tc);
}

// ============================================================================
// AVX2 Double
// ============================================================================
double compute_pairhmm_avx2_double(const TestCase& tc) {
    return PairHMMComputer<AVX2DoubleTraits>::compute(tc);
}

// ============================================================================
// AVX512 Float
// ============================================================================
float compute_pairhmm_avx512_float(const TestCase& tc) {
    return PairHMMComputer<AVX512FloatTraits>::compute(tc);
}

// ============================================================================
// AVX512 Double
// ============================================================================
double compute_pairhmm_avx512_double(const TestCase& tc) {
    return PairHMMComputer<AVX512DoubleTraits>::compute(tc);
}

}  // namespace intra
}  // namespace pairhmm

