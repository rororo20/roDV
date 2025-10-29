#include "pairhmm_inter_api.h"
#include "../common/cpu_features.h"
#include "pairhmm_inter.h"
#include <algorithm>
#include <cmath>
#include <simd_traits.h>
#include <string.h>
namespace pairhmm {
namespace inter {
constexpr float LOG10_INITIAL_CONSTANT_F = 36.1236000061;
constexpr double LOG10_INITIAL_CONSTANT_D = 307.050595577260822;

#if defined(__AVX512F__)
bool compute_inter_pairhmm_AVX512_float(TestCase *tc, uint32_t num,double *results) {
  if (num < AVX512FloatTraits::simd_width)
    return false;
  MultiTestCase<AVX512FloatTraits> mtc;
  memcpy(mtc.test_cases, tc, num * sizeof(TestCase));
  InterPairHMMComputer<AVX512FloatTraits>::precompute(mtc);
  InterPairHMMComputer<AVX512FloatTraits>::compute(mtc);
  for(uint32_t i = 0; i < AVX512FloatTraits::simd_width; i++) {
    results[i] =  log10(mtc.results[i]) - LOG10_INITIAL_CONSTANT_F;
  }
  InterPairHMMComputer<AVX512FloatTraits>::finalize(mtc);
  return true;
}

bool compute_inter_pairhmm_AVX512_double(TestCase *tc, uint32_t num,double *results) { 
    if(num < AVX512DoubleTraits::simd_width)
        return false;
    MultiTestCase<AVX512DoubleTraits> mtc;
    memcpy(mtc.test_cases, tc, num * sizeof(TestCase));
    InterPairHMMComputer<AVX512DoubleTraits>::precompute(mtc);
    InterPairHMMComputer<AVX512DoubleTraits>::compute(mtc);
    for(uint32_t i = 0; i < AVX512DoubleTraits::simd_width; i++) {
        results[i] =  log10(mtc.results[i]) - LOG10_INITIAL_CONSTANT_D;
    }
    InterPairHMMComputer<AVX512DoubleTraits>::finalize(mtc);
    return true;
 }
#elif defined(__AVX2__)
bool compute_inter_pairhmm_AVX2_float(TestCase *tc, uint32_t num,double *results) {
    if(num < AVX2FloatTraits::simd_width)
        return false;
    MultiTestCase<AVX2FloatTraits> mtc;
    memcpy(mtc.test_cases, tc, num * sizeof(TestCase));
    InterPairHMMComputer<AVX2FloatTraits>::precompute(mtc);
    InterPairHMMComputer<AVX2FloatTraits>::compute(mtc);
    for(uint32_t i = 0; i < AVX2FloatTraits::simd_width; i++) {
        results[i] =  log10(mtc.results[i]) - LOG10_INITIAL_CONSTANT_F;
    }
    InterPairHMMComputer<AVX2FloatTraits>::finalize(mtc);
    return true;
}
bool compute_inter_pairhmm_AVX2_double(TestCase *tc, uint32_t num,double *results) {
    if(num < AVX2DoubleTraits::simd_width)
        return false;
    MultiTestCase<AVX2DoubleTraits> mtc;
    memcpy(mtc.test_cases, tc, num * sizeof(TestCase));
    InterPairHMMComputer<AVX2DoubleTraits>::precompute(mtc);
    InterPairHMMComputer<AVX2DoubleTraits>::compute(mtc);
    for(uint32_t i = 0; i < AVX2DoubleTraits::simd_width; i++) {
        results[i] =  log10(mtc.results[i]) - LOG10_INITIAL_CONSTANT_D;
    }
    InterPairHMMComputer<AVX2DoubleTraits>::finalize(mtc);
    return true;
}
#endif
} // namespace inter
} // namespace pairhmm
