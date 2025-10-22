#include "pairhmm_inter.h"
#include "../common/context.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

using namespace pairhmm::common; // 使用 common 命名空间的 Context

namespace pairhmm {
namespace inter {

template <typename Traits>
void InterPairHMMComputer<Traits>::compute(const MultiTestCase<Traits> &tc) {

  uint32_t hap_lens[Traits::simd_width];
  for (uint32_t i = 0; i < Traits::simd_width; i++) {
    hap_lens[i] = tc.test_cases[i].haplen;
  }
  SimdType mm[tc.max_haplen + 1];
  SimdType ii[tc.max_haplen + 1];
  SimdType dd[tc.max_haplen + 1];

  initialize_matrices(tc, mm, ii, dd, hap_lens);
  return;
}
template <typename Traits>
void InterPairHMMComputer<Traits>::initialize_matrices(
    const MultiTestCase<Traits> &tc, SimdType *mm, SimdType *ii, SimdType *dd,
    uint32_t *hap_lens) {
  for (int i = 0; i <= tc.max_haplen; ++i) {
    mm[i] = Traits::setzero();
    ii[i] = Traits::setzero();
    dd[i] = Traits::set_init_d(hap_lens);
  }
}
// 显式实例化
#if defined(__AVX512F__)
template class InterPairHMMComputer<AVX512FloatTraits>;
template class InterPairHMMComputer<AVX512DoubleTraits>;
#elif defined(__AVX2__)
template class InterPairHMMComputer<AVX2FloatTraits>;
template class InterPairHMMComputer<AVX2DoubleTraits>;
#endif

} // namespace inter
} // namespace pairhmm
