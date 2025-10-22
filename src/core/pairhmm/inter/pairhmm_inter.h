#ifndef PAIRHMM_INTER_H_
#define PAIRHMM_INTER_H_

#include "../common/common.h"
#include "simd_traits.h"
#include <cstdint>

namespace pairhmm {
namespace inter {

using namespace pairhmm::common;

template <typename Traits> struct MultiTestCase {
  int min_haplen;
  int max_haplen;
  int min_rslen;
  int max_rslen;
  TestCase test_cases[Traits::simd_width];
};

/**
 * @brief Inter-PairHMM 计算器
 *
 * 使用模板实现多个 reads 与多个单倍型的并发计算
 * 支持 AVX2 和 AVX512 指令集
 *
 * @tparam Traits SIMD 特征类
 */
template <typename Traits> class InterPairHMMComputer {
public:
  using MainType = typename Traits::MainType;
  using SimdType = typename Traits::SimdType;
  using SimdIntType = typename Traits::SimdIntType;
  using MaskType = typename Traits::MaskType;

  static constexpr uint32_t simd_width = Traits::simd_width;

  static void compute(const MultiTestCase<Traits> &tc);

private:
  static void initialize_matrices(const MultiTestCase<Traits> &tc, SimdType *mm,
                                  SimdType *ii, SimdType *dd,
                                  uint32_t *hap_lens);
};

} // namespace inter
} // namespace pairhmm

#endif // PAIRHMM_INTER_H_
