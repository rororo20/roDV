#ifndef PAIRHMM_INTER_SIMD_TRAITS_AVX2_H_
#define PAIRHMM_INTER_SIMD_TRAITS_AVX2_H_

#include "../common/common.h"
#include "../common/context.h"
#include <cstdint>
#include <immintrin.h>
#include <x86intrin.h>

using namespace pairhmm::common;
namespace pairhmm {
namespace inter {

/**
 * @brief AVX2 SIMD Traits for Inter-PairHMM
 *
 * 支持多个 reads 与多个单倍型的并发计算
 * 使用 AVX2 指令集（256-bit）
 */

// ============================================================================
// AVX2 Float Traits (256-bit, 8 个 float)
// ============================================================================
struct AVX2FloatTraits {
  using MainType = float;
  using SimdType = __m256;
  using SimdIntType = __m256i;
  using MaskType = uint32_t;

  static constexpr uint32_t simd_width = 8;
  static constexpr uint32_t simd_bits = 256;
  static constexpr uint32_t alignment = 32;
  // 基础 SIMD 操作
  static inline SimdType set1(MainType v) { return _mm256_set1_ps(v); }
  static inline SimdType setzero() { return _mm256_setzero_ps(); }
  static inline SimdType add(SimdType a, SimdType b) {
    return _mm256_add_ps(a, b);
  }
  static inline SimdType sub(SimdType a, SimdType b) {
    return _mm256_sub_ps(a, b);
  }
  static inline SimdType mul(SimdType a, SimdType b) {
    return _mm256_mul_ps(a, b);
  }
  static inline SimdType div(SimdType a, SimdType b) {
    return _mm256_div_ps(a, b);
  }
  static inline SimdType load(const MainType *ptr) {
    return _mm256_load_ps(ptr);
  }
  static inline void store(MainType *ptr, SimdType v) {
    _mm256_store_ps(ptr, v);
  }
  static inline SimdType set_init_d(const uint32_t *hap_lens) {
    MainType init_const = Context<MainType>::INITIAL_CONSTANT;
    return _mm256_set_ps(init_const / static_cast<MainType>(hap_lens[7]),
                         init_const / static_cast<MainType>(hap_lens[6]),
                         init_const / static_cast<MainType>(hap_lens[5]),
                         init_const / static_cast<MainType>(hap_lens[4]),
                         init_const / static_cast<MainType>(hap_lens[3]),
                         init_const / static_cast<MainType>(hap_lens[2]),
                         init_const / static_cast<MainType>(hap_lens[1]),
                         init_const / static_cast<MainType>(hap_lens[0]));
  }
  // 特殊操作：用于生成长度掩码
  static inline SimdIntType generate_length_mask(uint32_t read_idx,
                                                 const uint32_t *lens) {
    __m256i idx = _mm256_set1_epi32(read_idx);
    __m256i len = _mm256_set_epi32(lens[7], lens[6], lens[5], lens[4], lens[3],
                                   lens[2], lens[1], lens[0]);
    return _mm256_cmpgt_epi32(len, idx);
  }
};

// ============================================================================
// AVX2 Double Traits (256-bit, 4 个 double)
// ============================================================================
struct AVX2DoubleTraits {
  using MainType = double;
  using SimdType = __m256d;
  using SimdIntType = __m256i;
  using MaskType = uint64_t;

  static constexpr uint32_t simd_width = 4;
  static constexpr uint32_t simd_bits = 256;
  // 基础 SIMD 操作
  static inline SimdType set1(MainType v) { return _mm256_set1_pd(v); }
  static inline SimdType setzero() { return _mm256_setzero_pd(); }
  static inline SimdType add(SimdType a, SimdType b) {
    return _mm256_add_pd(a, b);
  }
  static inline SimdType sub(SimdType a, SimdType b) {
    return _mm256_sub_pd(a, b);
  }
  static inline SimdType mul(SimdType a, SimdType b) {
    return _mm256_mul_pd(a, b);
  }
  static inline SimdType div(SimdType a, SimdType b) {
    return _mm256_div_pd(a, b);
  }
  static inline SimdType load(const MainType *ptr) {
    return _mm256_load_pd(ptr);
  }
  static inline void store(MainType *ptr, SimdType v) {
    _mm256_store_pd(ptr, v);
  }
  static inline SimdType set_init_d(const uint32_t *hap_lens) {
    MainType init_const = Context<MainType>::INITIAL_CONSTANT;
    return _mm256_set_pd(init_const / static_cast<MainType>(hap_lens[0]),
                         init_const / static_cast<MainType>(hap_lens[1]),
                         init_const / static_cast<MainType>(hap_lens[2]),
                         init_const / static_cast<MainType>(hap_lens[3]));
  }
  // 特殊操作：用于生成长度掩码
  static inline SimdIntType generate_length_mask(uint32_t read_idx,
                                                 const uint32_t *lens) {
    __m256i idx = _mm256_set1_epi64x(read_idx);
    __m256i len = _mm256_set_epi64x(
        static_cast<uint64_t>(lens[3]), static_cast<uint64_t>(lens[2]),
        static_cast<uint64_t>(lens[1]), static_cast<uint64_t>(lens[0]));
    return _mm256_cmpgt_epi64(len, idx);
  }

  
};

} // namespace inter
} // namespace pairhmm

#endif // PAIRHMM_INTER_SIMD_TRAITS_AVX2_H_
