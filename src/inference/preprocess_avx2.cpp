#include "preprocess.h"

#include <immintrin.h>

namespace rodv {
namespace inference {

void preprocess_inception_avx2(const uint8_t *src, float *dst, size_t n) {
  const __m256 scale = _mm256_set1_ps(1.0f / 128.0f);
  const __m256 bias = _mm256_set1_ps(-1.0f);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i bytes = _mm_loadl_epi64(
        reinterpret_cast<const __m128i *>(src + i));
    const __m256i i32 = _mm256_cvtepu8_epi32(bytes);
    const __m256 f = _mm256_cvtepi32_ps(i32);
    _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(f, scale, bias));
  }
  preprocess_inception_scalar(src + i, dst + i, n - i);
}

} // namespace inference
} // namespace rodv
