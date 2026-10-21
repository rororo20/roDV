#include "preprocess.h"

#include <immintrin.h>

namespace rodv {
namespace inference {

void preprocess_inception_avx512(const uint8_t *src, float *dst, size_t n) {
  const __m512 scale = _mm512_set1_ps(1.0f / 128.0f);
  const __m512 bias = _mm512_set1_ps(-1.0f);
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m128i bytes = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(src + i));
    const __m512i i32 = _mm512_cvtepu8_epi32(bytes);
    const __m512 f = _mm512_cvtepi32_ps(i32);
    _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(f, scale, bias));
  }
  preprocess_inception_scalar(src + i, dst + i, n - i);
}

} // namespace inference
} // namespace rodv
