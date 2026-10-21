#ifndef RODV_INFERENCE_PREPROCESS_H_
#define RODV_INFERENCE_PREPROCESS_H_

#include <cstddef>
#include <cstdint>

namespace rodv {
namespace inference {

// Matches deepvariant/dv_utils.py preprocess_images:
//   float(uint8 - 128) / 128   which is   uint8/128 - 1
// Division by 128 is exact in float32 for every uint8, so SIMD must match
// the scalar path bit-for-bit.
void preprocess_inception_scalar(const uint8_t *src, float *dst, size_t n);

// Dispatches to AVX-512, AVX2, or scalar. n may be any length.
void preprocess_inception(const uint8_t *src, float *dst, size_t n);

} // namespace inference
} // namespace rodv

#endif
