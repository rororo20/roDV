#include "preprocess.h"

#include "pairhmm/common/cpu_features.h"

namespace rodv {
namespace inference {

void preprocess_inception_avx2(const uint8_t *src, float *dst, size_t n);
void preprocess_inception_avx512(const uint8_t *src, float *dst, size_t n);

void preprocess_inception(const uint8_t *src, float *dst, size_t n) {
  using pairhmm::common::CpuFeatures;
  if (CpuFeatures::hasAVX512Support()) {
    preprocess_inception_avx512(src, dst, n);
  } else if (CpuFeatures::hasAVX2Support()) {
    preprocess_inception_avx2(src, dst, n);
  } else {
    preprocess_inception_scalar(src, dst, n);
  }
}

} // namespace inference
} // namespace rodv
