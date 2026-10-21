#include "preprocess.h"

namespace rodv {
namespace inference {

void preprocess_inception_scalar(const uint8_t *src, float *dst, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = static_cast<float>(src[i]) * (1.0f / 128.0f) - 1.0f;
  }
}

} // namespace inference
} // namespace rodv
