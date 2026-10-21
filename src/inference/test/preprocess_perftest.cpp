#include "preprocess.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
  constexpr int kHeight = 100;
  constexpr int kWidth = 221;
  constexpr int kChannels = 7;
  constexpr int kBatch = 16;
  constexpr int kIters = 200;
  const size_t n =
      static_cast<size_t>(kBatch) * kHeight * kWidth * kChannels;

  std::vector<uint8_t> src(n);
  std::vector<float> dst(n);
  for (size_t i = 0; i < n; ++i) {
    src[i] = static_cast<uint8_t>(i);
  }

  rodv::inference::preprocess_inception(src.data(), dst.data(), n);

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < kIters; ++i) {
    rodv::inference::preprocess_inception(src.data(), dst.data(), n);
  }
  const auto stop = std::chrono::steady_clock::now();
  const double sec = std::chrono::duration<double>(stop - start).count();
  const double examples = static_cast<double>(kIters) * kBatch;
  const double bytes =
      static_cast<double>(kIters) * static_cast<double>(n) * (1.0 + 4.0);
  std::printf("preprocess %dx%dx%d batch %d: %.1f examples/s, %.1f GB/s\n",
              kHeight, kWidth, kChannels, kBatch, examples / sec,
              bytes / sec / 1e9);
  // Keep the stores live so the timed loop cannot be deleted.
  volatile float sink = dst[0] + dst[n - 1];
  return sink == 0.0f ? 1 : 0;
}
