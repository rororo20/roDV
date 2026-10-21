#include "preprocess.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

void expect_matches_scalar(const std::vector<uint8_t> &src) {
  std::vector<float> got(src.size());
  std::vector<float> want(src.size());
  rodv::inference::preprocess_inception(src.data(), got.data(), src.size());
  rodv::inference::preprocess_inception_scalar(src.data(), want.data(),
                                              src.size());
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i], want[i]) << "index " << i;
  }
}

} // namespace

TEST(PreprocessTest, EndpointsAreExact) {
  const uint8_t src[] = {0, 128, 255};
  float dst[3] = {};
  rodv::inference::preprocess_inception(src, dst, 3);
  EXPECT_EQ(dst[0], -1.0f);
  EXPECT_EQ(dst[1], 0.0f);
  EXPECT_EQ(dst[2], 255.0f / 128.0f - 1.0f);
}

TEST(PreprocessTest, TailAndFullWgsImageMatchScalar) {
  std::vector<uint8_t> src(100 * 221 * 7);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<uint8_t>(i * 17 + 3);
  }
  expect_matches_scalar(src);

  for (size_t n = 0; n <= 64; ++n) {
    src.resize(n);
    for (size_t i = 0; i < n; ++i) {
      src[i] = static_cast<uint8_t>(i + 9);
    }
    expect_matches_scalar(src);
  }
}
