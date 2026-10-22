#include "pileup.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace {

std::unique_ptr<bam1_t, void (*)(bam1_t *)> make_read(const std::string &seq) {
  auto read = std::unique_ptr<bam1_t, void (*)(bam1_t *)>(bam_init1(), bam_destroy1);
  const uint32_t cigar = bam_cigar_gen(seq.size(), BAM_CMATCH);
  std::string qual(seq.size(), static_cast<char>(30));
  EXPECT_GE(bam_set1(read.get(), 1, "r", 0, 0, 100, 60, 1, &cigar, -1, -1, 0,
                     seq.size(), seq.data(), qual.data(), 0),
            0);
  return read;
}

} // namespace

TEST(PileupTest, ReferenceBaseAtTheVariantIsTheCenterPixel) {
  rodv::make_examples::Candidate candidate;
  candidate.pos = 103;
  candidate.kind = rodv::make_examples::AlleleKind::kSnp;
  candidate.ref = "T";
  candidate.alt = "A";
  const std::string ref = "ACGT";
  auto read = make_read("ACGA");
  std::vector<bam1_t *> reads = {read.get()};
  const auto image = rodv::make_examples::WgsPileup().encode_one(
      candidate, /*ref_start=*/100, ref, reads);
  constexpr int kWidth = rodv::make_examples::WgsPileup::kWidth;
  constexpr int kChannels = rodv::make_examples::WgsPileup::kOutputChannels;
  const int center = (kWidth - 1) / 2;
  const uint8_t *pixel = image.data() + center * kChannels;
  EXPECT_EQ(pixel[0], 100); // T
  EXPECT_EQ(pixel[1], 254); // reference base quality capped at 40
  EXPECT_EQ(pixel[3], 70);  // positive strand
  EXPECT_EQ(pixel[5], static_cast<uint8_t>(254.0f * 0.2f));

  const uint8_t *read_pixel =
      image.data() + (5 * kWidth + (103 - (103 - center))) * kChannels;
  EXPECT_EQ(read_pixel[0], 250); // read base A at the variant
  EXPECT_EQ(read_pixel[4], 254); // supports alt
}
