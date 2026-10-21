#include "realigner.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace {

struct BamDeleter {
  void operator()(bam1_t *read) const { bam_destroy1(read); }
};

std::unique_ptr<bam1_t, BamDeleter>
make_read(int64_t pos, const std::string &seq, const std::vector<uint32_t> &cigar) {
  auto read = std::unique_ptr<bam1_t, BamDeleter>(bam_init1());
  std::string qual(seq.size(), static_cast<char>(30));
  EXPECT_GE(bam_set1(read.get(), 1, "r", 0, 0, pos, 60, cigar.size(),
                     cigar.data(), -1, -1, 0, seq.size(), seq.data(),
                     qual.data(), 0),
            0);
  return read;
}

uint32_t op(int len, int code) { return bam_cigar_gen(len, code); }

std::string cigar_text(const std::vector<uint32_t> &cigar) {
  std::string text;
  for (uint32_t raw : cigar) {
    text += std::to_string(bam_cigar_oplen(raw));
    const char codes[] = "MIDNSHP=X";
    text.push_back(codes[bam_cigar_op(raw)]);
  }
  return text;
}

} // namespace

TEST(RealignerTest, WindowsMergeNearbyCandidates) {
  const auto windows =
      rodv::make_examples::assembly_windows({10, 20, 300}, 80, 1000);
  ASSERT_EQ(windows.size(), 2u);
  EXPECT_EQ(windows[0], (std::pair<int64_t, int64_t>{0, 100}));
  EXPECT_EQ(windows[1], (std::pair<int64_t, int64_t>{220, 380}));
}

TEST(RealignerTest, AffineAlignmentPlacesTheInsertion) {
  rodv::make_examples::RealignerOptions options;
  const auto cigar =
      rodv::make_examples::align_affine("ACGTTAC", "ACGTAC", options);
  ASSERT_EQ(cigar.size(), 3u);
  EXPECT_EQ(cigar[0].op, BAM_CMATCH);
  EXPECT_EQ(cigar[1].op, BAM_CINS);
  EXPECT_EQ(cigar[1].length, 1);
  EXPECT_EQ(cigar[0].length + cigar[2].length, 6);
}

TEST(RealignerTest, ComposeProjectsReadThroughHaplotype) {
  const std::vector<rodv::make_examples::CigarOp> read_to_hap = {
      rodv::make_examples::align_op(BAM_CMATCH, 7)};
  const std::vector<rodv::make_examples::CigarOp> hap_to_ref = {
      rodv::make_examples::align_op(BAM_CMATCH, 4),
      rodv::make_examples::align_op(BAM_CINS, 1),
      rodv::make_examples::align_op(BAM_CMATCH, 2)};
  const auto cigar = rodv::make_examples::compose_cigars(read_to_hap, hap_to_ref);
  ASSERT_EQ(cigar.size(), 3u);
  EXPECT_EQ(cigar[0].length, 4);
  EXPECT_EQ(cigar[1].op, BAM_CINS);
  EXPECT_EQ(cigar[1].length, 1);
  EXPECT_EQ(cigar[2].length, 2);
}

TEST(RealignerTest, AssemblyFindsTheInsertedHaplotype) {
  rodv::make_examples::RealignerOptions options;
  options.min_k = 3;
  options.max_k = 3;
  options.min_edge_weight = 2;
  const std::string ref = "ACGTAC";
  auto left = make_read(0, "ACGTTAC", {op(4, BAM_CMATCH), op(1, BAM_CINS),
                                       op(2, BAM_CMATCH)});
  auto right = make_read(0, "ACGTTAC", {op(4, BAM_CMATCH), op(1, BAM_CINS),
                                        op(2, BAM_CMATCH)});
  const auto haplotypes = rodv::make_examples::assemble_haplotypes(
      ref, {left.get(), right.get()}, options);
  ASSERT_EQ(haplotypes.size(), 2u);
  EXPECT_EQ(haplotypes[0], "ACGTAC");
  EXPECT_EQ(haplotypes[1], "ACGTTAC");
}

TEST(RealignerTest, ShiftedInsertionIsMovedBack) {
  rodv::make_examples::RealignerOptions options;
  options.min_k = 3;
  options.max_k = 3;
  options.min_edge_weight = 2;
  const std::string ref = "ACGTAC";
  // 5M1I puts the extra base at the end. Assembly + alignment should move it
  // next to the reference T.
  auto left = make_read(0, "ACGTTAC", {op(5, BAM_CMATCH), op(1, BAM_CINS),
                                       op(1, BAM_CMATCH)});
  auto right = make_read(0, "ACGTTAC", {op(5, BAM_CMATCH), op(1, BAM_CINS),
                                        op(1, BAM_CMATCH)});
  const auto aligned = rodv::make_examples::realign_reads(
      ref, 0, {left.get(), right.get()}, options);
  ASSERT_EQ(aligned.size(), 2u);
  EXPECT_TRUE(aligned[0].realigned);
  EXPECT_EQ(aligned[0].pos, 0);
  // 5M1I1M parks the extra T at the right. Affine alignment left-aligns it
  // against the reference T, which is 3M1I3M rather than 4M1I2M.
  EXPECT_EQ(cigar_text(aligned[0].cigar), "3M1I3M");
  EXPECT_EQ(cigar_text(aligned[1].cigar), "3M1I3M");
}
