#include "read_filter.h"
#include "snp_counter.h"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

struct BamDeleter {
  void operator()(bam1_t *read) const { bam_destroy1(read); }
};

std::unique_ptr<bam1_t, BamDeleter>
make_read(int64_t pos, const std::string &seq, const std::string &qual,
          std::vector<uint32_t> cigar = {}, uint16_t flag = 0,
          uint8_t mapq = 60) {
  if (cigar.empty()) {
    cigar.push_back(bam_cigar_gen(seq.size(), BAM_CMATCH));
  }
  // bam_set1 stores quality bytes unchanged. Callers pass FASTQ ASCII
  // (Phred+33); convert to the Phred values sam_read1 would produce.
  std::string raw_qual(qual.size(), '\0');
  for (size_t i = 0; i < qual.size(); ++i) {
    raw_qual[i] = static_cast<char>(qual[i] - 33);
  }
  auto read = std::unique_ptr<bam1_t, BamDeleter>(bam_init1());
  const int rc =
      bam_set1(read.get(), 1, "r", flag, 0, pos, mapq, cigar.size(),
               cigar.data(), -1, -1, 0, seq.size(), seq.data(),
               raw_qual.data(), 0);
  EXPECT_GE(rc, 0);
  return read;
}

uint32_t op(int len, int code) { return bam_cigar_gen(len, code); }

} // namespace

TEST(ReadFilterTest, DropsNonPrimaryLowMapqAndQcFail) {
  rodv::make_examples::ReadFilter filter;
  auto primary = make_read(0, "A", "=", {op(1, BAM_CMATCH)});
  EXPECT_TRUE(rodv::make_examples::keep_read(primary.get(), filter));

  auto low_mapq = make_read(0, "A", "=", {op(1, BAM_CMATCH)}, 0, 4);
  EXPECT_FALSE(rodv::make_examples::keep_read(low_mapq.get(), filter));

  auto dup = make_read(0, "A", "=", {op(1, BAM_CMATCH)}, BAM_FDUP);
  EXPECT_FALSE(rodv::make_examples::keep_read(dup.get(), filter));
  filter.keep_duplicates = true;
  EXPECT_TRUE(rodv::make_examples::keep_read(dup.get(), filter));

  auto secondary = make_read(0, "A", "=", {op(1, BAM_CMATCH)}, BAM_FSECONDARY);
  EXPECT_FALSE(rodv::make_examples::keep_read(secondary.get(), filter));

  auto supplementary =
      make_read(0, "A", "=", {op(1, BAM_CMATCH)}, BAM_FSUPPLEMENTARY);
  EXPECT_FALSE(rodv::make_examples::keep_read(supplementary.get(), filter));

  auto qcfail = make_read(0, "A", "=", {op(1, BAM_CMATCH)}, BAM_FQCFAIL);
  EXPECT_FALSE(rodv::make_examples::keep_read(qcfail.get(), filter));
}

TEST(SnpCounterTest, TwoAltReadsClearCountAndFraction) {
  const char *ref = "ACGTACGT";
  const int64_t start = 100;
  std::vector<rodv::make_examples::LocusCount> counts;
  // Position 103 is T. Two reads carry A, eight carry the reference.
  for (int i = 0; i < 2; ++i) {
    auto read = make_read(start, "ACGAACGT", ">>>>>>>>");
    rodv::make_examples::accumulate_snps(read.get(), start, ref, 8, counts);
  }
  for (int i = 0; i < 8; ++i) {
    auto read = make_read(start, "ACGTACGT", ">>>>>>>>");
    rodv::make_examples::accumulate_snps(read.get(), start, ref, 8, counts);
  }

  const auto found = rodv::make_examples::snp_candidates(counts, start, ref);
  ASSERT_EQ(found.size(), 1u);
  EXPECT_EQ(found[0].pos, 103);
  EXPECT_EQ(found[0].ref, 'T');
  EXPECT_EQ(found[0].alt, 'A');
  EXPECT_EQ(found[0].alt_count, 2);
  EXPECT_EQ(found[0].total, 10);
}

TEST(SnpCounterTest, LowQualityAltAndSingleAltAreNotCandidates) {
  const char *ref = "ACGT";
  std::vector<rodv::make_examples::LocusCount> counts;
  // Phred is encoded as ASCII+33. '*' is Phred 9, below the default of 10.
  auto low = make_read(0, "AGGT", ">*>>");
  rodv::make_examples::accumulate_snps(low.get(), 0, ref, 4, counts);
  // One high-quality alt is under min_count.
  auto once = make_read(0, "AGGT", ">>>>");
  rodv::make_examples::accumulate_snps(once.get(), 0, ref, 4, counts);

  EXPECT_EQ(counts[1].alt[2], 1); // G
  EXPECT_EQ(counts[1].ref, 0);
  EXPECT_TRUE(rodv::make_examples::snp_candidates(counts, 0, ref).empty());
}

TEST(SnpCounterTest, InsertionReplacesAnchorMatch) {
  const char *ref = "ACGTACGT";
  std::vector<rodv::make_examples::LocusCount> counts;
  // 3M 1I 5M. Inserted G is anchored on ref[2] and replaces that match.
  auto read = make_read(0, "ACGGTACGT", ">>>>>>>>>",
                        {op(3, BAM_CMATCH), op(1, BAM_CINS), op(5, BAM_CMATCH)});
  rodv::make_examples::accumulate_snps(read.get(), 0, ref, 8, counts);

  EXPECT_EQ(counts[2].ref, 0);
  ASSERT_EQ(counts[2].indels.size(), 1u);
  EXPECT_TRUE(counts[2].indels[0].insertion);
  EXPECT_EQ(counts[2].indels[0].bases, "GG");
  EXPECT_EQ(counts[2].total(), 1);
  EXPECT_EQ(counts[3].ref, 1);
  EXPECT_TRUE(rodv::make_examples::candidates(counts, 0, ref).empty());
}

TEST(SnpCounterTest, TwoInsertionsBecomeOneAlt) {
  const char *ref = "ACGTACGT";
  std::vector<rodv::make_examples::LocusCount> counts;
  const std::vector<uint32_t> cigar = {op(3, BAM_CMATCH), op(1, BAM_CINS),
                                       op(5, BAM_CMATCH)};
  for (int i = 0; i < 2; ++i) {
    auto read = make_read(0, "ACGGTACGT", ">>>>>>>>>", cigar);
    rodv::make_examples::accumulate_snps(read.get(), 0, ref, 8, counts);
  }
  const auto found = rodv::make_examples::candidates(counts, 0, ref);
  ASSERT_EQ(found.size(), 1u);
  EXPECT_EQ(found[0].pos, 2);
  EXPECT_EQ(found[0].kind, rodv::make_examples::AlleleKind::kInsertion);
  EXPECT_EQ(found[0].ref, "G");
  EXPECT_EQ(found[0].alt, "GG");
  EXPECT_EQ(found[0].alt_count, 2);
}

TEST(SnpCounterTest, TwoDeletionsPadTheReferenceAllele) {
  const char *ref = "ACGTACGT";
  std::vector<rodv::make_examples::LocusCount> counts;
  // 3M 1D 4M deletes ref[3] (T). Anchor is ref[2] (G): REF=GT ALT=G.
  const std::vector<uint32_t> cigar = {op(3, BAM_CMATCH), op(1, BAM_CDEL),
                                       op(4, BAM_CMATCH)};
  for (int i = 0; i < 2; ++i) {
    auto read = make_read(0, "ACGACGT", ">>>>>>>", cigar);
    rodv::make_examples::accumulate_snps(read.get(), 0, ref, 8, counts);
  }
  const auto found = rodv::make_examples::candidates(counts, 0, ref);
  ASSERT_EQ(found.size(), 1u);
  EXPECT_EQ(found[0].pos, 2);
  EXPECT_EQ(found[0].kind, rodv::make_examples::AlleleKind::kDeletion);
  EXPECT_EQ(found[0].ref, "GT");
  EXPECT_EQ(found[0].alt, "G");
  EXPECT_EQ(found[0].alt_count, 2);
}

TEST(SnpCounterTest, LowQualityInsertionAndSoftClipAreNotCandidates) {
  const char *ref = "ACGTACGT";
  std::vector<rodv::make_examples::LocusCount> counts;
  auto low = make_read(0, "ACGGTACGT", ">>>*>>>>>",
                       {op(3, BAM_CMATCH), op(1, BAM_CINS), op(5, BAM_CMATCH)});
  rodv::make_examples::accumulate_snps(low.get(), 0, ref, 8, counts);
  EXPECT_TRUE(counts[2].indels.empty());

  std::vector<rodv::make_examples::LocusCount> clipped;
  auto clip = make_read(0, "ACGTT", ">>>>>",
                        {op(3, BAM_CMATCH), op(2, BAM_CSOFT_CLIP)});
  rodv::make_examples::accumulate_snps(clip.get(), 0, ref, 8, clipped);
  EXPECT_EQ(clipped[2].ref, 0);
  EXPECT_EQ(clipped[2].soft_clip, 1);
  EXPECT_TRUE(rodv::make_examples::candidates(clipped, 0, ref).empty());
}
