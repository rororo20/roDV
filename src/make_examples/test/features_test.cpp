#include "cigar_norm.h"
#include "pileup.h"
#include "runtime/pipeline.h"
#include "runtime/read_dispatch.h"
#include "runtime/region_pool.h"
#include "runtime/region_worker.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

std::unique_ptr<bam1_t, void (*)(bam1_t *)>
make_read(int64_t pos, const std::string &seq, std::vector<uint32_t> cigar = {}) {
  auto read = std::unique_ptr<bam1_t, void (*)(bam1_t *)>(bam_init1(), bam_destroy1);
  if (cigar.empty()) {
    cigar.push_back(bam_cigar_gen(seq.size(), BAM_CMATCH));
  }
  std::string qual(seq.size(), static_cast<char>(30));
  EXPECT_GE(bam_set1(read.get(), 1, "r", 0, 0, pos, 60, cigar.size(), cigar.data(),
                     -1, -1, 0, seq.size(), seq.data(), qual.data(), 0),
            0);
  return read;
}

std::string cigar_text(const bam1_t *read) {
  std::string text;
  const uint32_t *cigar = bam_get_cigar(read);
  for (uint32_t i = 0; i < read->core.n_cigar; ++i) {
    text += std::to_string(bam_cigar_oplen(cigar[i]));
    text.push_back("MIDNSHP=X"[bam_cigar_op(cigar[i])]);
  }
  return text;
}

} // namespace

TEST(CigarNormTest, ShiftsInsertionLeftThroughTheRepeat) {
  auto read = make_read(0, "ACGTT", {bam_cigar_gen(4, BAM_CMATCH), bam_cigar_gen(1, BAM_CINS)});
  EXPECT_TRUE(rodv::make_examples::left_align_cigar(read.get(), 0, "ACGT"));
  EXPECT_EQ(cigar_text(read.get()), "3M1I1M");
}

TEST(PileupFeatureTest, AltImageCarriesTheAlternateBase) {
  rodv::make_examples::Candidate candidate;
  candidate.pos = 2;
  candidate.kind = rodv::make_examples::AlleleKind::kInsertion;
  candidate.ref = "G";
  candidate.alt = "GT";
  const auto image = rodv::make_examples::WgsPileup().encode_one(
      candidate, 0, "ACGTAC", {}, {});
  constexpr int kWidth = rodv::make_examples::WgsPileup::kWidth;
  constexpr int kOut = rodv::make_examples::WgsPileup::kOutputChannels;
  const int center = (kWidth - 1) / 2;
  const uint8_t *alt = image.data() + center * kOut + 7;
  EXPECT_EQ(alt[0], 180); // anchor G on the alt-aligned reference row
  const uint8_t *inserted = image.data() + (center + 1) * kOut + 7;
  EXPECT_EQ(inserted[0], 100); // inserted T
}

TEST(PileupFeatureTest, PhasePutsAltReadAboveRefRead) {
  auto alt = make_read(100, "A");
  auto ref = make_read(90, "ACGTACGT");
  rodv::make_examples::Candidate candidate;
  candidate.pos = 100;
  candidate.kind = rodv::make_examples::AlleleKind::kSnp;
  candidate.ref = "C";
  candidate.alt = "A";
  const auto image = rodv::make_examples::WgsPileup().encode_one(
      candidate, 90, "ACGTACGT", {ref.get(), alt.get()}, {}, true);
  constexpr int kWidth = rodv::make_examples::WgsPileup::kWidth;
  constexpr int kOut = rodv::make_examples::WgsPileup::kOutputChannels;
  const int center = (kWidth - 1) / 2;
  const uint8_t *first_read =
      image.data() + (5 * kWidth + center) * kOut;
  EXPECT_EQ(first_read[0], 250); // A from the alt-supporting read
}

TEST(PileupFeatureTest, DeepTrioDrawsParentBelowTheChild) {
  auto child = make_read(100, "A");
  auto parent = make_read(100, "C");
  rodv::make_examples::Candidate candidate;
  candidate.pos = 100;
  candidate.kind = rodv::make_examples::AlleleKind::kSnp;
  candidate.ref = "A";
  candidate.alt = "C";
  std::vector<bam1_t *> parents = {parent.get()};
  const auto image = rodv::make_examples::WgsPileup().encode_one(
      candidate, 100, "AC", {child.get()}, parents, false);
  constexpr int kWidth = rodv::make_examples::WgsPileup::kWidth;
  constexpr int kOut = rodv::make_examples::WgsPileup::kOutputChannels;
  const int center = (kWidth - 1) / 2;
  const int parent_row = rodv::make_examples::WgsPileup::kRefRows +
                         (rodv::make_examples::WgsPileup::kHeight -
                          rodv::make_examples::WgsPileup::kRefRows) /
                             2;
  const uint8_t *parent_pixel =
      image.data() + (static_cast<size_t>(parent_row) * kWidth + center) * kOut;
  EXPECT_EQ(parent_pixel[0], 30); // C
}

TEST(CallModeTest, LabelsAndGvcfBlocks) {
  auto ref_read = make_read(0, "AAAA");
  auto alt_read = make_read(0, "AAAT");
  auto alt_read2 = make_read(0, "AAAT");
  rodv::make_examples::RegionPool pool(1);
  auto resource = pool.acquire();
  bam1_t *copies[] = {bam_init1(), bam_init1(), bam_init1()};
  ASSERT_NE(bam_copy1(copies[0], ref_read.get()), nullptr);
  ASSERT_NE(bam_copy1(copies[1], alt_read.get()), nullptr);
  ASSERT_NE(bam_copy1(copies[2], alt_read2.get()), nullptr);
  resource->reads = {copies[0], copies[1], copies[2]};
  auto job = std::make_shared<rodv::make_examples::RegionSource>();
  job->source_id = 0;
  job->tid = 0;
  job->start = 0;
  job->end = 4;
  job->ref = "AAAA";
  job->resource = resource;

  rodv::make_examples::BlockingQueue<std::shared_ptr<rodv::make_examples::RegionSource>>
      jobs(2);
  rodv::make_examples::BlockingQueue<rodv::make_examples::ExampleRecord> results(2);
  jobs.push(job);
  auto finish = std::make_shared<rodv::make_examples::RegionSource>();
  finish->finish = true;
  finish->source_id = 1;
  jobs.push(finish);

  rodv::make_examples::TruthVariant truth;
  truth.pos = 3;
  truth.alt = "T";
  truth.genotype = 1;
  rodv::make_examples::PileupStage pileup;
  std::thread worker([&] {
    rodv::make_examples::RegionWorker().run(jobs, results, pool, pileup, {}, {}, 10,
                                            {}, {truth}, true, false);
  });
  auto record = results.pop();
  results.pop();
  worker.join();
  ASSERT_EQ(record.candidates.size(), 1u);
  EXPECT_EQ(record.candidates[0].alt, "T");
  ASSERT_EQ(record.labels.size(), 1u);
  EXPECT_EQ(record.labels[0], 1);
  EXPECT_FALSE(record.ref_blocks.empty());
  EXPECT_EQ(record.ref_blocks[0].start, 0);
}
