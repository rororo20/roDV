#include "blocking_queue.h"
#include "region_pipeline.h"
#include "snp_region.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Job {
  int value = 0;
  int delay_ms = 0;
};

struct Result {
  int value = 0;
};

} // namespace

TEST(BlockingQueueTest, PreservesOrderUnderBackpressure) {
  rodv::make_examples::BlockingQueue<int> queue(2);
  std::thread producer([&] {
    for (int i = 0; i < 6; ++i) {
      queue.push(i);
    }
  });
  std::vector<int> got;
  for (int i = 0; i < 6; ++i) {
    got.push_back(queue.pop());
  }
  producer.join();
  EXPECT_EQ(got, (std::vector<int>{0, 1, 2, 3, 4, 5}));
}

TEST(RegionPipelineTest, SlowFirstJobStillEmitsInOrder) {
  std::vector<std::shared_ptr<Job>> jobs;
  for (int i = 0; i < 8; ++i) {
    auto job = std::make_shared<Job>();
    job->value = i;
    job->delay_ms = i == 0 ? 40 : 0;
    jobs.push_back(job);
  }

  const std::vector<Result> results =
      rodv::make_examples::run_ordered<Job, Result>(
          jobs,
          [](const Job &job) {
            if (job.delay_ms > 0) {
              std::this_thread::sleep_for(
                  std::chrono::milliseconds(job.delay_ms));
            }
            Result result;
            result.value = job.value * 10;
            return result;
          },
          4);

  ASSERT_EQ(results.size(), 8u);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(results[static_cast<size_t>(i)].value, i * 10);
  }
}

TEST(SnpRegionTest, ParallelCountsStayInRegionOrder) {
  const std::string ref = "ACGTACGT";
  std::vector<std::unique_ptr<bam1_t, void (*)(bam1_t *)>> storage;
  auto make = [&](int64_t pos, const std::string &seq) {
    auto read =
        std::unique_ptr<bam1_t, void (*)(bam1_t *)>(bam_init1(), bam_destroy1);
    const uint32_t cigar = bam_cigar_gen(seq.size(), BAM_CMATCH);
    std::string qual(seq.size(), static_cast<char>(30));
    EXPECT_GE(bam_set1(read.get(), 1, "r", 0, 0, pos, 60, 1, &cigar, -1, -1, 0,
                       seq.size(), seq.data(), qual.data(), 0),
              0);
    storage.push_back(std::move(read));
    return storage.back().get();
  };

  std::vector<rodv::make_examples::SnpRegionInput> regions(3);
  for (auto &region : regions) {
    region.start = 1000;
    region.ref = ref;
  }
  // Region 0: T>A at offset 3, twice. Region 1: no alt. Region 2: C>G at
  // offset 1, twice. Region 0 is listed first and must stay first.
  regions[0].reads = {make(1000, "ACGAACGT"), make(1000, "ACGAACGT")};
  regions[1].reads = {make(1000, "ACGTACGT"), make(1000, "ACGTACGT")};
  regions[2].reads = {make(1000, "AGGTACGT"), make(1000, "AGGTACGT")};

  const auto results =
      rodv::make_examples::count_snp_regions(regions, /*nthreads=*/3);
  ASSERT_EQ(results.size(), 3u);
  ASSERT_EQ(results[0].candidates.size(), 1u);
  EXPECT_EQ(results[0].candidates[0].pos, 1003);
  EXPECT_EQ(results[0].candidates[0].alt, "A");
  EXPECT_EQ(results[0].candidates[0].ref, "T");
  EXPECT_TRUE(results[1].candidates.empty());
  ASSERT_EQ(results[2].candidates.size(), 1u);
  EXPECT_EQ(results[2].candidates[0].pos, 1001);
  EXPECT_EQ(results[2].candidates[0].alt, "G");
  EXPECT_EQ(results[2].start, 1000);
}

TEST(SnpRegionTest, CountsTheRealignedInsertion) {
  rodv::make_examples::RealignerOptions realigner;
  realigner.min_k = 3;
  realigner.max_k = 3;
  const std::string ref = "ACGTAC";
  const uint32_t cigar[] = {bam_cigar_gen(5, BAM_CMATCH),
                            bam_cigar_gen(1, BAM_CINS),
                            bam_cigar_gen(1, BAM_CMATCH)};
  std::string qual(7, static_cast<char>(30));
  auto make = [&]() {
    auto read =
        std::unique_ptr<bam1_t, void (*)(bam1_t *)>(bam_init1(), bam_destroy1);
    EXPECT_GE(bam_set1(read.get(), 1, "r", 0, 0, 0, 60, 3, cigar, -1, -1, 0, 7,
                       "ACGTTAC", qual.data(), 0),
              0);
    return read;
  };
  auto left = make();
  auto right = make();
  rodv::make_examples::SnpRegionInput region;
  region.start = 0;
  region.ref = ref;
  region.reads = {left.get(), right.get()};

  const auto results = rodv::make_examples::count_snp_regions(
      {region}, /*nthreads=*/1, {}, 10, {}, realigner);
  ASSERT_EQ(results.size(), 1u);
  ASSERT_EQ(results[0].candidates.size(), 1u);
  EXPECT_EQ(results[0].candidates[0].kind,
            rodv::make_examples::AlleleKind::kInsertion);
  EXPECT_EQ(results[0].candidates[0].pos, 2);
  EXPECT_EQ(results[0].candidates[0].ref, "G");
  EXPECT_EQ(results[0].candidates[0].alt, "GT");
}
