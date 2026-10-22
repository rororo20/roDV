#include "apply_alignment.h"
#include "runtime/example_writer.h"
#include "runtime/pipeline.h"
#include "runtime/read_dispatch.h"
#include "runtime/reference_cache.h"
#include "runtime/region_pool.h"
#include "runtime/region_worker.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

struct BamDeleter {
  void operator()(bam1_t *read) const { bam_destroy1(read); }
};

std::unique_ptr<bam1_t, BamDeleter> make_read(int64_t pos, const std::string &seq,
                                              uint16_t flag = 0,
                                              uint8_t mapq = 60) {
  auto read = std::unique_ptr<bam1_t, BamDeleter>(bam_init1());
  const uint32_t cigar = bam_cigar_gen(seq.size(), BAM_CMATCH);
  std::string qual(seq.size(), static_cast<char>(30));
  EXPECT_GE(bam_set1(read.get(), 1, "r", flag, 0, pos, mapq, 1, &cigar, -1, -1,
                     0, seq.size(), seq.data(), qual.data(), 0),
            0);
  return read;
}

class VectorReadSource : public rodv::make_examples::ReadSource {
public:
  explicit VectorReadSource(std::vector<bam1_t *> reads) : reads_(std::move(reads)) {}

  bool next(int *tid, bam1_t **read) override {
    if (index_ >= reads_.size()) {
      return false;
    }
    *tid = 0;
    *read = reads_[index_++];
    return true;
  }

private:
  std::vector<bam1_t *> reads_;
  size_t index_ = 0;
};

class SlowPileup : public rodv::make_examples::PileupStage {
public:
  std::vector<uint8_t>
  encode(int64_t source_id, int64_t, const std::vector<rodv::make_examples::Candidate> &,
         const std::string &, const std::vector<bam1_t *> &,
         const std::vector<bam1_t *> &, bool) override {
    if (source_id == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    return {static_cast<uint8_t>(source_id)};
  }
};

} // namespace

TEST(RegionPoolTest, RecyclesWithoutOverlap) {
  rodv::make_examples::RegionPool pool(1);
  EXPECT_EQ(pool.capacity(), 2);
  std::atomic<int> held{0};
  std::atomic<int> max_held{0};
  const auto loop = [&] {
    for (int i = 0; i < 20; ++i) {
      auto resource = pool.acquire();
      const int now = ++held;
      int seen = max_held.load();
      while (now > seen && !max_held.compare_exchange_weak(seen, now)) {
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      --held;
      pool.release(std::move(resource));
    }
  };
  std::thread left(loop);
  std::thread right(loop);
  left.join();
  right.join();
  EXPECT_LE(max_held.load(), pool.capacity());
  EXPECT_EQ(pool.in_use(), 0);
  auto again = pool.acquire();
  EXPECT_EQ(pool.in_use(), 1);
  pool.release(std::move(again));
}

TEST(ReferenceCacheTest, BlocksUntilContigReady) {
  rodv::make_examples::ReferenceCache cache;
  cache.hold_publish();
  cache.start({"ACGT"});
  std::atomic<bool> ready{false};
  std::string bases;
  std::thread waiter([&] {
    bases = cache.get(0);
    ready = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_FALSE(ready.load());
  cache.release_publish();
  waiter.join();
  EXPECT_TRUE(ready.load());
  EXPECT_EQ(bases, "ACGT");
  cache.join();
}

TEST(ReadDispatchTest, IdsIncreaseAndReadsAreCopies) {
  auto a = make_read(0, "AAAA");
  auto b = make_read(1000, "AAAA");
  auto c = make_read(2000, "AAAA");
  VectorReadSource source({a.get(), b.get(), c.get()});
  rodv::make_examples::ReferenceCache cache;
  cache.start({std::string(3000, 'A')});
  rodv::make_examples::RegionPool pool(2);
  rodv::make_examples::BlockingQueue<std::shared_ptr<rodv::make_examples::RegionSource>>
      jobs(4);
  rodv::make_examples::ReadDispatch().run(source, cache, pool, jobs, 1000, {}, 1);

  std::vector<std::shared_ptr<rodv::make_examples::RegionSource>> got;
  for (int i = 0; i < 4; ++i) {
    got.push_back(jobs.pop());
  }
  ASSERT_EQ(got.size(), 4u);
  EXPECT_EQ(got[0]->source_id, 0);
  EXPECT_EQ(got[1]->source_id, 1);
  EXPECT_EQ(got[2]->source_id, 2);
  EXPECT_TRUE(got[3]->finish);
  EXPECT_EQ(got[3]->source_id, 3);

  const uint16_t original = a->core.flag;
  got[0]->resource->reads[0]->core.flag = BAM_FDUP;
  EXPECT_EQ(a->core.flag, original);
  EXPECT_NE(got[0]->resource->reads[0]->core.flag, original);
  cache.join();
}

TEST(ReadDispatchTest, DropsFilteredReads) {
  auto good = make_read(0, "AAAA");
  auto dup = make_read(10, "AAAA", BAM_FDUP);
  auto low = make_read(20, "AAAA", 0, 4);
  VectorReadSource source({good.get(), dup.get(), low.get()});
  rodv::make_examples::ReferenceCache cache;
  cache.start({std::string(1000, 'A')});
  rodv::make_examples::RegionPool pool(1);
  rodv::make_examples::BlockingQueue<std::shared_ptr<rodv::make_examples::RegionSource>>
      jobs(4);
  rodv::make_examples::ReadDispatch().run(source, cache, pool, jobs, 1000, {}, 1);

  auto region = jobs.pop();
  auto finish = jobs.pop();
  ASSERT_FALSE(region->finish);
  ASSERT_EQ(region->resource->reads.size(), 1u);
  EXPECT_EQ(region->resource->reads[0]->core.pos, 0);
  EXPECT_TRUE(finish->finish);
  EXPECT_EQ(finish->source_id, 1);
  cache.join();
}

TEST(ExampleWriterTest, EmitsInSourceIdOrder) {
  rodv::make_examples::BlockingQueue<rodv::make_examples::ExampleRecord> queue(4);
  rodv::make_examples::MemorySink sink;
  rodv::make_examples::ExampleWriter writer(queue, sink);
  writer.start();

  rodv::make_examples::ExampleRecord second;
  second.source_id = 1;
  second.start = 1000;
  rodv::make_examples::ExampleRecord first;
  first.source_id = 0;
  first.start = 0;
  queue.push(std::move(second));
  queue.push(std::move(first));
  rodv::make_examples::ExampleRecord finish;
  finish.finish = true;
  finish.source_id = 2;
  queue.push(std::move(finish));
  writer.stop();

  const auto records = sink.records();
  ASSERT_EQ(records.size(), 2u);
  EXPECT_EQ(records[0].source_id, 0);
  EXPECT_EQ(records[1].source_id, 1);
}

TEST(RegionWorkerTest, MatchesInMemoryCounter) {
  auto left = make_read(100, "ACGAACGT");
  auto right = make_read(100, "ACGAACGT");
  rodv::make_examples::RegionPool pool(1);
  auto resource = pool.acquire();
  bam1_t *copies[] = {bam_init1(), bam_init1()};
  ASSERT_NE(bam_copy1(copies[0], left.get()), nullptr);
  ASSERT_NE(bam_copy1(copies[1], right.get()), nullptr);
  resource->reads = {copies[0], copies[1]};

  auto job = std::make_shared<rodv::make_examples::RegionSource>();
  job->source_id = 0;
  job->start = 100;
  job->end = 108;
  job->ref = "ACGTACGT";
  job->resource = resource;

  rodv::make_examples::BlockingQueue<std::shared_ptr<rodv::make_examples::RegionSource>>
      jobs(2);
  rodv::make_examples::BlockingQueue<rodv::make_examples::ExampleRecord> results(2);
  jobs.push(job);
  auto finish = std::make_shared<rodv::make_examples::RegionSource>();
  finish->finish = true;
  finish->source_id = 1;
  jobs.push(finish);

  rodv::make_examples::PileupStage pileup;
  std::thread worker([&] {
    rodv::make_examples::RegionWorker().run(jobs, results, pool, pileup, {}, {}, 10,
                                            {});
  });
  auto record = results.pop();
  auto end = results.pop();
  worker.join();
  EXPECT_TRUE(end.finish);

  std::vector<const bam1_t *> raw = {left.get(), right.get()};
  const auto realigned =
      rodv::make_examples::realign_reads(job->ref, job->start, raw, {});
  std::vector<rodv::make_examples::BamPtr> owned;
  std::vector<const bam1_t *> countable;
  for (size_t i = 0; i < raw.size(); ++i) {
    auto copy = rodv::make_examples::with_alignment(raw[i], realigned[i]);
    if (copy) {
      countable.push_back(copy.get());
      owned.push_back(std::move(copy));
    } else {
      countable.push_back(raw[i]);
    }
  }
  std::vector<rodv::make_examples::LocusCount> counts;
  for (const bam1_t *read : countable) {
    rodv::make_examples::accumulate_snps(read, 100, job->ref.data(), 8, counts);
  }
  const auto expected = rodv::make_examples::candidates(counts, 100, job->ref.data());
  ASSERT_EQ(record.candidates.size(), expected.size());
  ASSERT_EQ(record.candidates.size(), 1u);
  EXPECT_EQ(record.candidates[0].pos, expected[0].pos);
  EXPECT_EQ(record.candidates[0].alt, expected[0].alt);
  EXPECT_EQ(record.candidates[0].ref, expected[0].ref);
}

TEST(PipelineTest, SlowFirstRegionStillWritesFirst) {
  auto a = make_read(0, "AAAA");
  auto b = make_read(1000, "CCCC");
  auto c = make_read(2000, "GGGG");
  VectorReadSource source({a.get(), b.get(), c.get()});
  rodv::make_examples::ReferenceCache cache;
  cache.start({std::string(3000, 'A')});
  rodv::make_examples::PipelineConfig config;
  config.nthreads = 2;
  config.region_bp = 1000;
  rodv::make_examples::MakeExamplesPipeline pipeline(
      config, std::make_shared<SlowPileup>());
  rodv::make_examples::MemorySink sink;
  pipeline.run(source, cache, sink);
  cache.join();

  const auto records = sink.records();
  ASSERT_EQ(records.size(), 3u);
  EXPECT_EQ(records[0].source_id, 0);
  EXPECT_EQ(records[1].source_id, 1);
  EXPECT_EQ(records[2].source_id, 2);
  EXPECT_EQ(records[0].tensors, std::vector<uint8_t>{0});
  EXPECT_EQ(records[2].tensors, std::vector<uint8_t>{2});
}
