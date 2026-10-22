#include "runtime/pipeline.h"

#include "pileup.h"
#include "region_pipeline.h"

#include <thread>
#include <vector>

namespace rodv {
namespace make_examples {

MakeExamplesPipeline::MakeExamplesPipeline(PipelineConfig config,
                                           std::shared_ptr<PileupStage> pileup)
    : config_(std::move(config)),
      pileup_(pileup ? std::move(pileup) : std::make_shared<WgsPileup>()) {}

void MakeExamplesPipeline::run(ReadSource &reads, ReferenceCache &cache,
                               ExampleSink &sink, ReadSource *parent) {
  const int nthreads = clamp_nthreads(config_.nthreads);
  RegionPool pool(nthreads);
  BlockingQueue<std::shared_ptr<RegionSource>> jobs(
      static_cast<size_t>(nthreads) * 2);
  BlockingQueue<ExampleRecord> results(static_cast<size_t>(nthreads) * 2);

  ExampleWriter writer(results, sink);
  writer.start();

  RegionWorker worker;
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(nthreads));
  for (int i = 0; i < nthreads; ++i) {
    workers.emplace_back([&, this] {
      worker.run(jobs, results, pool, *pileup_, config_.realigner, config_.filter,
                 config_.min_base_quality, config_.thresholds, config_.truth,
                 config_.gvcf, config_.phase);
    });
  }

  std::thread dispatch([&] {
    ReadDispatch().run(reads, cache, pool, jobs, config_.region_bp, config_.filter,
                       nthreads, parent);
  });

  dispatch.join();
  for (std::thread &thread : workers) {
    thread.join();
  }
  writer.stop();
}

} // namespace make_examples
} // namespace rodv
