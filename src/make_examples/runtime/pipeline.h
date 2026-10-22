#ifndef RODV_MAKE_EXAMPLES_PIPELINE_H_
#define RODV_MAKE_EXAMPLES_PIPELINE_H_

#include "read_filter.h"
#include "realigner.h"
#include "runtime/example_writer.h"
#include "runtime/read_dispatch.h"
#include "runtime/reference_cache.h"
#include "runtime/region_worker.h"

#include <memory>

namespace rodv {
namespace make_examples {

struct PipelineConfig {
  int nthreads = 1;
  int64_t region_bp = 1000;
  ReadFilter filter{};
  RealignerOptions realigner{};
  CandidateThresholds thresholds{};
  int min_base_quality = 10;
  bool gvcf = false;
  bool phase = false;
  std::vector<TruthVariant> truth;
};

// Starts dispatch, workers, and the writer. ReferenceCache must already be
// start()'d. Does not run call_variants.
class MakeExamplesPipeline {
public:
  MakeExamplesPipeline(PipelineConfig config,
                       std::shared_ptr<PileupStage> pileup);

  void run(ReadSource &reads, ReferenceCache &cache, ExampleSink &sink,
           ReadSource *parent = nullptr);

private:
  PipelineConfig config_;
  std::shared_ptr<PileupStage> pileup_;
};

} // namespace make_examples
} // namespace rodv

#endif
