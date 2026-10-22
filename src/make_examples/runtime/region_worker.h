#ifndef RODV_MAKE_EXAMPLES_REGION_WORKER_H_
#define RODV_MAKE_EXAMPLES_REGION_WORKER_H_

#include "blocking_queue.h"
#include "realigner.h"
#include "runtime/pileup_stage.h"
#include "runtime/region_pool.h"

namespace rodv {
namespace make_examples {

// One call is one worker loop. Share one instance across threads; it keeps
// no per-thread state. Does not touch an hts file.
class RegionWorker {
public:
  void run(BlockingQueue<std::shared_ptr<RegionSource>> &jobs,
           BlockingQueue<ExampleRecord> &results, RegionPool &pool,
           PileupStage &pileup, const RealignerOptions &realigner,
           const ReadFilter &filter, int min_base_quality,
           const CandidateThresholds &thresholds,
           const std::vector<TruthVariant> &truth = {}, bool gvcf = false,
           bool phase = false) const;
};

} // namespace make_examples
} // namespace rodv

#endif
