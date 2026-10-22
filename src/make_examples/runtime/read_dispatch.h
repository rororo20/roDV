#ifndef RODV_MAKE_EXAMPLES_READ_DISPATCH_H_
#define RODV_MAKE_EXAMPLES_READ_DISPATCH_H_

#include "blocking_queue.h"
#include "read_filter.h"
#include "runtime/reference_cache.h"
#include "runtime/region_pool.h"

namespace rodv {
namespace make_examples {

// Test double and, later, the htslib BAM reader. Only the dispatch thread
// calls next().
class ReadSource {
public:
  virtual ~ReadSource() = default;
  virtual bool next(int *tid, bam1_t **read) = 0;
};

// Slices the read stream into region_bp windows, copies kept reads, and emits
// RegionSource records with increasing source_id. Pushes nworkers finish
// markers; only the first carries the region count for the writer.
class ReadDispatch {
public:
  void run(ReadSource &source, ReferenceCache &cache, RegionPool &pool,
           BlockingQueue<std::shared_ptr<RegionSource>> &jobs, int64_t region_bp,
           const ReadFilter &filter, int nworkers, ReadSource *parent = nullptr);
};

} // namespace make_examples
} // namespace rodv

#endif
