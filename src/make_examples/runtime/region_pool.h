#ifndef RODV_MAKE_EXAMPLES_REGION_POOL_H_
#define RODV_MAKE_EXAMPLES_REGION_POOL_H_

#include "blocking_queue.h"
#include "region_pipeline.h"
#include "runtime/region_types.h"

#include <memory>

namespace rodv {
namespace make_examples {

// nthreads*2 reusable region slots, matching hc's n_resource.
class RegionPool {
public:
  explicit RegionPool(int nthreads)
      : capacity_(clamp_nthreads(nthreads) * 2), free_(capacity_) {
    for (int i = 0; i < capacity_; ++i) {
      auto resource = std::make_shared<RegionResource>();
      resource->id = i;
      free_.push(resource);
    }
  }

  std::shared_ptr<RegionResource> acquire() { return free_.pop(); }

  void release(std::shared_ptr<RegionResource> resource) {
    if (!resource) {
      return;
    }
    resource->reset();
    free_.push(std::move(resource));
  }

  int capacity() const { return capacity_; }

  int in_use() const { return capacity_ - static_cast<int>(free_.size()); }

private:
  int capacity_;
  BlockingQueue<std::shared_ptr<RegionResource>> free_;
};

} // namespace make_examples
} // namespace rodv

#endif
