#include "runtime/read_dispatch.h"

#include <algorithm>

#if defined(__linux__)
#include <pthread.h>
#endif

namespace rodv {
namespace make_examples {

void ReadDispatch::run(ReadSource &source, ReferenceCache &cache, RegionPool &pool,
                       BlockingQueue<std::shared_ptr<RegionSource>> &jobs,
                       int64_t region_bp, const ReadFilter &filter, int nworkers,
                       ReadSource *parent) {
#if defined(__linux__)
  pthread_setname_np(pthread_self(), "dv-dispatch");
#endif
  if (region_bp < 1) {
    region_bp = 1;
  }
  if (nworkers < 1) {
    nworkers = 1;
  }

  bool open = false;
  int open_tid = -1;
  int64_t open_start = 0;
  int64_t open_end = 0;
  std::shared_ptr<RegionResource> resource;
  int64_t next_id = 0;
  std::vector<bam1_t *> parent_reads;
  if (parent != nullptr) {
    int parent_tid = 0;
    bam1_t *parent_read = nullptr;
    while (parent->next(&parent_tid, &parent_read)) {
      if (parent_read == nullptr || !keep_read(parent_read, filter)) {
        continue;
      }
      bam1_t *copy = bam_init1();
      if (bam_copy1(copy, parent_read) == nullptr) {
        bam_destroy1(copy);
        continue;
      }
      copy->core.tid = parent_tid;
      parent_reads.push_back(copy);
    }
  }

  const auto emit = [&](bool force_empty_ok) {
    if (!open) {
      return;
    }
    open = false;
    if (resource->reads.empty() && !force_empty_ok) {
      pool.release(std::move(resource));
      return;
    }
    auto job = std::make_shared<RegionSource>();
    job->source_id = next_id++;
    job->tid = open_tid;
    job->start = open_start;
    job->end = open_end;
    const std::string bases = cache.get(open_tid);
    const int64_t begin = std::max<int64_t>(0, open_start);
    const int64_t end =
        std::min<int64_t>(static_cast<int64_t>(bases.size()), open_end);
    if (end > begin) {
      job->ref = bases.substr(static_cast<size_t>(begin),
                              static_cast<size_t>(end - begin));
    }
    job->resource = std::move(resource);
    for (bam1_t *pread : parent_reads) {
      if (pread->core.tid == job->tid && pread->core.pos >= job->start &&
          pread->core.pos < job->end) {
        bam1_t *copy = bam_init1();
        if (bam_copy1(copy, pread) != nullptr) {
          job->resource->parent_reads.push_back(copy);
        } else {
          bam_destroy1(copy);
        }
      }
    }
    jobs.push(std::move(job));
  };

  int tid = 0;
  bam1_t *read = nullptr;
  while (source.next(&tid, &read)) {
    if (read == nullptr || !keep_read(read, filter)) {
      continue;
    }
    const int64_t pos = read->core.pos;
    const int64_t window = (pos / region_bp) * region_bp;
    if (!open || tid != open_tid || pos >= open_end) {
      emit(false);
      open = true;
      open_tid = tid;
      open_start = window;
      open_end = window + region_bp;
      resource = pool.acquire();
    }
    bam1_t *copy = bam_init1();
    if (bam_copy1(copy, read) == nullptr) {
      bam_destroy1(copy);
      continue;
    }
    resource->reads.push_back(copy);
  }
  emit(false);

  auto finish = std::make_shared<RegionSource>();
  finish->finish = true;
  finish->source_id = next_id;
  jobs.push(std::move(finish));
  for (int i = 1; i < nworkers; ++i) {
    auto stop = std::make_shared<RegionSource>();
    stop->finish = true;
    stop->source_id = -1;
    jobs.push(std::move(stop));
  }
  for (bam1_t *read : parent_reads) {
    bam_destroy1(read);
  }
}

} // namespace make_examples
} // namespace rodv
